#!/usr/bin/env python3
"""Integrated Gradients (IG) interpretability across modality, category, and feature levels."""

from __future__ import annotations

import csv
from fnmatch import fnmatchcase
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "01_metadata_selection"))
sys.path.insert(0, str(PROJECT_ROOT / "04_Learning"))

from data_loader import DomainDataset, PAD_INDEX, VOCAB_SIZE, resolve_dataset_config  # noqa: E402
import Fusion  # noqa: E402
import Metadata_MLP  # noqa: E402

METHOD_ABBR = "IG"
IG_STEPS = 64
BATCH_SIZE = 128
# ponytail: fixed 64 Riemann steps and zero/PAD baselines; add configurable steps/baselines in config.yml when needed.


def forward_fusion_from_embeds(model, char_embed, metadata):
    x = char_embed.permute(0, 2, 1)
    text_feat = model.text_encoder.dropout(
        torch.cat(
            [
                F.max_pool1d(F.relu(conv(x)), kernel_size=x.size(2)).squeeze(2)
                for conv in model.text_encoder.convs_1d
            ],
            dim=1,
        )
    )
    meta_feat = model.metadata_encoder.extract_features(metadata)
    return model.fusion(text_feat, meta_feat)


def ig_mlp_batch(model, _train_ds, meta_batch, target_classes, device, steps=IG_STEPS):
    bsz, feat_dim = meta_batch.shape
    alphas = torch.linspace(1.0 / steps, 1.0, steps, device=device)
    scaled = (
        (alphas.view(-1, 1, 1) * meta_batch.unsqueeze(0))
        .reshape(steps * bsz, feat_dim)
        .detach()
        .requires_grad_(True)
    )
    targets = target_classes.unsqueeze(0).expand(steps, bsz).reshape(-1)
    logits = model(scaled)[torch.arange(steps * bsz, device=device), targets]
    grads = torch.autograd.grad(logits.sum(), scaled)[0].view(steps, bsz, feat_dim)
    return (meta_batch * grads.mean(dim=0)).detach()


def ig_fusion_batch(
    model, _train_ds, char_ids_batch, meta_batch, target_classes, device, steps=IG_STEPS
):
    bsz = char_ids_batch.size(0)
    with torch.no_grad():
        embed = model.text_encoder.embedding(char_ids_batch)
        base_ids = torch.full_like(char_ids_batch, PAD_INDEX)
        base_embed = model.text_encoder.embedding(base_ids)
        base_meta = torch.zeros_like(meta_batch)
        diff_embed = embed - base_embed
        diff_meta = meta_batch - base_meta

    grad_e_sum = torch.zeros_like(embed)
    grad_m_sum = torch.zeros_like(meta_batch)
    alphas = torch.linspace(1.0 / steps, 1.0, steps, device=device)
    ar = torch.arange(bsz, device=device)

    for alpha in alphas:
        step_e = (base_embed + alpha * diff_embed).detach().requires_grad_(True)
        step_m = (base_meta + alpha * diff_meta).detach().requires_grad_(True)
        logits = forward_fusion_from_embeds(model, step_e, step_m)[ar, target_classes]
        ge, gm = torch.autograd.grad(logits.sum(), (step_e, step_m))
        grad_e_sum += ge.detach()
        grad_m_sum += gm.detach()

    return (
        diff_embed * (grad_e_sum / steps),
        diff_meta * (grad_m_sum / steps),
    )


def _is_signed(mode):
    return mode.startswith("signed")


def _apply_mode(tensor, mode):
    if mode == "absolute":
        return tensor.abs()
    if mode == "positive_only":
        return tensor.clamp(min=0.0)
    if _is_signed(mode):
        return tensor
    raise ValueError(f"Unsupported contribution mode: {mode!r}")


def rank_rows(example, level, items_scores, total_score, limit=None, include_pct=True, extra_cols=None):
    ordered = sorted(items_scores, key=lambda item: item[1], reverse=True)
    if limit is not None:
        ordered = ordered[:limit]
    rows = []
    for rank, item in enumerate(ordered, start=1):
        name, score = item[0], item[1]
        row = {
            "example": example,
            "level": level,
            "value": name,
            "contribution_score": score,
            "contribution_rank": rank,
        }
        if include_pct:
            row["contribution_percentage"] = (
                (100.0 * score / total_score) if total_score > 0 else 0.0
            )
        if len(item) > 2 and item[2] is not None:
            row["modality_value"] = item[2]
        if extra_cols:
            row.update(extra_cols)
        rows.append(row)
    return rows


def select_examples(
    mlp_model,
    fusion_model,
    test_ds,
    domains,
    task,
    k,
    device,
    *,
    need_mlp=True,
    need_fusion=True,
):
    labels = test_ds.labels
    mlp_probs = fusion_probs = None

    with torch.no_grad():
        if need_mlp:
            mlp_list = [
                mlp_model(test_ds.metadata[s : s + 512].to(device)).softmax(dim=1).cpu()
                for s in range(0, len(test_ds), 512)
            ]
            mlp_probs = torch.cat(mlp_list, dim=0)
            mlp_preds = mlp_probs.argmax(dim=1)
        if need_fusion:
            fus_list = [
                fusion_model(
                    test_ds.inputs[s : s + 512].to(device),
                    test_ds.metadata[s : s + 512].to(device),
                )
                .softmax(dim=1)
                .cpu()
                for s in range(0, len(test_ds), 512)
            ]
            fusion_probs = torch.cat(fus_list, dim=0)
            fusion_preds = fusion_probs.argmax(dim=1)

    mask = (labels == 1) if task == "binary_classification" else torch.ones_like(labels, dtype=torch.bool)
    if need_mlp:
        mask = mask & (mlp_preds == labels)
    if need_fusion:
        mask = mask & (fusion_preds == labels)

    candidate_idx = torch.where(mask)[0].tolist()
    if k is not None and k > 0:
        candidate_idx.sort(
            key=lambda i: (
                (mlp_probs[i, labels[i]].item() if mlp_probs is not None else 0.0)
                + (fusion_probs[i, labels[i]].item() if fusion_probs is not None else 0.0)
            ),
            reverse=True,
        )
        picked, seen = [], set()
        for idx in candidate_idx:
            dom = domains[idx]
            if dom not in seen:
                seen.add(dom)
                picked.append(idx)
                if len(picked) == k:
                    break
        if len(picked) < k:
            raise RuntimeError(f"Found only {len(picked)} matching test examples, needed {k}")
        return picked

    return candidate_idx


def _load_split_ds(csv_path, domain_col, target_col, features, label_to_index, imputer, scaler, max_length, nrows=None):
    frame = pd.read_csv(csv_path, usecols=[domain_col, target_col, *features], nrows=nrows)
    domains = frame[domain_col].astype(str).tolist()
    raw_targets = frame[target_col].astype(str).tolist()
    labels = np.array([label_to_index[t] for t in raw_targets], dtype=np.int64)
    numeric = frame[features].apply(pd.to_numeric, errors="raise")
    meta = scaler.transform(imputer.transform(numeric)).astype(np.float32)
    return DomainDataset(domains, labels, max_length=max_length, metadata=meta), domains, raw_targets


def run_interpretability(
    method_abbr,
    explain_modality_batch_fn,
    explain_mlp_batch_fn,
    *,
    need_train_bg=False,
    batch_size=BATCH_SIZE,
):
    config_path = PROJECT_ROOT / "config.yml"
    config = yaml.safe_load(config_path.read_text())
    interp = config["interpretability"]
    levels_cfg = interp["levels"]
    levels_to_run = interp.get("levels_to_run") or list(levels_cfg.keys())
    need_fusion = "modality" in levels_to_run
    need_mlp = ("category" in levels_to_run) or ("feature" in levels_to_run)

    out_cfg = interp["output"]
    base_fields = out_cfg["rows"] + [c for c in out_cfg["columns"] if c not in out_cfg["rows"]]
    contributions = [c["name"] for c in interp.get("contributions", [{"name": "absolute"}])]

    cat_patterns = {
        cat: patterns
        for cat, patterns in config["metadata_categories"].items()
        if cat != "all"
    }
    max_length = config["learning_combinations"]["char_metadata"]["data"]["max_length"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for ds_entry in interp["datasets"]:
        dataset_name = ds_entry["name"]
        ds_cfg = resolve_dataset_config(config, dataset_name)
        task = ds_cfg["task"]
        k = (
            interp["examples_to_explain"]["binary"]
            if task == "binary_classification"
            else interp["examples_to_explain"]["multi_class"]
        )
        include_extra = (k is not None and k < 0) or any(m == "signed-all" for m in contributions)

        mlp_model_name = levels_cfg["category"]["model"][0]
        _, mlp_learning, _ = Metadata_MLP.load_config(
            config_path, dataset_name, mlp_model_name
        )
        mlp_ckpt_path = PROJECT_ROOT / mlp_learning["output"]["directory"] / "best.pt"
        mlp_ckpt = torch.load(mlp_ckpt_path, map_location=device, weights_only=False)
        features = list(mlp_ckpt["metadata_features"])
        bundle = {
            "vocab_size": VOCAB_SIZE,
            "num_classes": int(mlp_ckpt["num_classes"]),
            "metadata_dim": len(features),
            "metadata_features": features,
        }
        mlp_model = None
        if need_mlp:
            mlp_model = Metadata_MLP.build_model(bundle, mlp_learning["model"], device)
            mlp_model.load_state_dict(mlp_ckpt["model_state_dict"])
            mlp_model.eval()

        fusion_model = None
        if need_fusion:
            mod_model_name = levels_cfg["modality"]["model"][0]
            mod_comb_name = levels_cfg["modality"]["combination"][0]
            fusion_out_dir = Fusion.format_template(
                config["learning_combinations"]["char_metadata"]["output"]["directory"],
                combination_name=mod_comb_name,
                model_name=mod_model_name,
                dataset_name=dataset_name,
                scenario_name=ds_cfg["scenario"],
            )
            fusion_ckpt_path = PROJECT_ROOT / fusion_out_dir / "best.pt"
            fusion_ckpt = torch.load(fusion_ckpt_path, map_location=device, weights_only=False)
            fusion_learning = {
                "encoder_type": "char",
                "encoder_model": fusion_ckpt.get(
                    "char_model_config",
                    Fusion.resolve_char_model_config(config["models"], mod_model_name),
                ),
                "metadata_model_config": fusion_ckpt.get(
                    "metadata_model_config",
                    Fusion.resolve_metadata_model_config(config["models"]),
                ),
                "combination": fusion_ckpt.get(
                    "combination",
                    Fusion.resolve_combination(config, mod_comb_name),
                ),
            }
            fusion_model = Fusion.build_model(bundle, fusion_learning, device)
            fusion_state = {
                 key.replace("char_encoder.", "text_encoder.", 1): val
                for key, val in fusion_ckpt["model_state_dict"].items()
            }
            fusion_model.load_state_dict(fusion_state)
            fusion_model.eval()

        ds_dir = Path(ds_cfg["path"])
        dom_col = ds_cfg["columns"]["domain"]
        tgt_col = ds_cfg["columns"]["target"]
        imputer = mlp_ckpt["metadata_imputer"]
        scaler = mlp_ckpt["metadata_scaler"]
        label_to_index = mlp_ckpt["label_to_index"]

        test_ds, domains, raw_targets = _load_split_ds(
            ds_dir / ds_cfg["splits"]["test"],
            dom_col,
            tgt_col,
            features,
            label_to_index,
            imputer,
            scaler,
            max_length,
        )
        train_ds = (
            _load_split_ds(
                ds_dir / ds_cfg["splits"]["train"],
                dom_col,
                tgt_col,
                features,
                label_to_index,
                imputer,
                scaler,
                max_length,
                nrows=512,
            )[0]
            if need_train_bg
            else None
        )
        picked_indices = select_examples(
            mlp_model,
            fusion_model,
            test_ds,
            domains,
            task,
            k,
            device,
            need_mlp=need_mlp,
            need_fusion=need_fusion,
        )
        cat_indices = {
            cat: [
                i
                for i, feat in enumerate(features)
                if any(fnmatchcase(feat, pat) for pat in pats)
            ]
            for cat, pats in cat_patterns.items()
        }
        top_n = levels_cfg["feature"]["top"]

        rows_by_mode = {mode: [] for mode in contributions}
        for start in range(0, len(picked_indices), batch_size):
            batch_idxs = picked_indices[start : start + batch_size]
            idx_t = torch.as_tensor(batch_idxs, dtype=torch.long)
            char_ids_b = test_ds.inputs[idx_t].to(device)
            meta_b = test_ds.metadata[idx_t].to(device)
            targets_b = test_ds.labels[idx_t].to(device)

            if need_fusion:
                raw_e_b, raw_m_b = explain_modality_batch_fn(
                    fusion_model, train_ds, char_ids_b, meta_b, targets_b, device
                )
                str_vals = (char_ids_b != PAD_INDEX).float().mean(dim=1).cpu().tolist()
                meta_vals = meta_b.mean(dim=1).cpu().tolist()
            if need_mlp:
                raw_feat_b = explain_mlp_batch_fn(
                    mlp_model, train_ds, meta_b, targets_b, device
                )

            for mode in contributions:
                include_pct = not _is_signed(mode)
                if need_fusion:
                    mod_e_b = _apply_mode(raw_e_b, mode).sum(dim=(1, 2)).cpu().tolist()
                    mod_m_b = _apply_mode(raw_m_b, mode).sum(dim=1).cpu().tolist()
                if need_mlp:
                    feat_attr_b = _apply_mode(raw_feat_b, mode).cpu()

                for b_i, orig_idx in enumerate(batch_idxs):
                    domain = domains[orig_idx]
                    extra_cols = (
                        {"class": raw_targets[orig_idx]}
                        if (include_extra and task == "multi_class_classification")
                        else None
                    )
                    if need_fusion:
                        me, mm = float(mod_e_b[b_i]), float(mod_m_b[b_i])
                        if include_extra:
                            mod_scores = [
                                ("string", me, float(str_vals[b_i])),
                                ("metadata", mm, float(meta_vals[b_i])),
                            ]
                        else:
                            mod_scores = [("string", me), ("metadata", mm)]
                        rows_by_mode[mode].extend(
                            rank_rows(
                                domain,
                                "modality",
                                mod_scores,
                                me + mm,
                                include_pct=include_pct,
                                extra_cols=extra_cols,
                            )
                        )
                    if need_mlp:
                        feat_attr = feat_attr_b[b_i]
                        feat_total = float(feat_attr.sum().item())
                        if "category" in levels_to_run:
                            cat_scores = [
                                (cat, float(feat_attr[idxs].sum().item()))
                                for cat, idxs in cat_indices.items()
                            ]
                            rows_by_mode[mode].extend(
                                rank_rows(
                                    domain,
                                    "category",
                                    cat_scores,
                                    feat_total,
                                    include_pct=include_pct,
                                    extra_cols=extra_cols,
                                )
                            )
                        if "feature" in levels_to_run:
                            feat_scores = [
                                (feat, float(feat_attr[i].item()))
                                for i, feat in enumerate(features)
                            ]
                            rows_by_mode[mode].extend(
                                rank_rows(
                                    domain,
                                    "feature",
                                    feat_scores,
                                    feat_total,
                                    limit=top_n,
                                    include_pct=include_pct,
                                    extra_cols=extra_cols,
                                )
                            )

        rows_per_example = (
            (2 if "modality" in levels_to_run else 0)
            + (len(cat_patterns) if "category" in levels_to_run else 0)
            + (top_n if "feature" in levels_to_run else 0)
        )
        expected_rows = len(picked_indices) * rows_per_example
        for mode, rows in rows_by_mode.items():
            fieldnames = (
                [f for f in base_fields if f != "contribution_percentage"]
                if _is_signed(mode)
                else list(base_fields)
            )
            if include_extra and "modality" in levels_to_run:
                fieldnames.append("modality_value")
            if include_extra and task == "multi_class_classification":
                fieldnames.append("class")

            out_path = PROJECT_ROOT / out_cfg["file"].format(
                contribution=mode, method=method_abbr, dataset=dataset_name
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            print(
                f"Wrote {len(rows)} rows ({len(picked_indices)} examples) to {out_path}",
                flush=True,
            )
            assert out_path.is_file() and len(rows) == expected_rows, (
                f"Expected {expected_rows} rows, got {len(rows)}"
            )


def main():
    run_interpretability(
        METHOD_ABBR,
        ig_fusion_batch,
        ig_mlp_batch,
    )


if __name__ == "__main__":
    main()
