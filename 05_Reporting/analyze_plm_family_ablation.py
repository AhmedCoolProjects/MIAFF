#!/usr/bin/env python3
"""Investigate PLM macro-F1 degradation in malware-family classification under MIAFF-base."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "04_Learning"))
sys.path.insert(0, str(PROJECT_ROOT / "05_Reporting"))

from inference_benchmark import build_bundle_and_model, ckpt_paths  # noqa: E402

MODELS = ["SecureBERT", "CySecBERT", "SecBERT", "DomURLsBERT"]
DATASETS = ["threatfox_malware_family", "dibhran_malware_family"]


def analyze_dataset_model(model_name: str, dataset_name: str, config: dict, device: torch.device):
    # Load MIAFF_base bundle once; it contains both text and metadata
    path_miaff = ckpt_paths(model_name, dataset_name)["MIAFF_base"]
    ckpt_miaff = torch.load(path_miaff, map_location=device, weights_only=False)
    loader, forward_miaff, net_miaff, task, bundle = build_bundle_and_model(
        model_name, dataset_name, "MIAFF_base", config, ckpt_miaff, device,
        batch_size=512, max_length=64, num_workers=0
    )

    # Load baseline model
    path_base = ckpt_paths(model_name, dataset_name)["baseline"]
    ckpt_base = torch.load(path_base, map_location=device, weights_only=False)
    from BASELINE_LANG import MODEL_REGISTRY as REG
    from BASELINE_LANG import resolve_model_config as resolve_lang
    model_cfg = resolve_lang(config["models"], model_name)
    params = {k: v for k, v in model_cfg.items() if k not in ("name", "pretrained_path")}
    net_base = REG[model_name](
        output_size=bundle["num_classes"],
        pretrained_path=model_cfg["pretrained_path"],
        **params
    )
    net_base.load_state_dict(ckpt_base["model_state_dict"], strict=True)
    net_base.to(device)
    net_base.eval()

    def forward_base(batch):
        return net_base(
            batch["input_ids"].to(device, non_blocking=True),
            batch["attention_mask"].to(device, non_blocking=True)
        )

    all_preds_base, all_preds_miaff, all_labels = [], [], []
    with torch.inference_mode():
        for batch in loader:
            logits_base = forward_base(batch)
            logits_miaff = forward_miaff(batch)
            all_preds_base.append(logits_base.argmax(dim=-1).cpu())
            all_preds_miaff.append(logits_miaff.argmax(dim=-1).cpu())
            all_labels.append(batch["label"].cpu())

    preds_base = torch.cat(all_preds_base).numpy()
    preds_miaff = torch.cat(all_preds_miaff).numpy()
    labels = torch.cat(all_labels).numpy()
    class_names = bundle["class_names"]

    # Overall metrics
    macro_f1_base = f1_score(labels, preds_base, average="macro", zero_division=0)
    acc_base = accuracy_score(labels, preds_base)
    macro_f1_miaff = f1_score(labels, preds_miaff, average="macro", zero_division=0)
    acc_miaff = accuracy_score(labels, preds_miaff)

    # Classification reports (per class)
    rep_base = classification_report(labels, preds_base, target_names=class_names, output_dict=True, zero_division=0)
    rep_miaff = classification_report(labels, preds_miaff, target_names=class_names, output_dict=True, zero_division=0)

    legit_idx = class_names.index("legit") if "legit" in class_names else -1
    num_classes = len(class_names)

    # Confusion breakdown
    # Mask for malware vs legit
    is_legit = (labels == legit_idx)
    is_malware = ~is_legit

    # Malware confusion
    mal_total = int(np.sum(is_malware))
    mal_pred_legit_base = int(np.sum(is_malware & (preds_base == legit_idx)))
    mal_pred_legit_miaff = int(np.sum(is_malware & (preds_miaff == legit_idx)))

    mal_pred_correct_base = int(np.sum(is_malware & (preds_base == labels)))
    mal_pred_correct_miaff = int(np.sum(is_malware & (preds_miaff == labels)))

    mal_pred_wrong_fam_base = mal_total - mal_pred_legit_base - mal_pred_correct_base
    mal_pred_wrong_fam_miaff = mal_total - mal_pred_legit_miaff - mal_pred_correct_miaff

    # Legit confusion
    legit_total = int(np.sum(is_legit))
    legit_pred_legit_base = int(np.sum(is_legit & (preds_base == legit_idx)))
    legit_pred_legit_miaff = int(np.sum(is_legit & (preds_miaff == legit_idx)))

    legit_pred_mal_base = legit_total - legit_pred_legit_base
    legit_pred_mal_miaff = legit_total - legit_pred_legit_miaff

    # Per-family analysis (excluding legit)
    family_stats = []
    for idx, cname in enumerate(class_names):
        if cname == "legit":
            continue
        supp = rep_base[cname]["support"]
        f1_b = rep_base[cname]["f1-score"]
        r_b = rep_base[cname]["recall"]
        p_b = rep_base[cname]["precision"]

        f1_m = rep_miaff[cname]["f1-score"]
        r_m = rep_miaff[cname]["recall"]
        p_m = rep_miaff[cname]["precision"]

        family_stats.append({
            "class": cname,
            "support": supp,
            "f1_base": f1_b,
            "f1_miaff": f1_m,
            "delta_f1": f1_m - f1_b,
            "r_base": r_b,
            "r_miaff": r_m,
            "delta_r": r_m - r_b,
            "p_base": p_b,
            "p_miaff": p_m,
            "delta_p": p_m - p_b,
        })

    df_fam = pd.DataFrame(family_stats)
    n_families = len(df_fam)
    n_improved = int((df_fam["delta_f1"] > 1e-5).sum())
    n_deteriorated = int((df_fam["delta_f1"] < -1e-5).sum())
    n_unchanged = int((df_fam["delta_f1"].abs() <= 1e-5).sum())
    n_zero_both = int(((df_fam["f1_base"] == 0) & (df_fam["f1_miaff"] == 0)).sum())
    n_collapsed_to_zero = int(((df_fam["f1_base"] > 0) & (df_fam["f1_miaff"] == 0)).sum())

    # Legit metrics
    legit_metrics = {
        "p_base": rep_base["legit"]["precision"],
        "r_base": rep_base["legit"]["recall"],
        "f1_base": rep_base["legit"]["f1-score"],
        "p_miaff": rep_miaff["legit"]["precision"],
        "r_miaff": rep_miaff["legit"]["recall"],
        "f1_miaff": rep_miaff["legit"]["f1-score"],
    }

    # Macro metrics for malware families only (excluding legit)
    mal_macro_f1_base = float(df_fam["f1_base"].mean())
    mal_macro_f1_miaff = float(df_fam["f1_miaff"].mean())
    mal_macro_r_base = float(df_fam["r_base"].mean())
    mal_macro_r_miaff = float(df_fam["r_miaff"].mean())
    mal_macro_p_base = float(df_fam["p_base"].mean())
    mal_macro_p_miaff = float(df_fam["p_miaff"].mean())

    # Support tertile analysis
    q33 = df_fam["support"].quantile(0.33)
    q66 = df_fam["support"].quantile(0.66)
    low_fam = df_fam[df_fam["support"] <= q33]
    mid_fam = df_fam[(df_fam["support"] > q33) & (df_fam["support"] <= q66)]
    high_fam = df_fam[df_fam["support"] > q66]

    res = {
        "model": model_name,
        "dataset": dataset_name,
        "overall": {
            "macro_f1_base": macro_f1_base,
            "macro_f1_miaff": macro_f1_miaff,
            "delta_macro_f1": macro_f1_miaff - macro_f1_base,
            "acc_base": acc_base,
            "acc_miaff": acc_miaff,
            "delta_acc": acc_miaff - acc_base,
        },
        "legit": legit_metrics,
        "malware_macro": {
            "f1_base": mal_macro_f1_base,
            "f1_miaff": mal_macro_f1_miaff,
            "delta_f1": mal_macro_f1_miaff - mal_macro_f1_base,
            "r_base": mal_macro_r_base,
            "r_miaff": mal_macro_r_miaff,
            "delta_r": mal_macro_r_miaff - mal_macro_r_base,
            "p_base": mal_macro_p_base,
            "p_miaff": mal_macro_p_miaff,
            "delta_p": mal_macro_p_miaff - mal_macro_p_base,
        },
        "family_counts": {
            "total_families": n_families,
            "improved": n_improved,
            "deteriorated": n_deteriorated,
            "unchanged": n_unchanged,
            "zero_both": n_zero_both,
            "collapsed_to_zero": n_collapsed_to_zero,
        },
        "confusion": {
            "malware_total": mal_total,
            "mal_to_legit_base": mal_pred_legit_base,
            "mal_to_legit_miaff": mal_pred_legit_miaff,
            "mal_to_legit_base_pct": mal_pred_legit_base / mal_total,
            "mal_to_legit_miaff_pct": mal_pred_legit_miaff / mal_total,
            "mal_correct_base": mal_pred_correct_base,
            "mal_correct_miaff": mal_pred_correct_miaff,
            "mal_wrong_fam_base": mal_pred_wrong_fam_base,
            "mal_wrong_fam_miaff": mal_pred_wrong_fam_miaff,
            "legit_total": legit_total,
            "legit_to_legit_base": legit_pred_legit_base,
            "legit_to_legit_miaff": legit_pred_legit_miaff,
            "legit_to_legit_base_pct": legit_pred_legit_base / legit_total,
            "legit_to_legit_miaff_pct": legit_pred_legit_miaff / legit_total,
            "legit_to_mal_base": legit_pred_mal_base,
            "legit_to_mal_miaff": legit_pred_mal_miaff,
        },
        "tertiles": {
            "low": {
                "count": len(low_fam),
                "support_max": float(q33),
                "f1_base": float(low_fam["f1_base"].mean()),
                "f1_miaff": float(low_fam["f1_miaff"].mean()),
                "r_base": float(low_fam["r_base"].mean()),
                "r_miaff": float(low_fam["r_miaff"].mean()),
            },
            "mid": {
                "count": len(mid_fam),
                "f1_base": float(mid_fam["f1_base"].mean()),
                "f1_miaff": float(mid_fam["f1_miaff"].mean()),
                "r_base": float(mid_fam["r_base"].mean()),
                "r_miaff": float(mid_fam["r_miaff"].mean()),
            },
            "high": {
                "count": len(high_fam),
                "support_min": float(q66),
                "f1_base": float(high_fam["f1_base"].mean()),
                "f1_miaff": float(high_fam["f1_miaff"].mean()),
                "r_base": float(high_fam["r_base"].mean()),
                "r_miaff": float(high_fam["r_miaff"].mean()),
            },
        },
        "family_details": family_stats,
    }
    return res


def main():
    config_path = PROJECT_ROOT / "config.yml"
    config = yaml.safe_load(config_path.read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running analysis on {device}...")

    all_results = []
    for d in DATASETS:
        for m in MODELS:
            print(f"\nProcessing {m} on {d}...")
            r = analyze_dataset_model(m, d, config, device)
            all_results.append(r)
            print(f"  Macro-F1: {r['overall']['macro_f1_base']:.4f} -> {r['overall']['macro_f1_miaff']:.4f} (delta={r['overall']['delta_macro_f1']:.4f})")
            print(f"  Accuracy: {r['overall']['acc_base']:.4f} -> {r['overall']['acc_miaff']:.4f} (delta={r['overall']['delta_acc']:.4f})")
            print(f"  Malware families: {r['family_counts']['improved']} improved, {r['family_counts']['deteriorated']} deteriorated, {r['family_counts']['unchanged']} unchanged ({r['family_counts']['collapsed_to_zero']} collapsed to 0)")
            print(f"  Malware -> Legit (Evasion): {r['confusion']['mal_to_legit_base']} ({r['confusion']['mal_to_legit_base_pct']:.1%}) -> {r['confusion']['mal_to_legit_miaff']} ({r['confusion']['mal_to_legit_miaff_pct']:.1%})")
            print(f"  Legit -> Legit (True Neg): {r['confusion']['legit_to_legit_base']} ({r['confusion']['legit_to_legit_base_pct']:.1%}) -> {r['confusion']['legit_to_legit_miaff']} ({r['confusion']['legit_to_legit_miaff_pct']:.1%})")

    out_file = PROJECT_ROOT / "09_output/05_Reporting/plm_family_ablation_investigation.json"
    out_file.write_text(json.dumps(all_results, indent=2))
    print(f"\nWrote full investigation results to {out_file}")


if __name__ == "__main__":
    main()
