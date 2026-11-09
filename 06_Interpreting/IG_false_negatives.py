#!/usr/bin/env python3
"""Preflight or compute true-class IG modality scores for malicious false negatives."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "01_metadata_selection"))
sys.path.insert(0, str(PROJECT_ROOT / "04_Learning"))
sys.path.insert(0, str(PROJECT_ROOT / "06_Interpreting"))

from data_loader import VOCAB_SIZE, resolve_dataset_config  # noqa: E402
import Fusion  # noqa: E402
from IG import IG_STEPS, _load_split_ds, ig_fusion_batch  # noqa: E402

DATASETS = (
    "dibhran_malware_binary",
    "dibhran_phishing_binary",
    "dibhran_malware_family",
)
MODEL = "CharCNN"
COMBINATION = "N_cat_N_ce"
BATCH_SIZE = 128
OUTPUT_DIR = (
    PROJECT_ROOT
    / "09_output/06_Interpreting/Interpretability/false-negatives"
)


def load_dataset(dataset_name: str, device: torch.device):
    config_path = PROJECT_ROOT / "config.yml"
    config = yaml.safe_load(config_path.read_text())
    ds_cfg = resolve_dataset_config(config, dataset_name)
    learning = config["learning_combinations"]["char_metadata"]
    checkpoint_path = PROJECT_ROOT / Fusion.format_template(
        learning["output"]["directory"],
        combination_name=COMBINATION,
        model_name=MODEL,
        dataset_name=dataset_name,
        scenario_name=ds_cfg["scenario"],
    ) / "best.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    features = list(checkpoint["metadata_features"])
    label_to_index = dict(checkpoint["label_to_index"])
    index_to_label = {index: label for label, index in label_to_index.items()}
    if len(index_to_label) != len(label_to_index) or "legit" not in label_to_index:
        raise ValueError(f"{dataset_name}: invalid label mapping {label_to_index}")
    if ds_cfg["task"] == "binary_classification" and set(label_to_index) != {
        "legit",
        "malicious",
    }:
        raise ValueError(f"{dataset_name}: unexpected binary labels {label_to_index}")

    bundle = {
        "vocab_size": VOCAB_SIZE,
        "num_classes": int(checkpoint["num_classes"]),
        "metadata_dim": len(features),
        "metadata_features": features,
    }
    model_config = {
        "encoder_type": "char",
        "encoder_model": checkpoint["char_model_config"],
        "metadata_model_config": checkpoint["metadata_model_config"],
        "combination": checkpoint["combination"],
    }
    model = Fusion.build_model(bundle, model_config, device)
    state = {
        key.replace("char_encoder.", "text_encoder.", 1): value
        for key, value in checkpoint["model_state_dict"].items()
    }
    model.load_state_dict(state)
    model.eval()

    dataset_dir = Path(ds_cfg["path"])
    test_ds, domains, raw_targets = _load_split_ds(
        dataset_dir / ds_cfg["splits"]["test"],
        ds_cfg["columns"]["domain"],
        ds_cfg["columns"]["target"],
        features,
        label_to_index,
        checkpoint["metadata_imputer"],
        checkpoint["metadata_scaler"],
        learning["data"]["max_length"],
    )
    return checkpoint_path, model, test_ds, domains, raw_targets, label_to_index, index_to_label


def predict(model, test_ds, device: torch.device) -> torch.Tensor:
    predictions = []
    with torch.no_grad():
        for start in range(0, len(test_ds), 512):
            predictions.append(
                model(
                    test_ds.inputs[start : start + 512].to(device),
                    test_ds.metadata[start : start + 512].to(device),
                )
                .argmax(dim=1)
                .cpu()
            )
    return torch.cat(predictions)


def existing_tp_count(dataset_name: str) -> int:
    path = (
        PROJECT_ROOT
        / "09_output/06_Interpreting/Interpretability/signed-all"
        / f"IG_{dataset_name}.csv"
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="") as stream:
        rows = csv.DictReader(stream)
        return sum(
            row["value"] == "string"
            and (dataset_name != "dibhran_malware_family" or row.get("class") != "legit")
            for row in rows
        )


def write_fn_ig(
    dataset_name: str,
    model,
    test_ds,
    domains,
    raw_targets,
    predictions: torch.Tensor,
    fn_indices: list[int],
    index_to_label: dict[int, str],
    device: torch.device,
) -> Path:
    rows = []
    for start in range(0, len(fn_indices), BATCH_SIZE):
        batch_indices = fn_indices[start : start + BATCH_SIZE]
        index_tensor = torch.as_tensor(batch_indices, dtype=torch.long)
        char_ids = test_ds.inputs[index_tensor].to(device)
        metadata = test_ds.metadata[index_tensor].to(device)
        true_classes = test_ds.labels[index_tensor].to(device)
        string_attr, metadata_attr = ig_fusion_batch(
            model,
            None,
            char_ids,
            metadata,
            true_classes,
            device,
            steps=IG_STEPS,
        )
        string_scores = string_attr.sum(dim=(1, 2)).cpu().tolist()
        metadata_scores = metadata_attr.sum(dim=1).cpu().tolist()
        for offset, sample_id in enumerate(batch_indices):
            rows.append(
                {
                    "sample_id": sample_id,
                    "example": domains[sample_id],
                    "true_label": raw_targets[sample_id],
                    "predicted_label": index_to_label[int(predictions[sample_id])],
                    "string_contribution_score": float(string_scores[offset]),
                    "metadata_contribution_score": float(metadata_scores[offset]),
                }
            )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"IG_FN_{dataset_name}.csv"
    with output_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if len(rows) != len(fn_indices):
        raise RuntimeError(f"{dataset_name}: wrote {len(rows)} rows for {len(fn_indices)} FNs")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Compute and save FN IG scores; omit for inference-only preflight.",
    )
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Mode: {'FN IG execution' if args.execute else 'preflight only'}; device={device}")

    total_fn = 0
    for dataset_name in DATASETS:
        (
            checkpoint_path,
            model,
            test_ds,
            domains,
            raw_targets,
            label_to_index,
            index_to_label,
        ) = load_dataset(dataset_name, device)
        predictions = predict(model, test_ds, device)
        labels = test_ds.labels
        legit_index = label_to_index["legit"]
        malicious = labels != legit_index
        true_positive = malicious & (predictions == labels)
        false_negative = malicious & (predictions == legit_index)
        other_error = malicious & (predictions != labels) & (predictions != legit_index)
        tp_count = int(true_positive.sum())
        fn_indices = torch.where(false_negative)[0].tolist()
        fn_count = len(fn_indices)
        total_fn += fn_count

        saved_tp_count = existing_tp_count(dataset_name)
        if saved_tp_count != tp_count:
            raise RuntimeError(
                f"{dataset_name}: saved TP IG count {saved_tp_count} != inferred TP count {tp_count}"
            )
        print(
            f"{dataset_name}: checkpoint={checkpoint_path}; labels={label_to_index}; "
            f"test={len(test_ds)} malicious={int(malicious.sum())} TP={tp_count} "
            f"FN_to_legit={fn_count} other_malicious_errors={int(other_error.sum())}; "
            f"IG_steps={IG_STEPS} IG_batches={(fn_count + BATCH_SIZE - 1) // BATCH_SIZE}"
        )

        if args.execute:
            output_path = write_fn_ig(
                dataset_name,
                model,
                test_ds,
                domains,
                raw_targets,
                predictions,
                fn_indices,
                index_to_label,
                device,
            )
            print(f"Wrote {fn_count} FN rows to {output_path}")

    print(
        f"Total FN samples={total_fn}; IG gradient steps={total_fn * IG_STEPS}; "
        f"execute={args.execute}"
    )


if __name__ == "__main__":
    main()
