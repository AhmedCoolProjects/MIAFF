#!/usr/bin/env python3
"""Preflight or compute true-class fusion (CharCNN + MIAFF N_cat_N_ce) per-feature IG."""

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
from IG_categories_fusion_all import (  # noqa: E402
    BATCH_SIZE,
    CATEGORIES,
    COMBINATION,
    DATASETS,
    MODEL,
    load_dataset,
    predict,
)

OUTPUT_DIR = PROJECT_ROOT / "09_output/06_Interpreting/Interpretability/features_fusion"


def write_ig(
    dataset_name: str,
    model,
    test_ds,
    domains,
    raw_targets,
    predictions: torch.Tensor,
    sample_indices: list[int],
    index_to_label: dict[int, str],
    features: list[str],
    category_indices: dict[str, list[int]],
    device: torch.device,
) -> Path:
    feature_index = {feature: i for i, feature in enumerate(features)}
    rows = []
    max_residual = 0.0
    for start in range(0, len(sample_indices), BATCH_SIZE):
        batch_indices = sample_indices[start : start + BATCH_SIZE]
        index_tensor = torch.as_tensor(batch_indices, dtype=torch.long)
        char_ids = test_ds.inputs[index_tensor].to(device)
        metadata = test_ds.metadata[index_tensor].to(device)
        true_classes = test_ds.labels[index_tensor].to(device)
        string_attr, metadata_attr = ig_fusion_batch(
            model, None, char_ids, metadata, true_classes, device, steps=IG_STEPS
        )
        string_scores = string_attr.sum(dim=(1, 2)).cpu().tolist()
        metadata_scores = metadata_attr.sum(dim=1).cpu().tolist()
        feature_matrix = metadata_attr.cpu().tolist()
        category_totals = {
            category: metadata_attr[:, indices].sum(dim=1).cpu().tolist()
            for category, indices in category_indices.items()
        }
        for offset, sample_id in enumerate(batch_indices):
            per_feature = {
                feature: float(feature_matrix[offset][feature_index[feature]])
                for feature in features
            }
            for category, indices in category_indices.items():
                cat_sum = sum(per_feature[features[i]] for i in indices)
                residual = abs(cat_sum - category_totals[category][offset])
                max_residual = max(max_residual, residual)
                if residual > 1e-3:
                    raise ValueError(
                        f"{dataset_name}:{sample_id}:{category}: feature sum residual "
                        f"{residual:.3g} exceeds tolerance"
                    )
            true_label = raw_targets[sample_id]
            predicted_label = index_to_label[int(predictions[sample_id])]
            row = {
                "sample_id": sample_id,
                "example": domains[sample_id],
                "true_label": true_label,
                "predicted_label": predicted_label,
                "classification": "Correct" if true_label == predicted_label else "Incorrect",
            }
            row.update(per_feature)
            row["string_contribution_score"] = float(string_scores[offset])
            row["metadata_contribution_score"] = float(metadata_scores[offset])
            rows.append(row)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"IG_FEATURES_FUSION_{dataset_name}.csv"
    with output_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if len(rows) != len(sample_indices):
        raise RuntimeError(f"{dataset_name}: wrote {len(rows)} rows for {len(sample_indices)} samples")
    print(f"{dataset_name}: max |sum(features)-category| = {max_residual:.3g}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Compute and save fusion per-feature IG; omit for inference-only preflight.",
    )
    args = parser.parse_args()
    if IG_STEPS != 64:
        raise ValueError(f"Expected 64 IG steps, found {IG_STEPS}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"mode={'execute' if args.execute else 'preflight'} device={device}")

    total_samples = 0
    for dataset_name in DATASETS:
        (
            checkpoint_path,
            model,
            test_ds,
            domains,
            raw_targets,
            label_to_index,
            index_to_label,
            category_indices,
        ) = load_dataset(dataset_name, device)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        features = list(checkpoint["metadata_features"])
        predictions = predict(model, test_ds, device)
        malicious = test_ds.labels != label_to_index["legit"]
        sample_indices = torch.where(malicious)[0].tolist()
        correct = malicious & (predictions == test_ds.labels)
        incorrect = malicious & (predictions != test_ds.labels)
        total_samples += len(sample_indices)
        print(
            f"{dataset_name}: checkpoint={checkpoint_path} test={len(test_ds)} "
            f"malicious={len(sample_indices)} Correct={int(correct.sum())} "
            f"Incorrect={int(incorrect.sum())} IG_steps={IG_STEPS} "
            f"IG_batches={(len(sample_indices) + BATCH_SIZE - 1) // BATCH_SIZE} "
            f"n_features={len(features)}"
        )
        if args.execute:
            output_path = write_ig(
                dataset_name,
                model,
                test_ds,
                domains,
                raw_targets,
                predictions,
                sample_indices,
                index_to_label,
                features,
                category_indices,
                device,
            )
            print(f"Wrote {len(sample_indices)} rows to {output_path}")

    print(f"total_malicious={total_samples} execute={args.execute}")


if __name__ == "__main__":
    main()
