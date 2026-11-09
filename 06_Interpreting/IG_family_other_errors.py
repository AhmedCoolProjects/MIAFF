#!/usr/bin/env python3
"""Preflight or compute true-class IG for malware-family cross-family errors."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from IG import IG_STEPS, ig_fusion_batch
from IG_false_negatives import (
    BATCH_SIZE,
    PROJECT_ROOT,
    existing_tp_count,
    load_dataset,
    predict,
)

DATASET = "dibhran_malware_family"
OUTPUT_PATH = (
    PROJECT_ROOT
    / "09_output/06_Interpreting/Interpretability/incorrect"
    / f"IG_INCORRECT_{DATASET}.csv"
)
EXISTING_FN_PATH = (
    PROJECT_ROOT
    / "09_output/06_Interpreting/Interpretability/false-negatives"
    / f"IG_FN_{DATASET}.csv"
)


def write_ig(
    model,
    test_ds,
    domains,
    raw_targets,
    predictions: torch.Tensor,
    sample_indices: list[int],
    index_to_label: dict[int, str],
    device: torch.device,
) -> None:
    rows = []
    for start in range(0, len(sample_indices), BATCH_SIZE):
        batch_indices = sample_indices[start : start + BATCH_SIZE]
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

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if len(rows) != len(sample_indices):
        raise RuntimeError(f"Wrote {len(rows)} rows for {len(sample_indices)} samples")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Compute and save IG; omit for inference-only preflight.",
    )
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (
        checkpoint_path,
        model,
        test_ds,
        domains,
        raw_targets,
        label_to_index,
        index_to_label,
    ) = load_dataset(DATASET, device)
    predictions = predict(model, test_ds, device)
    labels = test_ds.labels
    legit_index = label_to_index["legit"]
    malicious = labels != legit_index
    correct = malicious & (predictions == labels)
    false_negative = malicious & (predictions == legit_index)
    other_error = malicious & (predictions != labels) & (predictions != legit_index)
    other_indices = torch.where(other_error)[0].tolist()

    if IG_STEPS != 64:
        raise ValueError(f"Expected 64 IG steps, found {IG_STEPS}")
    if existing_tp_count(DATASET) != int(correct.sum()):
        raise RuntimeError("Existing TP IG count does not match current checkpoint predictions")
    if not EXISTING_FN_PATH.is_file():
        raise FileNotFoundError(EXISTING_FN_PATH)
    with EXISTING_FN_PATH.open(newline="") as stream:
        existing_fn_ids = {int(row["sample_id"]) for row in csv.DictReader(stream)}
    inferred_fn_ids = set(torch.where(false_negative)[0].tolist())
    if existing_fn_ids != inferred_fn_ids:
        raise RuntimeError("Existing FN IG sample IDs do not match current checkpoint predictions")
    if len(other_indices) != 947:
        raise RuntimeError(f"Expected 947 family-to-family errors, found {len(other_indices)}")

    print(
        f"mode={'execute' if args.execute else 'preflight'} device={device} "
        f"checkpoint={checkpoint_path} test={len(test_ds)} malicious={int(malicious.sum())} "
        f"correct={int(correct.sum())} FN_to_legit={int(false_negative.sum())} "
        f"family_to_family={len(other_indices)} IG_steps={IG_STEPS} "
        f"IG_batches={(len(other_indices) + BATCH_SIZE - 1) // BATCH_SIZE}"
    )
    if args.execute:
        write_ig(
            model,
            test_ds,
            domains,
            raw_targets,
            predictions,
            other_indices,
            index_to_label,
            device,
        )
        print(f"Wrote {len(other_indices)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
