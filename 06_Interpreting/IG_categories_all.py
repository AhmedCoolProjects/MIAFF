#!/usr/bin/env python3
"""Preflight or compute true-class Metadata_MLP IG for all malicious test samples."""

from __future__ import annotations

import argparse
import csv
from fnmatch import fnmatchcase
from pathlib import Path
import sys

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "01_metadata_selection"))
sys.path.insert(0, str(PROJECT_ROOT / "04_Learning"))
sys.path.insert(0, str(PROJECT_ROOT / "06_Interpreting"))

from data_loader import VOCAB_SIZE, resolve_dataset_config  # noqa: E402
from IG import IG_STEPS, _load_split_ds, ig_mlp_batch  # noqa: E402
import Metadata_MLP  # noqa: E402

DATASETS = (
    "dibhran_malware_binary",
    "dibhran_phishing_binary",
    "dibhran_malware_family",
)
CATEGORIES = ("dns", "rdap", "tls", "ip", "cross")
BATCH_SIZE = 128
OUTPUT_DIR = PROJECT_ROOT / "09_output/06_Interpreting/Interpretability/categories"


def load_dataset(dataset_name: str, device: torch.device):
    config_path = PROJECT_ROOT / "config.yml"
    config = yaml.safe_load(config_path.read_text())
    ds_cfg = resolve_dataset_config(config, dataset_name)
    _, learning, _ = Metadata_MLP.load_config(config_path, dataset_name, "Metadata_MLP")
    checkpoint_path = PROJECT_ROOT / learning["output"]["directory"] / "best.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    features = list(checkpoint["metadata_features"])
    label_to_index = dict(checkpoint["label_to_index"])
    index_to_label = {index: label for label, index in label_to_index.items()}
    if "legit" not in label_to_index or len(index_to_label) != len(label_to_index):
        raise ValueError(f"{dataset_name}: invalid label mapping {label_to_index}")

    bundle = {
        "vocab_size": VOCAB_SIZE,
        "num_classes": int(checkpoint["num_classes"]),
        "metadata_dim": len(features),
        "metadata_features": features,
    }
    model = Metadata_MLP.build_model(bundle, learning["model"], device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    category_indices = {
        category: [
            index
            for index, feature in enumerate(features)
            if any(
                fnmatchcase(feature, pattern)
                for pattern in config["metadata_categories"][category]
            )
        ]
        for category in CATEGORIES
    }
    membership = [
        sum(index in indices for indices in category_indices.values())
        for index in range(len(features))
    ]
    if not all(count == 1 for count in membership):
        raise ValueError(
            f"{dataset_name}: category patterns do not partition metadata features; "
            f"membership counts={dict(zip(features, membership))}"
        )

    dataset_dir = Path(ds_cfg["path"])
    test_ds, domains, raw_targets = _load_split_ds(
        dataset_dir / ds_cfg["splits"]["test"],
        ds_cfg["columns"]["domain"],
        ds_cfg["columns"]["target"],
        features,
        label_to_index,
        checkpoint["metadata_imputer"],
        checkpoint["metadata_scaler"],
        config["learning_combinations"]["char_metadata"]["data"]["max_length"],
    )
    return (
        checkpoint_path,
        model,
        test_ds,
        domains,
        raw_targets,
        label_to_index,
        index_to_label,
        category_indices,
    )


def predict(model, test_ds, device: torch.device) -> torch.Tensor:
    predictions = []
    with torch.no_grad():
        for start in range(0, len(test_ds), 512):
            predictions.append(
                model(test_ds.metadata[start : start + 512].to(device))
                .argmax(dim=1)
                .cpu()
            )
    return torch.cat(predictions)


def write_ig(
    dataset_name: str,
    model,
    test_ds,
    domains,
    raw_targets,
    predictions: torch.Tensor,
    sample_indices: list[int],
    index_to_label: dict[int, str],
    category_indices: dict[str, list[int]],
    device: torch.device,
) -> Path:
    rows = []
    for start in range(0, len(sample_indices), BATCH_SIZE):
        batch_indices = sample_indices[start : start + BATCH_SIZE]
        index_tensor = torch.as_tensor(batch_indices, dtype=torch.long)
        metadata = test_ds.metadata[index_tensor].to(device)
        true_classes = test_ds.labels[index_tensor].to(device)
        feature_attr = ig_mlp_batch(
            model,
            None,
            metadata,
            true_classes,
            device,
            steps=IG_STEPS,
        )
        category_scores = {
            category: feature_attr[:, indices].sum(dim=1).cpu().tolist()
            for category, indices in category_indices.items()
        }
        for offset, sample_id in enumerate(batch_indices):
            true_label = raw_targets[sample_id]
            predicted_label = index_to_label[int(predictions[sample_id])]
            row = {
                "sample_id": sample_id,
                "example": domains[sample_id],
                "true_label": true_label,
                "predicted_label": predicted_label,
                "classification": "Correct" if true_label == predicted_label else "Incorrect",
            }
            row.update(
                {
                    f"{category}_contribution_score": float(category_scores[category][offset])
                    for category in CATEGORIES
                }
            )
            rows.append(row)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"IG_CATEGORIES_{dataset_name}.csv"
    with output_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if len(rows) != len(sample_indices):
        raise RuntimeError(f"{dataset_name}: wrote {len(rows)} rows for {len(sample_indices)} samples")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Compute and save category IG; omit for inference-only preflight.",
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
            f"category_features={{{', '.join(f'{k}: {len(v)}' for k, v in category_indices.items())}}}"
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
                category_indices,
                device,
            )
            print(f"Wrote {len(sample_indices)} rows to {output_path}")

    print(f"total_malicious={total_samples} execute={args.execute}")


if __name__ == "__main__":
    main()
