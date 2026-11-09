#!/usr/bin/env python3
"""Build domain data loaders for character and language models."""

from __future__ import annotations

import argparse
from fnmatch import fnmatchcase
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


CHARACTERS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
PAD_INDEX = 0
UNKNOWN_INDEX = 1
CHAR_TO_INDEX = {character: index + 2 for index, character in enumerate(CHARACTERS)}
VOCAB_SIZE = len(CHARACTERS) + 2

def split_url(url):
    """DomURLs_BERT preprocessing: mark domain/IP and optional path."""
    from urllib.parse import urlparse
    import re

    def is_ip_address_or_domain(input_string):
        domain_pattern = r"^[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        if str(input_string).isspace() or input_string is None:
            return "[UNK]"
        input_string = input_string.strip("[]")
        parts = input_string.split(":")
        ip_string = input_string if len(parts) > 2 else parts[0]
        try:
            import ipaddress

            ip = ipaddress.ip_address(ip_string)
            return "[IP]" if ip.version == 4 else "[IPv6]"
        except ValueError:
            pass
        if re.match(domain_pattern, input_string):
            return "[DOMAIN]"
        return "[DOMAIN]"

    original_url = url
    edited_url = False
    if "http" not in url:
        url = "http://" + url
    if "/../" in url:
        url = url.replace("/../", "/")
        edited_url = True
    try:
        url_obj = urlparse(url)
        domain_or_ip = url_obj.netloc
        if edited_url:
            path = original_url.partition(domain_or_ip)[2]
        else:
            path = url_obj.path + ("?" + url_obj.query if url_obj.query else "")
        token = is_ip_address_or_domain(domain_or_ip)
        if path:
            return f"{token} {domain_or_ip} [PATH] {path}"
        return f"{token} {domain_or_ip}"
    except Exception:
        return f"[DOMAIN] {original_url}"


class DomURLsBERTDataset(Dataset):
    def __init__(
        self,
        domains,
        labels,
        tokenizer,
        max_length=64,
        metadata=None,
        preprocess_input=False,
    ):
        if max_length < 1:
            raise ValueError("max_length must be positive")
        if len(domains) != len(labels):
            raise ValueError("domains and labels must have equal lengths")
        if metadata is not None and len(metadata) != len(labels):
            raise ValueError("metadata and labels must have equal lengths")
        self.domains = [str(domain) for domain in domains]
        self.labels = torch.as_tensor(labels, dtype=torch.long)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.preprocess_input = preprocess_input
        self.metadata = (
            torch.as_tensor(metadata, dtype=torch.float32)
            if metadata is not None
            else None
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        text = self.domains[index]
        encoded = self.tokenizer(
            split_url(text) if self.preprocess_input else text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        sample = {
            "input_ids": encoded["input_ids"].flatten(),
            "attention_mask": encoded["attention_mask"].flatten(),
            "label": self.labels[index],
        }
        if self.metadata is not None:
            sample["metadata"] = self.metadata[index]
        return sample



class DomainDataset(Dataset):
    def __init__(self, domains, labels, max_length=64, metadata=None):
        if max_length < 1:
            raise ValueError("max_length must be positive")
        if len(domains) != len(labels):
            raise ValueError("domains and labels must have equal lengths")
        if metadata is not None and len(metadata) != len(labels):
            raise ValueError("metadata and labels must have equal lengths")

        encoded = np.full((len(domains), max_length), PAD_INDEX, dtype=np.int64)
        for row, domain in enumerate(domains):
            text = str(domain)
            limit = min(len(text), max_length)
            encoded[row, :limit] = [
                CHAR_TO_INDEX.get(character, UNKNOWN_INDEX) for character in text[:limit]
            ]

        self.inputs = torch.from_numpy(encoded)
        self.labels = torch.as_tensor(labels, dtype=torch.long)
        self.metadata = (
            torch.as_tensor(metadata, dtype=torch.float32)
            if metadata is not None
            else None
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        sample = {"input": self.inputs[index], "label": self.labels[index]}
        if self.metadata is not None:
            sample["metadata"] = self.metadata[index]
        return sample


def _load_env_if_present():
    import os
    env_file = Path(__file__).resolve().parents[1] / ".env"
    if env_file.is_file():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("'\""))


_load_env_if_present()


def _resolve_path(value, config_dir):
    import os
    import re

    def _repl(match):
        var_name, default_val = match.group(1), match.group(2)
        return os.environ.get(var_name, default_val if default_val is not None else "")

    expanded = re.sub(r"\$\{([A-Za-z0-9_]+)(?::-([^}]*))?\}", _repl, str(value))
    expanded = os.path.expandvars(expanded)
    path = Path(expanded).expanduser()
    return path if path.is_absolute() else (config_dir / path).resolve()

def resolve_dataset_config(config, dataset_name):
    matches = []
    for scenario_name, scenario in config.get("scenarios", {}).items():
        for dataset in scenario.get("datasets", []):
            if dataset.get("name") == dataset_name:
                inherited = {
                    key: value for key, value in scenario.items() if key != "datasets"
                }
                matches.append({**inherited, **dataset, "scenario": scenario_name})
    if len(matches) != 1:
        raise ValueError(
            f"Dataset {dataset_name!r} must occur exactly once across scenarios"
        )
    dataset = matches[0]
    if dataset.get("task") not in {"binary_classification", "multi_class_classification"}:
        raise ValueError(
            "Supported tasks are binary_classification and multi_class_classification"
        )
    return dataset


def _read_dataset_config(config_path, dataset_name):
    config_path = Path(config_path).resolve()
    with config_path.open() as stream:
        config = yaml.safe_load(stream)
    return resolve_dataset_config(config, dataset_name), config_path.parent


def build_dataloaders(
    config_path,
    dataset_name,
    *,
    include_selected_metadata=False,
    metadata_patterns=None,
    batch_size=256,
    max_length=64,
    num_workers=0,
    seed=42,
):
    """Return train/dev/test loaders and the fitted train-only preprocessing state."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    if metadata_patterns is not None and not include_selected_metadata:
        raise ValueError("metadata_patterns requires include_selected_metadata=True")

    config, config_dir = _read_dataset_config(config_path, dataset_name)
    dataset_dir = _resolve_path(config["path"], config_dir)
    columns = config["columns"]
    domain_column = columns["domain"]
    target_column = columns["target"]
    task = config["task"]
    metadata_features = []

    if include_selected_metadata:
        feature_path = _resolve_path(config["selected_infra_metadata"], config_dir)
        metadata_features = [
            line.strip()
            for line in feature_path.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        if metadata_patterns is not None:
            if (
                isinstance(metadata_patterns, str)
                or not isinstance(metadata_patterns, (list, tuple))
                or not metadata_patterns
                or any(
                    not isinstance(pattern, str) or not pattern
                    for pattern in metadata_patterns
                )
            ):
                raise ValueError(
                    "metadata_patterns must be a non-empty list of strings"
                )
            metadata_features = [
                feature
                for feature in metadata_features
                if any(fnmatchcase(feature, pattern) for pattern in metadata_patterns)
            ]
            if not metadata_features:
                raise ValueError("No selected metadata features match metadata_patterns")
        if not metadata_features or len(metadata_features) != len(set(metadata_features)):
            raise ValueError("Selected metadata file must contain unique feature names")

    required_columns = [domain_column, target_column, *metadata_features]
    frames = {}
    for split, filename in config["splits"].items():
        split_path = dataset_dir / filename
        available = set(pd.read_csv(split_path, nrows=0).columns)
        missing = set(required_columns) - available
        if missing:
            raise ValueError(f"{split} split lacks columns: {sorted(missing)}")
        frame = pd.read_csv(split_path, usecols=required_columns)
        if frame[domain_column].isna().any() or frame[target_column].isna().any():
            raise ValueError(f"{split} split contains missing domains or labels")
        frames[split] = frame

    expected_splits = {"train", "dev", "test"}
    if set(frames) != expected_splits:
        raise ValueError(f"splits must be exactly {sorted(expected_splits)}")

    if task == "binary_classification":
        negative_label = config["labels"]["negative"]
        positive_label = config["labels"]["positive"]
        label_to_index = {negative_label: 0, positive_label: 1}
        class_names = [negative_label, positive_label]
        for split, frame in frames.items():
            observed = set(frame[target_column].astype(str).unique())
            if observed != set(label_to_index):
                raise ValueError(
                    f"{split} labels are {sorted(observed)}, expected {sorted(label_to_index)}"
                )
    else:
        if config.get("labels") is not None:
            raise ValueError("multi_class_classification must not define binary labels")
        class_names = sorted(frames["train"][target_column].astype(str).unique().tolist())
        if len(class_names) < 3:
            raise ValueError("multi_class_classification requires at least 3 classes")
        label_to_index = {name: index for index, name in enumerate(class_names)}
        for split, frame in frames.items():
            unknown = set(frame[target_column].astype(str).unique()) - set(label_to_index)
            if unknown:
                raise ValueError(f"{split} contains classes unseen in train: {sorted(unknown)}")

    metadata_by_split = {split: None for split in frames}
    imputer = scaler = None
    if include_selected_metadata:
        numeric = {
            split: frame[metadata_features].apply(pd.to_numeric, errors="raise")
            for split, frame in frames.items()
        }
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        train_metadata = imputer.fit_transform(numeric["train"])
        scaler.fit(train_metadata)
        for split, values in numeric.items():
            transformed = scaler.transform(imputer.transform(values)).astype(np.float32)
            if transformed.shape[1] != len(metadata_features):
                raise ValueError("Imputation changed the metadata feature count")
            metadata_by_split[split] = transformed

    datasets = {}
    loaders = {}
    generator = torch.Generator().manual_seed(seed)
    for split, frame in frames.items():
        labels = (
            frame[target_column].astype(str).map(label_to_index).to_numpy(dtype=np.int64)
        )
        datasets[split] = DomainDataset(
            frame[domain_column].astype(str).to_numpy(),
            labels,
            max_length=max_length,
            metadata=metadata_by_split[split],
        )
        loaders[split] = DataLoader(
            datasets[split],
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            generator=generator if split == "train" else None,
        )

    return {
        "loaders": loaders,
        "datasets": datasets,
        "vocab_size": VOCAB_SIZE,
        "num_classes": len(class_names),
        "class_names": class_names,
        "label_to_index": label_to_index,
        "task": task,
        "metadata_dim": len(metadata_features),
        "metadata_features": metadata_features,
        "metadata_imputer": imputer,
        "metadata_scaler": scaler,
    }

def build_lang_dataloaders(
    config_path,
    dataset_name,
    *,
    pretrained_path="amahdaouy/DomURLs_BERT",
    include_selected_metadata=False,
    metadata_patterns=None,
    batch_size=128,
    max_length=64,
    num_workers=0,
    seed=42,
):
    """Return tokenizer-based loaders for DomURLsBERT."""
    from transformers import AutoTokenizer

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    if metadata_patterns is not None and not include_selected_metadata:
        raise ValueError("metadata_patterns requires include_selected_metadata=True")

    config, config_dir = _read_dataset_config(config_path, dataset_name)
    dataset_dir = _resolve_path(config["path"], config_dir)
    columns = config["columns"]
    domain_column = columns["domain"]
    target_column = columns["target"]
    task = config["task"]
    metadata_features = []

    if include_selected_metadata:
        feature_path = _resolve_path(config["selected_infra_metadata"], config_dir)
        metadata_features = [
            line.strip()
            for line in feature_path.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        if metadata_patterns is not None:
            if (
                isinstance(metadata_patterns, str)
                or not isinstance(metadata_patterns, (list, tuple))
                or not metadata_patterns
                or any(
                    not isinstance(pattern, str) or not pattern
                    for pattern in metadata_patterns
                )
            ):
                raise ValueError(
                    "metadata_patterns must be a non-empty list of strings"
                )
            metadata_features = [
                feature
                for feature in metadata_features
                if any(fnmatchcase(feature, pattern) for pattern in metadata_patterns)
            ]
            if not metadata_features:
                raise ValueError("No selected metadata features match metadata_patterns")
        if not metadata_features or len(metadata_features) != len(set(metadata_features)):
            raise ValueError("Selected metadata file must contain unique feature names")

    required_columns = [domain_column, target_column, *metadata_features]
    frames = {}
    for split, filename in config["splits"].items():
        split_path = dataset_dir / filename
        available = set(pd.read_csv(split_path, nrows=0).columns)
        missing = set(required_columns) - available
        if missing:
            raise ValueError(f"{split} split lacks columns: {sorted(missing)}")
        frame = pd.read_csv(split_path, usecols=required_columns)
        if frame[domain_column].isna().any() or frame[target_column].isna().any():
            raise ValueError(f"{split} split contains missing domains or labels")
        frames[split] = frame

    expected_splits = {"train", "dev", "test"}
    if set(frames) != expected_splits:
        raise ValueError(f"splits must be exactly {sorted(expected_splits)}")

    if task == "binary_classification":
        negative_label = config["labels"]["negative"]
        positive_label = config["labels"]["positive"]
        label_to_index = {negative_label: 0, positive_label: 1}
        class_names = [negative_label, positive_label]
        for split, frame in frames.items():
            observed = set(frame[target_column].astype(str).unique())
            if observed != set(label_to_index):
                raise ValueError(
                    f"{split} labels are {sorted(observed)}, expected {sorted(label_to_index)}"
                )
    else:
        if config.get("labels") is not None:
            raise ValueError("multi_class_classification must not define binary labels")
        class_names = sorted(frames["train"][target_column].astype(str).unique().tolist())
        if len(class_names) < 3:
            raise ValueError("multi_class_classification requires at least 3 classes")
        label_to_index = {name: index for index, name in enumerate(class_names)}
        for split, frame in frames.items():
            unknown = set(frame[target_column].astype(str).unique()) - set(label_to_index)
            if unknown:
                raise ValueError(f"{split} contains classes unseen in train: {sorted(unknown)}")

    metadata_by_split = {split: None for split in frames}
    imputer = scaler = None
    if include_selected_metadata:
        numeric = {
            split: frame[metadata_features].apply(pd.to_numeric, errors="raise")
            for split, frame in frames.items()
        }
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        train_metadata = imputer.fit_transform(numeric["train"])
        scaler.fit(train_metadata)
        for split, values in numeric.items():
            transformed = scaler.transform(imputer.transform(values)).astype(np.float32)
            if transformed.shape[1] != len(metadata_features):
                raise ValueError("Imputation changed the metadata feature count")
            metadata_by_split[split] = transformed

    tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
    datasets = {}
    loaders = {}
    generator = torch.Generator().manual_seed(seed)
    for split, frame in frames.items():
        labels = (
            frame[target_column].astype(str).map(label_to_index).to_numpy(dtype=np.int64)
        )
        datasets[split] = DomURLsBERTDataset(
            frame[domain_column].astype(str).to_numpy(),
            labels,
            tokenizer=tokenizer,
            max_length=max_length,
            metadata=metadata_by_split[split],
            preprocess_input=pretrained_path == "amahdaouy/DomURLs_BERT",
        )
        loaders[split] = DataLoader(
            datasets[split],
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            generator=generator if split == "train" else None,
        )

    return {
        "loaders": loaders,
        "datasets": datasets,
        "num_classes": len(class_names),
        "class_names": class_names,
        "label_to_index": label_to_index,
        "task": task,
        "pretrained_path": pretrained_path,
        "metadata_dim": len(metadata_features),
        "metadata_features": metadata_features,
        "metadata_imputer": imputer,
        "metadata_scaler": scaler,
    }



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "config.yml",
    )
    parser.add_argument("--dataset", default="threatfox_malware_binary")
    parser.add_argument("--include-selected-metadata", action="store_true")
    parser.add_argument("--metadata-pattern", action="append")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    bundle = build_dataloaders(
        args.config,
        args.dataset,
        include_selected_metadata=args.include_selected_metadata,
        metadata_patterns=args.metadata_pattern,
        batch_size=args.batch_size,
        max_length=args.max_length,
        num_workers=args.num_workers,
    )
    batch = next(iter(bundle["loaders"]["train"]))
    expected_keys = {"input", "label", "metadata"} if args.include_selected_metadata else {"input", "label"}
    assert set(batch) == expected_keys
    assert batch["input"].shape[1] == args.max_length
    print(
        f"train={len(bundle['datasets']['train'])} "
        f"input={tuple(batch['input'].shape)} "
        f"metadata_dim={bundle['metadata_dim']}"
    )


if __name__ == "__main__":
    main()