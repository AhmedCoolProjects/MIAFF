#!/usr/bin/env python3
"""Run metadata-only MLP ablations that keep or drop one metadata category."""

from __future__ import annotations

import argparse
from fnmatch import fnmatchcase
from pathlib import Path
import sys

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "01_metadata_selection"))
sys.path.insert(0, str(PROJECT_ROOT / "04_Learning"))

from data_loader import resolve_dataset_config  # noqa: E402
from Metadata_MLP import (  # noqa: E402
    MODEL_NAME,
    build_model,
    execute_experiment,
    load_config,
    release_memory,
    smoke_test,
)


def _unique_strings(value, field):
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
        or len(value) != len(set(value))
    ):
        raise ValueError(f"{field} must be a non-empty list of unique strings")
    return value


def format_ablation_template(value, field, **values):
    if not isinstance(value, str) or not value:
        raise ValueError(f"ablation.logging.wandb.{field} must be a non-empty string")
    try:
        return value.format(**values)
    except KeyError as error:
        raise ValueError(
            f"Unsupported template field in ablation.logging.wandb.{field}"
        ) from error


def load_ablation_plan(config_path):
    config_path = Path(config_path).resolve()
    with config_path.open() as stream:
        config = yaml.safe_load(stream)
    try:
        ablation = config["ablation"]
        metadata_categories = config["metadata_categories"]
    except (KeyError, TypeError) as error:
        raise ValueError("ablation and metadata_categories configuration is required") from error
    if not isinstance(ablation, dict) or not isinstance(metadata_categories, dict):
        raise ValueError("ablation and metadata_categories must be mappings")

    models = _unique_strings(ablation.get("model"), "ablation.model")
    if models != [MODEL_NAME] or models != config.get("learning_mlp", {}).get(
        "model_names"
    ):
        raise ValueError(
            f"ablation.model and learning_mlp.model_names must both be [{MODEL_NAME!r}]"
        )

    category_modes = {}
    patterns = {}
    for mode in ("only", "drop"):
        categories = _unique_strings(
            ablation.get("categories", {}).get(mode),
            f"ablation.categories.{mode}",
        )
        category_modes[mode] = categories
        for category in categories:
            if category == "all" or category not in metadata_categories:
                raise ValueError(f"Unknown metadata category {category!r}")
            if category not in patterns:
                patterns[category] = _unique_strings(
                    metadata_categories[category],
                    f"metadata_categories.{category}",
                )
    try:
        ablation_wandb = ablation["logging"]["wandb"]
    except (KeyError, TypeError) as error:
        raise ValueError("ablation.logging.wandb configuration is required") from error
    if not isinstance(ablation_wandb, dict):
        raise ValueError("ablation.logging.wandb must be a mapping")
    missing = {"project", "group", "run_name", "tags"} - set(ablation_wandb)
    if missing:
        raise ValueError(f"Invalid ablation W&B settings; missing={sorted(missing)}")
    template_values = {
        "mode": "only",
        "category": "rdap",
        "model": MODEL_NAME,
        "dataset": "dataset",
        "scenario": "scenario",
    }
    for field in ("project", "group", "run_name"):
        format_ablation_template(
            ablation_wandb[field],
            field,
            **template_values,
        )
    _unique_strings(ablation_wandb["tags"], "ablation.logging.wandb.tags")


    datasets = []
    for scenario in config.get("scenarios", {}).values():
        datasets.extend(dataset["name"] for dataset in scenario.get("datasets", []))
    _unique_strings(datasets, "configured datasets")
    return (
        config,
        config_path,
        models,
        category_modes,
        patterns,
        datasets,
        ablation_wandb,
    )


def select_requested(configured, requested, field):
    if not requested:
        return configured
    selected = _unique_strings(requested, field)
    unknown = set(selected) - set(configured)
    if unknown:
        raise ValueError(f"Unknown {field}: {sorted(unknown)}")
    return selected


def selected_features(config, config_path, dataset_name, patterns, mode):
    dataset = resolve_dataset_config(config, dataset_name)
    feature_path = Path(dataset["selected_infra_metadata"]).expanduser()
    if not feature_path.is_absolute():
        feature_path = config_path.parent / feature_path
    features = [
        line.strip()
        for line in feature_path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(features) != len(set(features)):
        raise ValueError(
            f"Dataset {dataset_name!r} contains duplicate selected metadata features"
        )
    matches = [
        feature
        for feature in features
        if any(fnmatchcase(feature, pattern) for pattern in patterns)
    ]
    if mode == "only":
        selected = matches
    elif mode == "drop":
        dropped = set(matches)
        selected = [feature for feature in features if feature not in dropped]
    else:
        raise ValueError(f"Unsupported ablation mode {mode!r}")
    if not selected:
        raise ValueError(
            f"Dataset {dataset_name!r} has no selected features for mode={mode} "
            f"patterns={patterns}"
        )
    return selected


def self_check(
    config,
    config_path,
    datasets,
    models,
    category_modes,
    patterns,
    ablation_wandb,
):
    counts = {}
    for mode, categories in category_modes.items():
        for dataset_name in datasets:
            for model_name in models:
                for category in categories:
                    selected = selected_features(
                        config,
                        config_path,
                        dataset_name,
                        patterns[category],
                        mode,
                    )
                    dataset_config, learning, _, experiment = prepare_experiment(
                        config_path,
                        dataset_name,
                        model_name,
                        mode,
                        category,
                        selected,
                        ablation_wandb,
                    )
                    assert dataset_config["task"] in {
                        "binary_classification",
                        "multi_class_classification",
                    }
                    assert experiment["mode"] == mode
                    assert experiment["category"] == category
                    wandb_config = learning["logging"]["wandb"]
                    assert wandb_config["project"] == "miaff"
                    assert wandb_config["group"] == "mlp_ablation"
                    assert wandb_config["run_name"] == f"{mode}_{category}"
                    assert wandb_config["tags"] == ["metadata", "ablation"]
                    metric_key = (
                        "binary_metrics"
                        if dataset_config["task"] == "binary_classification"
                        else "multi_class_metrics"
                    )
                    assert learning["testing"]["metrics"] == [
                        metric.lower()
                        for metric in config["learning_mlp"]["testing"][metric_key]
                    ]
                    assert learning["logging"]["model"] == config["learning_mlp"][
                        "logging"
                    ]["model"]
                    assert learning["logging"]["system"] == config["learning_mlp"][
                        "logging"
                    ]["system"]
                    expected_output = (
                        Path("09_output")
                        / "06_Interpreting"
                        / "Ablation"
                        / model_name
                        / f"{mode}_{category}"
                        / dataset_name
                    )
                    assert Path(learning["output"]["directory"]) == expected_output
                    model = build_model(
                        {"metadata_dim": len(selected), "num_classes": 3},
                        learning["model"],
                        torch.device("cpu"),
                    )
                    with torch.inference_mode():
                        output = model(torch.zeros(2, len(selected)))
                    assert output.shape == (2, 3)
                    counts[(mode, dataset_name, category)] = len(selected)
    print(
        f"ablation self-check passed: {len(counts)} mode/dataset/category runs; "
        f"feature_counts={counts}"
    )


def prepare_experiment(
    config_path,
    dataset_name,
    model_name,
    mode,
    category,
    selected,
    ablation_wandb,
):
    dataset_config, learning, config_path = load_config(
        config_path,
        dataset_name,
        model_name,
    )
    template_values = {
        "mode": mode,
        "category": category,
        "model": model_name,
        "dataset": dataset_name,
        "scenario": dataset_config["scenario"],
    }
    wandb_config = {
        **learning.get("logging", {}).get("wandb", {}),
        **ablation_wandb,
    }
    for field in ("project", "group", "run_name"):
        wandb_config[field] = format_ablation_template(
            wandb_config[field],
            field,
            **template_values,
        )
    wandb_config["tags"] = [
        format_ablation_template(tag, "tags", **template_values)
        for tag in ablation_wandb["tags"]
    ]
    learning["logging"]["wandb"] = wandb_config
    run_name = wandb_config["run_name"]
    learning["output"]["directory"] = str(
        Path("09_output")
        / "06_Interpreting"
        / "Ablation"
        / model_name
        / run_name
        / dataset_name
    )
    experiment = {
        "type": "metadata_category_ablation",
        "mode": mode,
        "category": category,
        "patterns": list(selected),
    }
    return dataset_config, learning, config_path, experiment


def run_ablation(
    config,
    config_path,
    datasets,
    models,
    category_modes,
    patterns,
    ablation_wandb,
):
    total = sum(
        len(datasets) * len(models) * len(categories)
        for categories in category_modes.values()
    )
    index = 0
    for mode, categories in category_modes.items():
        for dataset_name in datasets:
            for model_name in models:
                for category in categories:
                    index += 1
                    selected = selected_features(
                        config,
                        config_path,
                        dataset_name,
                        patterns[category],
                        mode,
                    )
                    print(
                        f"=== {mode}_{category} / {model_name} / {dataset_name} "
                        f"({index}/{total}) ==="
                    )
                    dataset_config, learning, resolved_path, experiment = (
                        prepare_experiment(
                            config_path,
                            dataset_name,
                            model_name,
                            mode,
                            category,
                            selected,
                            ablation_wandb,
                        )
                    )
                    execute_experiment(
                        dataset_config,
                        learning,
                        resolved_path,
                        dataset_name,
                        metadata_patterns=selected,
                        experiment=experiment,
                    )
                    release_memory()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yml")
    parser.add_argument("--dataset", action="append")
    parser.add_argument("--category", action="append")
    parser.add_argument("--mode", choices=("only", "drop", "all"), default="all")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    (
        config,
        config_path,
        models,
        category_modes,
        patterns,
        datasets,
        ablation_wandb,
    ) = load_ablation_plan(args.config)
    datasets = select_requested(datasets, args.dataset, "ablation datasets")
    if args.mode != "all":
        category_modes = {
            args.mode: select_requested(
                category_modes[args.mode],
                args.category,
                f"ablation.categories.{args.mode}",
            )
        }
    elif args.category:
        category_modes = {
            mode: select_requested(
                categories,
                args.category,
                f"ablation.categories.{mode}",
            )
            for mode, categories in category_modes.items()
        }
    if args.self_check:
        self_check(
            config,
            config_path,
            datasets,
            models,
            category_modes,
            patterns,
            ablation_wandb,
        )
        return
    if args.smoke_test:
        # ponytail: smoke mode checks selected runs without W&B/artifact writes;
        # use normal mode when full checkpoint and reporting artifacts are required.
        for mode, categories in category_modes.items():
            for dataset_name in datasets:
                for model_name in models:
                    for category in categories:
                        selected = selected_features(
                            config,
                            config_path,
                            dataset_name,
                            patterns[category],
                            mode,
                        )
                        print(
                            f"=== smoke {mode}_{category} / {model_name} / "
                            f"{dataset_name} ==="
                        )
                        smoke_test(
                            config_path,
                            dataset_name,
                            model_name,
                            metadata_patterns=selected,
                        )
        return
    run_ablation(
        config,
        config_path,
        datasets,
        models,
        category_modes,
        patterns,
        ablation_wandb,
    )


if __name__ == "__main__":
    main()