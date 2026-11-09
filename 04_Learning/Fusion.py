#!/usr/bin/env python3
"""Train text and metadata encoders jointly with configurable fusion."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "01_metadata_selection"))
sys.path.insert(0, str(PROJECT_ROOT / "02_NN_Models"))

from data_loader import (  # noqa: E402
    build_dataloaders,
    build_lang_dataloaders,
    resolve_dataset_config,
)
from BASELINE_Char import (  # noqa: E402
    MODEL_REGISTRY as CHAR_MODEL_REGISTRY,
    SUPPORTED_BINARY_METRICS,
    SUPPORTED_MODEL_INFO,
    SUPPORTED_MULTICLASS_METRICS,
    SUPPORTED_SYSTEM_METRICS,
    SYSTEM_METRIC_UNITS,
    calculate_metrics,
    configured_model_info,
    load_wandb_api_key,
    measure_stage,
    release_memory,
    resolve_device,
    resolve_model_config as resolve_char_model_config,
    set_seed,
    summarize_model,
)
from BASELINE_LANG import (  # noqa: E402
    MODEL_REGISTRY as LANG_MODEL_REGISTRY,
    resolve_model_config as resolve_lang_model_config,
)
from Metadata_MLP import (  # noqa: E402
    Metadata_MLP,
    resolve_model_config as resolve_metadata_model_config,
)

IMPLEMENTED_SAMPLING_STRATEGIES = {"natural"}
IMPLEMENTED_FUSION_METHODS = {"concat", "gated_residual"}
IMPLEMENTED_CLASS_PRIOR_CORRECTIONS = {"none"}
IMPLEMENTED_LOSS_OBJECTIVES = {
    "cross_entropy",
    "focal",
    "class_weighted_cross_entropy",
    "logit_adjusted_cross_entropy",
}


class ConcatenationFusion(nn.Module):
    def __init__(
        self,
        text_dim,
        metadata_dim,
        num_classes,
        *,
        head_hidden_dim,
        drop_prob,
    ):
        super().__init__()
        in_dim = text_dim + metadata_dim
        self.classifier = (
            nn.Sequential(
                nn.Linear(in_dim, head_hidden_dim),
                nn.GELU(),
                nn.Dropout(drop_prob),
                nn.Linear(head_hidden_dim, num_classes),
            )
            if head_hidden_dim > 0
            else nn.Linear(in_dim, num_classes)
        )

    def forward(self, text_features, metadata_features):
        return self.classifier(torch.cat((text_features, metadata_features), dim=1))


class GatedResidualFusion(nn.Module):
    def __init__(
        self,
        text_dim,
        metadata_dim,
        num_classes,
        *,
        fusion_dim,
        head_hidden_dim,
        drop_prob,
    ):
        super().__init__()
        self.text_projection = nn.Linear(text_dim, fusion_dim)
        self.metadata_projection = nn.Linear(metadata_dim, fusion_dim)
        self.gate = nn.Linear(fusion_dim * 2, fusion_dim)
        self.norm = nn.LayerNorm(fusion_dim)
        self.classifier = (
            nn.Sequential(
                nn.Linear(fusion_dim, head_hidden_dim),
                nn.GELU(),
                nn.Dropout(drop_prob),
                nn.Linear(head_hidden_dim, num_classes),
            )
            if head_hidden_dim > 0
            else nn.Linear(fusion_dim, num_classes)
        )

    def forward(self, text_features, metadata_features):
        text_features = self.text_projection(text_features)
        metadata_features = self.metadata_projection(metadata_features)
        gate = torch.sigmoid(
            self.gate(torch.cat((text_features, metadata_features), dim=1))
        )
        fused = self.norm(text_features + gate * metadata_features)
        return self.classifier(fused)


class TextMetadataFusion(nn.Module):
    def __init__(self, text_encoder, metadata_encoder, fusion):
        super().__init__()
        self.text_encoder = text_encoder
        self.metadata_encoder = metadata_encoder
        self.fusion = fusion

    def forward(self, inputs, metadata):
        text_features = (
            self.text_encoder.extract_features(**inputs)
            if isinstance(inputs, dict)
            else self.text_encoder.extract_features(inputs)
        )
        return self.fusion(
            text_features,
            self.metadata_encoder.extract_features(metadata),
        )


class FocalLoss(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        cross_entropy = F.cross_entropy(logits, targets, reduction="none")
        return (((1 - torch.exp(-cross_entropy)) ** self.gamma) * cross_entropy).mean()


class LogitAdjustedCrossEntropy(nn.Module):
    def __init__(self, class_counts, tau):
        super().__init__()
        priors = class_counts / class_counts.sum()
        self.register_buffer("adjustment", tau * priors.log())

    def forward(self, logits, targets):
        return F.cross_entropy(logits + self.adjustment, targets)



def _unique_strings(value, field):
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
        or len(value) != len(set(value))
    ):
        raise ValueError(f"{field} must be a non-empty list of unique strings")
    return value


def format_template(
    value,
    *,
    combination_name,
    model_name,
    dataset_name,
    scenario_name,
):
    if not isinstance(value, str):
        raise ValueError("Output and W&B template values must be strings")
    try:
        return value.format(
            combination=combination_name,
            model=model_name,
            dataset=dataset_name,
            scenario=scenario_name,
        )
    except KeyError as error:
        raise ValueError(f"Unsupported template field in {value!r}") from error


def resolve_metadata_patterns(config, category_names):
    categories = config.get("metadata_categories")
    if not isinstance(categories, dict):
        raise ValueError("metadata_categories must be a mapping")
    patterns = []
    for category_name in _unique_strings(
        category_names, "learning_combinations.char_metadata.metadata_categories"
    ):
        values = categories.get(category_name)
        _unique_strings(values, f"metadata_categories.{category_name}")
        for pattern in values:
            if pattern not in patterns:
                patterns.append(pattern)
    return patterns


def resolve_combination(config, combination_name):
    combinations = config.get("combinations")
    components = config.get("combination_components")
    if not isinstance(combinations, dict) or combination_name not in combinations:
        raise ValueError(f"Unknown combination {combination_name!r}")
    if not isinstance(components, dict):
        raise ValueError("combination_components must be a mapping")

    combination = combinations[combination_name]
    if not isinstance(combination, dict):
        raise ValueError(f"combinations.{combination_name} must be a mapping")
    references = {
        "sampling_strategy": (
            "sampling_strategies",
            IMPLEMENTED_SAMPLING_STRATEGIES,
        ),
        "fusion_method": ("fusion_methods", IMPLEMENTED_FUSION_METHODS),
        "class_prior_correction": (
            "class_prior_corrections",
            IMPLEMENTED_CLASS_PRIOR_CORRECTIONS,
        ),
        "loss_objective": ("loss_objectives", IMPLEMENTED_LOSS_OBJECTIVES),
    }
    resolved = {"name": combination_name}
    for field, (registry_name, implemented) in references.items():
        name = combination.get(field)
        registry = components.get(registry_name)
        if not isinstance(registry, dict) or name not in registry:
            raise ValueError(
                f"Combination {combination_name!r} references unknown {field} {name!r}"
            )
        if name not in implemented:
            raise ValueError(
                f"Combination {combination_name!r} selects unsupported "
                f"{field} {name!r}"
            )
        settings = registry[name]
        if not isinstance(settings, dict):
            raise ValueError(f"{registry_name}.{name} must be a mapping")
        resolved[field] = {"name": name, **settings}

    fusion = resolved["fusion_method"]
    for field in ("head_hidden_dim",):
        if not isinstance(fusion.get(field), int) or fusion[field] < 0:
            raise ValueError(f"fusion_methods.{fusion['name']}.{field} must be non-negative")
    if not isinstance(fusion.get("drop_prob"), (int, float)) or not 0 <= fusion[
        "drop_prob"
    ] < 1:
        raise ValueError(
            f"fusion_methods.{fusion['name']}.drop_prob must be in [0, 1)"
        )
    if fusion["name"] == "gated_residual" and (
        not isinstance(fusion.get("fusion_dim"), int) or fusion["fusion_dim"] < 1
    ):
        raise ValueError("fusion_methods.gated_residual.fusion_dim must be positive")

    loss = resolved["loss_objective"]
    if loss["name"] == "cross_entropy":
        smoothing = loss.get("label_smoothing")
        if not isinstance(smoothing, (int, float)) or not 0 <= smoothing < 1:
            raise ValueError(
                "loss_objectives.cross_entropy.label_smoothing must be in [0, 1)"
            )
    elif loss["name"] == "focal":
        gamma = loss.get("gamma")
        if not isinstance(gamma, (int, float)) or gamma < 0:
            raise ValueError("loss_objectives.focal.gamma must be non-negative")
    elif loss["name"] == "class_weighted_cross_entropy":
        if loss.get("weighting") != "inverse_frequency":
            raise ValueError(
                "class_weighted_cross_entropy.weighting must be inverse_frequency"
            )
        if loss.get("normalize") != "mean_one":
            raise ValueError(
                "class_weighted_cross_entropy.normalize must be mean_one"
            )
    elif loss["name"] == "logit_adjusted_cross_entropy":
        tau = loss.get("tau")
        if not isinstance(tau, (int, float)) or tau < 0:
            raise ValueError(
                "loss_objectives.logit_adjusted_cross_entropy.tau "
                "must be non-negative"
            )
    return resolved


def configured_runs(config_path, encoder_type="char"):
    with Path(config_path).resolve().open() as stream:
        config = yaml.safe_load(stream)
    learning_key = f"{encoder_type}_metadata"
    models_key = f"{encoder_type}_models"
    registry = (
        CHAR_MODEL_REGISTRY if encoder_type == "char" else LANG_MODEL_REGISTRY
    )
    try:
        learning = config["learning_combinations"][learning_key]
    except (KeyError, TypeError) as error:
        raise ValueError(
            f"learning_combinations.{learning_key} configuration is required"
        ) from error
    model_names = _unique_strings(
        learning.get(models_key),
        f"learning_combinations.{learning_key}.{models_key}",
    )
    unknown_models = set(model_names) - set(registry)
    if unknown_models:
        raise ValueError(
            f"Unknown {encoder_type} models: {sorted(unknown_models)}"
        )
    combination_names = _unique_strings(
        learning.get("combinations"),
        f"learning_combinations.{learning_key}.combinations",
    )
    for combination_name in combination_names:
        resolve_combination(config, combination_name)
    resolve_metadata_patterns(config, learning.get("metadata_categories"))
    if learning.get("metadata_model") != "Metadata_MLP":
        raise ValueError(f"{learning_key}.metadata_model must be Metadata_MLP")
    return model_names, combination_names


def load_config(
    config_path,
    dataset_name,
    model_name,
    combination_name,
    encoder_type="char",
):
    config_path = Path(config_path).resolve()
    with config_path.open() as stream:
        config = yaml.safe_load(stream)
    learning_key = f"{encoder_type}_metadata"
    resolve_encoder_config = (
        resolve_char_model_config
        if encoder_type == "char"
        else resolve_lang_model_config
    )
    try:
        learning = config["learning_combinations"][learning_key]
        dataset = resolve_dataset_config(config, dataset_name)
        encoder_model = resolve_encoder_config(config["models"], model_name)
        metadata_model = resolve_metadata_model_config(config["models"])
    except (KeyError, TypeError) as error:
        raise ValueError(
            f"Missing fusion, dataset {dataset_name!r}, or model configuration"
        ) from error

    enabled_models, enabled_combinations = configured_runs(config_path, encoder_type)
    if model_name not in enabled_models:
        raise ValueError(f"{encoder_type.title()} model {model_name!r} is not enabled")
    if combination_name not in enabled_combinations:
        raise ValueError(f"Combination {combination_name!r} is not enabled")

    combination = resolve_combination(config, combination_name)
    metadata_patterns = resolve_metadata_patterns(
        config, learning["metadata_categories"]
    )
    data = learning["data"]
    training = learning["training"]
    validation = learning["validation"]
    testing = learning["testing"]
    if data["batch_size"] < 1 or data["max_length"] < 1 or data["num_workers"] < 0:
        raise ValueError(
            "batch_size and max_length must be positive; num_workers must be non-negative"
        )
    if training.get("optimizer", "").lower() != "adamw":
        raise ValueError("Fusion supports optimizer: adamw")
    if training["epochs"] < 1 or training["learning_rate"] <= 0:
        raise ValueError("epochs and learning_rate must be positive")
    if training["weight_decay"] < 0 or training["gradient_clip_norm"] < 0:
        raise ValueError("weight_decay and gradient_clip_norm must be non-negative")
    if validation.get("monitor") != "macro_f1":
        raise ValueError("validation.monitor must be macro_f1")
    if validation["patience"] < 1 or validation["min_delta"] < 0:
        raise ValueError("patience must be positive and min_delta non-negative")
    if testing.get("checkpoint") != "best":
        raise ValueError("testing.checkpoint must be best")

    task = dataset["task"]
    metric_key = (
        "binary_metrics" if task == "binary_classification" else "multi_class_metrics"
    )
    metrics = [metric.lower() for metric in testing.get(metric_key, [])]
    allowed = (
        SUPPORTED_BINARY_METRICS
        if task == "binary_classification"
        else SUPPORTED_MULTICLASS_METRICS
    )
    unknown_metrics = set(metrics) - allowed
    if not metrics or unknown_metrics:
        raise ValueError(
            f"Invalid learning_combinations.{learning_key}.testing.{metric_key}: "
            f"unsupported={sorted(unknown_metrics)}"
        )
    testing["metrics"] = metrics

    template_values = {
        "combination_name": combination_name,
        "model_name": model_name,
        "dataset_name": dataset_name,
        "scenario_name": dataset["scenario"],
    }
    learning["output"]["directory"] = format_template(
        learning["output"]["directory"], **template_values
    )
    logging = learning.get("logging", {})
    unknown_model_fields = set(logging.get("model", [])) - SUPPORTED_MODEL_INFO
    unknown_system_fields = set(logging.get("system", [])) - SUPPORTED_SYSTEM_METRICS
    if unknown_model_fields or unknown_system_fields:
        raise ValueError(
            f"Unsupported logging fields: model={sorted(unknown_model_fields)}, "
            f"system={sorted(unknown_system_fields)}"
        )
    wandb_config = logging.get("wandb")
    if wandb_config and wandb_config.get("enabled", True):
        missing = {"project", "group", "run_name", "tags"} - set(wandb_config)
        if missing or not isinstance(wandb_config["tags"], list):
            raise ValueError(f"Invalid W&B settings; missing={sorted(missing)}")
        wandb_config["group"] = format_template(
            wandb_config["group"], **template_values
        )
        wandb_config["run_name"] = format_template(
            wandb_config["run_name"], **template_values
        )
        wandb_config["tags"] = [
            format_template(tag, **template_values) for tag in wandb_config["tags"]
        ]

    learning["encoder_type"] = encoder_type
    learning["encoder_model"] = encoder_model
    learning["metadata_model_config"] = metadata_model
    learning["combination"] = combination
    learning["metadata_patterns"] = metadata_patterns
    return dataset, learning, config_path


def build_model(bundle, learning, device):
    encoder_type = learning["encoder_type"]
    encoder_config = learning["encoder_model"]
    encoder_parameters = {
        key: value for key, value in encoder_config.items() if key != "name"
    }
    registry = (
        CHAR_MODEL_REGISTRY if encoder_type == "char" else LANG_MODEL_REGISTRY
    )
    if encoder_type == "char":
        encoder_parameters["vocab_size"] = bundle["vocab_size"]
    text_encoder = registry[encoder_config["name"]](
        output_size=bundle["num_classes"],
        **encoder_parameters,
    )
    classifier_name = "fc" if hasattr(text_encoder, "fc") else "classifier"
    setattr(text_encoder, classifier_name, nn.Identity())

    metadata_config = learning["metadata_model_config"]
    metadata_parameters = {
        key: value for key, value in metadata_config.items() if key != "name"
    }
    metadata_encoder = Metadata_MLP(
        in_dim=bundle["metadata_dim"],
        output_size=bundle["num_classes"],
        **metadata_parameters,
    )
    metadata_encoder.classifier = nn.Identity()

    fusion_config = learning["combination"]["fusion_method"]
    fusion_parameters = {
        key: value
        for key, value in fusion_config.items()
        if key != "name"
    }
    fusion_class = {
        "concat": ConcatenationFusion,
        "gated_residual": GatedResidualFusion,
    }[fusion_config["name"]]
    fusion = fusion_class(
        text_encoder.out_dim,
        metadata_encoder.out_dim,
        bundle["num_classes"],
        **fusion_parameters,
    )
    return TextMetadataFusion(text_encoder, metadata_encoder, fusion).to(device)


def build_criterion(combination, class_counts):
    loss = combination["loss_objective"]
    if loss["name"] == "cross_entropy":
        return nn.CrossEntropyLoss(label_smoothing=loss["label_smoothing"])
    if loss["name"] == "focal":
        return FocalLoss(loss["gamma"])
    if class_counts.ndim != 1 or torch.any(class_counts <= 0):
        raise ValueError("Every class must have a positive training sample count")
    class_counts = class_counts.to(dtype=torch.float32)
    if loss["name"] == "class_weighted_cross_entropy":
        weights = class_counts.reciprocal()
        weights /= weights.mean()
        return nn.CrossEntropyLoss(weight=weights)
    if loss["name"] == "logit_adjusted_cross_entropy":
        return LogitAdjustedCrossEntropy(class_counts, loss["tau"])
    raise ValueError(f"Unsupported loss objective {loss['name']!r}")


def batch_inputs(batch, device):
    if "input" in batch:
        return batch["input"].to(device, non_blocking=True)
    return {
        "input_ids": batch["input_ids"].to(device, non_blocking=True),
        "attention_mask": batch["attention_mask"].to(device, non_blocking=True),
    }


def train_epoch(model, loader, optimizer, criterion, device, gradient_clip_norm):
    model.train()
    total_loss = 0.0
    sample_count = 0
    for batch in loader:
        inputs = batch_inputs(batch, device)
        metadata = batch["metadata"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(inputs, metadata), labels)
        loss.backward()
        if gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()
        total_loss += loss.item() * len(labels)
        sample_count += len(labels)
    return total_loss / sample_count


@torch.inference_mode()
def evaluate(model, loader, criterion, device, *, task, class_names):
    model.eval()
    total_loss = 0.0
    labels, predictions, probabilities = [], [], []
    for batch in loader:
        inputs = batch_inputs(batch, device)
        metadata = batch["metadata"].to(device, non_blocking=True)
        targets = batch["label"].to(device, non_blocking=True)
        logits = model(inputs, metadata)
        total_loss += criterion(logits, targets).item() * len(targets)
        labels.append(targets.cpu())
        predictions.append(logits.argmax(dim=1).cpu())
        probabilities.append(logits.softmax(dim=1).cpu())
    labels = torch.cat(labels).numpy()
    predictions = torch.cat(predictions).numpy()
    probabilities = torch.cat(probabilities).numpy()
    return {
        "loss": total_loss / len(labels),
        **calculate_metrics(
            labels,
            predictions,
            probabilities,
            task=task,
            class_names=class_names,
        ),
    }


def initialize_wandb(
    learning,
    dataset_name,
    dataset_config,
    bundle,
    model_info,
    device,
    output_dir,
):
    logging = learning.get("logging", {})
    wandb_config = logging.get("wandb")
    if not wandb_config or not wandb_config.get("enabled", True):
        return None
    mode = wandb_config.get("mode")
    if mode not in {"disabled", "offline"}:
        load_wandb_api_key()
    try:
        import wandb
    except ImportError as error:
        raise RuntimeError("W&B logging is configured but wandb is not installed") from error
    if mode not in {"disabled", "offline"}:
        wandb.login(key=os.environ["WANDB_API_KEY"], relogin=True)

    init_args = {
        "project": wandb_config["project"],
        "group": wandb_config["group"],
        "name": wandb_config["run_name"],
        "tags": wandb_config["tags"],
        "dir": str(output_dir),
        "config": {
            "dataset": dataset_name,
            "scenario": dataset_config["scenario"],
            "task": dataset_config["task"],
            f"{learning['encoder_type']}_model": dict(learning["encoder_model"]),
            "metadata_model": dict(learning["metadata_model_config"]),
            "combination": dict(learning["combination"]),
            "metadata_categories": list(learning["metadata_categories"]),
            "metadata_patterns": list(learning["metadata_patterns"]),
            "metadata_features": list(bundle["metadata_features"]),
            "data": dict(learning["data"]),
            "training": dict(learning["training"]),
            "validation": dict(learning["validation"]),
            "testing_metrics": list(learning["testing"]["metrics"]),
            "device": str(device),
            "num_classes": bundle["num_classes"],
            "label_to_index": bundle["label_to_index"],
            "split_sizes": {
                split: len(dataset) for split, dataset in bundle["datasets"].items()
            },
            "model_info": configured_model_info(learning, model_info),
            "system_metrics": list(logging.get("system", [])),
            "system_metric_units": {
                name: SYSTEM_METRIC_UNITS[name] for name in logging.get("system", [])
            },
        },
    }
    if mode:
        init_args["mode"] = mode
    if wandb_config.get("entity"):
        init_args["entity"] = wandb_config["entity"]
    return wandb.init(**init_args)


def run_experiment(
    config_path,
    dataset_name,
    model_name,
    combination_name,
    encoder_type="char",
):
    dataset_config, learning, config_path = load_config(
        config_path,
        dataset_name,
        model_name,
        combination_name,
        encoder_type,
    )
    data = learning["data"]
    training = learning["training"]
    validation = learning["validation"]
    testing = learning["testing"]

    set_seed(training["seed"])
    device = resolve_device(training["device"])
    loader_arguments = {
        "include_selected_metadata": True,
        "metadata_patterns": learning["metadata_patterns"],
        "batch_size": data["batch_size"],
        "max_length": data["max_length"],
        "num_workers": data["num_workers"],
        "seed": training["seed"],
    }
    if encoder_type == "lang":
        loader_arguments["pretrained_path"] = learning["encoder_model"][
            "pretrained_path"
        ]
        bundle = build_lang_dataloaders(
            config_path,
            dataset_name,
            **loader_arguments,
        )
    else:
        bundle = build_dataloaders(
            config_path,
            dataset_name,
            **loader_arguments,
        )
    model = build_model(bundle, learning, device)
    class_counts = torch.bincount(
        bundle["datasets"]["train"].labels,
        minlength=bundle["num_classes"],
    ).to(device)
    criterion = build_criterion(learning["combination"], class_counts)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training["learning_rate"],
        weight_decay=training["weight_decay"],
    )

    output_dir = Path(learning["output"]["directory"])
    if not output_dir.is_absolute():
        output_dir = config_path.parent / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best.pt"
    split_sizes = {
        split: len(dataset) for split, dataset in bundle["datasets"].items()
    }
    model_info = summarize_model(model)
    wandb_run = initialize_wandb(
        learning, dataset_name, dataset_config, bundle, model_info, device, output_dir
    )
    print(
        f"combination={combination_name} model={model_name} "
        f"metadata_features={bundle['metadata_dim']} "
        f"parameters={model_info['total_parameters']:,}"
    )

    system_fields = learning.get("logging", {}).get("system", [])
    stage_metrics = {name: 0.0 for name in system_fields}
    history = []
    best_score = float("-inf")
    stale_epochs = 0
    for epoch in range(1, training["epochs"] + 1):
        train_loss, train_seconds, train_peak_memory = measure_stage(
            lambda: train_epoch(
                model,
                bundle["loaders"]["train"],
                optimizer,
                criterion,
                device,
                training["gradient_clip_norm"],
            ),
            device,
        )
        validation_metrics, val_seconds, val_peak_memory = measure_stage(
            lambda: evaluate(
                model,
                bundle["loaders"]["dev"],
                criterion,
                device,
                task=dataset_config["task"],
                class_names=bundle["class_names"],
            ),
            device,
        )
        if "time_train" in stage_metrics:
            stage_metrics["time_train"] += train_seconds
        if "time_val" in stage_metrics:
            stage_metrics["time_val"] += val_seconds
        if "peak_memory_train" in stage_metrics:
            stage_metrics["peak_memory_train"] = max(
                stage_metrics["peak_memory_train"], train_peak_memory
            )
        if "peak_memory_val" in stage_metrics:
            stage_metrics["peak_memory_val"] = max(
                stage_metrics["peak_memory_val"], val_peak_memory
            )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_time_seconds": train_seconds,
                "train_peak_memory_mib": train_peak_memory,
                "validation_time_seconds": val_seconds,
                "validation_peak_memory_mib": val_peak_memory,
                **{
                    f"validation_{key}": value
                    for key, value in validation_metrics.items()
                },
            }
        )
        score = validation_metrics[validation["monitor"]]
        improved = score > best_score + validation["min_delta"]
        if improved:
            best_score = score
            stale_epochs = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    f"{encoder_type}_model_config": dict(learning["encoder_model"]),
                    "metadata_model_config": dict(
                        learning["metadata_model_config"]
                    ),
                    "combination": dict(learning["combination"]),
                    "metadata_categories": list(learning["metadata_categories"]),
                    "metadata_patterns": list(learning["metadata_patterns"]),
                    "metadata_features": list(bundle["metadata_features"]),
                    "metadata_imputer": bundle["metadata_imputer"],
                    "metadata_scaler": bundle["metadata_scaler"],
                    "num_classes": bundle["num_classes"],
                    "label_to_index": bundle["label_to_index"],
                    "model_info": model_info,
                    "validation_metrics": validation_metrics,
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1

        print(
            f"epoch={epoch:02d} train_loss={train_loss:.6f} "
            f"val_loss={validation_metrics['loss']:.6f} "
            f"val_macro_f1={validation_metrics['macro_f1']:.6f}"
        )
        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch,
                    "train/loss": train_loss,
                    **{
                        f"validation/{key}": value
                        for key, value in validation_metrics.items()
                    },
                    "validation/best_macro_f1": best_score,
                    "validation/is_best": int(improved),
                    "system/time_train_epoch": train_seconds,
                    "system/time_val_epoch": val_seconds,
                    "system/peak_memory_train_epoch": train_peak_memory,
                    "system/peak_memory_val_epoch": val_peak_memory,
                },
                step=epoch,
            )
        if stale_epochs >= validation["patience"]:
            print(f"early stopping after {epoch} epochs")
            break

    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics, test_seconds, test_peak_memory = measure_stage(
        lambda: evaluate(
            model,
            bundle["loaders"]["test"],
            criterion,
            device,
            task=dataset_config["task"],
            class_names=bundle["class_names"],
        ),
        device,
    )
    if "time_test" in stage_metrics:
        stage_metrics["time_test"] = test_seconds
    if "peak_memory_test" in stage_metrics:
        stage_metrics["peak_memory_test"] = test_peak_memory
    requested_test_metrics = {
        metric: test_metrics[metric] for metric in testing["metrics"]
    }
    results = {
        "dataset": dataset_name,
        "scenario": dataset_config["scenario"],
        "model": model_name,
        "combination": combination_name,
        "input": "domain_and_selected_metadata",
        "metadata_categories": list(learning["metadata_categories"]),
        "metadata_features": list(bundle["metadata_features"]),
        "device": str(device),
        "model_info": model_info,
        "split_sizes": split_sizes,
        "system_metrics": stage_metrics,
        "system_metric_units": {
            name: SYSTEM_METRIC_UNITS[name] for name in system_fields
        },
        "best_epoch": checkpoint["epoch"],
        "validation": checkpoint["validation_metrics"],
        "test": requested_test_metrics,
    }
    (output_dir / "metrics.json").write_text(json.dumps(results, indent=2) + "\n")
    if wandb_run is not None:
        wandb_run.log(
            {
                **{
                    f"best_validation/{key}": value
                    for key, value in checkpoint["validation_metrics"].items()
                },
                **{f"test/{key}": value for key, value in requested_test_metrics.items()},
                **{f"system/{key}": value for key, value in stage_metrics.items()},
            },
            step=len(history) + 1,
        )
        wandb_run.summary["best_epoch"] = checkpoint["epoch"]
        wandb_run.summary["best_validation_macro_f1"] = checkpoint[
            "validation_metrics"
        ]["macro_f1"]
        for field, value in configured_model_info(learning, model_info).items():
            if isinstance(value, dict):
                for name, count in value.items():
                    wandb_run.summary[f"model/{field}/{name}"] = count
            else:
                wandb_run.summary[f"model/{field}"] = value
        for key, value in stage_metrics.items():
            wandb_run.summary[f"system/{key}"] = value
        for key, value in requested_test_metrics.items():
            wandb_run.summary[f"test/{key}"] = value
        wandb_run.finish()

    print(f"best_epoch={results['best_epoch']} final_test_metrics={requested_test_metrics}")
    print(f"system_metrics={stage_metrics}")
    print(f"outputs={output_dir}")
    return results


def self_check_losses():
    class_counts = torch.tensor([8.0, 3.0, 1.0])
    logits = torch.tensor([[1.5, -0.5, 0.0], [-0.5, 0.5, 1.0]])
    targets = torch.tensor([0, 2])

    focal = FocalLoss(0)(logits, targets)
    assert torch.allclose(focal, F.cross_entropy(logits, targets))

    weighted = build_criterion(
        {
            "loss_objective": {
                "name": "class_weighted_cross_entropy",
            }
        },
        class_counts,
    )
    expected_weights = class_counts.reciprocal()
    expected_weights /= expected_weights.mean()
    assert torch.allclose(weighted.weight, expected_weights)

    adjusted = LogitAdjustedCrossEntropy(class_counts, tau=1.0)
    expected_adjustment = (class_counts / class_counts.sum()).log()
    assert torch.allclose(adjusted.adjustment, expected_adjustment)


def self_check(config_path, dataset_name, encoder_type="char"):
    model_names, combination_names = configured_runs(config_path, encoder_type)
    self_check_losses()
    batch_size = 2
    bundle = {
        "vocab_size": 96,
        "num_classes": 3,
        "metadata_dim": 11,
    }
    if encoder_type == "char":
        inputs = torch.randint(0, bundle["vocab_size"], (batch_size, 64))
    else:
        inputs = {
            "input_ids": torch.zeros((batch_size, 64), dtype=torch.long),
            "attention_mask": torch.ones((batch_size, 64), dtype=torch.long),
        }
    metadata = torch.randn(batch_size, bundle["metadata_dim"])
    class_counts = torch.tensor([8.0, 3.0, 1.0])
    targets = torch.tensor([0, 2])
    checked = []
    for model_name in model_names:
        for combination_name in combination_names:
            _, learning, _ = load_config(
                config_path,
                dataset_name,
                model_name,
                combination_name,
                encoder_type,
            )
            model = build_model(bundle, learning, torch.device("cpu"))
            model.eval()
            with torch.inference_mode():
                output = model(inputs, metadata)
            assert output.shape == (batch_size, bundle["num_classes"])
            assert model.text_encoder.out_dim > 0
            assert isinstance(model.metadata_encoder.classifier, nn.Identity)
            assert model.metadata_encoder.out_dim == 64
            criterion = build_criterion(learning["combination"], class_counts)
            assert torch.isfinite(criterion(output, targets))
            checked.append(f"{combination_name}/{model_name}")
            del model
            release_memory()
    print(
        f"{encoder_type} fusion self-check passed: "
        f"{len(checked)} configurations"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yml")
    parser.add_argument("--dataset", default="threatfox_malware_binary")
    parser.add_argument("--encoder", choices=("char", "lang"), default="char")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()

    if args.self_check:
        self_check(args.config, args.dataset, args.encoder)
        return

    model_names, combination_names = configured_runs(args.config, args.encoder)
    total = len(model_names) * len(combination_names)
    index = 0
    for combination_name in combination_names:
        for model_name in model_names:
            index += 1
            print(
                f"=== {combination_name} / {model_name} "
                f"({index}/{total}) ==="
            )
            run_experiment(
                args.config,
                args.dataset,
                model_name,
                combination_name,
                args.encoder,
            )
            release_memory()


if __name__ == "__main__":
    main()
