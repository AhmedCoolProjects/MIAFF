#!/usr/bin/env python3
"""Train, validate, and test configurable input-only language baselines."""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "01_metadata_selection"))
sys.path.insert(0, str(PROJECT_ROOT / "03_LANG_Models"))

from CySecBERT import CySecBERT  # noqa: E402
from DomURLsBERT import DomURLsBERT  # noqa: E402
from SecBERT import SecBERT  # noqa: E402
from SecureBERT import SecureBERT  # noqa: E402
from URLBERT import URLBERT  # noqa: E402
from data_loader import build_lang_dataloaders, resolve_dataset_config  # noqa: E402


SUPPORTED_BINARY_METRICS = {
    "macro_f1",
    "binary_f1",
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "ap",
    "auroc",
    "mcc",
    "tp",
    "fp",
    "tn",
    "fn",
}
SUPPORTED_MULTICLASS_METRICS = {
    "macro_f1",
    "weighted_f1",
    "accuracy",
    "balanced_accuracy",
    "macro_precision",
    "macro_recall",
    "ap",
    "auroc",
    "mcc",
}

SUPPORTED_MODEL_INFO = {
    "total_parameters",
    "trainable_parameters",
    "non_trainable_parameters",
    "parameter_size_mb",
    "parameters_by_module",
}
SUPPORTED_SYSTEM_METRICS = {
    "peak_memory_train",
    "peak_memory_val",
    "peak_memory_test",
    "time_train",
    "time_val",
    "time_test",
}
SYSTEM_METRIC_UNITS = {
    "peak_memory_train": "MiB",
    "peak_memory_val": "MiB",
    "peak_memory_test": "MiB",
    "time_train": "seconds",
    "time_val": "seconds",
    "time_test": "seconds",
}

MODEL_REGISTRY = {
    "CySecBERT": CySecBERT,
    "DomURLsBERT": DomURLsBERT,
    "SecBERT": SecBERT,
    "SecureBERT": SecureBERT,
    "URLBERT": URLBERT,
}
MODEL_PARAMETERS = {
    name: {"pretrained_path", "drop_prob"} for name in MODEL_REGISTRY
}


def resolve_model_config(models, model_name):
    if model_name not in MODEL_REGISTRY or model_name not in models:
        raise ValueError(
            f"Unknown model {model_name!r}; choose one of {sorted(MODEL_REGISTRY)}"
        )
    expected = MODEL_PARAMETERS[model_name]
    parameters = {
        key: value
        for key, value in {**models.get("defaults", {}), **models[model_name]}.items()
        if key in expected
    }
    missing = expected - set(parameters)
    if missing:
        raise ValueError(f"Invalid parameters for {model_name}: missing={sorted(missing)}")
    return {"name": model_name, **parameters}


def format_template(value, *, model_name, dataset_name, scenario_name):
    if not isinstance(value, str):
        raise ValueError("Model-dependent output and W&B values must be strings")
    try:
        return value.format(
            model=model_name, dataset=dataset_name, scenario=scenario_name
        )
    except KeyError as error:
        raise ValueError(f"Unsupported template field in {value!r}") from error

def model_names_from_config(config, learning_key="learning_lang"):
    names = config.get(learning_key, {}).get("model_names")
    if not isinstance(names, list) or not names:
        raise ValueError(f"{learning_key}.model_names must be a non-empty list")
    if any(not isinstance(name, str) for name in names) or len(names) != len(set(names)):
        raise ValueError(f"{learning_key}.model_names must contain unique model names")
    unknown = set(names) - set(MODEL_REGISTRY)
    if unknown:
        raise ValueError(f"Unknown models in {learning_key}.model_names: {sorted(unknown)}")
    return names


def configured_model_names(config_path, learning_key="learning_lang"):
    with Path(config_path).resolve().open() as stream:
        return model_names_from_config(yaml.safe_load(stream), learning_key=learning_key)



def load_config(config_path, dataset_name, model_name):
    config_path = Path(config_path).resolve()
    with config_path.open() as stream:
        config = yaml.safe_load(stream)
    try:
        learning = config["learning_lang"]
        models = config["models"]
        dataset = resolve_dataset_config(config, dataset_name)
    except (KeyError, TypeError) as error:
        raise ValueError(f"Missing dataset {dataset_name!r}, model, or learning_lang configuration") from error
    if model_name not in model_names_from_config(config, learning_key="learning_lang"):
        raise ValueError(f"Model {model_name!r} is not enabled in learning_lang.model_names")
    learning["model"] = resolve_model_config(models, model_name)
    include_metadata = learning["data"].get("include_selected_metadata", False)
    if include_metadata:
        raise ValueError("Language baselines are input-only; set include_selected_metadata: false")
    if learning["training"].get("optimizer", "").lower() != "adamw":
        raise ValueError("This script supports optimizer: adamw")
    if learning["validation"].get("monitor") != "macro_f1":
        raise ValueError("validation.monitor must be macro_f1")
    if learning["testing"].get("checkpoint") != "best":
        raise ValueError("testing.checkpoint must be best")
    task = dataset["task"]
    metric_key = (
        "binary_metrics" if task == "binary_classification" else "multi_class_metrics"
    )
    if metric_key not in learning["testing"]:
        raise ValueError(f"learning.testing.{metric_key} is required for {task}")
    training = learning["training"]
    validation = learning["validation"]
    if training["epochs"] < 1 or training["learning_rate"] <= 0:
        raise ValueError("epochs and learning_rate must be positive")
    if training["weight_decay"] < 0 or training["gradient_clip_norm"] < 0:
        raise ValueError("weight_decay and gradient_clip_norm must be non-negative")
    if validation["patience"] < 1 or validation["min_delta"] < 0:
        raise ValueError("patience must be positive and min_delta non-negative")

    learning["output"]["directory"] = format_template(
        learning["output"]["directory"],
        model_name=model_name,
        dataset_name=dataset_name,
        scenario_name=dataset["scenario"],
    )
    logging = learning.get("logging", {})
    wandb_config = logging.get("wandb")
    if wandb_config and wandb_config.get("enabled", True):
        missing = {"project", "group", "run_name", "tags"} - set(wandb_config)
        if missing:
            raise ValueError(f"Missing W&B settings: {sorted(missing)}")
        if not isinstance(wandb_config["tags"], list):
            raise ValueError("learning.logging.wandb.tags must be a list")
        wandb_config["tags"] = [
            format_template(
                tag,
                model_name=model_name,
                dataset_name=dataset_name,
                scenario_name=dataset["scenario"],
            )
            for tag in wandb_config["tags"]
        ]
        resolved_group = format_template(
            wandb_config["group"],
            model_name=model_name,
            dataset_name=dataset_name,
            scenario_name=dataset["scenario"],
        )
        if resolved_group != model_name:
            raise ValueError("W&B group must resolve to the selected model name")
        wandb_config["group"] = resolved_group
        wandb_config["run_name"] = "baseline"

    model_fields = logging.get("model", [])
    system_fields = logging.get("system", [])
    unknown_model_fields = set(model_fields) - SUPPORTED_MODEL_INFO
    unknown_system_fields = set(system_fields) - SUPPORTED_SYSTEM_METRICS
    if unknown_model_fields:
        raise ValueError(f"Unsupported model logging fields: {sorted(unknown_model_fields)}")
    if unknown_system_fields:
        raise ValueError(f"Unsupported system logging fields: {sorted(unknown_system_fields)}")

    metrics = [metric.lower() for metric in learning["testing"][metric_key]]
    if not metrics:
        raise ValueError(f"learning.testing.{metric_key} must not be empty")
    allowed = (
        SUPPORTED_BINARY_METRICS
        if task == "binary_classification"
        else SUPPORTED_MULTICLASS_METRICS
    )
    unknown = set(metrics) - allowed
    if unknown:
        raise ValueError(f"Unsupported testing metrics for {task}: {sorted(unknown)}")
    learning["testing"]["metrics"] = metrics
    return dataset, learning, config_path


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(value):
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    return device


def build_model(bundle, model_config, device):
    model_name = model_config["name"]
    parameters = {key: value for key, value in model_config.items() if key != "name"}
    return MODEL_REGISTRY[model_name](
        output_size=bundle["num_classes"],
        **parameters,
    ).to(device)


def summarize_model(model):
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "non_trainable_parameters": total - trainable,
        "parameter_size_mb": sum(
            parameter.numel() * parameter.element_size()
            for parameter in model.parameters()
        ) / (1024**2),
        "parameters_by_module": {
            name: sum(parameter.numel() for parameter in module.parameters())
            for name, module in model.named_children()
        },
    }


def load_wandb_api_key(env_path=PROJECT_ROOT / ".env"):
    env_path = Path(env_path)
    if not env_path.is_file():
        raise RuntimeError(f"W&B is enabled but {env_path} does not exist")
    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if separator and key.strip() == "WANDB_API_KEY":
            value = value.strip().strip("'\"")
            if not value:
                break
            os.environ["WANDB_API_KEY"] = value
            return
    raise RuntimeError(f"WANDB_API_KEY is missing from {env_path}")


def configured_model_info(learning, model_info):
    fields = learning.get("logging", {}).get("model", [])
    return {field: model_info[field] for field in fields}


def initialize_wandb(
    learning,
    dataset_name,
    dataset_config,
    bundle,
    model_info,
    device,
    output_dir,
    *,
    mode_override=None,
):
    logging = learning.get("logging", {})
    wandb_config = logging.get("wandb")
    if not wandb_config or not wandb_config.get("enabled", True):
        return None
    mode = mode_override or wandb_config.get("mode")
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
            "model": dict(learning["model"]),
            "data": dict(learning["data"]),
            "training": dict(learning["training"]),
            "validation": dict(learning["validation"]),
            "testing_metrics": list(learning["testing"]["metrics"]),
            "device": str(device),
            "pretrained_path": bundle["pretrained_path"],
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


def measure_stage(operation, device):
    using_cuda = device.type == "cuda"
    if using_cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    result = operation()
    if using_cuda:
        torch.cuda.synchronize(device)
        peak_memory = torch.cuda.max_memory_allocated(device) / (1024**2)
    else:
        peak_memory = 0.0
    return result, time.perf_counter() - started, peak_memory






def train_epoch(model, loader, optimizer, criterion, device, gradient_clip_norm):
    model.train()
    total_loss = 0.0
    sample_count = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(input_ids=input_ids, attention_mask=attention_mask), labels)
        loss.backward()
        if gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()
        total_loss += loss.item() * len(labels)
        sample_count += len(labels)
    return total_loss / sample_count


def calculate_metrics(labels, predictions, probabilities, *, task, class_names):
    metrics = {
        "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0),
        "accuracy": accuracy_score(labels, predictions),
        "balanced_accuracy": balanced_accuracy_score(labels, predictions),
        "mcc": matthews_corrcoef(labels, predictions),
    }
    if task == "binary_classification":
        tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
        metrics.update(
            {
                "binary_f1": f1_score(labels, predictions, zero_division=0),
                "precision": precision_score(labels, predictions, zero_division=0),
                "recall": recall_score(labels, predictions, zero_division=0),
                "ap": average_precision_score(labels, probabilities[:, 1]),
                "auroc": roc_auc_score(labels, probabilities[:, 1]),
                "tp": int(tp),
                "fp": int(fp),
                "tn": int(tn),
                "fn": int(fn),
            }
        )
        return metrics

    classes = np.arange(len(class_names))
    binary_targets = label_binarize(labels, classes=classes)
    metrics.update(
        {
            "weighted_f1": f1_score(
                labels, predictions, average="weighted", zero_division=0
            ),
            "macro_precision": precision_score(
                labels, predictions, average="macro", zero_division=0
            ),
            "macro_recall": recall_score(
                labels, predictions, average="macro", zero_division=0
            ),
            "ap": average_precision_score(
                binary_targets, probabilities, average="macro"
            ),
            "auroc": roc_auc_score(
                labels,
                probabilities,
                labels=classes,
                multi_class="ovr",
                average="macro",
            ),
        }
    )
    return metrics


@torch.inference_mode()
def evaluate(model, loader, criterion, device, *, task, class_names):
    model.eval()
    total_loss = 0.0
    labels, predictions, probabilities = [], [], []
    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        targets = batch["label"].to(device, non_blocking=True)
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
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



def run_experiment(config_path, dataset_name, model_name):
    dataset_config, learning, config_path = load_config(
        config_path, dataset_name, model_name
    )
    model_config = learning["model"]
    data_config = learning["data"]
    training = learning["training"]
    validation = learning["validation"]
    testing = learning["testing"]

    set_seed(training["seed"])
    device = resolve_device(training["device"])
    bundle = build_lang_dataloaders(
        config_path,
        dataset_name,
        pretrained_path=model_config["pretrained_path"],
        include_selected_metadata=data_config["include_selected_metadata"],
        batch_size=data_config["batch_size"],
        max_length=data_config["max_length"],
        num_workers=data_config["num_workers"],
        seed=training["seed"],
    )
    model = build_model(bundle, model_config, device)
    criterion = torch.nn.CrossEntropyLoss()
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
        f"model_parameters={model_info['total_parameters']:,} "
        f"trainable={model_info['trainable_parameters']:,} "
        f"size_mb={model_info['parameter_size_mb']:.3f}"
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
                    "model_config": dict(model_config),
                    "pretrained_path": bundle["pretrained_path"],
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
        "model": model_config["name"],
        "input": "domain_only",
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


def smoke_test(config_path, dataset_name, model_name):
    dataset_config, learning, config_path = load_config(
        config_path, dataset_name, model_name
    )
    set_seed(learning["training"]["seed"])
    device = resolve_device(learning["training"]["device"])
    bundle = build_lang_dataloaders(
        config_path,
        dataset_name,
        pretrained_path=learning["model"]["pretrained_path"],
        include_selected_metadata=False,
        batch_size=2,
        max_length=learning["data"]["max_length"],
        num_workers=0,
        seed=learning["training"]["seed"],
    )
    model = build_model(bundle, learning["model"], device)
    model_info = summarize_model(model)
    assert model_info["total_parameters"] == model_info["trainable_parameters"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    batch = next(iter(bundle["loaders"]["train"]))
    load_wandb_api_key()
    (loss, train_seconds, train_peak_memory) = measure_stage(
        lambda: train_epoch(model, [batch], optimizer, criterion, device, 1.0),
        device,
    )
    assert set(batch) == {"input_ids", "attention_mask", "label"}
    assert train_seconds > 0 and train_peak_memory >= 0
    assert model(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device)).shape == (2, bundle["num_classes"])

    if dataset_config["task"] == "binary_classification":
        metrics = calculate_metrics(
            np.array([0, 0, 1, 1]),
            np.array([0, 1, 1, 1]),
            np.array([[0.9, 0.1], [0.3, 0.7], [0.2, 0.8], [0.1, 0.9]]),
            task="binary_classification",
            class_names=bundle["class_names"],
        )
        assert set(metrics) == SUPPORTED_BINARY_METRICS
    else:
        metrics = calculate_metrics(
            np.array([0, 1, 2, 0]),
            np.array([0, 1, 1, 0]),
            np.array(
                [
                    [0.7, 0.2, 0.1],
                    [0.2, 0.6, 0.2],
                    [0.1, 0.5, 0.4],
                    [0.8, 0.1, 0.1],
                ]
            ),
            task="multi_class_classification",
            class_names=["a", "b", "c"],
        )
        assert set(metrics) == SUPPORTED_MULTICLASS_METRICS
    checkpoint = {
        "epoch": 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": dict(learning["model"]),
        "pretrained_path": bundle["pretrained_path"],
        "num_classes": bundle["num_classes"],
        "label_to_index": bundle["label_to_index"],
        "validation_metrics": metrics,
    }
    with TemporaryDirectory() as directory:
        path = Path(directory) / "best.pt"
        torch.save(checkpoint, path)
        restored = torch.load(path, map_location="cpu", weights_only=False)
        wandb_run = initialize_wandb(
            learning,
            dataset_name,
            dataset_config,
            bundle,
            model_info,
            device,
            Path(directory),
            mode_override="disabled",
        )
        assert wandb_run is not None
        wandb_run.log(
            {
                "smoke/loss": loss,
                "system/time_train": train_seconds,
                "system/peak_memory_train": train_peak_memory,
            }
        )
        wandb_run.finish()
    assert set(restored) == set(checkpoint)
    print(f"smoke test passed: loss={loss:.6f}, batch={tuple(batch['input_ids'].shape)}")


def release_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yml")
    parser.add_argument("--dataset", default="threatfox_malware_binary")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    model_names = configured_model_names(args.config)
    for index, model_name in enumerate(model_names, start=1):
        print(f"=== {model_name} ({index}/{len(model_names)}) ===")
        if args.smoke_test:
            smoke_test(args.config, args.dataset, model_name)
        else:
            run_experiment(args.config, args.dataset, model_name)
        release_memory()


if __name__ == "__main__":
    main()