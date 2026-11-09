#!/usr/bin/env python3
"""Train, validate, and test an MLP on selected infrastructure metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import pandas as pd
import torch
import torch.nn as nn
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "01_metadata_selection"))

from data_loader import build_dataloaders, resolve_dataset_config  # noqa: E402
from BASELINE_Char import (  # noqa: E402
    SUPPORTED_BINARY_METRICS,
    SUPPORTED_MODEL_INFO,
    SUPPORTED_MULTICLASS_METRICS,
    SUPPORTED_SYSTEM_METRICS,
    SYSTEM_METRIC_UNITS,
    calculate_metrics,
    configured_model_info,
    format_template,
    initialize_wandb,
    measure_stage,
    release_memory,
    resolve_device,
    set_seed,
    summarize_model,
)

MODEL_NAME = "Metadata_MLP"
MODEL_PARAMETERS = {
    "hidden_dim",
    "out_dim",
    "drop_prob",
    "use_layernorm",
    "head_hidden_dim",
}


class Metadata_MLP(nn.Module):
    """MLP classifier over preprocessed infrastructure metadata."""

    def __init__(
        self,
        in_dim: int,
        output_size: int,
        hidden_dim: int,
        out_dim: int,
        drop_prob: float,
        use_layernorm: bool,
        head_hidden_dim: int,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.output_size = output_size
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop_prob),
            nn.Linear(hidden_dim, out_dim),
        )
        self.norm = nn.LayerNorm(out_dim) if use_layernorm else nn.Identity()
        self.classifier = (
            nn.Sequential(
                nn.Linear(out_dim, head_hidden_dim),
                nn.GELU(),
                nn.Dropout(drop_prob),
                nn.Linear(head_hidden_dim, output_size),
            )
            if head_hidden_dim > 0
            else nn.Linear(out_dim, output_size)
        )

    def extract_features(self, metadata):
        if isinstance(metadata, dict):
            metadata = metadata["metadata"]
        return self.norm(self.net(metadata))

    def forward(self, metadata):
        return self.classifier(self.extract_features(metadata))


MetaMLP = MetadataMLP = Metadata_MLP


def model_names_from_config(config):
    names = config.get("learning_mlp", {}).get("model_names")
    if names != [MODEL_NAME]:
        raise ValueError(f"learning_mlp.model_names must be [{MODEL_NAME!r}]")
    return names


def configured_model_names(config_path):
    with Path(config_path).resolve().open() as stream:
        return model_names_from_config(yaml.safe_load(stream))


def resolve_model_config(models):
    parameters = models.get(MODEL_NAME)
    if not isinstance(parameters, dict):
        raise ValueError(f"models.{MODEL_NAME} must be a mapping")
    missing = MODEL_PARAMETERS - set(parameters)
    unknown = set(parameters) - MODEL_PARAMETERS
    if missing or unknown:
        raise ValueError(
            f"Invalid parameters for {MODEL_NAME}: missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}"
        )
    if parameters["hidden_dim"] < 1 or parameters["out_dim"] < 1:
        raise ValueError("hidden_dim and out_dim must be positive")
    if parameters["head_hidden_dim"] < 0:
        raise ValueError("head_hidden_dim must be non-negative")
    if not 0 <= parameters["drop_prob"] < 1:
        raise ValueError("drop_prob must be in [0, 1)")
    if not isinstance(parameters["use_layernorm"], bool):
        raise ValueError("use_layernorm must be true or false")
    return {"name": MODEL_NAME, **parameters}


def load_config(config_path, dataset_name, model_name):
    config_path = Path(config_path).resolve()
    with config_path.open() as stream:
        config = yaml.safe_load(stream)
    try:
        learning = config["learning_mlp"]
        dataset = resolve_dataset_config(config, dataset_name)
        learning["model"] = resolve_model_config(config["models"])
    except (KeyError, TypeError) as error:
        raise ValueError(
            f"Missing dataset {dataset_name!r}, model, or learning_mlp configuration"
        ) from error
    if model_name not in model_names_from_config(config):
        raise ValueError(f"Model {model_name!r} is not enabled")

    data = learning["data"]
    training = learning["training"]
    validation = learning["validation"]
    testing = learning["testing"]
    if data.get("include_selected_metadata") is not True:
        raise ValueError("learning_mlp.data.include_selected_metadata must be true")
    if data["batch_size"] < 1 or data["num_workers"] < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative")
    if training.get("optimizer", "").lower() != "adamw":
        raise ValueError("This script supports optimizer: adamw")
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
            f"Invalid learning_mlp.testing.{metric_key}: "
            f"unsupported={sorted(unknown_metrics)}"
        )
    testing["metrics"] = metrics

    learning["output"]["directory"] = format_template(
        learning["output"]["directory"],
        model_name=model_name,
        dataset_name=dataset_name,
        scenario_name=dataset["scenario"],
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
        wandb_config["tags"] = [
            format_template(
                value,
                model_name=model_name,
                dataset_name=dataset_name,
                scenario_name=dataset["scenario"],
            )
            for value in wandb_config["tags"]
        ]
        wandb_config["group"] = format_template(
            wandb_config["group"],
            model_name=model_name,
            dataset_name=dataset_name,
            scenario_name=dataset["scenario"],
        )
        if wandb_config["group"] != model_name:
            raise ValueError("W&B group must resolve to the selected model name")
        wandb_config["run_name"] = "baseline"
    return dataset, learning, config_path


def build_model(bundle, model_config, device):
    parameters = {key: value for key, value in model_config.items() if key != "name"}
    return Metadata_MLP(
        in_dim=bundle["metadata_dim"],
        output_size=bundle["num_classes"],
        **parameters,
    ).to(device)


def train_epoch(model, loader, optimizer, criterion, device, gradient_clip_norm):
    model.train()
    total_loss = 0.0
    sample_count = 0
    for batch in loader:
        metadata = batch["metadata"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(metadata), labels)
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
        metadata = batch["metadata"].to(device, non_blocking=True)
        targets = batch["label"].to(device, non_blocking=True)
        logits = model(metadata)
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


def run_experiment(config_path, dataset_name, model_name=MODEL_NAME):
    dataset_config, learning, config_path = load_config(
        config_path, dataset_name, model_name
    )
    return execute_experiment(
        dataset_config,
        learning,
        config_path,
        dataset_name,
    )


def execute_experiment(
    dataset_config,
    learning,
    config_path,
    dataset_name,
    *,
    metadata_patterns=None,
    experiment=None,
):
    model_config = learning["model"]
    data = learning["data"]
    training = learning["training"]
    validation = learning["validation"]
    testing = learning["testing"]

    set_seed(training["seed"])
    device = resolve_device(training["device"])
    bundle = build_dataloaders(
        config_path,
        dataset_name,
        include_selected_metadata=True,
        metadata_patterns=metadata_patterns,
        batch_size=data["batch_size"],
        num_workers=data["num_workers"],
        seed=training["seed"],
    )
    model = build_model(bundle, model_config, device)
    criterion = nn.CrossEntropyLoss()
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
    if wandb_run is not None:
        wandb_run.config.update(
            {
                "metadata_dim": bundle["metadata_dim"],
                "metadata_features": bundle["metadata_features"],
                **({"experiment": dict(experiment)} if experiment else {}),
            }
        )
    print(
        f"metadata_features={bundle['metadata_dim']} "
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
                    "metadata_dim": bundle["metadata_dim"],
                    "metadata_features": bundle["metadata_features"],
                    "metadata_imputer": bundle["metadata_imputer"],
                    "metadata_scaler": bundle["metadata_scaler"],
                    "num_classes": bundle["num_classes"],
                    "label_to_index": bundle["label_to_index"],
                    "model_info": model_info,
                    "validation_metrics": validation_metrics,
                    **({"experiment": dict(experiment)} if experiment else {}),
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
        "input": "selected_metadata",
        "metadata_features": bundle["metadata_features"],
        **({"experiment": dict(experiment)} if experiment else {}),
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


def smoke_test(
    config_path,
    dataset_name,
    model_name=MODEL_NAME,
    *,
    metadata_patterns=None,
):
    dataset_config, learning, config_path = load_config(
        config_path, dataset_name, model_name
    )
    set_seed(learning["training"]["seed"])
    device = resolve_device(learning["training"]["device"])
    bundle = build_dataloaders(
        config_path,
        dataset_name,
        include_selected_metadata=True,
        metadata_patterns=metadata_patterns,
        batch_size=4,
        num_workers=0,
        seed=learning["training"]["seed"],
    )
    model = build_model(bundle, learning["model"], device)
    batch = next(iter(bundle["loaders"]["train"]))
    assert set(batch) == {"input", "metadata", "label"}
    assert batch["metadata"].shape == (4, bundle["metadata_dim"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    loss = train_epoch(model, [batch], optimizer, criterion, device, 1.0)
    assert model(batch["metadata"].to(device)).shape == (4, bundle["num_classes"])
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": dict(learning["model"]),
        "metadata_features": bundle["metadata_features"],
        "metadata_imputer": bundle["metadata_imputer"],
        "metadata_scaler": bundle["metadata_scaler"],
    }
    with TemporaryDirectory() as directory:
        path = Path(directory) / "best.pt"
        torch.save(checkpoint, path)
        restored = torch.load(path, map_location="cpu", weights_only=False)
    assert set(restored) == set(checkpoint)
    print(
        f"smoke test passed: task={dataset_config['task']} loss={loss:.6f} "
        f"batch={tuple(batch['metadata'].shape)}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yml")
    parser.add_argument("--dataset", default="threatfox_malware_binary")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    for model_name in configured_model_names(args.config):
        if args.smoke_test:
            smoke_test(args.config, args.dataset, model_name)
        else:
            run_experiment(args.config, args.dataset, model_name)
        release_memory()


if __name__ == "__main__":
    main()
