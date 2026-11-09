#!/usr/bin/env python3
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path
def _load_env_key():
    env_file = Path(".env")
    if env_file.is_file():
        for line in env_file.read_text().splitlines():
            if line.startswith("WANDB_API_KEY="):
                os.environ["WANDB_API_KEY"] = line.split("=", 1)[1].strip().strip("'\"")
                break

_load_env_key()
import wandb
import yaml

# ponytail: fixed aggregation over all datasets in config.yml; add CLI filters if selective model/dataset queries needed

PROJECT = os.environ.get("WANDB_PROJECT_PATH", "jina-pro/miaff")
CONFIG_PATH = Path("config.yml")
OUTPUT_PATH = Path("efficiency_wandb.csv")

VARIANTS = {
    "baseline": "baseline",
    "N_cat_N_ce": "N_cat_N_ce",
}

METRIC_KEYS = {
    "macro_f1": "test/macro_f1",
    "training_time_min": "system/time_train",
    "test_evaluation_time_min": "system/time_test",
    "peak_train_memory_gib": "system/peak_memory_train",
    "peak_test_memory_gib": "system/peak_memory_test",
}


def load_config():
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)
    datasets = [
        d["name"]
        for scenario in cfg.get("scenarios", {}).values()
        for d in scenario.get("datasets", [])
    ]
    models = [m["name"] for m in cfg.get("effectiveness_efficiency", {}).get("approaches", [])]
    return models, datasets


def fetch_runs(api, models, datasets):
    runs_by_key = defaultdict(list)
    filters = {"state": "finished"}
    for run in api.runs(PROJECT, filters=filters):
        variant = None
        if run.name == VARIANTS["baseline"]:
            variant = "baseline"
        elif run.name == VARIANTS["N_cat_N_ce"]:
            variant = "N_cat_N_ce"
        if not variant:
            continue

        dataset = run.config.get("dataset")
        if dataset not in datasets:
            continue

        runs_by_key[(run.group, variant, dataset)].append(run)
    return runs_by_key


def extract_and_convert(run):
    res = {}
    for col, key in METRIC_KEYS.items():
        val = run.summary.get(key)
        if val is None:
            raise ValueError(f"Run {run.id} ({run.group}, {run.name}) missing summary key '{key}'")
        val = float(val)
        if col in ("training_time_min", "test_evaluation_time_min"):
            val = val / 60.0
        elif col in ("peak_train_memory_gib", "peak_test_memory_gib"):
            val = val / 1024.0
        res[col] = val
    return res


def main():
    if "WANDB_API_KEY" not in os.environ:
        print("Error: WANDB_API_KEY not set and not found in .env", file=sys.stderr)
        sys.exit(1)

    models, datasets = load_config()
    api = wandb.Api(timeout=60)
    runs_map = fetch_runs(api, models, datasets)

    errors = []
    rows = []

    for model in models:
        for variant in ("baseline", "N_cat_N_ce"):
            dataset_metrics = defaultdict(list)
            for ds in datasets:
                key = (model, variant, ds)
                matched = runs_map.get(key, [])
                if not matched:
                    errors.append(f"Missing run for model='{model}', variant='{variant}', dataset='{ds}'")
                elif len(matched) > 1:
                    ids = [r.id for r in matched]
                    errors.append(f"Duplicate runs for model='{model}', variant='{variant}', dataset='{ds}': {ids}")
                else:
                    try:
                        vals = extract_and_convert(matched[0])
                        for col, v in vals.items():
                            dataset_metrics[col].append(v)
                    except ValueError as err:
                        errors.append(str(err))

            if not errors:
                row = {"model": model, "variant": variant}
                for col in METRIC_KEYS:
                    vals = dataset_metrics[col]
                    row[col] = sum(vals) / len(vals)
                rows.append(row)

    if errors:
        print(f"Validation failed with {len(errors)} error(s):", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        sys.exit(1)

    fields = [
        "model",
        "variant",
        "macro_f1",
        "training_time_min",
        "test_evaluation_time_min",
        "peak_train_memory_gib",
        "peak_test_memory_gib",
    ]
    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Successfully generated {OUTPUT_PATH} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
