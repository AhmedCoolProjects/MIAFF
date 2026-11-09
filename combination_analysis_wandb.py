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

# ponytail: fixed loss objectives, model families, and 3 scenarios from config.yml; add CLI flags if subset queries are needed

PROJECT = os.environ.get("WANDB_PROJECT_PATH", "jina-pro/miaff")
CONFIG_PATH = Path("config.yml")
OUTPUT_PATH = Path("combination_loss_analysis.csv")

LOSS_OBJECTIVES = [
    ("baseline", "baseline"),
    ("N_cat_N_ce", "cross_entropy"),
    ("N_cat_N_focal", "focal"),
    ("N_cat_N_cwce", "weighted_ce"),
    ("N_cat_N_lace", "logit_adjusted_ce"),
]

NN_MODELS = [
    "CharCNN",
    "CharLSTM",
    "CharGRU",
    "CharBiLSTM",
    "CharBiGRU",
    "CharCNNBiLSTM",
]

PLM_MODELS = [
    "SecureBERT",
    "CySecBERT",
    "SecBERT",
    "DomURLsBERT",
]

SCENARIOS = ["malware", "phishing", "malware_family"]


def load_config():
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    models = [m["name"] for m in cfg.get("effectiveness_efficiency", {}).get("approaches", [])]
    scenarios = {}
    dataset_to_scenario = {}
    for scen_key in SCENARIOS:
        ds_list = [d["name"] for d in cfg["scenarios"][scen_key]["datasets"]]
        scenarios[scen_key] = ds_list
        for ds in ds_list:
            dataset_to_scenario[ds] = scen_key
    return models, scenarios, dataset_to_scenario


def fetch_runs(api, models, dataset_to_scenario):
    runs_by_key = defaultdict(list)
    valid_models = set(models)
    valid_runs = {r for r, _ in LOSS_OBJECTIVES}
    filters = {
        "state": "finished",
        "group": {"$in": list(valid_models)},
    }
    for run in api.runs(PROJECT, filters=filters):
        if run.group not in valid_models or run.name not in valid_runs:
            continue
        ds = run.config.get("dataset")
        if ds not in dataset_to_scenario:
            continue
        runs_by_key[(run.group, run.name, ds)].append(run)
    return runs_by_key


def main():
    if "WANDB_API_KEY" not in os.environ:
        print("Error: WANDB_API_KEY not found in .env or environment", file=sys.stderr)
        sys.exit(1)

    models, scenarios, dataset_to_scenario = load_config()
    if set(models) != set(NN_MODELS + PLM_MODELS):
        print(f"Error: config models {models} do not match expected NN+PLM sets", file=sys.stderr)
        sys.exit(1)

    api = wandb.Api(timeout=60)
    runs_map = fetch_runs(api, models, dataset_to_scenario)

    errors = []
    f1_scores = {}

    for model in models:
        for run_name, _ in LOSS_OBJECTIVES:
            for ds in dataset_to_scenario:
                key = (model, run_name, ds)
                matched = runs_map.get(key, [])
                if not matched:
                    errors.append(f"Missing run: model='{model}', run='{run_name}', dataset='{ds}'")
                elif len(matched) > 1:
                    ids = [r.id for r in matched]
                    errors.append(f"Duplicate runs: model='{model}', run='{run_name}', dataset='{ds}': {ids}")
                else:
                    val = matched[0].summary.get("test/macro_f1")
                    if val is None:
                        errors.append(
                            f"Missing 'test/macro_f1' in run {matched[0].id} ({model}, {run_name}, {ds})"
                        )
                    else:
                        f1_scores[key] = float(val)

    if errors:
        print(f"Validation failed with {len(errors)} error(s):", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        sys.exit(1)

    families = [("nn", NN_MODELS), ("plm", PLM_MODELS)]
    group_cols = [f"{scen}_{fam}" for scen in SCENARIOS for fam, _ in families]

    group_means = {}
    for run_name, _ in LOSS_OBJECTIVES:
        for scen in SCENARIOS:
            ds_list = scenarios[scen]
            for fam, fam_models in families:
                col = f"{scen}_{fam}"
                vals = [f1_scores[(m, run_name, ds)] for m in fam_models for ds in ds_list]
                group_means[(run_name, col)] = sum(vals) / len(vals)

    rows = []
    for run_name, obj_label in LOSS_OBJECTIVES:
        row = {"loss_objective": obj_label}
        for col in group_cols:
            base_avg = group_means[("baseline", col)]
            curr_avg = group_means[(run_name, col)]
            if run_name == "baseline":
                row[col] = base_avg * 100.0
            else:
                row[col] = ((curr_avg - base_avg) / base_avg) * 100.0
        rows.append(row)

    fields = ["loss_objective", *group_cols]
    with OUTPUT_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Successfully generated {OUTPUT_PATH}")
    print(f"NN models ({len(NN_MODELS)}): {', '.join(NN_MODELS)}")
    print(f"PLM models ({len(PLM_MODELS)}): {', '.join(PLM_MODELS)}")
    for scen in SCENARIOS:
        ds_list = scenarios[scen]
        n_ds = len(ds_list)
        print(
            f"Scenario '{scen}': {n_ds} datasets | "
            f"NN={len(NN_MODELS)*n_ds} runs/config, PLM={len(PLM_MODELS)*n_ds} runs/config"
        )


if __name__ == "__main__":
    main()
