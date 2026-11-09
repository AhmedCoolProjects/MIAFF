"""Effectiveness vs. efficiency reporting: per-model averages of efficiency
metrics and avg macro F1 for the baseline and MIAFF_base runs.

Reads config.yml -> reporting.effectiveness_efficiency.
  - datasets: "all" -> every dataset defined under scenarios
  - approaches: models (one CSV row each, rows: model)
  - output.columns: e.g. training_time_baseline, avg_macro_f1_MIAFF_base ...

Runs used per (model, dataset):
  - baseline   -> wandb run named "baseline" in group = model
  - MIAFF_base -> wandb run named "N_cat_N_ce" (the MIAFF_base combination)

Efficiency metrics come from the system/ summary keys, macro F1 from test/.
Every value is averaged across all datasets for that model.

Output: config -> effectiveness_efficiency.output.file
"""

import csv
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent

MIAFF_BASE_RUN_NAME = "N_cat_N_ce"  # run named after the MIAFF_base combination
# CSV column prefix -> wandb system/ summary key suffix
METRICS = {
    "training_time": "time_train",
    "deployment_inference_time": "time_test",
    "training_peak_memory": "peak_memory_train",
    "deployment_inference_peak_memory": "peak_memory_test",
}

MIAFF_BASE_RUN_NAME = "N_cat_N_ce"  # run named after the MIAFF_base combination


def load_config():
    return yaml.safe_load((ROOT / "config.yml").read_text())


def load_wandb():
    for line in (ROOT / ".env").read_text().splitlines():
        if line.startswith("WANDB_API_KEY="):
            os.environ["WANDB_API_KEY"] = line.split("=", 1)[1].strip().strip("'\"")
    import wandb
    return wandb.Api()


def get_metric(run, key):
    value = run.summary.get(key)
    return float(value) if value is not None else None


def all_dataset_names(config):
    eff = config["effectiveness_efficiency"]
    if eff.get("datasets") != "all":
        return [d["name"] for d in eff["datasets"]]
    return [
        d["name"]
        for sc in config.get("scenarios", {}).values()
        for d in sc.get("datasets", [])
    ]


def collect_runs(api, project, approaches, datasets):
    """For each (model, dataset, run) fetch system/* and test/macro_f1.
    Returns {(model, dataset, run_name): {metric: value}}; run_name is
    "baseline" or "MIAFF_base"."""
    def run_label(name):
        return "MIAFF_base" if name == MIAFF_BASE_RUN_NAME else "baseline"

    groups = [a["name"] for a in approaches]
    wanted = {"baseline", MIAFF_BASE_RUN_NAME}

    values = {}
    runs = api.runs(
        project,
        filters={"state": "finished", "group": {"$in": groups}},
    )
    for r in runs:
        ds = r.config.get("dataset")
        if ds not in datasets or r.name not in wanted:
            continue
        vals = {key: get_metric(r, f"system/{key}") for key in METRICS.values()}
        vals["macro_f1"] = get_metric(r, "test/macro_f1")
        values[(r.group, ds, run_label(r.name))] = vals
    return values


def avg_metric(values, model, run, key):
    opts = []
    for (m, ds, r), vals in values.items():
        if m == model and r == run and vals.get(key) is not None:
            opts.append(vals[key])
    return sum(opts) / len(opts) if opts else ""


def main():
    config = load_config()
    eff = config["effectiveness_efficiency"]
    approaches = [a["name"] for a in eff["approaches"]]
    datasets = all_dataset_names(config)
    header = ["model"] + [c for c in eff["output"]["columns"] if c != "model"]

    wandb_cfg = config.get("learning_mlp", {}).get("logging", {}).get("wandb", {})
    project = f"{os.environ.get('WANDB_ENTITY', 'jina-pro')}/{wandb_cfg.get('project', 'miaff')}"

    api = load_wandb()
    values = collect_runs(api, project, eff["approaches"], datasets)
    print(f"Fetched {len(values)} (model, dataset, run) entries from {project}")

    rows = []
    for model in approaches:
        row = {"model": model}
        for col in header:
            if col == "model":
                continue
            # column format: <metric>_<run>  (run in {"baseline", "MIAFF_base"})
            sorted_matches = [
                run for run in ("baseline", "MIAFF_base") if col.endswith(run)
            ]
            if not sorted_matches:
                raise ValueError(f"column {col!r} does not end with baseline or MIAFF_base")
            run = sorted_matches[0]
            if col == f"avg_macro_f1_{run}":
                row[col] = avg_metric(values, model, run, "macro_f1")
                continue
            metric = col[: -len(f"_{run}")]
            sys_key = METRICS.get(metric)
            if sys_key is None:
                raise ValueError(f"unknown metric column {col!r}")
            row[col] = avg_metric(values, model, run, sys_key)
        rows.append(row)

    out_path = ROOT / eff["output"]["file"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
