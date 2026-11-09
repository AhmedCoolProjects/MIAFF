"""Effectiveness reporting: builds per-(model, dataset) rows from wandb runs (online API).

Reads config.yml -> reporting.effectiveness. Each CSV row is one (model, dataset)
pair with a `dataset` column; columns are the metrics for the configured runs:
  - baseline            -> wandb run named "baseline" in group = model
  - MIAFF_base          -> wandb run named = the configured combination (run_name)
  - MIAFF_best_macro_f1 -> fusion run with the best test macro_f1 for that
                           model/dataset (all metrics taken from that run)
"""

import csv
import os
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_wandb():
    for line in (ROOT / ".env").read_text().splitlines():
        if line.startswith("WANDB_API_KEY="):
            os.environ["WANDB_API_KEY"] = line.split("=", 1)[1].strip().strip("'\"")
    import wandb
    return wandb.Api()


def run_metric(run, metric):
    return run.summary.get(f"test/{metric}", run.summary.get(metric))


def fetch_runs(api, project, datasets):
    """Pull finished runs and index them by (group = model, run name, dataset)."""
    runs = api.runs(project, filters={"state": "finished"})
    index = []
    for r in runs:
        ds = r.config.get("dataset")
        if ds in datasets:
            index.append({"group": r.group, "run": r.name, "dataset": ds, "_run": r})
    return index


def build_rows(model, tables, run_index):
    """One row per (model, dataset)."""
    datasets = [d["name"] for d in tables["datasets"]]
    metric_list = tables["metrics"]
    columns = [c for c in tables["columns"] if c not in ("dataset", "model")]

    rows = []
    for ds in datasets:
        row = {"model": model, "dataset": ds}

        # resolve the best fusion run for this model/dataset once
        best_item, best_score = None, float("-inf")
        for item in run_index:
            if item["dataset"] != ds or item["group"] != model or item["run"] == "baseline":
                continue
            score = run_metric(item["_run"], metric_list[0])
            if score is not None and score > best_score:
                best_item, best_score = item, score

        for run in tables["runs"]:
            run_name = run["name"]
            if run_name == "MIAFF_best_macro_f1":
                source = best_item
            else:
                want = run.get("run_name", run_name)
                source = next(
                    (i for i in run_index
                     if i["dataset"] == ds and i["group"] == model and i["run"] == want),
                    None)
            item = source["_run"] if source else None
            for metric in metric_list:
                col = f"{metric}_{run_name}"
                if col in columns:
                    row[col] = run_metric(item, metric) if item else ""
            col = "MIAFF_best_macro_f1_run_name"
            if run_name == "MIAFF_best_macro_f1" and col in columns:
                row[col] = best_item["run"] if best_item else ""

        rows.append(row)
    return rows


def main():
    config = yaml.safe_load((ROOT / "config.yml").read_text())
    eff = config["reporting"]["effectiveness"]
    full_tables = eff["tables"] if isinstance(eff["tables"], list) else [eff["tables"]]
    all_datasets = {d["name"] for t in full_tables for d in t["datasets"]}

    wandb_cfg = config.get("learning_mlp", {}).get("logging", {}).get("wandb", {})
    project = f"{os.environ.get('WANDB_ENTITY', 'jina-pro')}/{wandb_cfg.get('project', 'miaff')}"

    api = load_wandb()
    run_index = fetch_runs(api, project, all_datasets)
    print(f"Fetched {len(run_index)} finished runs for {len(all_datasets)} datasets from {project}")

    for tables in full_tables:
        title = tables["title"]
        rows = []
        for group_name, models in tables["models"].items():
            for m in models:
                for row in build_rows(m["name"], tables, run_index):
                    row["model"] = f"{group_name}:{m['name']}"
                    rows.append(row)

        # single flat header, first the id columns then the metric_run columns
        header = ["model", "dataset"] + [c for c in tables["columns"] if c != "dataset"]
        out_path = ROOT / eff["output"]["file"].replace("{table.title}", title)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
