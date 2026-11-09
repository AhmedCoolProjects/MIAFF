import csv
import os
from pathlib import Path
import wandb
import yaml

# ponytail: queries single metric (first in ablation.reporting.metrics); add per-metric file suffix when multiple metrics are configured
ROOT = Path(__file__).resolve().parent.parent

for line in (ROOT / ".env").read_text().splitlines():
    if line.startswith("WANDB_API_KEY="):
        os.environ["WANDB_API_KEY"] = line.split("=", 1)[1].strip().strip("'\"")

config = yaml.safe_load((ROOT / "config.yml").read_text())
ablation = config["ablation"]
rep = ablation["reporting"]
metric = rep["metrics"][0]
row_cols = rep["output"]["rows"]
val_cols = rep["output"]["columns"]
fieldnames = row_cols + [c for c in val_cols if c not in row_cols]

datasets = [
    d["name"]
    for sc in config.get("scenarios", {}).values()
    for d in sc.get("datasets", [])
] if rep.get("datasets") == "all" else rep["datasets"]

project = ablation["logging"]["wandb"]["project"]
ablation_group = ablation["logging"]["wandb"]["group"]
mlp_run_name = config.get("learning_mlp", {}).get("logging", {}).get("wandb", {}).get("run_name", "baseline")

api = wandb.Api()

for model in ablation["model"]:
    runs = api.runs(
        f"jina-pro/{project}",
        filters={"state": "finished", "group": {"$in": [model, ablation_group]}},
    )
    scores = {ds: {} for ds in datasets}
    for r in runs:
        ds = r.config.get("dataset")
        if ds not in scores:
            continue
        val = r.summary.get(f"test/{metric}", r.summary.get(metric))
        if val is None:
            continue
        if r.group == model and r.name == mlp_run_name:
            scores[ds]["all_categories"] = float(val)
        elif r.group == ablation_group:
            scores[ds][r.name] = float(val)

    rows = []
    for ds in datasets:
        s = scores[ds]
        all_val = s.get("all_categories")
        row = {"dataset": ds}
        for col in val_cols:
            if col.startswith("delta_") and col.endswith("_vs_all"):
                target = col[len("delta_") : -len("_vs_all")]
                t_val = s.get(target)
                row[col] = (t_val - all_val) if (t_val is not None and all_val is not None) else ""
            else:
                row[col] = s.get(col, "")
        rows.append(row)

    out_path = ROOT / rep["output"]["file"].format(model=model)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out_path}")

    assert out_path.is_file() and len(rows) == len(datasets), "Missing output rows"
    assert all(r["all_categories"] != "" for r in rows), "Missing baseline all_categories scores"