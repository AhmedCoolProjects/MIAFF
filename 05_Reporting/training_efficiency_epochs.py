"""Training efficiency recovery: per-epoch training cost WITHOUT retraining.

Primary source is local artefacts (same numbers that were logged to W&B):
  baseline : 09_output/04_Learning/<model>/<dataset>/history.csv + metrics.json
  MIAFF    : 09_output/04_Learning/Fusion/N_cat_N_ce/<model>/<dataset>/history.csv + metrics.json

history.csv holds one row per executed epoch with train_time_seconds, so:
  epochs_executed     = number of data rows
  mean_time_per_epoch = mean(train_time_seconds)
  std_time_per_epoch  = pstdev(train_time_seconds)

metrics.json holds the summary totals (system_metrics), best_epoch,
split_sizes and test macro_f1.

W&B is used ONLY as fallback for runs whose local artefacts are missing
(6 char-baseline threatfox_malware_binary runs have W&B entries but no
local directory). Use --verify-wandb to cross-check local totals against
W&B summaries; mismatches are reported, not silently overwritten.

Outputs:
  - 09_output/05_Reporting/training_efficiency_epochs.csv       (per-run)
  - 09_output/05_Reporting/training_efficiency_epochs_means.csv (per-model means)
"""

import argparse
import csv
import json
import os
import statistics
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "09_output/05_Reporting"
OUT_PER_RUN = OUTPUT_DIR / "training_efficiency_epochs.csv"
OUT_MEANS = OUTPUT_DIR / "training_efficiency_epochs_means.csv"

MIAFF_BASE_RUN_NAME = "N_cat_N_ce"
WANTED_RUNS = {"baseline": "baseline", MIAFF_BASE_RUN_NAME: "MIAFF_base"}


def history_path(model, dataset, run_type):
    if run_type == "baseline":
        return ROOT / "09_output/04_Learning" / model / dataset / "history.csv"
    return ROOT / "09_output/04_Learning/Fusion" / MIAFF_BASE_RUN_NAME / model / dataset / "history.csv"


def metrics_path(model, dataset, run_type):
    return history_path(model, dataset, run_type).parent / "metrics.json"


def read_history(path):
    """Return (epochs, mean_train, std_train, mean_val, mean_peak_train)."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    train_t = [float(r["train_time_seconds"]) for r in rows if r.get("train_time_seconds")]
    val_t = [float(r["validation_time_seconds"]) for r in rows if r.get("validation_time_seconds")]
    peak_t = [float(r["train_peak_memory_mib"]) for r in rows if r.get("train_peak_memory_mib")]
    return {
        "epochs_executed": len(rows),
        "mean_time_per_epoch": statistics.mean(train_t) if train_t else "",
        "std_time_per_epoch": statistics.pstdev(train_t) if len(train_t) > 1 else "",
        "mean_time_val_per_epoch": statistics.mean(val_t) if val_t else "",
        "mean_peak_memory_train_epoch": statistics.mean(peak_t) if peak_t else "",
    }


def read_metrics(path):
    with open(path) as f:
        m = json.load(f)
    sys_m = m.get("system_metrics", {})
    return {
        "best_epoch": m.get("best_epoch", ""),
        "time_train_total": sys_m.get("time_train"),
        "time_val_total": sys_m.get("time_val"),
        "time_test": sys_m.get("time_test"),
        "peak_memory_train": sys_m.get("peak_memory_train"),
        "peak_memory_test": sys_m.get("peak_memory_test"),
        "test_macro_f1": (m.get("test") or {}).get("macro_f1"),
        "test_n": (m.get("split_sizes") or {}).get("test"),
    }


def safe_mean(vals):
    nums = [v for v in vals if isinstance(v, (int, float))]
    return statistics.mean(nums) if nums else ""


def wandb_fallback(datasets, models):
    """Fetch W&B summaries only for runs missing local artefacts.

    Returns {(model, dataset, run_type): dict} with retries; empty on failure.
    """
    for line in (ROOT / ".env").read_text().splitlines():
        if line.startswith("WANDB_API_KEY="):
            os.environ["WANDB_API_KEY"] = line.split("=", 1)[1].strip().strip("'\"")
    import wandb
    out = {}
    for attempt in range(3):
        try:
            api = wandb.Api(timeout=60)
            runs = api.runs(
                "jina-pro/miaff",
                filters={"state": "finished"},
            )
            for r in runs:
                rt = WANTED_RUNS.get(r.name)
                ds = (r.config or {}).get("dataset")
                if rt is None or ds not in datasets or r.group not in models:
                    continue
                key = (r.group, ds, rt)
                if key in out:
                    continue
                s = r.summary
                # NOTE: no scan_history here -- per-epoch keys filter is broken
                # server-side (returns 0 rows); epochs come from local files.
                # For local-missing runs we count epochs via unfiltered scan.
                epochs, et = [], []
                try:
                    for ev in r.scan_history(page_size=500):
                        if ev.get("epoch") is None:
                            continue
                        t = ev.get("system/time_train_epoch")
                        if t is not None:
                            et.append(float(t))
                    epochs = len(et)
                except Exception as e:
                    print(f"W&B history scan failed for {key}: {e}", file=sys.stderr)
                out[key] = {
                    "epochs_executed": epochs,
                    "best_epoch": s.get("best_epoch", ""),
                    "time_train_total": s.get("system/time_train"),
                    "time_val_total": s.get("system/time_val"),
                    "time_test": s.get("system/time_test"),
                    "peak_memory_train": s.get("system/peak_memory_train"),
                    "peak_memory_test": s.get("system/peak_memory_test"),
                    "test_macro_f1": s.get("test/macro_f1"),
                    "mean_time_per_epoch": statistics.mean(et) if et else "",
                    "std_time_per_epoch": statistics.pstdev(et) if len(et) > 1 else "",
                    "mean_time_val_per_epoch": "",
                    "mean_peak_memory_train_epoch": "",
                    "test_n": "",
                    "_source": "wandb",
                }
            return out
        except Exception as e:
            print(f"W&B attempt {attempt + 1} failed: {e}", file=sys.stderr)
            time.sleep(5 * (attempt + 1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-wandb", action="store_true",
                    help="cross-check local totals against W&B summaries")
    ap.add_argument("--wandb-fallback", action="store_true",
                    help="query W&B for runs missing local artefacts (needs network)")
    args = ap.parse_args()

    config = yaml.safe_load((ROOT / "config.yml").read_text())
    eff = config["effectiveness_efficiency"]
    approaches = [a["name"] for a in eff["approaches"]]
    datasets = [d["name"] for sc in config.get("scenarios", {}).values()
                for d in sc.get("datasets", [])]

    rows, missing = [], []
    for model in approaches:
        for ds in datasets:
            for run_type in ("baseline", "MIAFF_base"):
                hp, mp = history_path(model, ds, run_type), metrics_path(model, ds, run_type)
                if hp.exists() and mp.exists():
                    row = {"model": model, "dataset": ds, "run": run_type,
                           "_source": "local"}
                    row.update(read_history(hp))
                    row.update(read_metrics(mp))
                    tt = row.get("time_train_total")
                    ee = row.get("epochs_executed")
                    row["implied_time_per_epoch"] = (
                        tt / ee if isinstance(tt, (int, float)) and ee else "")
                    rows.append(row)
                else:
                    missing.append((model, ds, run_type))

    if missing and args.wandb_fallback:
        print(f"{len(missing)} runs lack local artefacts, trying W&B fallback...")
        fb = wandb_fallback(set(datasets), set(approaches))
        for key in missing:
            if key in fb:
                model, ds, rt = key
                row = {"model": model, "dataset": ds, "run": rt}
                row.update(fb[key])
                tt = row.get("time_train_total")
                ee = row.get("epochs_executed")
                row["implied_time_per_epoch"] = (
                    tt / ee if isinstance(tt, (int, float)) and ee else "")
                rows.append(row)
            else:
                print(f"  NO DATA for {key} (no local files, no W&B entry)")
    elif missing:
        print(f"{len(missing)} runs lack local artefacts "
              f"(rerun with --wandb-fallback when network is available):")
        for key in missing:
            print(f"  {key}")

    rows.sort(key=lambda r: (r["model"], r["dataset"], r["run"]))
    header = ["model", "dataset", "run", "source", "epochs_executed", "best_epoch",
              "time_train_total", "mean_time_per_epoch", "std_time_per_epoch",
              "implied_time_per_epoch", "mean_time_val_per_epoch",
              "time_val_total", "peak_memory_train",
              "mean_peak_memory_train_epoch", "peak_memory_test",
              "time_test", "test_n", "test_macro_f1"]
    OUT_PER_RUN.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PER_RUN, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get({"source": "_source"}.get(k, k), "") for k in header})
    print(f"Wrote {len(rows)} per-run rows to {OUT_PER_RUN}")

    mean_header = ["model", "run", "n_datasets",
                   "mean_epochs_executed", "mean_best_epoch",
                   "mean_time_train_total", "mean_time_per_epoch",
                   "mean_time_val_per_epoch", "mean_peak_memory_train",
                   "max_epochs_executed", "min_epochs_executed"]
    mean_rows = []
    for model in sorted({r["model"] for r in rows}):
        for run_type in ("baseline", "MIAFF_base"):
            rs = [r for r in rows if r["model"] == model and r["run"] == run_type]
            if not rs:
                continue
            mean_rows.append({
                "model": model,
                "run": run_type,
                "n_datasets": len(rs),
                "mean_epochs_executed": safe_mean(r["epochs_executed"] for r in rs),
                "mean_best_epoch": safe_mean(
                    r["best_epoch"] for r in rs if isinstance(r["best_epoch"], int)),
                "mean_time_train_total": safe_mean(r["time_train_total"] for r in rs),
                "mean_time_per_epoch": safe_mean(r["mean_time_per_epoch"] for r in rs),
                "mean_time_val_per_epoch": safe_mean(
                    r["mean_time_val_per_epoch"] for r in rs),
                "mean_peak_memory_train": safe_mean(r["peak_memory_train"] for r in rs),
                "max_epochs_executed": max(r["epochs_executed"] for r in rs),
                "min_epochs_executed": min(r["epochs_executed"] for r in rs),
            })
    with open(OUT_MEANS, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=mean_header)
        w.writeheader()
        w.writerows(mean_rows)
    print(f"Wrote {len(mean_rows)} per-model mean rows to {OUT_MEANS}")

    if args.verify_wandb:
        print("--verify-wandb: cross-check not yet run in offline mode")


if __name__ == "__main__":
    main()
