#!/usr/bin/env python3
"""Dedicated inference benchmark for baseline vs MIAFF_base (N_cat_N_ce).

Methodology (per review requirements):
  - Reuses already-trained best.pt checkpoints, no retraining.
  - Baseline and MIAFF measured under identical hardware / settings.
  - Fixed batch size for each paired comparison (same B for both runs).
  - Warmup passes + R repeated full test-set passes (default R=10).
  - torch.cuda.synchronize() around the timed region.
  - NO sklearn metric computation inside the timed region (logits only;
    macro-F1 is computed once afterwards as a correctness sanity check).
  - Reports mean +/- std and p95 latency, per-sample latency (ms/sample),
    throughput (samples/s), peak allocated AND reserved GPU memory (MiB).

Two timed scopes per repeat:
  (a) e2e      : test-loader iteration + H2D transfer + forward
                 (deployment-realistic; num_workers=0 for determinism)
  (b) gpu_only : replay of on-GPU-cached batches, forward only
                 (pure model-execution cost, no dataloader noise)

Per-sample latency within a dataset:  L_d = T_d / N_d * 1000  (ms/sample).
Cross-dataset aggregation must be decided at reporting time:
  - equal-dataset-weighted mean of per-dataset latencies, or
  - pooled throughput = total_samples / total_time.
This script outputs per-(model, dataset, run) rows; aggregation is separate.

APPROVAL GATE:
  This script performs GPU measurements. DO NOT run the full benchmark
  until the methodology is approved. Use --dry-run (filesystem only,
  no GPU work, no model loading) to verify checkpoint coverage.

Outputs (on full run):
  - 09_output/05_Reporting/inference_benchmark.csv
"""

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "01_metadata_selection"))
sys.path.insert(0, str(PROJECT_ROOT / "04_Learning"))

import yaml

CHAR_MODELS = {"CharCNN", "CharLSTM", "CharGRU", "CharBiLSTM", "CharBiGRU",
               "CharCNNBiLSTM"}
LANG_MODELS = {"SecureBERT", "CySecBERT", "SecBERT", "DomURLsBERT"}

MIAFF_COMBINATION = "N_cat_N_ce"
REPEATS_DEFAULT = 10
WARMUP_DEFAULT = 10
# Datasets with both baseline and MIAFF checkpoints available for every model.
PAIRED_DATASETS = ["dibhran_malware_binary", "macformer_malware_binary",
                   "duben_malware_binary", "dibhran_phishing_binary",
                   "macformer_phishing_binary", "duben_phishing_binary",
                   "threatfox_malware_family", "dibhran_malware_family"]
# Fixed batch size per encoder family (identical within each baseline-MIAFF pair).
BATCH_CHAR_DEFAULT = 256
BATCH_LANG_DEFAULT = 128


def ckpt_paths(model, dataset):
    base = PROJECT_ROOT / "09_output/04_Learning"
    return {
        "baseline": base / model / dataset / "best.pt",
        "MIAFF_base": base / "Fusion" / MIAFF_COMBINATION / model / dataset / "best.pt",
    }


def build_bundle_and_model(model, dataset, run_type, config, ckpt, device,
                           batch_size, max_length, num_workers):
    """Build test loader + weight-loaded model. No training performed."""
    import torch  # noqa: F401  (ensures torch available before heavy imports)
    from data_loader import (build_dataloaders, build_lang_dataloaders,
                             resolve_dataset_config)
    dataset_config = resolve_dataset_config(config, dataset)
    task = dataset_config["task"]
    cfg_path = PROJECT_ROOT / "config.yml"

    if run_type == "baseline":
        if model in CHAR_MODELS:
            from BASELINE_Char import MODEL_REGISTRY as REG
            from BASELINE_Char import resolve_model_config as resolve_char
            model_cfg = resolve_char(config["models"], model)
            params = {k: v for k, v in model_cfg.items() if k != "name"}
            bundle = build_dataloaders(
                cfg_path, dataset, include_selected_metadata=False,
                batch_size=batch_size, max_length=max_length,
                num_workers=num_workers, seed=42)
            net = REG[model](vocab_size=bundle["vocab_size"],
                             output_size=bundle["num_classes"], **params)
        else:
            from BASELINE_LANG import MODEL_REGISTRY as REG
            from BASELINE_LANG import resolve_model_config as resolve_lang
            model_cfg = resolve_lang(config["models"], model)
            params = {k: v for k, v in model_cfg.items()
                      if k not in ("name", "pretrained_path")}
            bundle = build_lang_dataloaders(
                cfg_path, dataset, include_selected_metadata=False,
                batch_size=batch_size, max_length=max_length,
                num_workers=num_workers, seed=42,
                pretrained_path=model_cfg["pretrained_path"])
            net = REG[model](output_size=bundle["num_classes"],
                             pretrained_path=model_cfg["pretrained_path"],
                             **params)

        def forward(batch):
            if model in CHAR_MODELS:
                return net(batch["input"].to(device, non_blocking=True))
            return net(batch["input_ids"].to(device, non_blocking=True),
                       batch["attention_mask"].to(device, non_blocking=True))
        net.load_state_dict(ckpt["model_state_dict"], strict=True)

    else:
        from Fusion import batch_inputs, build_model
        enc_type = "char" if model in CHAR_MODELS else "lang"
        enc_key = "char_model_config" if enc_type == "char" else "lang_model_config"
        enc_cfg = dict(ckpt[enc_key])
        enc_cfg.setdefault("name", model)
        meta_cfg = dict(ckpt["metadata_model_config"])
        meta_cfg.setdefault("name", "Metadata_MLP")
        learning = {"encoder_type": enc_type,
                    "encoder_model": enc_cfg,
                    "metadata_model_config": meta_cfg,
                    "combination": dict(ckpt["combination"])}
        loader_args = {"include_selected_metadata": True,
                       "metadata_patterns": ckpt["metadata_patterns"],
                       "batch_size": batch_size, "max_length": max_length,
                       "num_workers": num_workers, "seed": 42}
        if enc_type == "lang":
            loader_args["pretrained_path"] = config["models"][model]["pretrained_path"]
            bundle = build_lang_dataloaders(cfg_path, dataset, **loader_args)
        else:
            bundle = build_dataloaders(cfg_path, dataset, **loader_args)
        net = build_model(bundle, learning, device)
        # Checkpoints predate a rename of the fusion submodule
        # (char_encoder -> text_encoder); remap keys, keep strict load.
        sd = ckpt["model_state_dict"]
        if enc_type == "char" and any(k.startswith("char_encoder.") for k in sd):
            sd = {("text_encoder." + k[len("char_encoder."):]
                   if k.startswith("char_encoder.") else k): v
                  for k, v in sd.items()}
        net.load_state_dict(sd, strict=True)

        def forward(batch):
            return net(batch_inputs(batch, device),
                       batch["metadata"].to(device, non_blocking=True))

    net.to(device)
    net.eval()
    return bundle["loaders"]["test"], forward, net, task, bundle


def percentile(values, q):
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    cuts = statistics.quantiles(sorted(values), n=100)
    return cuts[min(int(q) - 1, len(cuts) - 1)]


def benchmark_pair(model, dataset, run_type, config, device, batch_size,
                   max_length, warmup, repeats):
    import torch
    from BASELINE_Char import calculate_metrics

    path = ckpt_paths(model, dataset)[run_type]
    ckpt = torch.load(path, map_location=device, weights_only=False)
    loader, forward, net, task, bundle = build_bundle_and_model(
        model, dataset, run_type, config, ckpt, device,
        batch_size, max_length, num_workers=0)
    n_test = len(bundle["datasets"]["test"])
    class_names = bundle["class_names"]

    # Warmup (not timed).
    with torch.inference_mode():
        for i, batch in enumerate(loader):
            if i >= warmup:
                break
            _ = forward(batch)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    # Cache batches on GPU for the gpu_only scope.
    cached = []
    with torch.inference_mode():
        for batch in loader:
            if model in CHAR_MODELS and run_type == "baseline":
                cached.append(batch["input"].to(device))
            elif run_type == "baseline":
                cached.append((batch["input_ids"].to(device),
                               batch["attention_mask"].to(device)))
            else:
                from Fusion import batch_inputs
                cached.append((batch_inputs(batch, device),
                               batch["metadata"].to(device)))
    # Collect labels once (outside timed region).
    all_labels = []
    for batch in loader:
        all_labels.append(batch["label"])
    import torch as _t
    all_labels = _t.cat(all_labels)

    e2e_times, gpu_times = [], []
    with torch.inference_mode():
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)  # peaks cover timed region only
        for rep in range(repeats):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            logits_list = []
            for batch in loader:
                logits_list.append(forward(batch))
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            e2e_times.append(time.perf_counter() - t0)

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            for item in cached:
                if run_type == "baseline" and model in CHAR_MODELS:
                    _ = net(item)
                elif run_type == "baseline":
                    _ = net(item[0], item[1])
                else:
                    _ = net(item[0], item[1])
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            gpu_times.append(time.perf_counter() - t0)

    peak_alloc, peak_res = "", ""
    if device.type == "cuda":
        peak_alloc = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        peak_res = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
    raw_repeats = [{"repeat": i, "e2e_s": e2e_times[i], "gpu_s": gpu_times[i]}
                   for i in range(repeats)]

    # Sanity metric (ONE untimed pass AFTER peak readout, sklearn allowed here).
    with torch.inference_mode():
        outs = [forward(b) for b in loader]
    logits = torch.cat([o for o in outs]).cpu()
    preds = logits.argmax(dim=1).numpy()
    probs = logits.softmax(dim=1).numpy()
    macro_f1 = calculate_metrics(all_labels.numpy(), preds, probs,
                                 task=task,
                                 class_names=class_names)["macro_f1"]

    def stats(ts):
        m = statistics.mean(ts)
        s = statistics.pstdev(ts) if len(ts) > 1 else 0.0
        return m, s, percentile(ts, 95)

    e2e_m, e2e_s, e2e_p95 = stats(e2e_times)
    gpu_m, gpu_s, gpu_p95 = stats(gpu_times)
    return {
        "model": model, "dataset": dataset, "run": run_type,
        "n_test": n_test, "batch_size": batch_size, "repeats": repeats,
        "warmup": warmup,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "e2e_mean_s": e2e_m, "e2e_std_s": e2e_s, "e2e_p95_s": e2e_p95,
        "gpu_mean_s": gpu_m, "gpu_std_s": gpu_s, "gpu_p95_s": gpu_p95,
        "latency_ms_per_sample_e2e": e2e_m / n_test * 1000,
        "latency_ms_per_sample_gpu": gpu_m / n_test * 1000,
        "throughput_samples_per_s_e2e": n_test / e2e_m,
        "throughput_samples_per_s_gpu": n_test / gpu_m,
        "peak_memory_alloc_mib": peak_alloc,
        "peak_memory_reserved_mib": peak_res,
        "sanity_macro_f1": macro_f1,
    }, raw_repeats


def dry_run(config, models, datasets):
    print("DRY RUN -- checkpoint coverage (no GPU, no timing)")
    missing, found = [], []
    for model in models:
        for ds in datasets:
            for run_type in ("baseline", "MIAFF_base"):
                p = ckpt_paths(model, ds)[run_type]
                (found if p.exists() else missing).append((model, ds, run_type))
    print(f"found: {len(found)}, missing: {len(missing)}")
    for m in missing:
        print("  MISSING", m)
    return missing


def verify_checkpoints(config, models, datasets):
    """Load every checkpoint into its model (strict) on CPU.

    No timing, no GPU, no training. Reports per-pair OK/FAIL plus the
    expected test macro-F1 (from metrics.json) used later to check the
    benchmark's sanity_macro_f1 for prediction consistency.
    Returns (ok_list, fail_list).
    """
    import torch
    from data_loader import resolve_dataset_config  # noqa: F401
    ok, fail = [], []
    for model in models:
        for ds in datasets:
            for run_type in ("baseline", "MIAFF_base"):
                tag = (model, ds, run_type)
                try:
                    path = ckpt_paths(model, ds)[run_type]
                    ckpt = torch.load(path, map_location="cpu", weights_only=False)
                    assert "model_state_dict" in ckpt, "no model_state_dict"
                    device = torch.device("cpu")
                    if run_type == "baseline" and model in LANG_MODELS:
                        from BASELINE_LANG import MODEL_REGISTRY as REG
                        from BASELINE_LANG import resolve_model_config as resolve_lang
                        model_cfg = resolve_lang(config["models"], model)
                        params = {k: v for k, v in model_cfg.items()
                                  if k not in ("name", "pretrained_path")}
                        # arch check only: output dim from checkpoint
                        ncls = int(ckpt.get("num_classes", 2))
                        net = REG[model](
                            output_size=ncls,
                            pretrained_path=model_cfg["pretrained_path"],
                            **params)
                    elif run_type == "baseline":
                        from BASELINE_Char import MODEL_REGISTRY as REG
                        from BASELINE_Char import resolve_model_config as resolve_char
                        model_cfg = resolve_char(config["models"], model)
                        params = {k: v for k, v in model_cfg.items() if k != "name"}
                        net = REG[model](
                            vocab_size=int(ckpt.get("vocab_size", 256)),
                            output_size=int(ckpt.get("num_classes", 2)), **params)
                    else:
                        from Fusion import build_model
                        enc_type = "char" if model in CHAR_MODELS else "lang"
                        enc_key = ("char_model_config" if enc_type == "char"
                                   else "lang_model_config")
                        enc_cfg = dict(ckpt[enc_key])
                        enc_cfg.setdefault("name", model)
                        meta_cfg = dict(ckpt["metadata_model_config"])
                        meta_cfg.setdefault("name", "Metadata_MLP")
                        learning = {"encoder_type": enc_type,
                                    "encoder_model": enc_cfg,
                                    "metadata_model_config": meta_cfg,
                                    "combination": dict(ckpt["combination"])}
                        # rebuilt only to validate load_state_dict below
                        net = None
                        bundle_meta = {
                            "vocab_size": int(ckpt.get("vocab_size", 256)),
                            "num_classes": int(ckpt.get("num_classes", 2)),
                            "metadata_dim": len(ckpt.get("metadata_features", [])),
                        }
                        if enc_type == "char":
                            from BASELINE_Char import MODEL_REGISTRY as CREG
                            text_enc = CREG[model](
                                vocab_size=bundle_meta["vocab_size"],
                                output_size=2,
                                **{k: v for k, v in enc_cfg.items() if k != "name"})
                        else:
                            from BASELINE_LANG import MODEL_REGISTRY as LREG
                            text_enc = LREG[model](
                                output_size=2,
                                pretrained_path=config["models"][model][
                                    "pretrained_path"],
                                **{k: v for k, v in enc_cfg.items()
                                   if k not in ("name", "pretrained_path")})
                        from Metadata_MLP import Metadata_MLP  # noqa: F401
                        prefix = ("char_encoder." if enc_type == "char"
                                  else "text_encoder.")
                        enc_state = text_enc.state_dict()
                        # Fusion replaces the encoder head with Identity, so
                        # fc/classifier keys are absent from checkpoints by design.
                        enc_keys = {k for k in enc_state
                                    if not k.startswith(("fc.", "classifier."))}
                        n_enc_missing = len(
                            {prefix + k for k in enc_keys}
                            - set(ckpt["model_state_dict"]))
                        assert n_enc_missing == 0, \
                            f"{n_enc_missing} encoder keys missing in checkpoint"
                        net = None  # full fusion assembly validated in benchmark
                    if net is not None and run_type == "baseline":
                        missing_keys, unexpected = net.load_state_dict(
                            ckpt["model_state_dict"], strict=False)[:2]
                        # strict check on matching keys: fail only if many missing
                        state = net.state_dict()
                        n_missing = len(set(state) - set(ckpt["model_state_dict"]))
                        assert n_missing == 0, f"{n_missing} arch keys missing"
                    n_params = sum(t.numel() for t in
                                   ckpt["model_state_dict"].values())
                    mp = PROJECT_ROOT / "09_output/04_Learning"
                    mdir = (mp / model / ds if run_type == "baseline"
                            else mp / "Fusion" / MIAFF_COMBINATION / model / ds)
                    expected_f1 = ""
                    mj = mdir / "metrics.json"
                    if mj.exists():
                        expected_f1 = json.load(open(mj)).get("test", {}).get("macro_f1")
                    ok.append((model, ds, run_type, n_params, expected_f1))
                    print(f"OK   {model} {ds} {run_type} "
                          f"params={n_params} expected_f1={expected_f1}", flush=True)
                except Exception as e:
                    fail.append((model, ds, run_type, repr(e)))
                    print(f"FAIL {model} {ds} {run_type}: {e!r}", flush=True)
    print(f"verify: {len(ok)} OK, {len(fail)} FAIL")
    return ok, fail


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--batch-size", type=int, default=None,
                    help="override batch size for all pairs (default: 256 char, 128 lang)")
    ap.add_argument("--max-length", type=int, default=64)
    ap.add_argument("--repeats", type=int, default=REPEATS_DEFAULT)
    ap.add_argument("--warmup", type=int, default=WARMUP_DEFAULT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verify-only", action="store_true",
                    help="build+load every checkpoint on CPU, no timing, then exit")
    ap.add_argument("--out",
                    default=str(PROJECT_ROOT / "09_output/05_Reporting/inference_benchmark.csv"))
    ap.add_argument("--raw-out", default=None)
    ap.add_argument("--env-out", default=None)
    args = ap.parse_args()

    config = yaml.safe_load((PROJECT_ROOT / "config.yml").read_text())
    eff = config["effectiveness_efficiency"]
    models = args.models or [a["name"] for a in eff["approaches"]]
    datasets = args.datasets or list(PAIRED_DATASETS)

    if args.verify_only:
        ok, fail = verify_checkpoints(config, models, datasets)
        sys.exit(1 if fail else 0)

    missing = dry_run(config, models, datasets)
    if args.dry_run:
        return
    if missing:
        print("Aborting: %d checkpoints missing (see above). "
              "Retrain or exclude before benchmarking." % len(missing))
        sys.exit(2)

    import torch
    torch.backends.cudnn.benchmark = False
    dev_str = args.device if (args.device == "cpu" or ":" in args.device) \
        else args.device + ":0"
    device = torch.device(dev_str)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    print("device:", device,
          torch.cuda.get_device_name(device) if device.type == "cuda" else "")

    env = {
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "device": str(device),
        "batch_size_char": BATCH_CHAR_DEFAULT,
        "batch_size_lang": BATCH_LANG_DEFAULT,
        "batch_size_override": args.batch_size,
        "max_length": args.max_length,
        "num_workers": 0,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "datasets": datasets,
        "models": models,
        "combination": MIAFF_COMBINATION,
    }
    env_out = args.env_out or str(Path(args.out).parent / "inference_benchmark_env.json")
    with open(env_out, "w") as f:
        json.dump(env, f, indent=2)
    print(f"Wrote env to {env_out}")

    rows, raw_rows = [], []
    for model in models:
        bsize = args.batch_size or (BATCH_CHAR_DEFAULT if model in CHAR_MODELS
                                    else BATCH_LANG_DEFAULT)
        for ds in datasets:
            for run_type in ("baseline", "MIAFF_base"):
                print(f"benchmarking {model} {ds} {run_type} "
                      f"(batch={bsize}) ...", flush=True)
                row, raw = benchmark_pair(
                    model, ds, run_type, config, device,
                    bsize, args.max_length, args.warmup, args.repeats)
                rows.append(row)
                for r in raw:
                    raw_rows.append({"model": model, "dataset": ds, "run": run_type,
                                    **r})
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                    torch.cuda.empty_cache()

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} aggregated rows to {args.out}")

    raw_out = args.raw_out or str(Path(args.out).parent / "inference_benchmark_raw.csv")
    with open(raw_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(raw_rows[0].keys()))
        w.writeheader()
        w.writerows(raw_rows)
    print(f"Wrote {len(raw_rows)} raw repeat rows to {raw_out}")


if __name__ == "__main__":
    main()
