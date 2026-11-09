#!/usr/bin/env python3
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import yaml

# ponytail: static 4-panel layout from efficiency_wandb.csv; parameterize CLI args if needed for other CSVs

ROOT = Path(__file__).resolve().parent.parent
INPUT_CSV = ROOT / "efficiency_wandb.csv"
CONFIG_PATH = ROOT / "config.yml"
OUTPUT_DIR = ROOT / "09_output" / "05_Reporting" / "figures"
OUTPUT_STEM = "effectiveness_efficiency"

NN_MODELS = {
    "CharCNN",
    "CharLSTM",
    "CharGRU",
    "CharBiLSTM",
    "CharBiGRU",
    "CharCNNBiLSTM",
}

CHARTS = [
    ("training_time_min", "(a) Training Time (s)", 60.0),
    ("test_evaluation_time_min", "(b) Test Evaluation Time (s)", 60.0),
    ("peak_train_memory_gib", "(c) Peak Training GPU Memory (GiB)", 1.0),
    ("peak_test_memory_gib", "(d) Peak Test GPU Memory (GiB)", 1.0),
]


def _make_tick_label(last_value: float):
    last_value = max(last_value, 1e-12)

    def _formatter(value, _pos):
        if value <= 0:
            return "0"
        if abs(value - last_value) / last_value < 1e-6:
            if last_value < 1:
                return f"{last_value:.2g}"
            return f"{last_value:.1f}" if last_value < 100 else f"{int(round(last_value)):d}"
        exp = round(np.log10(value))
        if abs(10**exp - value) / (10**exp) < 1e-6:
            return f"{10**exp:g}" if exp < 0 else f"{int(round(value)):d}"
        return ""

    return _formatter


def load_data():
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    colors = cfg.get("approaches_colors", {})
    models = [m["name"] for m in cfg.get("effectiveness_efficiency", {}).get("approaches", [])]

    by_model = {}
    with INPUT_CSV.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            m = row["model"]
            v = row["variant"]
            by_model.setdefault(m, {})[v] = {
                k: float(row[k]) for k in row if k not in ("model", "variant")
            }
    return models, colors, by_model


def main():
    models, colors, by_model = load_data()

    fig, axes = plt.subplots(1, 4, figsize=(18, 5.5), sharey=True)

    for ax, (col, xlabel, factor) in zip(axes, CHARTS):
        ax.set_xscale("log")
        ax.xaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())

        max_x, min_x = -np.inf, np.inf
        for model in models:
            variants = by_model.get(model, {})
            if "baseline" not in variants or "N_cat_N_ce" not in variants:
                continue
            b = variants["baseline"]
            e = variants["N_cat_N_ce"]
            bx, by = b[col] * factor, b["macro_f1"]
            ex, ey = e[col] * factor, e["macro_f1"]

            max_x = max(max_x, bx, ex)
            min_x = min(min_x, bx, ex)

            color = colors.get(model, "#888888")
            marker = "o" if model in NN_MODELS else "s"

            ax.annotate(
                "",
                xy=(ex, ey),
                xytext=(bx, by),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=color,
                    alpha=0.85,
                    linewidth=1.5,
                    shrinkA=6,
                    shrinkB=7,
                    mutation_scale=11,
                ),
                zorder=3,
            )
            ax.scatter(
                [bx],
                [by],
                facecolors="none",
                edgecolors=color,
                marker=marker,
                s=120,
                linewidths=1.6,
                zorder=4,
            )
            ax.scatter(
                [ex],
                [ey],
                facecolors=color,
                edgecolors=color,
                marker=marker,
                s=120,
                linewidths=0.6,
                zorder=5,
            )

        if np.isfinite(max_x) and max_x > 0:
            start_exp = max(int(np.floor(np.log10(min_x))), -3) if (np.isfinite(min_x) and min_x > 0) else -2
            ticks = []
            exp = start_exp
            while 10**exp <= max_x:
                ticks.append(10.0**exp)
                exp += 1
            custom_appended = False
            if not ticks or ticks[-1] < max_x:
                ticks.append(max_x)
                custom_appended = True
            if custom_appended and len(ticks) >= 2:
                prev_tick = ticks[-2]
                if np.log10(max_x / prev_tick) < 0.28:
                    ticks.remove(prev_tick)
            ax.set_xticks(ticks)
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(_make_tick_label(max_x)))

        ax.set_xlabel(xlabel, fontsize=16, labelpad=8)
        ax.set_ylim(0.60, 0.90)
        ax.set_yticks(np.arange(0.60, 0.91, 0.05))
        ax.tick_params(axis="y", labelsize=15, labelleft=True)
        ax.tick_params(axis="x", labelsize=14)
        ax.grid(True, linestyle="--", alpha=0.4, which="both")
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Average Macro-F1", fontsize=17)

    handles = []
    for model in models:
        color = colors.get(model, "#888888")
        marker = "o" if model in NN_MODELS else "s"
        handles.append(
            mlines.Line2D(
                [],
                [],
                color=color,
                marker=marker,
                linestyle="None",
                markersize=10,
                markerfacecolor=color,
                label=model,
            )
        )
    handles.append(
        mlines.Line2D(
            [],
            [],
            color="0.4",
            marker="o",
            linestyle="None",
            markersize=10,
            markerfacecolor="none",
            markeredgewidth=1.6,
            label="Baseline",
        )
    )
    handles.append(
        mlines.Line2D(
            [],
            [],
            color="0.4",
            marker="o",
            linestyle="None",
            markersize=10,
            markerfacecolor="0.4",
            markeredgewidth=0.8,
            label="MIAFF",
        )
    )

    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=6,
        frameon=True,
        framealpha=0.9,
        fontsize=14,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.90))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_png = OUTPUT_DIR / f"{OUTPUT_STEM}.png"
    out_pdf = OUTPUT_DIR / f"{OUTPUT_STEM}.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_png}")
    print(f"Saved {out_pdf}")


if __name__ == "__main__":
    main()
