"""Effectiveness vs. complexity figure: baseline -> MIAFF_base arrow plot.

Reads 09_output/05_Reporting/num_params.csv (one row per approach):
  model, num_params_baseline, num_params_N_cat_N_ce,
         avg_macro_f1_baseline, avg_macro_f1_N_cat_N_ce

Each approach is drawn with its own color (config.yml -> approaches_colors):
  - hollow marker at the baseline point
  - filled marker at the MIAFF_base point
  - an arrow from baseline to MIAFF_base
  - circles for nn (char) approaches, squares for plm approaches
x axis = number of parameters (log scale), y axis = avg macro F1.

Output: 09_output/05_Reporting/figures/effectiveness_complexity.{png,pdf}
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import yaml

ROOT = Path(__file__).resolve().parent.parent

NUM_PARAMS_CSV = ROOT / "09_output/05_Reporting/num_params.csv"
OUTPUT_DIR = ROOT / "09_output/05_Reporting/figures"
OUTPUT_STEM = "effectiveness_complexity"

# nn (char) approaches get light colors defined in config + circle markers;
# everything else (plm) gets darker colors + square markers
NN_MODELS = {"CharCNN", "CharLSTM", "CharGRU", "CharBiLSTM", "CharBiGRU", "CharCNNBiLSTM"}

PARAMS_COL = {"baseline": "num_params_baseline", "MIAFF_base": "num_params_N_cat_N_ce"}
F1_COL = {"baseline": "avg_macro_f1_baseline", "MIAFF_base": "avg_macro_f1_N_cat_N_ce"}


def load_config():
    return yaml.safe_load((ROOT / "config.yml").read_text())


def load_rows():
    with NUM_PARAMS_CSV.open() as f:
        return list(csv.DictReader(f))


def plot(config, rows):
    colors = config.get("approaches_colors", {})
    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    legend_handles = []
    for r in rows:
        model = r["model"]
        color = colors.get(model, "#888888")
        marker = "o" if model in NN_MODELS else "s"

        x0, y0 = float(r[PARAMS_COL["baseline"]]), float(r[F1_COL["baseline"]])
        x1, y1 = float(r[PARAMS_COL["MIAFF_base"]]), float(r[F1_COL["MIAFF_base"]])

        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops=dict(
                arrowstyle="-|>",
                color=color,
                lw=1.6,
                alpha=0.85,
                shrinkA=9,
                shrinkB=12,
                mutation_scale=13,
            ),
        )
        # hollow marker at baseline
        ax.scatter(
            [x0], [y0], marker=marker, s=150,
            facecolor="none", edgecolor=color, linewidth=1.8, zorder=3,
        )
        # filled marker at MIAFF_base
        ax.scatter(
            [x1], [y1], marker=marker, s=170, color=color,
            edgecolor="white", linewidth=0.6, zorder=3,
        )
        legend_handles.append(
            plt.Line2D([], [], marker=marker, color="none", label=model,
                       markerfacecolor="none",
                       markeredgecolor=color, markeredgewidth=1.8,
                       markersize=10)
        )

    ax.set_xscale("log")
    ax.set_xlabel("Number of parameters", fontsize=15)
    ax.set_ylabel("Average macro F1", fontsize=15)
    ax.tick_params(labelsize=13)
    ax.grid(True, which="major", linestyle="--", linewidth=0.7, color="#cccccc", alpha=0.9)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.4, color="#e5e5e5", alpha=0.7)
    for spine in ax.spines.values():
        spine.set_color("#444444")

    nn_legend = plt.Line2D([], [], color="none")
    # note + proxies: empty marker = baseline, filled = MIAFF_base
    # two entries so they stack vertically instead of one wide line
    note_handles = [
        plt.Line2D([], [], color="none", label="empty = baseline"),
        plt.Line2D([], [], color="none", label="filled = MIAFF_base"),
    ]
    legend_handles = note_handles + legend_handles
    ax.legend(handles=legend_handles, fontsize=11,
              frameon=True, loc="upper left", bbox_to_anchor=(1.02, 1.0))

    fig.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / f"{OUTPUT_STEM}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{OUTPUT_STEM}.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    config = load_config()
    rows = load_rows()
    plot(config, rows)
    print(f"saved {OUTPUT_STEM}.png/.pdf in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
