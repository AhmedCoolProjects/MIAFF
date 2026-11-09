#!/usr/bin/env python3
"""Plot correct-vs-incorrect signed fusion (CharCNN + MIAFF N_cat_N_ce) IG by metadata category."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "09_output/06_Interpreting/Interpretability/categories_fusion"
OUTPUT_STEM = ROOT / "interpretability_categories_fusion"
PANELS = (
    ("dibhran_malware_binary", "Malware Detection"),
    ("dibhran_phishing_binary", "Phishing Detection"),
    ("dibhran_malware_family", "Malware Family Classification"),
)
CATEGORIES = (
    ("dns", "DNS", 4.0),
    ("rdap", "RDAP", 3.0),
    ("tls", "TLS", 2.0),
    ("ip", "IP", 1.0),
    ("cross", "Cross", 0.0),
)
GROUPS = (
    ("Correct", "#4C72B0", 0.17, 0.22),
    ("Incorrect", "#C44E52", -0.17, 0.38),
)
BASE_COLUMNS = {
    "sample_id",
    "example",
    "true_label",
    "predicted_label",
    "classification",
}


def load_results(dataset: str) -> pd.DataFrame:
    path = DATA_DIR / f"IG_CATEGORIES_FUSION_{dataset}.csv"
    required = BASE_COLUMNS | {
        f"{category}_contribution_score" for category, _label, _center in CATEGORIES
    }
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns {sorted(missing)}")
    if frame["sample_id"].duplicated().any():
        raise ValueError(f"{path.name}: duplicate sample IDs")
    if (frame["true_label"] == "legit").any():
        raise ValueError(f"{path.name}: contains legit ground-truth samples")
    expected = np.where(
        frame["true_label"] == frame["predicted_label"], "Correct", "Incorrect"
    )
    if not np.array_equal(frame["classification"].to_numpy(), expected):
        raise ValueError(f"{path.name}: inconsistent classification labels")
    score_columns = [
        f"{category}_contribution_score" for category, _label, _center in CATEGORIES
    ]
    if not np.isfinite(frame[score_columns]).all().all():
        raise ValueError(f"{path.name}: non-finite attribution values")
    return frame


def density_jitter(
    values: np.ndarray,
    x_min: float,
    x_max: float,
    seed: int,
    max_width: float = 0.12,
) -> np.ndarray:
    """Deterministic vertical jitter scaled by local x-bin density; x is unchanged."""
    edges = np.linspace(x_min, x_max, 121)
    bins = np.clip(np.searchsorted(edges, values, side="right") - 1, 0, 119)
    occupancy = np.bincount(bins, minlength=120)
    width = max_width * np.sqrt(occupancy[bins] / max(occupancy.max(), 1))
    return np.random.default_rng(seed).uniform(-1.0, 1.0, len(values)) * width


def main() -> None:
    loaded = [(dataset, label, load_results(dataset)) for dataset, label in PANELS]
    score_columns = [
        f"{category}_contribution_score" for category, _label, _center in CATEGORIES
    ]
    scores = np.concatenate(
        [frame[score_columns].to_numpy().ravel() for _dataset, _label, frame in loaded]
    )
    x_min, x_max = float(scores.min()), float(scores.max())
    padding = 0.035 * (x_max - x_min)
    limits = (x_min - padding, x_max + padding)

    fig, axes = plt.subplots(3, 1, figsize=(3.5, 7.5), sharex=True, sharey=True)
    for panel_index, (dataset, label, frame) in enumerate(loaded):
        ax = axes[panel_index]
        counts = frame["classification"].value_counts()
        for category_index, (category, _display, center) in enumerate(CATEGORIES):
            column = f"{category}_contribution_score"
            for group_index, (group, color, offset, alpha) in enumerate(GROUPS):
                values = frame.loc[frame["classification"] == group, column].to_numpy()
                y = center + offset + density_jitter(
                    values,
                    *limits,
                    seed=42 + 100 * panel_index + 10 * category_index + group_index,
                )
                ax.scatter(
                    values,
                    y,
                    s=2.2,
                    c=color,
                    alpha=alpha,
                    edgecolors="none",
                    rasterized=True,
                    zorder=2,
                )

        ax.axvline(0.0, color="0.25", linestyle="--", linewidth=1.0, zorder=1)
        ax.set_xlim(*limits)
        ax.set_ylim(-0.48, 4.48)
        ax.set_yticks(
            [center for _category, _display, center in CATEGORIES],
            [display for _category, display, _center in CATEGORIES],
        )
        panel_label = (
            f"({chr(97 + panel_index)}) {label}\n"
            f"Correct n = {counts.get('Correct', 0):,}; "
            f"Incorrect n = {counts.get('Incorrect', 0):,}"
        )
        if panel_index == len(PANELS) - 1:
            panel_label += "\nSigned IG attribution"
        ax.set_xlabel(panel_label, fontsize=9.5, labelpad=4)
        ax.tick_params(axis="x", labelsize=9)
        ax.tick_params(axis="y", labelsize=9)
        ax.grid(axis="x", linestyle="--", alpha=0.25)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        print(
            f"{dataset}: Correct={counts.get('Correct', 0):,}, "
            f"Incorrect={counts.get('Incorrect', 0):,}"
        )

    handles = [
        Line2D([], [], marker="o", linestyle="none", color=color, markersize=5, label=group)
        for group, color, _offset, _alpha in GROUPS
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=2,
        frameon=True,
        framealpha=0.9,
        fontsize=9,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.975), h_pad=0.45)
    fig.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_STEM.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Shared raw-attribution x limits: [{limits[0]:.6g}, {limits[1]:.6g}]")
    print("No attribution normalization, transformation, clipping, or subsampling applied.")
    print(f"Saved {OUTPUT_STEM.with_suffix('.png')}")
    print(f"Saved {OUTPUT_STEM.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
