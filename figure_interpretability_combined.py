#!/usr/bin/env python3
"""Combine modality, category, and Top-5 feature IG beeswarms into one 3x3 figure.

Reuses existing attribution CSVs and the verified plotting logic from
interpretability_beeswarm.py, interpretability_categories.py, and
interpretability_features.py. No new IG computation; originals untouched.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

import interpretability_beeswarm as modality
import interpretability_categories as categories
import interpretability_features as features

ROOT = Path(__file__).resolve().parent
OUTPUT_STEM = ROOT / "figure_interpretability_combined"

CORRECT = "#4C72B0"
INCORRECT = "#C44E52"
POINT_SIZE = 2.2
JITTER_WIDTH = 0.12

ROW_DATASETS = (
    ("dibhran_malware_binary", "Malware Detection"),
    ("dibhran_phishing_binary", "Phishing Detection"),
    ("dibhran_malware_family", "Malware Family Classification"),
)


def jitter(values, limits, seed):
    edges = np.linspace(limits[0], limits[1], 121)
    bins = np.clip(np.searchsorted(edges, values, side="right") - 1, 0, 119)
    occ = np.bincount(bins, minlength=120)
    width = JITTER_WIDTH * np.sqrt(occ[bins] / max(occ.max(), 1))
    return np.random.default_rng(seed).uniform(-1.0, 1.0, len(values)) * width


def scatter(ax, values, center, offset, color, alpha, limits, seed):
    y = center + offset + jitter(np.asarray(values), limits, seed)
    ax.scatter(values, y, s=POINT_SIZE, c=color, alpha=alpha,
               edgecolors="none", rasterized=True, zorder=2)


def style_ax(ax, limits, ylim, yticks, ylabels, ticksize=7.5, yside="left"):
    ax.axvline(0.0, color="0.25", linestyle="--", linewidth=0.8, zorder=1)
    ax.set_xlim(*limits)
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks, ylabels)
    if yside == "right":
        ax.yaxis.tick_right()
        ax.yaxis.set_tick_params(labelleft=False, labelright=True)
    ax.tick_params(axis="x", labelsize=7.5)
    ax.tick_params(axis="y", labelsize=7.5)
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    ax.set_axisbelow(True)
    for spine in ("top", "right" if yside == "left" else "left"):
        ax.spines[spine].set_visible(False)


def main() -> None:
    modality.validate_label_mappings()
    # Load each column once to fix per-column shared x limits (no clipping).
    mod_loaded = []
    for dataset, _label in ROW_DATASETS:
        correct, n_correct = modality.load_results("IG", dataset)
        incorrect = modality.load_incorrect_results(dataset)
        mod_loaded.append((correct, n_correct, incorrect))
    cat_loaded = [categories.load_results(ds) for ds, _ in ROW_DATASETS]
    feat_loaded = [features.load_results(ds) for ds, _ in ROW_DATASETS]

    mod_scores = np.concatenate(
        [c["contribution_score"].to_numpy() for c, _n, _i in mod_loaded]
        + [i[["string_contribution_score", "metadata_contribution_score"]].to_numpy().ravel()
           for _c, _n, i in mod_loaded]
    )
    cat_scores = np.concatenate(
        [f[[f"{c}_contribution_score" for c, _l, _c in categories.CATEGORIES]].to_numpy().ravel()
         for f in cat_loaded]
    )
    feat_scores = np.concatenate(
        [f[top].to_numpy().ravel() for f, top in feat_loaded]
    )

    def lim(s):
        lo, hi = float(s.min()), float(s.max())
        pad = 0.035 * (hi - lo)
        return (lo - pad, hi + pad)

    mod_lim, cat_lim, feat_lim = lim(mod_scores), lim(cat_scores), lim(feat_scores)
    print(f"modality limits: [{mod_lim[0]:.6g}, {mod_lim[1]:.6g}]")
    print(f"category limits: [{cat_lim[0]:.6g}, {cat_lim[1]:.6g}]")
    print(f"feature limits: [{feat_lim[0]:.6g}, {feat_lim[1]:.6g}]")

    fig = plt.figure(figsize=(9.2, 4.6))
    gs = fig.add_gridspec(3, 5, width_ratios=[1.10, 0.55, 1.05, 0.08, 1.45],
                          hspace=0.55, wspace=0.1,
                          left=0.12, right=0.86, top=0.87, bottom=0.12)
    axes = np.empty((3, 3), dtype=object)
    for _r in range(3):
        axes[_r, 0] = fig.add_subplot(gs[_r, 0])
        axes[_r, 1] = fig.add_subplot(gs[_r, 2])
        axes[_r, 2] = fig.add_subplot(gs[_r, 4])
    short_rows = ("Malware", "Phishing", "Family")
    for row, ((dataset, row_label), (m_correct, m_n, m_incorrect),
              c_frame, (f_frame, f_top)) in enumerate(
            zip(ROW_DATASETS, mod_loaded, cat_loaded, feat_loaded)):
        # --- column (a): modality ---
        ax = axes[row, 0]
        for mi, (key, _display, center) in enumerate(modality.MODALITIES):
            scol = f"{key}_contribution_score"
            for gi, (group, color, off, alpha) in enumerate(
                    (("Correct", CORRECT, 0.17, 0.22), ("Incorrect", INCORRECT, -0.17, 0.38))):
                vals = (m_correct.loc[m_correct["value"] == key, "contribution_score"].to_numpy()
                        if group == "Correct" else m_incorrect[scol].to_numpy())
                scatter(ax, vals, center, off, color, alpha, mod_lim,
                        1000 + 100 * row + 10 * mi + gi)
        style_ax(ax, mod_lim, (-0.48, 1.48), [1.0, 0.0], ["String", "Metadata"])
        # --- column (b): categories ---
        ax = axes[row, 1]
        for ci, (cat, _display, center) in enumerate(categories.CATEGORIES):
            col = f"{cat}_contribution_score"
            for gi, (group, color, off, alpha) in enumerate(
                    (("Correct", CORRECT, 0.17, 0.22), ("Incorrect", INCORRECT, -0.17, 0.38))):
                vals = c_frame.loc[c_frame["classification"] == group, col].to_numpy()
                scatter(ax, vals, center, off, color, alpha, cat_lim,
                        2000 + 100 * row + 10 * ci + gi)
        style_ax(ax, cat_lim, (-0.48, 4.48),
                 [c for _c, _l, c in categories.CATEGORIES],
                 [l for _c, l, _c2 in categories.CATEGORIES])
        # --- column (c): top-5 features ---
        ax = axes[row, 2]
        centers = {f: float(4 - r) for r, f in enumerate(f_top)}
        for fi, feat in enumerate(f_top):
            for gi, (group, color, off, alpha) in enumerate(
                    (("Correct", CORRECT, 0.17, 0.22), ("Incorrect", INCORRECT, -0.17, 0.38))):
                vals = f_frame.loc[f_frame["classification"] == group, feat].to_numpy()
                scatter(ax, vals, centers[feat], off, color, alpha, feat_lim,
                        3000 + 100 * row + 10 * fi + gi)
        feat_labels = ["DNS-hosting IP gap" if "hosting" in lab else lab
                       for lab in [features.display_name(f) for f in f_top]]
        style_ax(ax, feat_lim, (-0.48, 4.48),
                 [centers[f] for f in f_top],
                 feat_labels, ticksize=7, yside="right")

        n_corr = m_n
        n_inc = len(m_incorrect)
        print(f"{dataset}: Correct={n_corr:,}, Incorrect={n_inc:,}")
        axes[row, 0].set_ylabel(short_rows[row], fontsize=10, fontweight="normal",
                                rotation="vertical", ha="center", va="center", labelpad=10)
        for col in range(3):
            if row == 2:
                axes[row, col].set_xlabel("Signed IG attribution", fontsize=8)
            else:
                axes[row, col].set_xlabel("")

    for ax, title in zip(axes[0], ("(a) Modality", "(b) Category", "(c) Features")):
        ax.set_title(title, fontsize=9, pad=5)

    handles = [Line2D([], [], marker="o", linestyle="none", color=c, markersize=5, label=g)
               for g, c in (("Correct", CORRECT), ("Incorrect", INCORRECT))]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0),
               ncol=2, frameon=True, framealpha=0.9, fontsize=8)
    fig.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=300)
    fig.savefig(OUTPUT_STEM.with_suffix(".pdf"), dpi=600)
    plt.close(fig)


if __name__ == "__main__":
    main()
