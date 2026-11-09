#!/usr/bin/env python3
"""Plot correct-vs-incorrect signed fusion (CharCNN + MIAFF N_cat_N_ce) IG for Top-5 metadata features."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "09_output/06_Interpreting/Interpretability/features_fusion"
OUTPUT_STEM = ROOT / "interpretability_features_fusion"
PANELS = (
    ("dibhran_malware_binary", "Malware Detection"),
    ("dibhran_phishing_binary", "Phishing Detection"),
    ("dibhran_malware_family", "Malware Family Classification"),
)
GROUPS = (
    ("Correct", "#4C72B0", 0.17, 0.22),
    ("Incorrect", "#C44E52", -0.17, 0.38),
)
TOP_K = 5
BASE_COLUMNS = {
    "sample_id",
    "example",
    "true_label",
    "predicted_label",
    "classification",
}


def load_results(dataset: str) -> tuple[pd.DataFrame, list[str]]:
    path = DATA_DIR / f"IG_FEATURES_FUSION_{dataset}.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    feature_columns = [
        column
        for column in frame.columns
        if column
        not in BASE_COLUMNS | {"string_contribution_score", "metadata_contribution_score"}
    ]
    if len(feature_columns) < TOP_K:
        raise ValueError(f"{path.name}: only {len(feature_columns)} feature columns")
    if frame["sample_id"].duplicated().any():
        raise ValueError(f"{path.name}: duplicate sample IDs")
    if (frame["true_label"] == "legit").any():
        raise ValueError(f"{path.name}: contains legit ground-truth samples")
    expected = np.where(
        frame["true_label"] == frame["predicted_label"], "Correct", "Incorrect"
    )
    if not np.array_equal(frame["classification"].to_numpy(), expected):
        raise ValueError(f"{path.name}: inconsistent classification labels")
    if not np.isfinite(frame[feature_columns]).all().all():
        raise ValueError(f"{path.name}: non-finite attribution values")
    ranking = (
        frame[feature_columns].abs().mean(axis=0).sort_values(ascending=False)
    )
    top_features = list(ranking.index[:TOP_K])
    print(f"{dataset}: Top-{TOP_K} = {', '.join(top_features)}")
    return frame, top_features


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


SHORT_NAMES = {
    "x_dns_ip_count_vs_hosting_ip_count_diff": "DNS-hosting IP gap",
    "rdap_is_new_domain_365_days": "RDAP new domain",
    "rdap_registered_for_one_year": "RDAP registered 1 yr",
    "rdap_recently_updated_90_days": "RDAP updated (90d)",
    "tls_short_validity_180_days": "TLS short validity",
    "dns_num_txt_records": "DNS TXT count",
    "dns_num_cname_records": "DNS CNAME count",
    "dns_num_a_records": "DNS A count",
    "dns_has_dnssec": "DNSSEC present",
    "rdap_entity_count": "RDAP entity count",
    "ip_num_ipv4": "IPv4 count",
}


def display_name(feature: str) -> str:
    return SHORT_NAMES.get(feature, feature.replace("_", " "))


def main() -> None:
    loaded = [(dataset, label, *load_results(dataset)) for dataset, label in PANELS]
    scores = np.concatenate(
        [frame[top].to_numpy().ravel() for _dataset, _label, frame, top in loaded]
    )
    x_min, x_max = float(scores.min()), float(scores.max())
    padding = 0.035 * (x_max - x_min)
    limits = (x_min - padding, x_max + padding)

    fig, axes = plt.subplots(3, 1, figsize=(5.5, 7.5), sharex=True, sharey=False)
    for panel_index, (dataset, label, frame, top_features) in enumerate(loaded):
        ax = axes[panel_index]
        centers = {feature: float(TOP_K - 1 - rank) for rank, feature in enumerate(top_features)}
        counts = frame["classification"].value_counts()
        for feature_index, feature in enumerate(top_features):
            center = centers[feature]
            for group_index, (group, color, offset, alpha) in enumerate(GROUPS):
                values = frame.loc[frame["classification"] == group, feature].to_numpy()
                y = center + offset + density_jitter(
                    values,
                    *limits,
                    seed=42 + 100 * panel_index + 10 * feature_index + group_index,
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
        ax.set_ylim(-0.48, TOP_K - 0.52)
        ax.set_yticks(
            [centers[feature] for feature in top_features],
            [display_name(feature) for feature in top_features],
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
        ax.tick_params(axis="y", labelsize=8)
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
