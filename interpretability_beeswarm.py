#!/usr/bin/env python3
"""Plot raw signed IG and SHAP modality attributions for malicious DomainRadar samples."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "09_output/06_Interpreting/Interpretability/signed-all"
FN_DATA_DIR = ROOT / "09_output/06_Interpreting/Interpretability/false-negatives"
INCORRECT_DATA_DIR = ROOT / "09_output/06_Interpreting/Interpretability/incorrect"
CORRECT_INCORRECT_OUTPUT_STEM = ROOT / "interpretability_beeswarm_correct_incorrect"
OUTPUT_STEMS = {
    "IG": ROOT / "interpretability_beeswarm",
    "SHAP": ROOT / "interpretability_beeswarm_shap",
}

PANELS = (
    ("dibhran_malware_binary", "Malware Detection", "#4C72B0"),
    ("dibhran_phishing_binary", "Phishing Detection", "#DD8452"),
    ("dibhran_malware_family", "Malware Family Classification", "#55A467"),
)
MODALITIES = (("string", "Domain String", 1.0), ("metadata", "Infrastructure Metadata", 0.0))
REQUIRED_COLUMNS = {
    "example",
    "level",
    "value",
    "contribution_score",
    "contribution_rank",
    "modality_value",
}


def validate_label_mappings() -> None:
    config = yaml.safe_load((ROOT / "config.yml").read_text())
    for scenario in ("malware", "phishing"):
        labels = config["scenarios"][scenario]["labels"]
        if labels != {"negative": "legit", "positive": "malicious"}:
            raise ValueError(f"Unexpected {scenario} label mapping: {labels}")
    family = config["scenarios"]["malware_family"]
    if family["task"] != "multi_class_classification" or family["columns"]["target"] != "class":
        raise ValueError("Unexpected malware-family task or target-column mapping")


def load_results(method: str, dataset: str) -> tuple[pd.DataFrame, int]:
    path = DATA_DIR / f"{method}_{dataset}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing required {method} results: {path}")

    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns {sorted(missing)}")
    if set(frame["level"]) != {"modality"}:
        raise ValueError(f"{path.name}: expected only level='modality'")
    if set(frame["value"]) != {"string", "metadata"}:
        raise ValueError(f"{path.name}: unexpected modality values {sorted(frame['value'].unique())}")
    if not np.isfinite(frame["contribution_score"]).all():
        raise ValueError(f"{path.name}: contribution_score contains non-finite values")

    if dataset == "dibhran_malware_family":
        if "class" not in frame.columns:
            raise ValueError(f"{path.name}: class column is required to exclude legit samples")
        frame = frame.loc[frame["class"] != "legit"].copy()
    # Binary signed-all files omit labels because IG.py already selects label index 1;
    # config.yml maps index 1 to malicious for both binary tasks.

    counts = frame["value"].value_counts()
    if counts.get("string", 0) != counts.get("metadata", 0):
        raise ValueError(f"{path.name}: unpaired modality row counts {counts.to_dict()}")
    if not ((frame["contribution_score"] < 0).any() and (frame["contribution_score"] > 0).any()):
        raise ValueError(f"{path.name}: signed attributions do not contain both signs")

    # Keep every malicious record; repeated domains with different metadata remain distinct samples.
    return frame, int(counts["string"])


def density_jitter(
    values: np.ndarray,
    x_min: float,
    x_max: float,
    seed: int,
    max_width: float = 0.31,
) -> np.ndarray:
    """Deterministic vertical jitter scaled by local x-bin density; x is unchanged."""
    n_bins = 120
    edges = np.linspace(x_min, x_max, n_bins + 1)
    bins = np.clip(np.searchsorted(edges, values, side="right") - 1, 0, n_bins - 1)
    occupancy = np.bincount(bins, minlength=n_bins)
    local_width = max_width * np.sqrt(occupancy[bins] / max(occupancy.max(), 1))
    return np.random.default_rng(seed).uniform(-1.0, 1.0, len(values)) * local_width


def load_fn_results(dataset: str) -> pd.DataFrame:
    path = FN_DATA_DIR / f"IG_FN_{dataset}.csv"
    required = {
        "sample_id",
        "example",
        "true_label",
        "predicted_label",
        "string_contribution_score",
        "metadata_contribution_score",
    }
    if not path.is_file():
        raise FileNotFoundError(f"Missing required FN IG results: {path}")
    frame = pd.read_csv(path)
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns {sorted(missing)}")
    if frame["sample_id"].duplicated().any():
        raise ValueError(f"{path.name}: duplicate sample IDs")
    if not (frame["predicted_label"] == "legit").all():
        raise ValueError(f"{path.name}: contains predictions other than legit")
    if (frame["true_label"] == "legit").any():
        raise ValueError(f"{path.name}: contains legit ground-truth samples")
    score_columns = ["string_contribution_score", "metadata_contribution_score"]
    if not np.isfinite(frame[score_columns]).all().all():
        raise ValueError(f"{path.name}: attribution scores contain non-finite values")
    return frame


def load_incorrect_results(dataset: str) -> pd.DataFrame:
    incorrect = load_fn_results(dataset)
    if dataset != "dibhran_malware_family":
        return incorrect

    path = INCORRECT_DATA_DIR / f"IG_INCORRECT_{dataset}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing family-to-family IG results: {path}")
    other = pd.read_csv(path)
    required = set(incorrect.columns)
    missing = required.difference(other.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns {sorted(missing)}")
    if other["sample_id"].duplicated().any():
        raise ValueError(f"{path.name}: duplicate sample IDs")
    if (other["true_label"] == "legit").any() or (other["predicted_label"] == "legit").any():
        raise ValueError(f"{path.name}: expected only malware-family-to-family errors")
    if (other["true_label"] == other["predicted_label"]).any():
        raise ValueError(f"{path.name}: contains correctly classified samples")
    score_columns = ["string_contribution_score", "metadata_contribution_score"]
    if not np.isfinite(other[score_columns]).all().all():
        raise ValueError(f"{path.name}: attribution scores contain non-finite values")
    if set(incorrect["sample_id"]) & set(other["sample_id"]):
        raise ValueError("FN and family-to-family IG sample IDs overlap")
    return pd.concat([incorrect, other], ignore_index=True)


def plot_correct_incorrect() -> None:
    loaded = []
    all_scores = []
    for dataset, label, _ in PANELS:
        correct, correct_count = load_results("IG", dataset)
        incorrect = load_incorrect_results(dataset)
        loaded.append((dataset, label, correct, correct_count, incorrect))
        all_scores.append(correct["contribution_score"].to_numpy())
        all_scores.extend(
            [
                incorrect["string_contribution_score"].to_numpy(),
                incorrect["metadata_contribution_score"].to_numpy(),
            ]
        )

    scores = np.concatenate(all_scores)
    x_min, x_max = float(scores.min()), float(scores.max())
    padding = 0.035 * (x_max - x_min)
    limits = (x_min - padding, x_max + padding)
    groups = (
        ("Correct", "#4C72B0", 0.17, 0.22),
        ("Incorrect", "#C44E52", -0.17, 0.38),
    )

    fig, axes = plt.subplots(3, 1, figsize=(3.5, 6.0), sharex=True, sharey=True)
    for panel_index, (dataset, label, correct, correct_count, incorrect) in enumerate(loaded):
        ax = axes[panel_index]
        for modality_index, (key, _display, center) in enumerate(MODALITIES):
            incorrect_column = f"{key}_contribution_score"
            for group_index, (group, color, offset, alpha) in enumerate(groups):
                values = (
                    correct.loc[correct["value"] == key, "contribution_score"].to_numpy()
                    if group == "Correct"
                    else incorrect[incorrect_column].to_numpy()
                )
                y = center + offset + density_jitter(
                    values,
                    *limits,
                    seed=42 + 100 * panel_index + 10 * modality_index + group_index,
                    max_width=0.12,
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
        ax.set_ylim(-0.48, 1.48)
        ax.set_yticks([1.0, 0.0], ["String", "Metadata"])
        panel_label = f"({chr(97 + panel_index)}) {label}"
        if panel_index == len(PANELS) - 1:
            panel_label += "\nSigned IG attribution"
        ax.set_xlabel(panel_label, fontsize=10, labelpad=4)
        ax.tick_params(axis="x", labelsize=9)
        ax.tick_params(axis="y", labelsize=9)
        ax.grid(axis="x", linestyle="--", alpha=0.25)
        ax.set_axisbelow(True)
        ax.text(
            0.97,
            0.52,
            f"Correct n = {correct_count:,}\nIncorrect n = {len(incorrect):,}",
            transform=ax.transAxes,
            ha="right",
            va="center",
            fontsize=8.5,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.0},
            zorder=3,
        )
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        print(f"IG {dataset}: Correct={correct_count:,}, Incorrect={len(incorrect):,}")

    handles = [
        Line2D([], [], marker="o", linestyle="none", color=color, markersize=5, label=group)
        for group, color, _offset, _alpha in groups
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
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.965), h_pad=0.35)
    fig.savefig(CORRECT_INCORRECT_OUTPUT_STEM.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(CORRECT_INCORRECT_OUTPUT_STEM.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Shared raw-attribution x limits: [{limits[0]:.6g}, {limits[1]:.6g}]")
    print(f"Saved {CORRECT_INCORRECT_OUTPUT_STEM.with_suffix('.png')}")
    print(f"Saved {CORRECT_INCORRECT_OUTPUT_STEM.with_suffix('.pdf')}")


def plot_method(method: str) -> None:
    loaded = [
        (dataset, label, color, *load_results(method, dataset))
        for dataset, label, color in PANELS
    ]
    all_scores = np.concatenate(
        [frame["contribution_score"].to_numpy() for _, _, _, frame, _ in loaded]
    )
    x_min, x_max = float(all_scores.min()), float(all_scores.max())
    padding = 0.035 * (x_max - x_min)
    limits = (x_min - padding, x_max + padding)

    fig, axes = plt.subplots(3, 1, figsize=(3.5, 5.8), sharex=True, sharey=True)
    for panel_index, (dataset, label, color, frame, sample_count) in enumerate(loaded):
        ax = axes[panel_index]
        for modality_index, (key, display, center) in enumerate(MODALITIES):
            values = frame.loc[frame["value"] == key, "contribution_score"].to_numpy()
            y = center + density_jitter(
                values, *limits, seed=42 + 10 * panel_index + modality_index
            )
            ax.scatter(
                values,
                y,
                s=2.2,
                c=color,
                alpha=0.24,
                edgecolors="none",
                rasterized=True,
                zorder=2,
            )

        ax.axvline(0.0, color="0.25", linestyle="--", linewidth=1.0, zorder=1)
        ax.set_xlim(*limits)
        ax.set_ylim(-0.43, 1.43)
        ax.set_yticks([1.0, 0.0], ["String", "Metadata"])
        panel_label = f"({chr(97 + panel_index)}) {label}"
        if panel_index == len(PANELS) - 1:
            panel_label += f"\nSigned {method} attribution"
        ax.set_xlabel(panel_label, fontsize=10, labelpad=4)
        ax.tick_params(axis="x", labelsize=9)
        ax.tick_params(axis="y", labelsize=9)
        ax.grid(axis="x", linestyle="--", alpha=0.25)
        ax.set_axisbelow(True)
        ax.text(
            0.97,
            0.95,
            f"n = {sample_count:,}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
        )
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

        print(
            f"{method} {dataset}: n={sample_count:,}, rows={len(frame):,}, "
            f"score_range=[{frame['contribution_score'].min():.6g}, "
            f"{frame['contribution_score'].max():.6g}]"
        )

    output_stem = OUTPUT_STEMS[method]
    fig.tight_layout(h_pad=0.35)
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"{method} shared raw-attribution x limits: [{limits[0]:.6g}, {limits[1]:.6g}]")
    print(f"Saved {output_stem.with_suffix('.png')}")
    print(f"Saved {output_stem.with_suffix('.pdf')}")


def main() -> None:
    validate_label_mappings()
    plot_correct_incorrect()
    print("No attribution normalization, transformation, clipping, or subsampling applied.")


if __name__ == "__main__":
    main()
