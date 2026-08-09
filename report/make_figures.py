#!/usr/bin/env python3
"""Regenerate all figures used by the TUHH project report from aggregate outputs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPORT_DIR = Path(__file__).resolve().parent
ROOT = REPORT_DIR.parent
FIGURE_DIR = REPORT_DIR / "figures"

BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
MAGENTA = "#CC79A7"
GRAY = "#6B7280"
LIGHT_GRAY = "#D1D5DB"


def load_json(relative_path: str) -> dict:
    with (ROOT / relative_path).open(encoding="utf-8") as handle:
        return json.load(handle)


def save_figure(fig: plt.Figure, filename: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / filename, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.labelsize": 9.5,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#4B5563",
            "axes.linewidth": 0.8,
            "xtick.color": "#374151",
            "ytick.color": "#374151",
            "text.color": "#111827",
            "figure.dpi": 150,
            "savefig.dpi": 300,
        }
    )


def plot_validity_boundary() -> None:
    summary = load_json("results/label_baseline_control/summary.json")
    values = {row["feature_set"]: row["auroc"] for row in summary["metrics"]}
    labels = [
        "Current-day\npassive",
        "Passive\nhistory",
        "Prior label\nrate",
        "Current-day +\nprior label",
        "History +\nprior label",
    ]
    keys = [
        "allday_raw",
        "history_raw",
        "prior_label_rate_only",
        "allday_raw_plus_prior_label_rate",
        "history_raw_plus_prior_label_rate",
    ]
    colors = [BLUE, BLUE, ORANGE, GREEN, GREEN]
    y = [values[key] for key in keys]

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    bars = ax.bar(np.arange(len(y)), y, color=colors, width=0.68)
    ax.axhline(0.5, color=GRAY, linestyle="--", linewidth=1, label="Chance")
    ax.set_ylim(0.50, 0.89)
    ax.set_ylabel("Held-out AUROC")
    ax.set_xticks(np.arange(len(labels)), labels)
    ax.set_title("Label history explains the apparent depression-classification advantage")
    ax.grid(axis="y", color=LIGHT_GRAY, linewidth=0.6, alpha=0.7)
    for bar, value in zip(bars, y):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.008,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    fig.tight_layout()
    save_figure(fig, "validity_boundary_auroc.png")


def plot_construct_overlap() -> None:
    correlations = pd.read_csv(
        ROOT / "results/construct_fragility_audit/same_day_construct_correlations.csv"
    )
    targets = [
        "target_dep_weekly",
        "feel_depressed",
        "feel_anxious",
        "phq4",
        "phq4_EMA",
        "phq4_anxiety_EMA",
        "phq4_depression_EMA",
        "pss4_EMA",
        "positive_affect_EMA",
        "negative_affect_EMA",
    ]
    display = [
        "Weekly depression",
        "Depressed",
        "Anxious",
        "PHQ-4",
        "PHQ-4 EMA",
        "PHQ-4 anxiety",
        "PHQ-4 depression",
        "PSS-4",
        "Positive affect",
        "Negative affect",
    ]
    matrix = pd.DataFrame(np.eye(len(targets)), index=targets, columns=targets)
    for row in correlations.itertuples(index=False):
        matrix.loc[row.target_a, row.target_b] = row.pearson_r
        matrix.loc[row.target_b, row.target_a] = row.pearson_r

    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    cmap = mpl.colormaps["RdBu_r"].copy()
    cmap.set_bad("#E5E7EB")
    image = ax.imshow(
        np.ma.masked_invalid(matrix.loc[targets, targets].to_numpy()),
        cmap=cmap,
        vmin=-1,
        vmax=1,
    )
    ax.set_xticks(np.arange(len(display)), display, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(display)), display)
    ax.set_title("Same-day correlation among repeated psychological measures")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Pearson correlation")
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    save_figure(fig, "construct_overlap_heatmap.png")


def plot_behavior_forecasting() -> None:
    summary = load_json("results/behavioural_weather_forecast/summary.json")["aggregate"]
    horizons = ["Tomorrow", "Next 7 days"]
    model_names = ["History + fingerprint", "+ current partial day"]
    regression = np.array(
        [
            [
                summary["regression"]["tomorrow"]["evening"]["mean_history_plus_fingerprint"],
                summary["regression"]["tomorrow"]["evening"]["mean_history_fingerprint_plus_today"],
            ],
            [
                summary["regression"]["next_week"]["evening"]["mean_history_plus_fingerprint"],
                summary["regression"]["next_week"]["evening"]["mean_history_fingerprint_plus_today"],
            ],
        ]
    )
    classification = np.array(
        [
            [
                summary["classification"]["tomorrow"]["evening"]["mean_history_plus_fingerprint"],
                summary["classification"]["tomorrow"]["evening"]["mean_history_fingerprint_plus_today"],
            ],
            [
                summary["classification"]["next_week"]["evening"]["mean_history_plus_fingerprint"],
                summary["classification"]["next_week"]["evening"]["mean_history_fingerprint_plus_today"],
            ],
        ]
    )

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.9))
    x = np.arange(len(horizons))
    width = 0.34
    for ax, values, title, ylabel, ylim in [
        (axes[0], regression, "Continuous targets", "Mean held-out $R^2$", (0, 0.60)),
        (axes[1], classification, "High/low targets", "Mean held-out AUROC", (0.50, 0.92)),
    ]:
        for index, (model, color) in enumerate(zip(model_names, [BLUE, ORANGE])):
            bars = ax.bar(x + (index - 0.5) * width, values[:, index], width, label=model, color=color)
            for bar, value in zip(bars, values[:, index]):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + (0.012 if ylabel.endswith("$R^2$") else 0.008),
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
        ax.set_xticks(x, horizons)
        ax.set_ylim(*ylim)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", color=LIGHT_GRAY, linewidth=0.6, alpha=0.7)
    handles, legend_labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        frameon=False,
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        fontsize=8,
    )
    fig.suptitle("Prospective behavior forecasting across 16 targets", fontweight="bold", y=1.02)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save_figure(fig, "behavior_forecasting_performance.png")


def plot_construct_audit() -> None:
    stability = pd.read_csv(
        ROOT / "results/referee_robustness/stability_robustness.csv"
    )
    profiles = pd.read_csv(
        ROOT / "results/referee_robustness/profile_robustness.csv"
    )
    regime_order = ["fixed_original", "expanding_refit", "disjoint_refit"]
    regime_labels = ["Fixed", "Expanding", "Disjoint"]
    series = [
        ("forecastability", "pooled", "Absolute error, raw", GRAY),
        (
            "forecastability",
            "residual_variability_entropy",
            "Absolute error, adjusted",
            ORANGE,
        ),
        (
            "history_skill",
            "residual_variability_entropy",
            "History benefit, adjusted",
            BLUE,
        ),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.2))
    x = np.arange(len(regime_order))
    offsets = [-0.22, 0, 0.22]
    for offset, (measure, analysis, label, color) in zip(offsets, series):
        subset = stability[
            (stability["scope"] == "core")
            & (stability["measure"] == measure)
            & (stability["analysis"] == analysis)
        ].set_index("regime")
        values = np.array([subset.loc[regime, "pearson_r"] for regime in regime_order])
        low = np.array([subset.loc[regime, "ci_low"] for regime in regime_order])
        high = np.array([subset.loc[regime, "ci_high"] for regime in regime_order])
        axes[0].errorbar(
            x + offset,
            values,
            yerr=np.vstack([values - low, high - values]),
            fmt="o",
            markersize=5,
            capsize=3,
            color=color,
            label=label,
        )

    profile_series = [
        ("forecastability", "Absolute error, raw", GRAY),
        ("forecastability_residual_sd_entropy", "Absolute error, adjusted", ORANGE),
        ("history_skill_residual_sd_entropy", "History benefit, adjusted", BLUE),
    ]
    for offset, (measure, label, color) in zip(offsets, profile_series):
        subset = profiles[
            (profiles["scope"] == "core")
            & (profiles["measure"] == measure)
            & (profiles["sample"] == "all_trajectories")
        ].set_index("regime")
        values = [subset.loc[regime, "delta"] for regime in regime_order]
        axes[1].plot(x + offset, values, "o", markersize=5, color=color, label=label)

    axes[0].axhline(0.30, color=MAGENTA, linestyle="--", linewidth=1, label="0.30 gate")
    axes[0].set_ylim(0, 0.58)
    axes[0].set_ylabel("Early-late Pearson $r$")
    axes[0].set_title("Scalar stability")
    axes[1].axhline(0.10, color=MAGENTA, linestyle="--", linewidth=1, label="0.10 gate")
    axes[1].set_ylim(0, 0.49)
    axes[1].set_ylabel(r"Same-minus-cross profile cosine ($\Delta$)")
    axes[1].set_title("Continuous profile advantage")
    for ax in axes:
        ax.set_xticks(x, regime_labels)
        ax.grid(axis="y", color=LIGHT_GRAY, linewidth=0.6, alpha=0.7)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("Construct audit across temporal refitting regimes", fontweight="bold", y=1.02)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    save_figure(fig, "forecastability_construct_audit.png")


def plot_feature_families() -> None:
    stability = pd.read_csv(
        ROOT / "results/structure_followup/feature_family_stability.csv"
    )
    profiles = pd.read_csv(
        ROOT / "results/structure_followup/feature_family_profiles.csv"
    )
    measures = [
        "lag1_history_residual",
        "short_window_history_residual",
        "recent_history_residual",
        "long_horizon_fingerprint_residual",
        "combined_own_history_residual",
        "cross_domain_increment_residual",
    ]
    labels = [
        "Yesterday",
        "7/14-day means",
        "Recent combined",
        "42-day fingerprint",
        "All own history",
        "Other-domain increment",
    ]
    regimes = ["fixed_refit", "expanding_refit", "disjoint_refit"]
    regime_labels = ["Fixed", "Expanding", "Disjoint"]
    colors = [BLUE, ORANGE, GREEN]
    y = np.arange(len(measures))

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 5.0), sharey=True)
    for index, (regime, regime_label, color) in enumerate(zip(regimes, regime_labels, colors)):
        s = stability[
            (stability["regime"] == regime) & stability["measure"].isin(measures)
        ].set_index("measure")
        p = profiles[
            (profiles["regime"] == regime) & profiles["measure"].isin(measures)
        ].set_index("measure")
        offset = (index - 1) * 0.17
        axes[0].plot([s.loc[m, "pearson_r"] for m in measures], y + offset, "o", color=color, label=regime_label)
        axes[1].plot([p.loc[m, "delta"] for m in measures], y + offset, "o", color=color, label=regime_label)

    axes[0].axvline(0.30, color=MAGENTA, linestyle="--", linewidth=1)
    axes[0].set_xlabel("Adjusted degree stability $r$")
    axes[0].set_title("Scalar gate")
    axes[1].axvline(0.10, color=MAGENTA, linestyle="--", linewidth=1)
    axes[1].set_xlabel(r"Adjusted profile advantage $\Delta$")
    axes[1].set_title("Profile gate")
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    for ax in axes:
        ax.set_xlim(0, 0.45)
        ax.grid(axis="x", color=LIGHT_GRAY, linewidth=0.6, alpha=0.7)
    axes[1].legend(frameon=False, loc="lower right")
    fig.suptitle("Post-hoc feature-family decomposition after SD and entropy adjustment", fontweight="bold", y=1.01)
    fig.tight_layout()
    save_figure(fig, "feature_family_decomposition.png")


def plot_typology() -> None:
    summary = load_json("results/structure_followup/typology_summary.json")
    validation = pd.DataFrame(summary["validation"])
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.8))
    for regime, label, color, zero_offset, marker in [
        ("fixed_original", "Fixed", BLUE, 0.07, "o"),
        ("disjoint_refit", "Disjoint", ORANGE, -0.07, "s"),
    ]:
        subset = validation[validation["regime"] == regime].sort_values("k")
        axes[0].plot(
            subset["k"],
            subset["median_adjusted_rand_index"],
            marker=marker,
            color=color,
            label=label,
        )
        axes[1].plot(
            subset["k"],
            subset["successful_cohorts"] + zero_offset,
            marker=marker,
            color=color,
            label=f"{label} (actual 0)",
        )
    axes[0].axhline(0.20, color=MAGENTA, linestyle="--", linewidth=1, label="Required")
    axes[0].set_ylabel("Median held-out ARI")
    axes[0].set_ylim(0, 0.24)
    axes[0].set_title("Assignment stability")
    axes[1].axhline(3, color=MAGENTA, linestyle="--", linewidth=1, label="Required")
    axes[1].set_ylabel("Held-out cohorts passing joint rule")
    axes[1].set_ylim(-0.2, 3.4)
    axes[1].set_title("Cross-cohort replication")
    axes[1].axhline(0, color="#6B7280", linewidth=0.8)
    axes[1].text(
        3.5,
        0.35,
        "0 successful cohorts\nfor every $k$ in both regimes",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#374151",
    )
    for ax in axes:
        ax.set_xticks([2, 3, 4, 5])
        ax.set_xlabel("Number of clusters ($k$)")
        ax.grid(axis="y", color=LIGHT_GRAY, linewidth=0.6, alpha=0.7)
    axes[1].legend(frameon=False, loc="upper right")
    fig.suptitle("No discrete profile typology passed the validation gate", fontweight="bold", y=1.02)
    fig.tight_layout()
    save_figure(fig, "typology_validation.png")


def main() -> None:
    set_style()
    plot_validity_boundary()
    plot_construct_overlap()
    plot_behavior_forecasting()
    plot_construct_audit()
    plot_feature_families()
    plot_typology()


if __name__ == "__main__":
    main()
