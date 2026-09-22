#!/usr/bin/env python3
"""Regenerate all figures used by the TUHH project report from aggregate outputs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


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
    regression_metrics = pd.read_csv(
        ROOT / "results/behavioural_weather_forecast/regression_metrics.csv"
    )
    classification_metrics = pd.read_csv(
        ROOT / "results/behavioural_weather_forecast/classification_metrics.csv"
    )
    horizons = ["Tomorrow", "Next 7 days"]
    models = [
        ("calendar_only", "Calendar only", GRAY),
        ("recent_history_only", "Recent history", BLUE),
        ("fingerprint_42d_only", "42-day baseline", GREEN),
        ("history_fingerprint_plus_today", "+ partial day", ORANGE),
    ]

    def mean_matrix(frame: pd.DataFrame, metric: str) -> np.ndarray:
        rows = []
        for horizon in ["tomorrow", "next_week"]:
            rows.append(
                [
                    frame[
                        frame["horizon"].eq(horizon)
                        & frame["issue_time"].eq("evening")
                        & frame["model"].eq(model)
                    ][metric].mean()
                    for model, _, _ in models
                ]
            )
        return np.asarray(rows)

    regression = mean_matrix(regression_metrics, "r2")
    classification = mean_matrix(classification_metrics, "auroc")

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.9))
    x = np.arange(len(horizons))
    width = 0.18
    for ax, values, title, ylabel, ylim in [
        (axes[0], regression, "Continuous targets", "Mean held-out $R^2$", (-0.05, 0.58)),
        (axes[1], classification, "High/low targets", "Mean held-out AUROC", (0.50, 0.92)),
    ]:
        ax.axhline(0 if ylabel.endswith("$R^2$") else 0.5, color="#9CA3AF", linewidth=0.8)
        for index, (_, label, color) in enumerate(models):
            bars = ax.bar(
                x + (index - 1.5) * width,
                values[:, index],
                width,
                label=label,
                color=color,
            )
            for bar, value in zip(bars, values[:, index]):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + (0.012 if ylabel.endswith("$R^2$") else 0.007),
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
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
        ncol=4,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        fontsize=7.5,
    )
    fig.suptitle(
        "Simple history drives most prospective behavior performance",
        fontweight="bold",
        y=1.02,
    )
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


def plot_android_pilot_workflow() -> None:
    absolute = provenance = two_stage = load_json("results/android_pilot/summary.json")

    boxes = [
        (
            "1  Self-instrumentation",
            "Author-operated Android logger\n37 calendar days",
            BLUE,
        ),
        (
            "2  Recorded context",
            "App | device | time\nInitial/settled volume | key evidence",
            GRAY,
        ),
        (
            "3  Provenance gate",
            f"{absolute['data']['n_sessions']} recorded sessions\n"
            f"{provenance['n_eligible']} eligible sessions",
            ORANGE,
        ),
        (
            "4  Future evaluation",
            f"{provenance['split']['n_train']} train -> "
            f"{provenance['split']['n_test']} future test\n"
            f"{two_stage['sample']['n_adjusted_test']} adjusted test sessions",
            GREEN,
        ),
    ]

    fig, ax = plt.subplots(figsize=(8.2, 3.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    width = 0.215
    gap = 0.032
    start = 0.02
    y = 0.22
    height = 0.58
    for index, (heading, detail, color) in enumerate(boxes):
        x = start + index * (width + gap)
        box = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            linewidth=1.4,
            edgecolor=color,
            facecolor="#F9FAFB",
        )
        ax.add_patch(box)
        ax.text(
            x + width / 2,
            y + height * 0.68,
            heading,
            ha="center",
            va="center",
            fontsize=8.2,
            fontweight="bold",
            color=color,
        )
        ax.text(
            x + width / 2,
            y + height * 0.35,
            detail,
            ha="center",
            va="center",
            fontsize=7.7,
            linespacing=1.35,
        )
        if index < len(boxes) - 1:
            arrow = FancyArrowPatch(
                (x + width + 0.006, y + height / 2),
                (x + width + gap - 0.006, y + height / 2),
                arrowstyle="-|>",
                mutation_scale=12,
                linewidth=1.1,
                color="#4B5563",
            )
            ax.add_patch(arrow)
    ax.set_title(
        "Independent Android pilot: instrumentation and locked evaluation path",
        pad=10,
    )
    fig.tight_layout()
    save_figure(fig, "android_pilot_workflow.png")


def plot_android_pilot_policy_cost() -> None:
    summary = load_json("results/android_pilot/summary.json")
    confusion = summary["stage_one_confusion_at_0_5"]
    matrix = np.array(
        [
            [confusion["true_negative"], confusion["false_positive"]],
            [confusion["false_negative"], confusion["true_positive"]],
        ]
    )
    stratified = pd.DataFrame(summary["policy_metrics_by_true_adjustment"])
    unchanged = stratified[stratified["adjusted"] == 0].set_index("model")["mae"]
    overall = pd.DataFrame(summary["policy_metrics"]).set_index("model")["mae"]

    fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.7))

    image = axes[0].imshow(matrix, cmap="Blues", vmin=0, vmax=matrix.max())
    for row in range(2):
        for column in range(2):
            axes[0].text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                fontsize=13,
                fontweight="bold",
                color="white" if matrix[row, column] > matrix.max() * 0.55 else "#111827",
            )
    axes[0].set_xticks([0, 1], ["No", "Adjust"])
    axes[0].set_yticks([0, 1], ["No", "Adjust"])
    axes[0].set_xlabel("Gate prediction")
    axes[0].set_ylabel("Observed adjustment")
    axes[0].set_title("A  Gate at 0.50")
    axes[0].text(
        0.5,
        -0.33,
        f"AUROC = {summary['stage_one_app_propensity']['auc']:.3f}",
        transform=axes[0].transAxes,
        ha="center",
        fontsize=8.5,
        color="#374151",
    )
    image.set_clim(0, matrix.max())

    labels = ["No action", "App-aware", "Proposed"]
    colors = [GREEN, BLUE, ORANGE]
    unchanged_values = [
        unchanged["keep_current"],
        unchanged["observed_app_sensitive"],
        unchanged["observed_two_stage"],
    ]
    overall_values = [
        overall["keep_current"],
        overall["observed_app_sensitive"],
        overall["observed_two_stage"],
    ]
    for ax, values, title in [
        (axes[1], unchanged_values, "B  Cost on 45\nunchanged sessions"),
        (axes[2], overall_values, "C  End-to-end error\nall 78 sessions"),
    ]:
        bars = ax.bar(np.arange(3), values, color=colors, width=0.68)
        ax.set_xticks(np.arange(3), labels, rotation=24, ha="right")
        ax.set_ylabel("MAE (volume percentage points)")
        ax.set_title(title, fontsize=9.5)
        ax.grid(axis="y", color=LIGHT_GRAY, linewidth=0.6, alpha=0.7)
        upper = max(values) * 1.22
        ax.set_ylim(0, upper)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + upper * 0.025,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold",
            )

    fig.suptitle(
        "Moderate adjustment ranking did not produce a useful intervention policy",
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    save_figure(fig, "android_pilot_policy_cost.png")


def plot_android_september_followup() -> None:
    summary = load_json("results/android_pilot/september_summary.json")
    selection = summary["adjustment_enrichment"]
    improvement_ci = summary["day_clustered_paired_mae"]["ci_95"]
    improvement = summary["paired_mae_improvement"]

    fig = plt.figure(figsize=(8.4, 3.5))
    grid = fig.add_gridspec(2, 2, height_ratios=[3.2, 1], wspace=0.38, hspace=0.18)
    error_ax = fig.add_subplot(grid[0, 0])
    interval_ax = fig.add_subplot(grid[1, 0])
    selection_ax = fig.add_subplot(grid[:, 1])

    error_values = [summary["keep_mae"], summary["hybrid_mae"]]
    bars = error_ax.bar([0, 1], error_values, color=[GRAY, BLUE], width=0.58)
    error_ax.set_xticks([0, 1], ["Keep current", "Saved hybrid"])
    error_ax.set_ylim(0, 12.2)
    error_ax.set_ylabel("MAE (volume points)")
    error_ax.set_title("A  Error on 110 later sessions")
    error_ax.grid(axis="y", color=LIGHT_GRAY, linewidth=0.6, alpha=0.7)
    for bar, value in zip(bars, error_values):
        error_ax.text(bar.get_x() + bar.get_width() / 2, value + 0.25,
                      f"{value:.2f}", ha="center", fontweight="bold")

    interval_ax.axvline(0, color=GRAY, linestyle="--", linewidth=1.1)
    interval_ax.plot(improvement_ci, [0, 0], color=BLUE, linewidth=3)
    interval_ax.scatter(improvement_ci, [0, 0], color=BLUE, s=20, zorder=3)
    interval_ax.scatter([improvement], [0], color=BLUE, s=60, zorder=4)
    interval_ax.set_xlim(-0.55, 2.2)
    interval_ax.set_ylim(-0.55, 0.55)
    interval_ax.set_yticks([])
    interval_ax.set_xlabel("Paired MAE improvement; 95% day-clustered CI")
    interval_ax.spines["left"].set_visible(False)
    interval_ax.spines["bottom"].set_visible(False)

    rates = [selection["selected_adjustment_rate"],
             selection["kept_adjustment_rate"]]
    counts = [
        (selection["selected_adjusted"], summary["n_interventions"]),
        (selection["kept_adjusted"], summary["n_future"] - summary["n_interventions"]),
    ]
    bars = selection_ax.bar([0, 1], rates, color=[ORANGE, GRAY], width=0.58)
    selection_ax.set_xticks([0, 1], ["Selected", "Not selected"])
    selection_ax.set_ylim(0, 0.9)
    selection_ax.set_ylabel("Observed adjustment rate")
    selection_ax.set_title("B  Adjustment-event selection")
    selection_ax.grid(axis="y", color=LIGHT_GRAY, linewidth=0.6, alpha=0.7)
    for bar, value, (numerator, denominator) in zip(bars, rates, counts):
        selection_ax.text(bar.get_x() + bar.get_width() / 2, value + 0.025,
                          f"{value:.1%}\n({numerator}/{denominator})",
                          ha="center", fontweight="bold")
    fig.suptitle("September saved-model follow-up: a selective but uncertain gain",
                 fontweight="bold", y=1.02)
    save_figure(fig, "android_september_followup.png")


def plot_sample_attrition_flow() -> None:
    """Show the separate RQ1/RQ3 branches and parallel RQ3 sensitivities."""
    fig, ax = plt.subplots(figsize=(10.0, 8.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def box(x, y, width, height, title, subtitle, edge, face):
        patch = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.012,rounding_size=0.012",
            linewidth=1.5,
            edgecolor=edge,
            facecolor=face,
        )
        ax.add_patch(patch)
        ax.text(
            x + width / 2,
            y + height * 0.63,
            title,
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
        )
        ax.text(
            x + width / 2,
            y + height * 0.27,
            subtitle,
            ha="center",
            va="center",
            fontsize=8.2,
            color="#374151",
        )

    def arrow(start, end, label=None, label_xy=None):
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=11,
                linewidth=1.2,
                color=GRAY,
                connectionstyle="arc3,rad=0.0",
            )
        )
        if label and label_xy:
            ax.text(
                *label_xy,
                label,
                ha="center",
                va="center",
                fontsize=7.8,
                color="#374151",
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.2},
            )

    ax.set_title(
        "Analysis cohorts and attrition are branched, not one linear funnel",
        fontsize=14,
        fontweight="bold",
        pad=10,
    )
    ax.text(
        0.5,
        0.965,
        "Counts denote person-year trajectories identified by institute-year and released user ID",
        ha="center",
        va="center",
        fontsize=9,
        color=GRAY,
    )

    box(0.33, 0.84, 0.34, 0.085, "Canonical daily table  n = 705", "Shared source population", "#374151", "#F3F4F6")
    box(0.03, 0.65, 0.35, 0.10, "RQ1 label cohort  n = 697", "At least 4 weekly depression labels", ORANGE, "#FFF7E6")
    box(0.62, 0.65, 0.35, 0.10, "RQ3 forecast cohort  n = 696", "42-day history + observed future behavior", BLUE, "#EAF5FB")
    box(0.62, 0.48, 0.35, 0.10, "Any person-level score  n = 598", "Both halves; core-6 or rich-10 coverage", BLUE, "#EAF5FB")
    box(0.62, 0.31, 0.35, 0.10, "Primary core cohort  n = 521", "Both halves for at least 6 of 8 core targets", GREEN, "#EAF7F3")
    box(0.36, 0.14, 0.27, 0.10, "Disjoint refit  n = 499", "Shorter non-overlapping blocks", MAGENTA, "#FBEFF6")
    box(0.70, 0.14, 0.27, 0.10, "Known single-year  n = 232", "Repeat candidates removed", ORANGE, "#FFF7E6")
    box(0.53, 0.01, 0.30, 0.075, "Both sensitivities  n = 215", "Disjoint and known single-year", "#374151", "#F3F4F6")

    arrow((0.43, 0.84), (0.205, 0.75), "-8: fewer than 4 labels", (0.25, 0.79))
    arrow((0.57, 0.84), (0.795, 0.75), "-9: no eligible forecast", (0.75, 0.79))
    arrow((0.795, 0.65), (0.795, 0.58), "-98: target-half / scope coverage", (0.79, 0.615))
    arrow((0.795, 0.48), (0.795, 0.41), "-77: rich-only scores", (0.79, 0.445))
    arrow((0.72, 0.31), (0.50, 0.24), "-22: disjoint coverage", (0.58, 0.275))
    arrow((0.87, 0.31), (0.84, 0.24), "-289: repeat candidates", (0.86, 0.275))
    arrow((0.50, 0.14), (0.61, 0.085))
    arrow((0.84, 0.14), (0.75, 0.085))

    ax.plot([0.38, 0.62], [0.70, 0.70], linestyle=(0, (3, 3)), color="#9CA3AF", linewidth=1)
    ax.text(
        0.50,
        0.70,
        "Separate branches\nOverlap n = 693\n4 RQ1-only; 3 RQ3-only",
        ha="center",
        va="center",
        fontsize=8.2,
        color="#374151",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "#F9FAFB", "edgecolor": "#D1D5DB"},
    )
    ax.text(
        0.66,
        0.115,
        "Parallel sensitivities from n = 521",
        ha="center",
        va="center",
        fontsize=7.8,
        color="#374151",
    )

    fig.savefig(
        FIGURE_DIR / "agent2_sample_attrition_flow.pdf",
        bbox_inches="tight",
        facecolor="white",
    )
    save_figure(fig, "agent2_sample_attrition_flow.png")


def main() -> None:
    set_style()
    plot_sample_attrition_flow()
    plot_validity_boundary()
    plot_construct_overlap()
    plot_behavior_forecasting()
    plot_construct_audit()
    plot_feature_families()
    plot_typology()
    plot_android_pilot_workflow()
    plot_android_pilot_policy_cost()
    plot_android_september_followup()


if __name__ == "__main__":
    main()
