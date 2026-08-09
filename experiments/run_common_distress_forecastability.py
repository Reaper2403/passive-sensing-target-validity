from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


COMPONENTS = [
    "baseline_depression",
    "baseline_anxiety",
    "baseline_stress",
    "baseline_loneliness",
]

PREDICTORS = ["core_forecastability", "core_history_skill"]

CORE_TARGETS = [
    "screen_unlock_count",
    "screen_unlock_duration",
    "steps_sum",
    "active_duration",
    "sedentary_duration",
    "sleep_duration",
    "sleep_in_bed",
    "sleep_efficiency",
]

BASE_COVARIATES = [
    "n_prediction_rows",
    "n_targets_observed",
    "n_unique_dates",
    "date_span_days",
    "mean_rows_per_target",
    "behavior_level_pc1",
    "behavior_level_pc2",
    "behavior_level_pc3",
]

COVARIATES = [
    *BASE_COVARIATES,
    "behavior_variability_pc1",
    "behavior_variability_pc2",
    "behavior_variability_pc3",
]


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 5:
        return np.nan
    x0 = x[mask] - x[mask].mean()
    y0 = y[mask] - y[mask].mean()
    denominator = np.sqrt(np.sum(x0 * x0) * np.sum(y0 * y0))
    if denominator <= 1e-12:
        return np.nan
    return float(np.sum(x0 * y0) / denominator)


def zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if not np.isfinite(std) or std <= 1e-12:
        return pd.Series(np.nan, index=series.index)
    return (series - series.mean()) / std


def design_frame(
    df: pd.DataFrame,
    include_institute: bool = True,
    numeric_covariates: list[str] | None = None,
) -> pd.DataFrame:
    if numeric_covariates is None:
        numeric_covariates = COVARIATES
    categorical = ["platform"]
    if include_institute:
        categorical.insert(0, "institute_year")
    parts = [pd.Series(1.0, index=df.index, name="intercept")]
    parts.append(
        pd.get_dummies(df[categorical].fillna("missing"), drop_first=False).astype(float)
    )
    for column in numeric_covariates:
        values = pd.to_numeric(df[column], errors="coerce")
        values = values.fillna(values.median())
        std = values.std(ddof=0)
        if np.isfinite(std) and std > 1e-12:
            parts.append(((values - values.mean()) / std).rename(column))
    return pd.concat(parts, axis=1)


def residualize(values: np.ndarray, design: np.ndarray) -> np.ndarray:
    return values - design @ (np.linalg.pinv(design) @ values)


def bootstrap_correlation(
    x: np.ndarray, y: np.ndarray, rng: np.random.Generator, n_bootstrap: int
) -> tuple[float, float]:
    values = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(x), len(x))
        value = pearson_r(x[idx], y[idx])
        if np.isfinite(value):
            values.append(value)
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = np.sum((y_true - y_true.mean()) ** 2)
    if denominator <= 1e-12:
        return np.nan
    return float(1.0 - np.sum((y_true - y_pred) ** 2) / denominator)


def build_common_distress(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    work = df.copy()
    standardized_columns = []
    for component in COMPONENTS:
        column = f"{component}_within_year_z"
        work[column] = work.groupby("institute_year")[component].transform(zscore)
        standardized_columns.append(column)
    observed = work[standardized_columns].notna().sum(axis=1)
    work["common_distress"] = work[standardized_columns].mean(axis=1, skipna=True)
    work.loc[observed < 3, "common_distress"] = np.nan

    pairs = []
    for i, first in enumerate(standardized_columns):
        for second in standardized_columns[i + 1 :]:
            sample = work[[first, second]].dropna()
            pairs.append(
                {
                    "component_a": first.replace("_within_year_z", ""),
                    "component_b": second.replace("_within_year_z", ""),
                    "n": int(len(sample)),
                    "pearson_r": pearson_r(
                        sample[first].to_numpy(dtype=float),
                        sample[second].to_numpy(dtype=float),
                    ),
                }
            )
    complete = work[standardized_columns].dropna()
    item_variances = complete.var(axis=0, ddof=1).sum()
    total_variance = complete.sum(axis=1).var(ddof=1)
    k = len(standardized_columns)
    alpha = float(k / (k - 1) * (1 - item_variances / total_variance))
    return work, pd.DataFrame(pairs), alpha


def pooled_associations(
    df: pd.DataFrame,
    rng: np.random.Generator,
    n_bootstrap: int,
    n_permutations: int,
    numeric_covariates: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    sample = df.dropna(subset=["common_distress", *PREDICTORS]).copy().reset_index(drop=True)
    design = design_frame(
        sample,
        include_institute=True,
        numeric_covariates=numeric_covariates,
    ).to_numpy(dtype=float)
    y = sample["common_distress"].to_numpy(dtype=float)
    y_residual = residualize(y, design)
    x_residuals = {
        predictor: residualize(sample[predictor].to_numpy(dtype=float), design)
        for predictor in PREDICTORS
    }
    rows = []
    for predictor, x_residual in x_residuals.items():
        r = pearson_r(x_residual, y_residual)
        low, high = bootstrap_correlation(x_residual, y_residual, rng, n_bootstrap)
        rows.append(
            {
                "predictor": predictor,
                "n": int(len(sample)),
                "partial_r": r,
                "bootstrap_ci_low": low,
                "bootstrap_ci_high": high,
            }
        )

    null_rows = []
    max_values = []
    strata = sample.groupby(["institute_year", "platform"], dropna=False).indices
    for permutation in range(n_permutations):
        shuffled = y_residual.copy()
        for positions in strata.values():
            positions = np.asarray(positions)
            shuffled[positions] = rng.permutation(shuffled[positions])
        values = [abs(pearson_r(x, shuffled)) for x in x_residuals.values()]
        maximum = float(np.nanmax(values))
        max_values.append(maximum)
        null_rows.append({"permutation": permutation, "max_abs_partial_r": maximum})
    max_values = np.asarray(max_values)
    associations = pd.DataFrame(rows)
    associations["max_stat_permutation_p"] = [
        float((1 + np.sum(max_values >= abs(value))) / (n_permutations + 1))
        for value in associations["partial_r"]
    ]
    residuals = {"common_distress": y_residual, **x_residuals}
    return associations, pd.DataFrame(null_rows), residuals


def cohort_associations(
    df: pd.DataFrame, numeric_covariates: list[str] | None = None
) -> pd.DataFrame:
    rows = []
    for institute_year, frame in df.groupby("institute_year"):
        sample = frame.dropna(subset=["common_distress", *PREDICTORS]).copy().reset_index(drop=True)
        design = design_frame(
            sample,
            include_institute=False,
            numeric_covariates=numeric_covariates,
        ).to_numpy(dtype=float)
        y = residualize(sample["common_distress"].to_numpy(dtype=float), design)
        for predictor in PREDICTORS:
            x = residualize(sample[predictor].to_numpy(dtype=float), design)
            rows.append(
                {
                    "institute_year": institute_year,
                    "predictor": predictor,
                    "n": int(len(sample)),
                    "partial_r": pearson_r(x, y),
                }
            )
    return pd.DataFrame(rows)


def repeated_holdout(
    df: pd.DataFrame, rng: np.random.Generator, repetitions: int
) -> pd.DataFrame:
    sample = df.dropna(subset=["common_distress", *PREDICTORS]).copy().reset_index(drop=True)
    base = design_frame(sample, include_institute=True).to_numpy(dtype=float)
    y = sample["common_distress"].to_numpy(dtype=float)
    strata = sample.groupby("institute_year").indices
    rows = []
    for repetition in range(repetitions):
        train_positions = []
        test_positions = []
        for positions in strata.values():
            positions = np.asarray(positions)
            shuffled = rng.permutation(positions)
            cut = max(1, int(np.floor(0.8 * len(shuffled))))
            train_positions.extend(shuffled[:cut])
            test_positions.extend(shuffled[cut:])
        train = np.asarray(train_positions)
        test = np.asarray(test_positions)
        baseline_beta = np.linalg.pinv(base[train]) @ y[train]
        baseline_prediction = base[test] @ baseline_beta
        baseline_r2 = r2_score(y[test], baseline_prediction)
        for predictor in PREDICTORS:
            augmented = np.column_stack([base, sample[predictor].to_numpy(dtype=float)])
            beta = np.linalg.pinv(augmented[train]) @ y[train]
            prediction = augmented[test] @ beta
            full_r2 = r2_score(y[test], prediction)
            rows.append(
                {
                    "repetition": repetition,
                    "predictor": predictor,
                    "n_train": int(len(train)),
                    "n_test": int(len(test)),
                    "baseline_r2": baseline_r2,
                    "full_r2": full_r2,
                    "delta_r2": full_r2 - baseline_r2,
                }
            )
    return pd.DataFrame(rows)


def attach_direct_variability_controls(
    users: pd.DataFrame, target_metrics_path: Path
) -> tuple[pd.DataFrame, list[str], list[str]]:
    if not target_metrics_path.exists():
        raise FileNotFoundError(
            "Direct-variability sensitivity requires the referee target metrics: "
            f"{target_metrics_path}"
        )
    metrics = pd.read_csv(target_metrics_path)
    metrics = metrics[
        metrics["regime"].eq("fixed_original")
        & metrics["target_name"].isin(CORE_TARGETS)
    ].copy()
    work = users.copy()
    control_columns: dict[str, list[str]] = {}
    for measure, prefix in [
        ("observed_sd_z", "direct_sd"),
        ("observed_entropy_z", "direct_entropy"),
    ]:
        pivot = metrics.pivot_table(
            index=["institute_year", "user_id"],
            columns="target_name",
            values=measure,
            aggfunc="mean",
        ).reindex(columns=CORE_TARGETS)
        renamed = [f"{prefix}_{target}" for target in CORE_TARGETS]
        pivot.columns = renamed
        work = work.merge(
            pivot.reset_index(), on=["institute_year", "user_id"], how="left"
        )
        control_columns[prefix] = renamed
    return work, control_columns["direct_sd"], control_columns["direct_entropy"]


def direct_variability_sensitivity(
    df: pd.DataFrame,
    sd_columns: list[str],
    entropy_columns: list[str],
    rng: np.random.Generator,
    n_bootstrap: int,
    n_permutations: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    specifications = {
        "direct_core_sd": [*BASE_COVARIATES, *sd_columns],
        "direct_core_sd_entropy": [
            *BASE_COVARIATES,
            *sd_columns,
            *entropy_columns,
        ],
        "variability_pc3_plus_direct_core_sd": [*COVARIATES, *sd_columns],
    }
    association_frames = []
    cohort_frames = []
    for name, covariates in specifications.items():
        associations, _, _ = pooled_associations(
            df,
            rng,
            n_bootstrap,
            n_permutations,
            numeric_covariates=covariates,
        )
        associations.insert(0, "control_set", name)
        association_frames.append(associations)
        cohort = cohort_associations(df, numeric_covariates=covariates)
        cohort.insert(0, "control_set", name)
        cohort_frames.append(cohort)
    return (
        pd.concat(association_frames, ignore_index=True),
        pd.concat(cohort_frames, ignore_index=True),
    )


def plot_results(
    residuals: dict[str, np.ndarray], cohort: pd.DataFrame, out_dir: Path
) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    outputs = []
    fig, ax = plt.subplots(figsize=(7, 6))
    x = residuals["core_forecastability"]
    y = residuals["common_distress"]
    ax.scatter(x, y, s=14, alpha=0.35, color="#2563eb")
    slope, intercept = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 100)
    ax.plot(xs, intercept + slope * xs, color="#b91c1c", lw=2)
    ax.set_xlabel("Core forecastability residual")
    ax.set_ylabel("Common distress residual")
    ax.set_title("Routine forecastability and common distress")
    fig.tight_layout()
    path = out_dir / "forecastability_common_distress.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    outputs.append(str(path))

    frame = cohort[cohort["predictor"].eq("core_forecastability")]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(frame["institute_year"], frame["partial_r"], color="#0f766e")
    ax.axhline(0, color="black", lw=1)
    ax.set_ylabel("Strictly controlled partial r")
    ax.set_title("Direction by institute-year")
    fig.tight_layout()
    path = out_dir / "cohort_replication.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    outputs.append(str(path))
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--holdout-repetitions", type=int, default=200)
    parser.add_argument("--seed", type=int, default=73)
    args = parser.parse_args()

    root = args.repo_root.resolve()
    out_dir = root / "results/distress_sensitivity"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    source = root / "results/forecastability/per_user_forecastability.csv"
    users = pd.read_csv(source)
    users, component_correlations, alpha = build_common_distress(users)
    associations, permutation_null, residuals = pooled_associations(
        users, rng, args.bootstrap, args.permutations
    )
    cohort = cohort_associations(users)
    holdout = repeated_holdout(users, rng, args.holdout_repetitions)
    sensitivity_users, direct_sd_columns, direct_entropy_columns = (
        attach_direct_variability_controls(
            users,
            root
            / "results/referee_robustness/target_half_metrics.csv",
        )
    )
    direct_associations, direct_cohort = direct_variability_sensitivity(
        sensitivity_users,
        direct_sd_columns,
        direct_entropy_columns,
        rng,
        args.bootstrap,
        args.permutations,
    )
    figures = plot_results(residuals, cohort, out_dir)

    users.to_csv(out_dir / "common_distress_user_table.csv", index=False)
    component_correlations.to_csv(out_dir / "component_correlations.csv", index=False)
    associations.to_csv(out_dir / "pooled_associations.csv", index=False)
    permutation_null.to_csv(out_dir / "permutation_null.csv", index=False)
    cohort.to_csv(out_dir / "cohort_associations.csv", index=False)
    holdout.to_csv(out_dir / "repeated_holdout.csv", index=False)
    direct_associations.to_csv(
        out_dir / "direct_variability_associations.csv", index=False
    )
    direct_cohort.to_csv(
        out_dir / "direct_variability_cohort_associations.csv", index=False
    )

    median_component_r = float(component_correlations["pearson_r"].median())
    primary = associations[associations["predictor"].eq("core_forecastability")].iloc[0]
    primary_cohort = cohort[cohort["predictor"].eq("core_forecastability")]
    primary_holdout = holdout[holdout["predictor"].eq("core_forecastability")]
    h1_pass = bool(alpha >= 0.70 and median_component_r >= 0.30)
    h2_pass = bool(
        primary["partial_r"] <= -0.10
        and primary["bootstrap_ci_high"] < 0
        and primary["max_stat_permutation_p"] < 0.05
    )
    same_direction = int((primary_cohort["partial_r"] < 0).sum())
    median_delta = float(primary_holdout["delta_r2"].median())
    positive_rate = float((primary_holdout["delta_r2"] > 0).mean())
    h3_pass = bool(same_direction >= 3 and median_delta > 0 and positive_rate >= 0.70)
    if h1_pass and h2_pass and h3_pass:
        decision = "common_distress_forecastability_association_supported"
    elif h1_pass and h2_pass:
        decision = "pooled_association_only"
    else:
        decision = "common_distress_association_not_supported"
    summary = {
        "contract": "experiments/README.md",
        "status": "post_hoc_validation",
        "n_users": int(users["common_distress"].notna().sum()),
        "h1_common_factor": {
            "passed": h1_pass,
            "cronbach_alpha": alpha,
            "median_component_r": median_component_r,
        },
        "h2_controlled_association": {"passed": h2_pass, **primary.to_dict()},
        "h3_internal_generalization": {
            "passed": h3_pass,
            "same_negative_direction_cohorts": same_direction,
            "n_cohorts": int(len(primary_cohort)),
            "median_holdout_delta_r2": median_delta,
            "positive_holdout_rate": positive_rate,
            "n_holdout_repetitions": args.holdout_repetitions,
        },
        "direct_variability_sensitivity": {
            "control_definition": (
                "Eight target-specific SDs averaged over test halves; direct controls "
                "are median-imputed within the regression design."
            ),
            "associations": direct_associations.to_dict(orient="records"),
            "cohort_associations": direct_cohort.to_dict(orient="records"),
        },
        "decision": decision,
        "independent_replication_required": True,
        "figures": [str(Path(path).relative_to(root)) for path in figures],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
