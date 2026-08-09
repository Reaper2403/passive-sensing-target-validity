from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.csv as arrow_csv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.run_referee_robustness_checks import (
    CORE_TARGETS,
    RidgeRegressor,
    assign_positions,
    bootstrap_correlation,
    normalized_entropy,
    pearson_r,
    residualize,
    row_cosines,
    zscore,
)


MODEL_COLUMNS = {
    "calendar": ["weekday", "is_weekend", "study_day"],
    "lag1_history": [
        "weekday",
        "is_weekend",
        "study_day",
        "yesterday",
    ],
    "short_window_history": [
        "weekday",
        "is_weekend",
        "study_day",
        "roll7_mean",
        "roll14_mean",
    ],
    "recent_history": [
        "weekday",
        "is_weekend",
        "study_day",
        "yesterday",
        "roll7_mean",
        "roll14_mean",
    ],
    "long_horizon_fingerprint": [
        "weekday",
        "is_weekend",
        "study_day",
        "roll42_mean",
        "roll42_std",
    ],
    "combined_own_history": [
        "weekday",
        "is_weekend",
        "study_day",
        "yesterday",
        "roll7_mean",
        "roll14_mean",
        "roll42_mean",
        "roll42_std",
    ],
}

FAMILIES = [
    "lag1_history",
    "short_window_history",
    "recent_history",
    "long_horizon_fingerprint",
    "combined_own_history",
    "cross_domain_history",
    "cross_domain_increment",
]

CORE_SOURCE_COLUMNS = {
    "screen_unlock_count": "f_screen:phone_screen_rapids_countepisodeunlock:allday",
    "screen_unlock_duration": "f_screen:phone_screen_rapids_sumdurationunlock:allday",
    "steps_sum": "f_steps:fitbit_steps_intraday_rapids_sumsteps:allday",
    "active_duration": "f_steps:fitbit_steps_intraday_rapids_sumdurationactivebout:allday",
    "sedentary_duration": (
        "f_steps:fitbit_steps_intraday_rapids_sumdurationsedentarybout:allday"
    ),
    "sleep_duration": "f_slp:fitbit_sleep_summary_rapids_sumdurationasleepmain:allday",
    "sleep_in_bed": "f_slp:fitbit_sleep_summary_rapids_sumdurationinbedmain:allday",
    "sleep_efficiency": (
        "f_slp:fitbit_sleep_summary_rapids_avgefficiencymain:allday"
    ),
}


def load_model_samples(repo_root: Path) -> pd.DataFrame:
    path = repo_root / "results/behavioural_weather_forecast/prediction_samples.parquet"
    samples = pd.read_parquet(path)
    samples = samples[
        samples["horizon"].eq("tomorrow")
        & samples["issue_time"].eq("evening")
        & samples["target_name"].isin(CORE_TARGETS)
    ]
    rows = []
    keys = [
        "institute_year",
        "user_id",
        "date",
        "target_name",
        "target_value",
        "study_day",
        "weekday",
        "is_weekend",
    ]
    for target in CORE_TARGETS:
        frame = samples[samples["target_name"].eq(target)].copy()
        rename = {
            f"{target}__yesterday": "yesterday",
            f"{target}__roll7_mean": "roll7_mean",
            f"{target}__roll14_mean": "roll14_mean",
            f"{target}__roll42_mean": "roll42_mean",
            f"{target}__roll42_std": "roll42_std",
        }
        rows.append(frame[[*keys, *rename]].rename(columns=rename))
    long = pd.concat(rows, ignore_index=True)

    cross_source = long[
        [
            "institute_year",
            "user_id",
            "date",
            "target_name",
            "yesterday",
            "roll7_mean",
            "roll14_mean",
        ]
    ]
    cross = cross_source.pivot_table(
        index=["institute_year", "user_id", "date"],
        columns="target_name",
        values=["yesterday", "roll7_mean", "roll14_mean"],
        aggfunc="first",
    )
    cross.columns = [f"cross__{target}__{feature}" for feature, target in cross.columns]
    long = long.merge(
        cross.reset_index(), on=["institute_year", "user_id", "date"], how="left"
    )
    return assign_positions(long)


def block_mask(samples: pd.DataFrame, regime: str, block: str) -> pd.Series:
    rank = samples["trajectory_rank"]
    count = samples["trajectory_n"]
    if regime == "fixed_refit":
        train_end = np.floor(0.65 * count).astype(int)
        early_end = train_end + np.ceil((count - train_end) / 2).astype(int)
        bounds = {
            "early_train": (np.zeros(len(samples), dtype=int), train_end),
            "early_test": (train_end, early_end),
            "late_train": (np.zeros(len(samples), dtype=int), train_end),
            "late_test": (early_end, count),
        }
    elif regime == "expanding_refit":
        train_end = np.floor(0.65 * count).astype(int)
        early_end = train_end + np.ceil((count - train_end) / 2).astype(int)
        bounds = {
            "early_train": (np.zeros(len(samples), dtype=int), train_end),
            "early_test": (train_end, early_end),
            "late_train": (np.zeros(len(samples), dtype=int), early_end),
            "late_test": (early_end, count),
        }
    elif regime == "disjoint_refit":
        cut_35 = np.floor(0.35 * count).astype(int)
        cut_50 = np.floor(0.50 * count).astype(int)
        cut_85 = np.floor(0.85 * count).astype(int)
        bounds = {
            "early_train": (np.zeros(len(samples), dtype=int), cut_35),
            "early_test": (cut_35, cut_50),
            "late_train": (cut_50, cut_85),
            "late_test": (cut_85, count),
        }
    else:
        raise ValueError(regime)
    low, high = bounds[block]
    return rank.ge(low) & rank.lt(high)


def fit_feature_models(samples: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_frames = []
    fit_rows = []
    regimes = ["fixed_refit", "expanding_refit", "disjoint_refit"]
    for target, target_samples in samples.groupby("target_name", sort=True):
        other_targets = [name for name in CORE_TARGETS if name != target]
        cross_columns = [
            f"cross__{other}__{feature}"
            for other in other_targets
            for feature in ["yesterday", "roll7_mean", "roll14_mean"]
        ]
        model_columns = {
            **MODEL_COLUMNS,
            "cross_domain_history": [
                *MODEL_COLUMNS["combined_own_history"],
                *cross_columns,
            ],
        }
        for regime in regimes:
            for half in ["early", "late"]:
                train = target_samples.loc[
                    block_mask(target_samples, regime, f"{half}_train")
                ]
                test = target_samples.loc[
                    block_mask(target_samples, regime, f"{half}_test")
                ].copy()
                y_train = np.log1p(train["target_value"].clip(lower=0).astype(float))
                output = test[
                    ["institute_year", "user_id", "date", "target_name", "target_value"]
                ].copy()
                output["regime"] = regime
                output["half"] = half
                for model_name, columns in model_columns.items():
                    model = RidgeRegressor(alpha=1.0).fit(train[columns], y_train)
                    output[f"prediction__{model_name}"] = model.predict(test[columns])
                    fit_rows.append(
                        {
                            "regime": regime,
                            "target_name": target,
                            "half": half,
                            "model": model_name,
                            "n_train_rows": int(len(train)),
                            "n_features": int(len(columns)),
                            "coefficient_l2_norm": float(np.linalg.norm(model.coef)),
                        }
                    )
                prediction_frames.append(output)
    return pd.concat(prediction_frames, ignore_index=True), pd.DataFrame(fit_rows)


def add_target_residual(
    metrics: pd.DataFrame,
    value_column: str,
    output_column: str,
    controls: list[str],
) -> None:
    metrics[output_column] = np.nan
    for _, frame in metrics.groupby(["regime", "target_name", "half"], sort=False):
        usable = frame[[value_column, *controls]].notna().all(axis=1)
        sample = frame.loc[usable]
        if len(sample) < 10:
            continue
        cohort = pd.get_dummies(
            sample["institute_year"], drop_first=True
        ).to_numpy(dtype=float)
        design = np.column_stack(
            [np.ones(len(sample)), cohort, sample[controls].to_numpy(dtype=float)]
        )
        residual = residualize(sample[value_column].to_numpy(dtype=float), design)
        metrics.loc[sample.index, output_column] = zscore(
            pd.Series(residual, index=sample.index)
        )


def make_feature_metrics(
    predictions: pd.DataFrame, min_half_rows: int
) -> pd.DataFrame:
    work = predictions.copy()
    work["observed_log"] = np.log1p(work["target_value"].clip(lower=0).astype(float))
    model_names = [*MODEL_COLUMNS, "cross_domain_history"]
    for model in model_names:
        work[f"error__{model}"] = (
            work["observed_log"] - work[f"prediction__{model}"]
        ).abs()

    entropy_edges = {}
    for key, frame in work.groupby(["regime", "target_name"]):
        entropy_edges[key] = np.unique(
            np.quantile(frame["observed_log"].dropna(), np.linspace(0, 1, 6))
        )

    rows = []
    group_columns = ["regime", "institute_year", "user_id", "target_name", "half"]
    for keys, frame in work.groupby(group_columns, sort=False):
        if len(frame) < min_half_rows:
            continue
        regime, institute_year, user_id, target, half = keys
        row = {
            "regime": regime,
            "institute_year": institute_year,
            "user_id": user_id,
            "target_name": target,
            "half": half,
            "n_obs": int(len(frame)),
            "observed_sd": float(frame["observed_log"].std(ddof=0)),
            "marginal_entropy": normalized_entropy(
                frame["observed_log"], entropy_edges[(regime, target)]
            ),
        }
        for model in model_names:
            row[f"mae__{model}"] = float(frame[f"error__{model}"].mean())
        rows.append(row)
    metrics = pd.DataFrame(rows)
    calendar = metrics["mae__calendar"].clip(lower=1e-8)
    for family in FAMILIES[:-1]:
        metrics[f"skill__{family}"] = 1.0 - metrics[f"mae__{family}"] / calendar
    metrics["skill__cross_domain_increment"] = 1.0 - (
        metrics["mae__cross_domain_history"]
        / metrics["mae__combined_own_history"].clip(lower=1e-8)
    )
    target_scale = (
        work.groupby(["regime", "target_name"])["observed_log"]
        .std(ddof=0)
        .rename("target_scale")
        .reset_index()
    )
    metrics = metrics.merge(target_scale, on=["regime", "target_name"], how="left")
    metrics["forecastability"] = (
        -metrics["mae__combined_own_history"] / metrics["target_scale"]
    )
    z_columns = [
        "observed_sd",
        "marginal_entropy",
        "forecastability",
        *[f"skill__{family}" for family in FAMILIES],
    ]
    for column in z_columns:
        metrics[f"{column}_z"] = metrics.groupby(
            ["regime", "target_name", "half"]
        )[column].transform(zscore)
    for construct in ["forecastability", *[f"skill__{family}" for family in FAMILIES]]:
        add_target_residual(
            metrics,
            f"{construct}_z",
            f"{construct}_residual_z",
            ["observed_sd_z", "marginal_entropy_z"],
        )
    return metrics


def make_training_moments(samples: pd.DataFrame) -> pd.DataFrame:
    """Estimate variability and marginal entropy only from each model's train block."""
    rows = []
    for target, target_samples in samples.groupby("target_name", sort=True):
        for regime in ["fixed_refit", "expanding_refit", "disjoint_refit"]:
            for half in ["early", "late"]:
                train = target_samples.loc[
                    block_mask(target_samples, regime, f"{half}_train")
                ].copy()
                train["observed_log"] = np.log1p(
                    train["target_value"].clip(lower=0).astype(float)
                )
                edges = np.unique(
                    np.quantile(train["observed_log"].dropna(), np.linspace(0, 1, 6))
                )
                for (institute_year, user_id), frame in train.groupby(
                    ["institute_year", "user_id"], sort=False
                ):
                    rows.append(
                        {
                            "regime": regime,
                            "institute_year": institute_year,
                            "user_id": user_id,
                            "target_name": target,
                            "half": half,
                            "n_train_obs": int(len(frame)),
                            "training_sd": float(
                                frame["observed_log"].std(ddof=0)
                            ),
                            "training_entropy": normalized_entropy(
                                frame["observed_log"], edges
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def composite_stability(
    metrics: pd.DataFrame,
    value_columns: dict[str, str],
    rng: np.random.Generator,
    n_bootstrap: int,
) -> pd.DataFrame:
    rows = []
    for regime in sorted(metrics["regime"].unique()):
        subset = metrics[metrics["regime"].eq(regime)]
        for measure, column in value_columns.items():
            composite = (
                subset.groupby(["institute_year", "user_id", "half"])
                .agg(value=(column, "mean"), n_targets=("target_name", "nunique"))
                .query("n_targets >= 6")
                .reset_index()
            )
            pivot = composite.pivot_table(
                index=["institute_year", "user_id"],
                columns="half",
                values="value",
                aggfunc="first",
            ).dropna()
            early = pivot["early"].to_numpy(dtype=float)
            late = pivot["late"].to_numpy(dtype=float)
            low, high = bootstrap_correlation(early, late, rng, n_bootstrap)
            rows.append(
                {
                    "regime": regime,
                    "measure": measure,
                    "n_trajectories": int(len(pivot)),
                    "pearson_r": pearson_r(early, late),
                    "ci_low": low,
                    "ci_high": high,
                }
            )
    return pd.DataFrame(rows)


def profile_stability(
    metrics: pd.DataFrame,
    value_columns: dict[str, str],
    rng: np.random.Generator,
    n_permutations: int,
) -> pd.DataFrame:
    rows = []
    for regime in sorted(metrics["regime"].unique()):
        subset = metrics[metrics["regime"].eq(regime)]
        for measure, column in value_columns.items():
            pivot = subset.pivot_table(
                index=["institute_year", "user_id", "half"],
                columns="target_name",
                values=column,
                aggfunc="first",
            )
            early = pivot.xs("early", level="half", drop_level=True).reindex(
                columns=CORE_TARGETS
            )
            late = pivot.xs("late", level="half", drop_level=True).reindex(
                columns=CORE_TARGETS
            )
            common = early.index.intersection(late.index)
            early = early.loc[common]
            late = late.loc[common]
            usable = (early.notna() & late.notna()).sum(axis=1) >= 6
            early = early.loc[usable]
            late = late.loc[usable]
            early_values = early.to_numpy(dtype=float)
            late_values = late.to_numpy(dtype=float)
            same_mean = float(np.nanmean(row_cosines(early_values, late_values)))
            institutes = np.asarray(early.index.get_level_values("institute_year"))
            null = []
            for _ in range(n_permutations):
                permutation = np.arange(len(late_values))
                for institute in sorted(set(institutes)):
                    positions = np.flatnonzero(institutes == institute)
                    permutation[positions] = rng.permutation(positions)
                null.append(
                    float(
                        np.nanmean(row_cosines(early_values, late_values[permutation]))
                    )
                )
            cross_mean = float(np.mean(null))
            rows.append(
                {
                    "regime": regime,
                    "measure": measure,
                    "n_trajectories": int(len(early)),
                    "same_person_mean": same_mean,
                    "matched_cross_person_mean": cross_mean,
                    "delta": same_mean - cross_mean,
                    "permutation_p": float(
                        (1 + np.sum(np.asarray(null) >= same_mean))
                        / (n_permutations + 1)
                    ),
                    "n_permutations": n_permutations,
                }
            )
    return pd.DataFrame(rows)


def training_moment_adjusted_results(
    metrics: pd.DataFrame,
    training_moments: pd.DataFrame,
    rng: np.random.Generator,
    n_bootstrap: int,
    n_permutations: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    work = metrics.merge(
        training_moments,
        on=["regime", "institute_year", "user_id", "target_name", "half"],
        how="left",
        validate="one_to_one",
    )
    for column in ["training_sd", "training_entropy"]:
        work[f"{column}_z"] = work.groupby(
            ["regime", "target_name", "half"]
        )[column].transform(zscore)
    constructs = {
        "forecastability_training_adjusted": "forecastability_z",
        "combined_skill_training_adjusted": "skill__combined_own_history_z",
    }
    value_columns = {}
    for measure, source in constructs.items():
        output = f"{measure}_z"
        add_target_residual(
            work,
            source,
            output,
            ["training_sd_z", "training_entropy_z"],
        )
        value_columns[measure] = output
    return (
        work,
        composite_stability(work, value_columns, rng, n_bootstrap),
        profile_stability(work, value_columns, rng, n_permutations),
    )


def conditional_entropy(states: np.ndarray, contexts: np.ndarray, n_states: int) -> float:
    if not len(states):
        return np.nan
    total = 0.0
    for context in np.unique(contexts):
        values = states[contexts == context]
        counts = np.bincount(values, minlength=n_states).astype(float)
        probabilities = counts[counts > 0] / counts.sum()
        entropy = -float(np.sum(probabilities * np.log(probabilities)))
        total += len(values) / len(states) * entropy
    return total / math.log(n_states)


def lz_entropy_rate(states: np.ndarray, n_states: int) -> float:
    n = len(states)
    if n < 20:
        return np.nan
    match_lengths = []
    sequence = states.tolist()
    for index in range(n):
        maximum = n - index
        length = 1
        while length <= maximum:
            candidate = sequence[index : index + length]
            found = any(
                sequence[start : start + length] == candidate
                for start in range(index)
            )
            if not found:
                break
            length += 1
        match_lengths.append(min(length, maximum + 1))
    estimate = n * math.log(n) / np.sum(match_lengths)
    return float(min(estimate / math.log(n_states), 1.5))


def load_core_daily(repo_root: Path, cache_path: Path) -> pd.DataFrame:
    if cache_path.exists():
        return pd.read_parquet(cache_path)
    frames = []
    source_columns = ["pid", "date", *CORE_SOURCE_COLUMNS.values()]
    for path in sorted(
        (repo_root / "data/raw/globem/1.1").glob("INS-W_*/FeatureData/rapids.csv")
    ):
        table = arrow_csv.read_csv(
            path,
            convert_options=arrow_csv.ConvertOptions(include_columns=source_columns),
        )
        frame = table.to_pandas().rename(columns={"pid": "user_id"})
        frame["institute_year"] = path.parts[-3]
        frame["date"] = pd.to_datetime(frame["date"])
        frame = frame.rename(columns={value: key for key, value in CORE_SOURCE_COLUMNS.items()})
        frames.append(frame)
    daily = pd.concat(frames, ignore_index=True)
    daily.to_parquet(cache_path, index=False)
    return daily


def longest_contiguous_states(frame: pd.DataFrame) -> np.ndarray:
    work = frame.sort_values("date").copy()
    block = work["date"].diff().dt.days.ne(1).cumsum()
    counts = block.value_counts()
    if counts.empty:
        return np.asarray([], dtype=int)
    return work.loc[block.eq(counts.idxmax()), "state"].to_numpy(dtype=int)


def sequence_metrics(daily: pd.DataFrame) -> pd.DataFrame:
    long = daily.melt(
        id_vars=["institute_year", "user_id", "date"],
        value_vars=CORE_TARGETS,
        var_name="target_name",
        value_name="value",
    ).dropna(subset=["value"])
    long["observed_log"] = np.log1p(long["value"].clip(lower=0).astype(float))
    long["state"] = -1
    for target, frame in long.groupby("target_name"):
        edges = np.unique(np.quantile(frame["observed_log"], np.linspace(0, 1, 6)))
        if len(edges) < 3:
            continue
        long.loc[frame.index, "state"] = np.digitize(
            frame["observed_log"], edges[1:-1], right=True
        )
    rows = []
    for keys, frame in long[long["state"].ge(0)].groupby(
        ["institute_year", "user_id", "target_name"], sort=False
    ):
        frame = frame.sort_values("date")
        states = frame["state"].to_numpy(dtype=int)
        n_states = 5
        counts = np.bincount(states, minlength=n_states).astype(float)
        probabilities = counts[counts > 0] / counts.sum()
        marginal = -float(np.sum(probabilities * np.log(probabilities))) / math.log(
            n_states
        )
        consecutive = frame["date"].diff().dt.days.eq(1).to_numpy()
        transition_states = states[consecutive]
        previous_states = states[np.flatnonzero(consecutive) - 1]
        weekday = frame["date"].dt.weekday.to_numpy(dtype=int)
        contiguous = longest_contiguous_states(frame)
        institute_year, user_id, target = keys
        rows.append(
            {
                "institute_year": institute_year,
                "user_id": user_id,
                "target_name": target,
                "n_days": int(len(frame)),
                "n_transitions": int(consecutive.sum()),
                "longest_contiguous_run": int(len(contiguous)),
                "full_marginal_entropy": marginal,
                "first_order_entropy": conditional_entropy(
                    transition_states, previous_states, n_states
                ),
                "weekday_conditional_entropy": conditional_entropy(
                    states, weekday, n_states
                ),
                "lz_entropy_rate_proxy": lz_entropy_rate(contiguous, n_states),
            }
        )
    return pd.DataFrame(rows)


def sequence_adjusted_results(
    metrics: pd.DataFrame,
    sequences: pd.DataFrame,
    rng: np.random.Generator,
    n_bootstrap: int,
    n_permutations: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = metrics.merge(
        sequences,
        on=["institute_year", "user_id", "target_name"],
        how="left",
    )
    controls = [
        "observed_sd_z",
        "full_marginal_entropy",
        "first_order_entropy",
        "weekday_conditional_entropy",
        "lz_entropy_rate_proxy",
    ]
    constructs = {
        "forecastability_sequence_adjusted": "forecastability_z",
        "combined_skill_sequence_adjusted": "skill__combined_own_history_z",
    }
    value_columns = {}
    for measure, source in constructs.items():
        output = f"{measure}_z"
        add_target_residual(work, source, output, controls)
        value_columns[measure] = output
    return (
        composite_stability(work, value_columns, rng, n_bootstrap),
        profile_stability(work, value_columns, rng, n_permutations),
    )


def diagnostic_tables(
    metrics: pd.DataFrame, predictions: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    coverage = (
        metrics.groupby(["regime", "half"], as_index=False)
        .agg(
            n_target_halves=("n_obs", "size"),
            median_test_obs=("n_obs", "median"),
            mean_test_obs=("n_obs", "mean"),
            median_observed_sd=("observed_sd", "median"),
            mean_marginal_entropy=("marginal_entropy", "mean"),
        )
        .sort_values(["regime", "half"])
    )

    calendar = predictions.copy()
    calendar["observed_log"] = np.log1p(
        calendar["target_value"].clip(lower=0).astype(float)
    )
    calendar["calendar_absolute_error"] = (
        calendar["observed_log"] - calendar["prediction__calendar"]
    ).abs()
    calendar["target_regime_mean_mae"] = calendar.groupby(
        ["regime", "target_name"]
    )["calendar_absolute_error"].transform("mean")
    calendar["normalized_calendar_mae"] = (
        calendar["calendar_absolute_error"]
        / calendar["target_regime_mean_mae"].clip(lower=1e-8)
    )
    calendar_cohort = (
        calendar.groupby(["regime", "institute_year"], as_index=False)
        .agg(
            n_rows=("normalized_calendar_mae", "size"),
            mean_normalized_calendar_mae=("normalized_calendar_mae", "mean"),
        )
        .sort_values(["regime", "institute_year"])
    )
    return coverage, calendar_cohort


def performance_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    work = predictions.copy()
    work["observed_log"] = np.log1p(work["target_value"].clip(lower=0).astype(float))
    rows = []
    models = [*MODEL_COLUMNS, "cross_domain_history"]
    for (regime, target), frame in work.groupby(["regime", "target_name"]):
        row = {"regime": regime, "target_name": target, "n_rows": int(len(frame))}
        for model in models:
            row[f"mae__{model}"] = float(
                np.mean(
                    np.abs(frame["observed_log"] - frame[f"prediction__{model}"])
                )
            )
        row["cross_domain_relative_mae_gain"] = 1.0 - (
            row["mae__cross_domain_history"] / row["mae__combined_own_history"]
        )
        rows.append(row)
    return pd.DataFrame(rows)


def plot_summary(
    stability: pd.DataFrame, profiles: pd.DataFrame, output_path: Path
) -> None:
    import matplotlib.pyplot as plt

    measures = [f"{family}_residual" for family in FAMILIES]
    degree = stability[stability["measure"].isin(measures)].pivot(
        index="measure", columns="regime", values="pearson_r"
    )
    profile = profiles[profiles["measure"].isin(measures)].pivot(
        index="measure", columns="regime", values="delta"
    )
    labels = [name.replace("_residual", "").replace("_", " ") for name in degree.index]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    degree.plot(kind="bar", ax=axes[0], width=0.8)
    axes[0].axhline(0.30, color="black", linestyle="--", linewidth=1)
    axes[0].set_title("Residualized degree stability")
    axes[0].set_ylabel("Early-late Pearson r")
    axes[0].set_xticklabels(labels, rotation=30, ha="right")
    profile.plot(kind="bar", ax=axes[1], width=0.8)
    axes[1].axhline(0.10, color="black", linestyle="--", linewidth=1)
    axes[1].set_title("Residualized profile advantage")
    axes[1].set_ylabel("Same-person minus matched cross-person cosine")
    axes[1].set_xticklabels(labels, rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--permutations", type=int, default=500)
    parser.add_argument("--headline-permutations", type=int, default=10000)
    parser.add_argument("--min-half-rows", type=int, default=4)
    parser.add_argument("--seed", type=int, default=211)
    args = parser.parse_args()

    root = args.repo_root.resolve()
    output = root / "results/structure_followup"
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    samples = load_model_samples(root)
    predictions, fits = fit_feature_models(samples)
    metrics = make_feature_metrics(predictions, args.min_half_rows)
    value_columns = {}
    for family in FAMILIES:
        value_columns[family] = f"skill__{family}_z"
        value_columns[f"{family}_residual"] = f"skill__{family}_residual_z"
    value_columns["forecastability"] = "forecastability_z"
    value_columns["forecastability_residual"] = "forecastability_residual_z"
    stability = composite_stability(metrics, value_columns, rng, args.bootstrap)
    profiles = profile_stability(metrics, value_columns, rng, args.permutations)
    performance = performance_summary(predictions)
    coverage, calendar_cohort = diagnostic_tables(metrics, predictions)

    training_moments = make_training_moments(samples)
    metrics, training_stability, training_profiles = training_moment_adjusted_results(
        metrics,
        training_moments,
        rng,
        args.bootstrap,
        args.headline_permutations,
    )

    daily = load_core_daily(root, output / "core_daily.parquet")
    sequences = sequence_metrics(daily)
    sequence_stability, sequence_profiles = sequence_adjusted_results(
        metrics, sequences, rng, args.bootstrap, args.headline_permutations
    )

    fits.to_csv(output / "feature_model_fits.csv", index=False)
    performance.to_csv(output / "feature_family_performance.csv", index=False)
    stability.to_csv(output / "feature_family_stability.csv", index=False)
    profiles.to_csv(output / "feature_family_profiles.csv", index=False)
    training_stability.to_csv(
        output / "training_moment_adjusted_stability.csv", index=False
    )
    training_profiles.to_csv(
        output / "training_moment_adjusted_profiles.csv", index=False
    )
    sequence_stability.to_csv(output / "sequence_adjusted_stability.csv", index=False)
    sequence_profiles.to_csv(output / "sequence_adjusted_profiles.csv", index=False)
    coverage.to_csv(output / "window_coverage_diagnostics.csv", index=False)
    calendar_cohort.to_csv(output / "calendar_cohort_diagnostics.csv", index=False)
    metrics.to_csv(output / "feature_target_half_metrics.csv", index=False)
    sequences.to_csv(output / "sequence_metrics.csv", index=False)
    predictions.to_parquet(output / "feature_predictions.parquet", index=False)
    plot_summary(stability, profiles, output / "feature_family_decomposition.png")

    residual_stability = stability[stability["measure"].str.endswith("_residual")]
    residual_profiles = profiles[profiles["measure"].str.endswith("_residual")]
    family_decisions = {}
    for family in FAMILIES:
        measure = f"{family}_residual"
        degrees = residual_stability[residual_stability["measure"].eq(measure)]
        profile_rows = residual_profiles[residual_profiles["measure"].eq(measure)]
        family_decisions[family] = bool(
            (degrees["pearson_r"] >= 0.30).all()
            and (degrees["ci_low"] > 0).all()
            and (profile_rows["delta"] >= 0.10).all()
            and (profile_rows["permutation_p"] < 0.05).all()
        )
    cross_performance = performance.groupby("regime").agg(
        mean_relative_gain=("cross_domain_relative_mae_gain", "mean"),
        positive_targets=("cross_domain_relative_mae_gain", lambda x: int((x > 0).sum())),
    )
    cross_gate = bool(
        (cross_performance["mean_relative_gain"] > 0).all()
        and (cross_performance["positive_targets"] >= 6).all()
        and family_decisions["cross_domain_increment"]
    )
    summary = {
        "status": "post_hoc_exploratory",
        "contract": "experiments/README.md",
        "entropy_boundary": (
            "Sequence metrics are discretized descriptive proxies, not a Song-style "
            "continuous-behaviour predictability ceiling."
        ),
        "source_rows": int(len(samples)),
        "family_decisions": family_decisions,
        "cross_domain_gate_passed": cross_gate,
        "cross_domain_performance": cross_performance.reset_index().to_dict(
            orient="records"
        ),
        "permutation_counts": {
            "headline_profiles": args.headline_permutations,
            "secondary_feature_profiles": args.permutations,
        },
        "training_moment_adjusted_stability": training_stability.to_dict(
            orient="records"
        ),
        "training_moment_adjusted_profiles": training_profiles.to_dict(
            orient="records"
        ),
        "training_moment_profile_gate_passed_all_regimes": bool(
            (training_profiles["delta"] >= 0.10).all()
            and (training_profiles["permutation_p"] < 0.05).all()
        ),
        "window_coverage_diagnostics": coverage.to_dict(orient="records"),
        "calendar_cohort_diagnostics": calendar_cohort.to_dict(orient="records"),
        "sequence_sample": {
            "n_trajectories": int(len(sequences)),
            "median_days": float(sequences["n_days"].median()),
            "median_transitions": float(sequences["n_transitions"].median()),
            "median_longest_run": float(sequences["longest_contiguous_run"].median()),
        },
        "sequence_adjusted_stability": sequence_stability.to_dict(orient="records"),
        "sequence_adjusted_profiles": sequence_profiles.to_dict(orient="records"),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
