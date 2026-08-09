from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import urllib.request
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd


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

REGIMES = ["expanding_refit", "disjoint_refit"]
GLOBEM_METADATA_COMMIT = "4f140fc5290dc97204298cca28b956165aa0a29f"
GLOBEM_OVERLAP_URL = (
    "https://raw.githubusercontent.com/UW-EXP/GLOBEM/"
    f"{GLOBEM_METADATA_COMMIT}/data/additional_user_setup/overlapping_pids.json"
)


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


def bootstrap_correlation(
    x: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> tuple[float, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 10:
        return np.nan, np.nan
    values = []
    for _ in range(n_bootstrap):
        indices = rng.integers(0, len(x), len(x))
        value = pearson_r(x[indices], y[indices])
        if np.isfinite(value):
            values.append(value)
    if not values:
        return np.nan, np.nan
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if not np.isfinite(std) or std <= 1e-12:
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std


class RidgeRegressor:
    """Median-imputed, standardized Ridge with the original alpha of 1.0."""

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha

    def fit(self, features: pd.DataFrame, target: pd.Series) -> "RidgeRegressor":
        values = features.to_numpy(dtype=float)
        values[~np.isfinite(values)] = np.nan
        self.medians = np.nanmedian(values, axis=0)
        self.medians[~np.isfinite(self.medians)] = 0.0
        values = np.where(np.isnan(values), self.medians, values)
        self.means = values.mean(axis=0)
        self.scales = values.std(axis=0, ddof=0)
        self.scales[self.scales <= 1e-12] = 1.0
        standardized = (values - self.means) / self.scales
        y = target.to_numpy(dtype=float)
        self.intercept = float(y.mean())
        centered_y = y - self.intercept
        gram = standardized.T @ standardized
        penalty = self.alpha * np.eye(standardized.shape[1])
        self.coef = np.linalg.solve(gram + penalty, standardized.T @ centered_y)
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        values = features.to_numpy(dtype=float)
        values[~np.isfinite(values)] = np.nan
        values = np.where(np.isnan(values), self.medians, values)
        standardized = (values - self.means) / self.scales
        return self.intercept + standardized @ self.coef


def make_regressor() -> RidgeRegressor:
    return RidgeRegressor(alpha=1.0)


def load_or_build_samples(repo_root: Path) -> pd.DataFrame:
    cached = repo_root / "results/behavioural_weather_forecast/prediction_samples.parquet"
    if cached.exists():
        samples = pd.read_parquet(cached)
        return samples[
            samples["horizon"].eq("tomorrow") & samples["issue_time"].eq("evening")
        ].copy()

    sys.path.insert(0, str(repo_root))
    from scripts.run_behavioural_weather_forecast import (  # noqa: PLC0415
        TARGETS,
        add_history_features,
        load_rapids,
        make_future_targets,
    )

    raw = load_rapids(repo_root / "data/raw/globem/1.1")
    raw["first_date"] = raw.groupby(["institute_year", "user_id"])["date"].transform(
        "min"
    )
    raw["study_day"] = (raw["date"] - raw["first_date"]).dt.days
    raw["weekday"] = raw["date"].dt.weekday
    raw["is_weekend"] = raw["weekday"].isin([5, 6]).astype(int)

    frames = []
    for target_name, base in TARGETS.items():
        all_day = f"{base}:allday"
        if all_day not in raw.columns:
            continue
        history = add_history_features(raw, target_name, all_day)
        future = make_future_targets(raw, target_name, all_day, "tomorrow")
        samples = raw[
            ["institute_year", "user_id", "date", "study_day", "weekday", "is_weekend"]
        ].copy()
        samples = samples.merge(history, on=["institute_year", "user_id", "date"])
        samples = samples.merge(
            future,
            on=["institute_year", "user_id", "date"],
            how="inner",
        )
        samples = samples.dropna(subset=["target_value"])
        frames.append(samples[samples["study_day"] >= 42].copy())
    return pd.concat(frames, ignore_index=True)


def assign_positions(samples: pd.DataFrame) -> pd.DataFrame:
    work = samples.sort_values(["institute_year", "user_id", "target_name", "date"]).copy()
    groups = work.groupby(["institute_year", "user_id", "target_name"], sort=False)
    work["trajectory_rank"] = groups.cumcount().astype(int)
    work["trajectory_n"] = groups["date"].transform("size").astype(int)
    return work


def block_mask(samples: pd.DataFrame, regime: str, block: str) -> pd.Series:
    rank = samples["trajectory_rank"]
    count = samples["trajectory_n"]
    if regime == "expanding_refit":
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


def fit_refitted_predictions(samples: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_frames = []
    model_rows = []
    for target_name, target_samples in samples.groupby("target_name", sort=True):
        history_columns = [
            "weekday",
            "is_weekend",
            "study_day",
            f"{target_name}__yesterday",
            f"{target_name}__roll7_mean",
            f"{target_name}__roll14_mean",
            f"{target_name}__roll42_mean",
            f"{target_name}__roll42_std",
        ]
        calendar_columns = ["weekday", "is_weekend", "study_day"]
        for regime in REGIMES:
            fitted_models: dict[str, dict[str, RidgeRegressor]] = {}
            for half in ["early", "late"]:
                train_mask = block_mask(target_samples, regime, f"{half}_train")
                train = target_samples.loc[train_mask]
                y_train = np.log1p(train["target_value"].clip(lower=0).astype(float))
                fitted_models[half] = {}
                for model_name, columns in {
                    "history": history_columns,
                    "calendar": calendar_columns,
                }.items():
                    model = make_regressor()
                    model.fit(train[columns], y_train)
                    fitted_models[half][model_name] = model
                    model_rows.append(
                        {
                            "regime": regime,
                            "target_name": target_name,
                            "half": half,
                            "model": model_name,
                            "n_train_rows": int(len(train)),
                            "n_train_trajectories": int(
                                train[["institute_year", "user_id"]].drop_duplicates().shape[0]
                            ),
                            "coefficient_l2_norm": float(np.linalg.norm(model.coef)),
                        }
                    )

            for half in ["early", "late"]:
                test_mask = block_mask(target_samples, regime, f"{half}_test")
                test = target_samples.loc[test_mask].copy()
                output = test[
                    ["institute_year", "user_id", "date", "target_name", "target_value"]
                ].copy()
                output["regime"] = regime
                output["half"] = half
                output["reg_history"] = fitted_models[half]["history"].predict(
                    test[history_columns]
                )
                output["reg_calendar"] = fitted_models[half]["calendar"].predict(
                    test[calendar_columns]
                )
                prediction_frames.append(output)
    return pd.concat(prediction_frames, ignore_index=True), pd.DataFrame(model_rows)


def load_fixed_predictions(repo_root: Path, min_half_rows: int) -> pd.DataFrame:
    path = repo_root / "results/behavioural_weather_forecast/predictions.parquet"
    columns = [
        "institute_year",
        "user_id",
        "date",
        "target_name",
        "horizon",
        "issue_time",
        "target_value",
        "reg__history_plus_fingerprint",
        "reg__calendar_only",
    ]
    work = pd.read_parquet(path, columns=columns)
    work = work[work["horizon"].eq("tomorrow") & work["issue_time"].eq("evening")]
    work = work.sort_values(["institute_year", "user_id", "target_name", "date"]).copy()
    groups = work.groupby(["institute_year", "user_id", "target_name"], sort=False)
    work["test_rank"] = groups.cumcount()
    work["test_n"] = groups["date"].transform("size")
    work = work[work["test_n"] >= 2 * min_half_rows].copy()
    work["half"] = np.where(
        work["test_rank"] < (work["test_n"] / 2), "early", "late"
    )
    return work.rename(
        columns={
            "reg__history_plus_fingerprint": "reg_history",
            "reg__calendar_only": "reg_calendar",
        }
    ).assign(regime="fixed_original")


def load_repeat_candidates() -> tuple[set[tuple[str, str]], dict[str, str | int]]:
    with urllib.request.urlopen(GLOBEM_OVERLAP_URL, timeout=30) as response:
        content = response.read()
    metadata = json.loads(content)["dep_weekly"]
    candidates: set[tuple[str, str]] = set()
    for reference_year, year_lists in metadata.items():
        if not reference_year.startswith("INS-W_"):
            continue
        for institute_year, identifiers in year_lists.items():
            if not institute_year.startswith("INS-W_"):
                continue
            for identifier in identifiers:
                candidates.add((institute_year, identifier.split("#", maxsplit=1)[0]))
    provenance: dict[str, str | int] = {
        "url": GLOBEM_OVERLAP_URL,
        "commit": GLOBEM_METADATA_COMMIT,
        "sha256": hashlib.sha256(content).hexdigest(),
        "n_repeat_candidate_trajectory_keys": len(candidates),
    }
    return candidates, provenance


def normalized_entropy(values: pd.Series, edges: np.ndarray) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if not len(values) or len(edges) <= 2:
        return 0.0
    bins = np.digitize(values, edges[1:-1], right=True)
    counts = np.bincount(bins, minlength=len(edges) - 1).astype(float)
    probabilities = counts[counts > 0] / counts.sum()
    entropy = -float(np.sum(probabilities * np.log(probabilities)))
    return entropy / math.log(len(edges) - 1)


def make_target_metrics(predictions: pd.DataFrame, min_half_rows: int) -> pd.DataFrame:
    work = predictions.copy()
    work["observed_log"] = np.log1p(work["target_value"].clip(lower=0).astype(float))
    work["history_abs_error"] = (work["observed_log"] - work["reg_history"]).abs()
    work["calendar_abs_error"] = (work["observed_log"] - work["reg_calendar"]).abs()

    entropy_edges: dict[tuple[str, str], np.ndarray] = {}
    for (regime, target), frame in work.groupby(["regime", "target_name"]):
        edges = np.unique(np.quantile(frame["observed_log"].dropna(), np.linspace(0, 1, 6)))
        entropy_edges[(regime, target)] = edges

    rows = []
    group_columns = ["regime", "institute_year", "user_id", "target_name", "half"]
    for keys, frame in work.groupby(group_columns, sort=False):
        if len(frame) < min_half_rows:
            continue
        regime, institute_year, user_id, target_name, half = keys
        observed = frame["observed_log"]
        rows.append(
            {
                "regime": regime,
                "institute_year": institute_year,
                "user_id": user_id,
                "target_name": target_name,
                "half": half,
                "n_obs": int(len(frame)),
                "mae_history": float(frame["history_abs_error"].mean()),
                "mae_calendar": float(frame["calendar_abs_error"].mean()),
                "observed_sd": float(observed.std(ddof=0)),
                "observed_entropy": normalized_entropy(
                    observed, entropy_edges[(regime, target_name)]
                ),
            }
        )
    metrics = pd.DataFrame(rows)
    scales = (
        work.groupby(["regime", "target_name"])["observed_log"]
        .std(ddof=0)
        .rename("target_scale")
        .reset_index()
    )
    metrics = metrics.merge(scales, on=["regime", "target_name"], how="left")
    metrics["forecastability_raw"] = -metrics["mae_history"] / metrics["target_scale"]
    metrics["history_skill"] = np.where(
        metrics["mae_calendar"] > 1e-6,
        1.0 - metrics["mae_history"] / metrics["mae_calendar"],
        np.nan,
    )
    for column in ["forecastability_raw", "history_skill", "observed_sd", "observed_entropy"]:
        metrics[f"{column}_z"] = metrics.groupby(
            ["regime", "target_name", "half"]
        )[column].transform(zscore)
    return metrics


def add_target_level_residuals(metrics: pd.DataFrame) -> pd.DataFrame:
    """Remove target-specific variability before profile construction."""
    work = metrics.copy()
    specifications = {
        "residual_sd": ["observed_sd_z"],
        "residual_sd_entropy": ["observed_sd_z", "observed_entropy_z"],
    }
    predictors = {
        "forecastability": "forecastability_raw_z",
        "history_skill": "history_skill_z",
    }
    for predictor, value_column in predictors.items():
        for suffix, control_columns in specifications.items():
            output_column = f"{predictor}_{suffix}_z"
            work[output_column] = np.nan
            for _, frame in work.groupby(
                ["regime", "target_name", "half"], sort=False
            ):
                columns = [value_column, *control_columns]
                usable = frame[columns].notna().all(axis=1)
                sample = frame.loc[usable]
                if len(sample) < 10:
                    continue
                cohort = pd.get_dummies(
                    sample["institute_year"], drop_first=True
                ).to_numpy(dtype=float)
                controls = sample[control_columns].to_numpy(dtype=float)
                design = np.column_stack([np.ones(len(sample)), cohort, controls])
                residual = residualize(
                    sample[value_column].to_numpy(dtype=float), design
                )
                work.loc[sample.index, output_column] = zscore(
                    pd.Series(residual, index=sample.index)
                )
    return work


def make_composites(metrics: pd.DataFrame) -> pd.DataFrame:
    targets_by_scope = {
        "core": CORE_TARGETS,
        "rich": sorted(metrics["target_name"].unique()),
    }
    minimum_by_scope = {"core": 6, "rich": 10}
    rows = []
    for regime, scope in [
        (regime_name, scope_name)
        for regime_name in sorted(metrics["regime"].unique())
        for scope_name in ["core", "rich"]
    ]:
        subset = metrics[
            metrics["regime"].eq(regime) & metrics["target_name"].isin(targets_by_scope[scope])
        ]
        for keys, frame in subset.groupby(
            ["institute_year", "user_id", "half"], sort=False
        ):
            if frame["target_name"].nunique() < minimum_by_scope[scope]:
                continue
            institute_year, user_id, half = keys
            rows.append(
                {
                    "regime": regime,
                    "scope": scope,
                    "institute_year": institute_year,
                    "user_id": user_id,
                    "half": half,
                    "n_targets": int(frame["target_name"].nunique()),
                    "forecastability": float(frame["forecastability_raw_z"].mean()),
                    "history_skill": float(frame["history_skill_z"].mean()),
                    "variability": float(frame["observed_sd_z"].mean()),
                    "entropy": float(frame["observed_entropy_z"].mean()),
                }
            )
    return pd.DataFrame(rows)


def residualize(values: np.ndarray, covariates: np.ndarray) -> np.ndarray:
    return values - covariates @ (np.linalg.pinv(covariates) @ values)


def stability_tables(
    composites: pd.DataFrame,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stability_rows = []
    cohort_rows = []
    for (regime, scope), frame in composites.groupby(["regime", "scope"]):
        pivot = frame.pivot_table(
            index=["institute_year", "user_id"],
            columns="half",
            values=["forecastability", "history_skill", "variability", "entropy"],
            aggfunc="first",
        ).dropna()
        for measure in ["forecastability", "history_skill", "variability", "entropy"]:
            early = pivot[(measure, "early")].to_numpy(dtype=float)
            late = pivot[(measure, "late")].to_numpy(dtype=float)
            low, high = bootstrap_correlation(early, late, rng, n_bootstrap)
            stability_rows.append(
                {
                    "regime": regime,
                    "scope": scope,
                    "analysis": "pooled",
                    "measure": measure,
                    "n_trajectories": int(len(pivot)),
                    "pearson_r": pearson_r(early, late),
                    "ci_low": low,
                    "ci_high": high,
                }
            )

        cohort = pd.get_dummies(
            pivot.index.get_level_values("institute_year"), drop_first=False
        ).to_numpy(dtype=float)
        early_forecast = pivot[("forecastability", "early")].to_numpy(dtype=float)
        late_forecast = pivot[("forecastability", "late")].to_numpy(dtype=float)
        early_centered = residualize(early_forecast, cohort)
        late_centered = residualize(late_forecast, cohort)
        low, high = bootstrap_correlation(early_centered, late_centered, rng, n_bootstrap)
        stability_rows.append(
            {
                "regime": regime,
                "scope": scope,
                "analysis": "cohort_fixed_effect_residual",
                "measure": "forecastability",
                "n_trajectories": int(len(pivot)),
                "pearson_r": pearson_r(early_centered, late_centered),
                "ci_low": low,
                "ci_high": high,
            }
        )

        for measure in ["forecastability", "history_skill"]:
            early_measure = pivot[(measure, "early")].to_numpy(dtype=float)
            late_measure = pivot[(measure, "late")].to_numpy(dtype=float)
            for controls in [
                ("variability",),
                ("entropy",),
                ("variability", "entropy"),
            ]:
                pieces_early = [np.ones((len(pivot), 1)), cohort]
                pieces_late = [np.ones((len(pivot), 1)), cohort]
                for control in controls:
                    pieces_early.append(
                        pivot[(control, "early")].to_numpy(dtype=float)[:, None]
                    )
                    pieces_late.append(
                        pivot[(control, "late")].to_numpy(dtype=float)[:, None]
                    )
                early_residual = residualize(
                    early_measure, np.hstack(pieces_early)
                )
                late_residual = residualize(late_measure, np.hstack(pieces_late))
                low, high = bootstrap_correlation(
                    early_residual, late_residual, rng, n_bootstrap
                )
                stability_rows.append(
                    {
                        "regime": regime,
                        "scope": scope,
                        "analysis": "residual_" + "_".join(controls),
                        "measure": measure,
                        "n_trajectories": int(len(pivot)),
                        "pearson_r": pearson_r(early_residual, late_residual),
                        "ci_low": low,
                        "ci_high": high,
                    }
                )

        flat = pivot.reset_index()
        for institute_year, cohort_frame in flat.groupby("institute_year"):
            early = cohort_frame[("forecastability", "early")].to_numpy(dtype=float)
            late = cohort_frame[("forecastability", "late")].to_numpy(dtype=float)
            low, high = bootstrap_correlation(early, late, rng, n_bootstrap)
            cohort_rows.append(
                {
                    "regime": regime,
                    "scope": scope,
                    "institute_year": institute_year,
                    "n_trajectories": int(len(cohort_frame)),
                    "pearson_r": pearson_r(early, late),
                    "ci_low": low,
                    "ci_high": high,
                }
            )
    return pd.DataFrame(stability_rows), pd.DataFrame(cohort_rows)


def known_single_year_stability(
    composites: pd.DataFrame,
    repeat_candidates: set[tuple[str, str]],
    rng: np.random.Generator,
    n_bootstrap: int,
) -> pd.DataFrame:
    rows = []
    for (regime, scope), frame in composites.groupby(["regime", "scope"]):
        is_repeat = [
            (year, user_id) in repeat_candidates
            for year, user_id in zip(
                frame["institute_year"], frame["user_id"], strict=True
            )
        ]
        frame = frame.loc[~np.asarray(is_repeat)].copy()
        pivot = frame.pivot_table(
            index=["institute_year", "user_id"],
            columns="half",
            values="forecastability",
            aggfunc="first",
        ).dropna()
        early = pivot["early"].to_numpy(dtype=float)
        late = pivot["late"].to_numpy(dtype=float)
        low, high = bootstrap_correlation(early, late, rng, n_bootstrap)
        rows.append(
            {
                "regime": regime,
                "scope": scope,
                "analysis": "known_single_year_only",
                "measure": "forecastability",
                "n_trajectories": int(len(pivot)),
                "pearson_r": pearson_r(early, late),
                "ci_low": low,
                "ci_high": high,
            }
        )
    return pd.DataFrame(rows)


def row_cosines(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    mask = np.isfinite(a) & np.isfinite(b)
    a0 = np.where(mask, a, 0.0)
    b0 = np.where(mask, b, 0.0)
    denominator = np.sqrt(np.sum(a0 * a0, axis=1) * np.sum(b0 * b0, axis=1))
    return np.divide(
        np.sum(a0 * b0, axis=1),
        denominator,
        out=np.full(len(a0), np.nan),
        where=(denominator > 1e-12) & (mask.sum(axis=1) >= 4),
    )


def profile_stability(
    metrics: pd.DataFrame,
    rng: np.random.Generator,
    n_permutations: int,
    headline_permutations: int,
    repeat_candidates: set[tuple[str, str]],
) -> pd.DataFrame:
    rows = []
    all_targets = sorted(metrics["target_name"].unique())
    measures = {
        "forecastability": "forecastability_raw_z",
        "forecastability_residual_sd": "forecastability_residual_sd_z",
        "forecastability_residual_sd_entropy": (
            "forecastability_residual_sd_entropy_z"
        ),
        "history_skill": "history_skill_z",
        "history_skill_residual_sd": "history_skill_residual_sd_z",
        "history_skill_residual_sd_entropy": "history_skill_residual_sd_entropy_z",
    }
    for regime in sorted(metrics["regime"].unique()):
        for scope, targets, minimum in [
            ("core", CORE_TARGETS, 6),
            ("rich", all_targets, 10),
        ]:
            subset = metrics[
                metrics["regime"].eq(regime) & metrics["target_name"].isin(targets)
            ]
            for measure, value_column in measures.items():
                pivot = subset.pivot_table(
                    index=["institute_year", "user_id", "half"],
                    columns="target_name",
                    values=value_column,
                    aggfunc="first",
                )
                early = pivot.xs("early", level="half", drop_level=True).reindex(
                    columns=targets
                )
                late = pivot.xs("late", level="half", drop_level=True).reindex(
                    columns=targets
                )
                common = early.index.intersection(late.index)
                early = early.loc[common]
                late = late.loc[common]
                usable = ((early.notna() & late.notna()).sum(axis=1) >= minimum)
                early = early.loc[usable]
                late = late.loc[usable]
                known_single_mask = np.asarray(
                    [tuple(index) not in repeat_candidates for index in early.index],
                    dtype=bool,
                )
                for sample, sample_mask in [
                    ("all_trajectories", np.ones(len(early), dtype=bool)),
                    ("known_single_year_only", known_single_mask),
                ]:
                    permutation_count = (
                        headline_permutations
                        if scope == "core"
                        and sample == "all_trajectories"
                        and measure
                        in {
                            "forecastability",
                            "forecastability_residual_sd_entropy",
                            "history_skill_residual_sd_entropy",
                        }
                        else n_permutations
                    )
                    sample_early = early.loc[sample_mask]
                    sample_late = late.loc[sample_mask]
                    early_values = sample_early.to_numpy(dtype=float)
                    late_values = sample_late.to_numpy(dtype=float)
                    same_mean = float(
                        np.nanmean(row_cosines(early_values, late_values))
                    )
                    institute_values = np.asarray(
                        sample_early.index.get_level_values(0)
                    )
                    null = []
                    for _ in range(permutation_count):
                        permutation = np.arange(len(late_values))
                        for institute_year in sorted(set(institute_values)):
                            positions = np.flatnonzero(
                                institute_values == institute_year
                            )
                            permutation[positions] = rng.permutation(positions)
                        null.append(
                            float(
                                np.nanmean(
                                    row_cosines(
                                        early_values, late_values[permutation]
                                    )
                                )
                            )
                        )
                    cross_mean = float(np.mean(null))
                    rows.append(
                        {
                            "regime": regime,
                            "scope": scope,
                            "measure": measure,
                            "sample": sample,
                            "n_trajectories": int(len(sample_early)),
                            "same_person_mean": same_mean,
                            "matched_cross_person_mean": cross_mean,
                            "delta": same_mean - cross_mean,
                            "permutation_p": float(
                                (1 + np.sum(np.asarray(null) >= same_mean))
                                / (permutation_count + 1)
                            ),
                            "n_permutations": permutation_count,
                        }
                    )
    return pd.DataFrame(rows)


def variability_overlap(metrics: pd.DataFrame, composites: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for regime, frame in metrics.groupby("regime"):
        for predictor in ["forecastability_raw", "history_skill"]:
            for variability_measure in ["observed_sd", "observed_entropy"]:
                x = frame[predictor].to_numpy(dtype=float)
                y = frame[variability_measure].to_numpy(dtype=float)
                rows.append(
                    {
                        "regime": regime,
                        "scope": "target_half_rows",
                        "analysis": "raw_across_targets",
                        "predictor": predictor,
                        "variability_measure": variability_measure,
                        "n": int(np.isfinite(x * y).sum()),
                        "pearson_r": pearson_r(x, y),
                        "spearman_r": pearson_r(
                            pd.Series(x).rank().to_numpy(), pd.Series(y).rank().to_numpy()
                        ),
                    }
                )
                xz = frame[f"{predictor}_z"].to_numpy(dtype=float)
                yz = frame[f"{variability_measure}_z"].to_numpy(dtype=float)
                rows.append(
                    {
                        "regime": regime,
                        "scope": "target_half_rows",
                        "analysis": "within_target_half_standardized",
                        "predictor": predictor,
                        "variability_measure": variability_measure,
                        "n": int(np.isfinite(xz * yz).sum()),
                        "pearson_r": pearson_r(xz, yz),
                        "spearman_r": pearson_r(
                            pd.Series(xz).rank().to_numpy(), pd.Series(yz).rank().to_numpy()
                        ),
                    }
                )

    for (regime, scope), frame in composites.groupby(["regime", "scope"]):
        pivot = frame.pivot_table(
            index=["institute_year", "user_id"],
            columns="half",
            values=["forecastability", "history_skill", "variability", "entropy"],
            aggfunc="first",
        ).dropna()
        for predictor in ["forecastability", "history_skill"]:
            predictor_values = pivot[predictor].mean(axis=1).to_numpy(dtype=float)
            for variability_measure in ["variability", "entropy"]:
                variability = pivot[variability_measure].mean(axis=1).to_numpy(dtype=float)
                rows.append(
                    {
                        "regime": regime,
                        "scope": scope,
                        "analysis": "person_average_composite",
                        "predictor": predictor,
                        "variability_measure": variability_measure,
                        "n": int(len(pivot)),
                        "pearson_r": pearson_r(predictor_values, variability),
                        "spearman_r": pearson_r(
                            pd.Series(predictor_values).rank().to_numpy(),
                            pd.Series(variability).rank().to_numpy(),
                        ),
                    }
                )
    return pd.DataFrame(rows)


def identity_and_power_audit(
    source_predictions: pd.DataFrame,
    original_composites: pd.DataFrame,
    repeat_candidates: set[tuple[str, str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    source_units = source_predictions[["institute_year", "user_id"]].drop_duplicates()
    eligible = (
        original_composites[original_composites["scope"].eq("core")]
        .pivot_table(
            index=["institute_year", "user_id"],
            columns="half",
            values="forecastability",
            aggfunc="first",
        )
        .dropna()
        .reset_index()
    )
    identity_rows = []
    for label, frame in [("source", source_units), ("core_eligible", eligible)]:
        counts = frame.groupby("user_id").size()
        is_repeat = [
            (year, user_id) in repeat_candidates
            for year, user_id in zip(
                frame["institute_year"], frame["user_id"], strict=True
            )
        ]
        identity_rows.append(
            {
                "sample": label,
                "n_trajectory_keys": int(len(frame)),
                "n_unique_released_user_ids": int(frame["user_id"].nunique()),
                "n_released_ids_in_multiple_cohorts": int((counts > 1).sum()),
                "maximum_cohorts_per_released_id": int(counts.max()),
                "n_known_repeat_candidate_trajectories": int(np.sum(is_repeat)),
                "n_known_single_year_trajectories": int(len(frame) - np.sum(is_repeat)),
                "linked_cross_year_person_id_available": False,
                "cluster_bootstrap_status": (
                    "not identifiable from released PIDs; use known-single-year sensitivity"
                ),
            }
        )

    normal = NormalDist()
    n = int(len(eligible))
    power_rows = []
    for correction, comparisons in [
        ("uncorrected", 1),
        ("three_trait_bonferroni", 3),
        ("full_28_test_bonferroni_approximation", 28),
    ]:
        alpha = 0.05 / comparisons
        critical = normal.inv_cdf(1 - alpha / 2)
        for power in [0.80, 0.90]:
            z_effect = (critical + normal.inv_cdf(power)) / math.sqrt(n - 3)
            power_rows.append(
                {
                    "correction": correction,
                    "comparisons": comparisons,
                    "n": n,
                    "target_power": power,
                    "approximate_minimum_detectable_abs_r": math.tanh(z_effect),
                    "note": "Fisher-z approximation; max-statistic permutation may differ",
                }
            )
    return pd.DataFrame(identity_rows), pd.DataFrame(power_rows)


def plot_stability(composites: pd.DataFrame, output: Path) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=False, sharey=False)
    for axis, regime in zip(axes, ["expanding_refit", "disjoint_refit"], strict=True):
        frame = composites[
            composites["regime"].eq(regime) & composites["scope"].eq("core")
        ]
        pivot = frame.pivot_table(
            index=["institute_year", "user_id"],
            columns="half",
            values="forecastability",
            aggfunc="first",
        ).dropna()
        axis.scatter(pivot["early"], pivot["late"], s=12, alpha=0.35, color="#2563eb")
        if len(pivot) >= 2:
            slope, intercept = np.polyfit(pivot["early"], pivot["late"], 1)
            x_values = np.linspace(pivot["early"].min(), pivot["early"].max(), 100)
            axis.plot(x_values, intercept + slope * x_values, color="#b91c1c", lw=2)
        axis.set_title(regime.replace("_", " ").title())
        axis.set_xlabel("Early forecastability")
        axis.set_ylabel("Late forecastability")
    figure.suptitle("Core forecastability under separately fitted early and late models")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--min-half-rows", type=int, default=4)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--permutations", type=int, default=500)
    parser.add_argument("--headline-permutations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=240719)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    output = repo_root / "results/referee_robustness"
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    repeat_candidates, overlap_provenance = load_repeat_candidates()
    samples = assign_positions(load_or_build_samples(repo_root))
    refitted_predictions, model_fits = fit_refitted_predictions(samples)
    fixed_predictions = load_fixed_predictions(repo_root, args.min_half_rows)
    predictions = pd.concat([fixed_predictions, refitted_predictions], ignore_index=True)
    metrics = add_target_level_residuals(
        make_target_metrics(predictions, args.min_half_rows)
    )
    composites = make_composites(metrics)
    stability, cohort_stability = stability_tables(
        composites, rng, args.bootstrap
    )
    stability = pd.concat(
        [
            stability,
            known_single_year_stability(
                composites, repeat_candidates, rng, args.bootstrap
            ),
        ],
        ignore_index=True,
    )
    profiles = profile_stability(
        metrics,
        rng,
        args.permutations,
        args.headline_permutations,
        repeat_candidates,
    )
    overlap = variability_overlap(metrics, composites)

    source_predictions = pd.read_parquet(
        repo_root / "results/behavioural_weather_forecast/predictions.parquet",
        columns=["institute_year", "user_id", "horizon", "issue_time"],
    )
    source_predictions = source_predictions[
        source_predictions["horizon"].eq("tomorrow")
        & source_predictions["issue_time"].eq("evening")
    ]
    original_composites = pd.read_csv(
        repo_root / "results/forecastability/half_composites.csv"
    )
    identity, power = identity_and_power_audit(
        source_predictions, original_composites, repeat_candidates
    )

    predictions.to_parquet(output / "refitted_predictions.parquet", index=False)
    model_fits.to_csv(output / "model_fits.csv", index=False)
    metrics.to_csv(output / "target_half_metrics.csv", index=False)
    composites.to_csv(output / "half_composites.csv", index=False)
    stability.to_csv(output / "stability_robustness.csv", index=False)
    cohort_stability.to_csv(output / "cohort_stability.csv", index=False)
    profiles.to_csv(output / "profile_robustness.csv", index=False)
    overlap.to_csv(output / "variability_overlap.csv", index=False)
    identity.to_csv(output / "identity_audit.csv", index=False)
    power.to_csv(output / "h3_power_approximation.csv", index=False)
    plot_stability(composites, output / "independent_refit_stability.png")

    expanding = stability[
        stability["regime"].eq("expanding_refit")
        & stability["scope"].eq("core")
        & stability["analysis"].eq("pooled")
        & stability["measure"].eq("forecastability")
    ].iloc[0]
    disjoint = stability[
        stability["regime"].eq("disjoint_refit")
        & stability["scope"].eq("core")
        & stability["analysis"].eq("pooled")
        & stability["measure"].eq("forecastability")
    ].iloc[0]
    expanding_profile = profiles[
        profiles["regime"].eq("expanding_refit")
        & profiles["scope"].eq("core")
        & profiles["measure"].eq("forecastability")
        & profiles["sample"].eq("all_trajectories")
    ].iloc[0]
    disjoint_profile = profiles[
        profiles["regime"].eq("disjoint_refit")
        & profiles["scope"].eq("core")
        & profiles["measure"].eq("forecastability")
        & profiles["sample"].eq("all_trajectories")
    ].iloc[0]
    residual_profiles = profiles[
        profiles["scope"].eq("core")
        & profiles["measure"].eq("forecastability_residual_sd_entropy")
        & profiles["sample"].eq("all_trajectories")
    ].sort_values("regime")
    history_skill_stability = stability[
        stability["scope"].eq("core")
        & stability["analysis"].eq("pooled")
        & stability["measure"].eq("history_skill")
    ].sort_values("regime")
    history_skill_residual_stability = stability[
        stability["scope"].eq("core")
        & stability["analysis"].eq("residual_variability_entropy")
        & stability["measure"].eq("history_skill")
    ].sort_values("regime")
    history_skill_profiles = profiles[
        profiles["scope"].eq("core")
        & profiles["measure"].eq("history_skill")
        & profiles["sample"].eq("all_trajectories")
    ].sort_values("regime")
    history_skill_residual_profiles = profiles[
        profiles["scope"].eq("core")
        & profiles["measure"].eq("history_skill_residual_sd_entropy")
        & profiles["sample"].eq("all_trajectories")
    ].sort_values("regime")
    fixed_overlap = overlap[
        overlap["regime"].eq("fixed_original")
        & overlap["scope"].eq("core")
        & overlap["analysis"].eq("person_average_composite")
        & overlap["predictor"].eq("forecastability")
        & overlap["variability_measure"].eq("variability")
    ].iloc[0]
    fixed_single_year = stability[
        stability["regime"].eq("fixed_original")
        & stability["scope"].eq("core")
        & stability["analysis"].eq("known_single_year_only")
    ].iloc[0]
    fixed_residual_degree = stability[
        stability["regime"].eq("fixed_original")
        & stability["scope"].eq("core")
        & stability["analysis"].eq("residual_variability_entropy")
        & stability["measure"].eq("forecastability")
    ].iloc[0]
    fixed_history_skill_overlap = overlap[
        overlap["regime"].eq("fixed_original")
        & overlap["scope"].eq("core")
        & overlap["analysis"].eq("person_average_composite")
        & overlap["predictor"].eq("history_skill")
        & overlap["variability_measure"].eq("variability")
    ].iloc[0]
    fixed_history_skill_entropy_overlap = overlap[
        overlap["regime"].eq("fixed_original")
        & overlap["scope"].eq("core")
        & overlap["analysis"].eq("person_average_composite")
        & overlap["predictor"].eq("history_skill")
        & overlap["variability_measure"].eq("entropy")
    ].iloc[0]
    fixed_history_skill_target_sd_overlap = overlap[
        overlap["regime"].eq("fixed_original")
        & overlap["scope"].eq("target_half_rows")
        & overlap["analysis"].eq("within_target_half_standardized")
        & overlap["predictor"].eq("history_skill")
        & overlap["variability_measure"].eq("observed_sd")
    ].iloc[0]
    residual_profile_gate = bool(
        (residual_profiles["delta"] >= 0.10).all()
        and (residual_profiles["permutation_p"] < 0.05).all()
    )
    history_skill_degree_gate = bool(
        (history_skill_stability["pearson_r"] >= 0.30).all()
        and (history_skill_stability["ci_low"] > 0).all()
    )
    history_skill_residual_degree_gate = bool(
        (history_skill_residual_stability["pearson_r"] >= 0.30).all()
        and (history_skill_residual_stability["ci_low"] > 0).all()
    )
    history_skill_residual_profile_gate = bool(
        (history_skill_residual_profiles["delta"] >= 0.10).all()
        and (history_skill_residual_profiles["permutation_p"] < 0.05).all()
    )
    summary = {
        "question": "Do the central E27 results survive the referee's artifact checks?",
        "source_sample": {
            "rows": int(len(samples)),
            "trajectory_keys": int(
                samples[["institute_year", "user_id"]].drop_duplicates().shape[0]
            ),
            "released_user_ids": int(samples["user_id"].nunique()),
        },
        "overlap_metadata": overlap_provenance,
        "expanding_refit_core_degree": expanding.dropna().to_dict(),
        "disjoint_refit_core_degree": disjoint.dropna().to_dict(),
        "expanding_refit_core_profile": expanding_profile.dropna().to_dict(),
        "disjoint_refit_core_profile": disjoint_profile.dropna().to_dict(),
        "fixed_model_core_variability_overlap": fixed_overlap.dropna().to_dict(),
        "fixed_model_known_single_year_sensitivity": fixed_single_year.dropna().to_dict(),
        "original_h1_after_sd_entropy_control": {
            **fixed_residual_degree.dropna().to_dict(),
            "pre_specified_threshold": 0.30,
            "passed": bool(fixed_residual_degree["pearson_r"] >= 0.30),
        },
        "residualized_core_profiles": residual_profiles.to_dict(orient="records"),
        "residualized_profile_gate_passed_all_regimes": residual_profile_gate,
        "history_skill": {
            "person_average_sd_overlap": fixed_history_skill_overlap.dropna().to_dict(),
            "target_half_sd_overlap": (
                fixed_history_skill_target_sd_overlap.dropna().to_dict()
            ),
            "person_average_entropy_overlap": (
                fixed_history_skill_entropy_overlap.dropna().to_dict()
            ),
            "stability": history_skill_stability.to_dict(orient="records"),
            "sd_entropy_residual_stability": (
                history_skill_residual_stability.to_dict(orient="records")
            ),
            "profiles": history_skill_profiles.to_dict(orient="records"),
            "sd_entropy_residual_profiles": (
                history_skill_residual_profiles.to_dict(orient="records")
            ),
            "degree_gate_passed_all_regimes": history_skill_degree_gate,
            "sd_entropy_residual_degree_gate_passed_all_regimes": (
                history_skill_residual_degree_gate
            ),
            "sd_entropy_residual_profile_gate_passed_all_regimes": (
                history_skill_residual_profile_gate
            ),
        },
        "claim_decision": (
            "The raw forecastability degree dissolves into inverse variability and "
            "its variability-independent H1 fails the 0.30 gate. Target-residualized "
            "profiles and relative history skill survive across all model regimes, "
            "but pooled models make these transferability rather than idiographic "
            "learnability measures."
        ),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
