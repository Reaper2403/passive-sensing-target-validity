from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

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

TRAIT_TARGETS = [
    "baseline_depression",
    "baseline_anxiety",
    "baseline_stress",
    "baseline_conscientiousness",
    "baseline_extraversion",
    "baseline_loneliness",
    "baseline_social_fit",
]

FORECAST_CONSTRUCTS = [
    "core_forecastability",
    "rich_forecastability",
    "core_history_skill",
    "rich_history_skill",
]


def coalesce(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    available = [column for column in columns if column in df.columns]
    if not available:
        return pd.Series(np.nan, index=df.index)
    out = df[available[0]].copy()
    for column in available[1:]:
        out = out.combine_first(df[column])
    return out


def load_baseline(raw_root: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(raw_root.glob("INS-W_*/SurveyData/pre.csv")):
        frame = pd.read_csv(path)
        frame["institute_year"] = path.parts[-3]
        frames.append(frame)
    survey = pd.concat(frames, ignore_index=True).rename(columns={"pid": "user_id"})
    survey["baseline_depression"] = coalesce(
        survey, ["BDI2_PRE", "CESD_10items_PRE", "CESD_9items_PRE"]
    )
    survey["baseline_anxiety"] = coalesce(survey, ["STAI_PRE", "STAIS_PRE"])
    survey["baseline_stress"] = coalesce(
        survey, ["PSS_14items_PRE", "PSS_10items_PRE", "CHIPS_PRE"]
    )
    survey["baseline_conscientiousness"] = coalesce(
        survey, ["BFI10_conscientiousness_PRE"]
    )
    survey["baseline_extraversion"] = coalesce(survey, ["BFI10_extroversion_PRE"])
    survey["baseline_loneliness"] = coalesce(survey, ["UCLA_10items_PRE"])
    survey["baseline_social_fit"] = coalesce(survey, ["SocialFit_PRE"])

    platform_frames = []
    for path in sorted(raw_root.glob("INS-W_*/ParticipantsInfoData/platform.csv")):
        frame = pd.read_csv(path).rename(columns={"pid": "user_id"})
        frame["institute_year"] = path.parts[-3]
        platform_frames.append(frame[["institute_year", "user_id", "platform"]])
    if platform_frames:
        survey = survey.merge(
            pd.concat(platform_frames, ignore_index=True),
            on=["institute_year", "user_id"],
            how="left",
        )
    else:
        survey["platform"] = "missing"
    keep = ["institute_year", "user_id", "platform", *TRAIT_TARGETS]
    return survey[keep].drop_duplicates(["institute_year", "user_id"])


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


def fisher_p(r: float, n: int) -> float:
    if not np.isfinite(r) or n < 5:
        return np.nan
    z = np.arctanh(np.clip(r, -0.999999, 0.999999)) * np.sqrt(max(n - 3, 1))
    return float(math.erfc(abs(z) / math.sqrt(2.0)))


def bh_fdr(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    out = np.full(values.shape, np.nan)
    mask = np.isfinite(values)
    valid = values[mask]
    if not len(valid):
        return out
    order = np.argsort(valid)
    ranked = valid[order]
    adjusted = ranked * len(ranked) / (np.arange(len(ranked)) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    restored = np.empty_like(valid)
    restored[order] = np.clip(adjusted, 0, 1)
    out[mask] = restored
    return out


def bootstrap_correlation(
    x: np.ndarray, y: np.ndarray, rng: np.random.Generator, n_bootstrap: int
) -> tuple[float, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 10:
        return np.nan, np.nan
    values = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(x), len(x))
        value = pearson_r(x[idx], y[idx])
        if np.isfinite(value):
            values.append(value)
    if not values:
        return np.nan, np.nan
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def zscore_group(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if not np.isfinite(std) or std <= 1e-12:
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std


def make_target_half_metrics(predictions: pd.DataFrame, min_half_rows: int) -> pd.DataFrame:
    work = predictions.copy()
    work["date"] = pd.to_datetime(work["date"])
    work["observed_log"] = np.log1p(work["target_value"].clip(lower=0).astype(float))
    work["history_abs_error"] = (
        work["observed_log"] - work["reg__history_plus_fingerprint"]
    ).abs()
    work["calendar_abs_error"] = (work["observed_log"] - work["reg__calendar_only"]).abs()
    work = work.sort_values(["institute_year", "user_id", "target_name", "date"])
    group = work.groupby(["institute_year", "user_id", "target_name"], sort=False)
    work["within_target_rank"] = group.cumcount()
    work["within_target_n"] = group["date"].transform("size")
    work = work[work["within_target_n"] >= 2 * min_half_rows].copy()
    work["half"] = np.where(
        work["within_target_rank"] < (work["within_target_n"] / 2), "early", "late"
    )

    scales = work.groupby("target_name")["observed_log"].std(ddof=0).rename("target_scale")
    metrics = (
        work.groupby(["institute_year", "user_id", "target_name", "half"], sort=False)
        .agg(
            n_obs=("date", "size"),
            mae_history=("history_abs_error", "mean"),
            mae_calendar=("calendar_abs_error", "mean"),
            first_date=("date", "min"),
            last_date=("date", "max"),
        )
        .reset_index()
        .merge(scales, on="target_name", how="left")
    )
    metrics = metrics[metrics["n_obs"] >= min_half_rows].copy()
    metrics["forecastability_raw"] = -metrics["mae_history"] / metrics["target_scale"]
    metrics["history_skill"] = np.where(
        metrics["mae_calendar"] > 1e-6,
        1.0 - metrics["mae_history"] / metrics["mae_calendar"],
        np.nan,
    )
    for column in ["forecastability_raw", "history_skill"]:
        metrics[f"{column}_z"] = metrics.groupby(["target_name", "half"])[column].transform(
            zscore_group
        )
    return metrics


def make_composites(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_targets = sorted(metrics["target_name"].unique())
    scopes = {"core": CORE_TARGETS, "rich": all_targets}
    minimums = {"core": 6, "rich": 10}
    rows = []
    for scope, targets in scopes.items():
        subset = metrics[metrics["target_name"].isin(targets)]
        for (institute_year, user_id, half), frame in subset.groupby(
            ["institute_year", "user_id", "half"], sort=False
        ):
            if frame["target_name"].nunique() < minimums[scope]:
                continue
            rows.append(
                {
                    "institute_year": institute_year,
                    "user_id": user_id,
                    "half": half,
                    "scope": scope,
                    "n_targets": int(frame["target_name"].nunique()),
                    "forecastability": float(frame["forecastability_raw_z"].mean()),
                    "history_skill": float(frame["history_skill_z"].mean()),
                }
            )
    half_composites = pd.DataFrame(rows)
    person_rows = []
    for (institute_year, user_id, scope), frame in half_composites.groupby(
        ["institute_year", "user_id", "scope"], sort=False
    ):
        if set(frame["half"]) != {"early", "late"}:
            continue
        person_rows.append(
            {
                "institute_year": institute_year,
                "user_id": user_id,
                f"{scope}_forecastability": float(frame["forecastability"].mean()),
                f"{scope}_history_skill": float(frame["history_skill"].mean()),
            }
        )
    person = pd.DataFrame(person_rows)
    if not person.empty:
        person = person.groupby(["institute_year", "user_id"], as_index=False).first()
    return half_composites, person


def behavior_profile_covariates(predictions: pd.DataFrame, n_components: int = 3) -> pd.DataFrame:
    work = predictions[["institute_year", "user_id", "target_name", "target_value"]].copy()
    work["observed_log"] = np.log1p(work["target_value"].clip(lower=0).astype(float))
    stats = (
        work.groupby(["institute_year", "user_id", "target_name"], sort=False)["observed_log"]
        .agg(["mean", "std"])
        .reset_index()
    )
    base_index = stats[["institute_year", "user_id"]].drop_duplicates().set_index(
        ["institute_year", "user_id"]
    )
    output = base_index.copy()
    for statistic, prefix in [("mean", "behavior_level"), ("std", "behavior_variability")]:
        pivot = stats.pivot_table(
            index=["institute_year", "user_id"],
            columns="target_name",
            values=statistic,
            aggfunc="first",
        ).reindex(base_index.index)
        standardized = pivot.apply(zscore_group, axis=0).fillna(0.0)
        values = standardized.to_numpy(dtype=float)
        _, _, vt = np.linalg.svd(values, full_matrices=False)
        scores = values @ vt[:n_components].T
        for component in range(min(n_components, scores.shape[1])):
            output[f"{prefix}_pc{component + 1}"] = scores[:, component]
    return output.reset_index()


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < 4:
        return np.nan
    a = a[mask]
    b = b[mask]
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator <= 1e-12:
        return np.nan
    return float(np.dot(a, b) / denominator)


def profile_similarity(
    metrics: pd.DataFrame,
    scope: str,
    value_column: str,
    rng: np.random.Generator,
    n_permutations: int,
) -> dict[str, float | int | str]:
    targets = CORE_TARGETS if scope == "core" else sorted(metrics["target_name"].unique())
    minimum = 6 if scope == "core" else 10
    subset = metrics[metrics["target_name"].isin(targets)]
    pivot = subset.pivot_table(
        index=["institute_year", "user_id", "half"],
        columns="target_name",
        values=value_column,
        aggfunc="first",
    )
    early = pivot.xs("early", level="half", drop_level=True)
    late = pivot.xs("late", level="half", drop_level=True)
    common = early.index.intersection(late.index)
    early = early.loc[common, targets]
    late = late.loc[common, targets]
    usable = ((early.notna() & late.notna()).sum(axis=1) >= minimum)
    early = early.loc[usable]
    late = late.loc[usable]

    early_values = early.to_numpy(dtype=float)
    late_values = late.to_numpy(dtype=float)

    def row_cosines(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        mask = np.isfinite(a) & np.isfinite(b)
        a0 = np.where(mask, a, 0.0)
        b0 = np.where(mask, b, 0.0)
        denominator = np.sqrt(np.sum(a0 * a0, axis=1) * np.sum(b0 * b0, axis=1))
        values = np.divide(
            np.sum(a0 * b0, axis=1),
            denominator,
            out=np.full(len(a0), np.nan),
            where=(denominator > 1e-12) & (mask.sum(axis=1) >= 4),
        )
        return values

    same_values = row_cosines(early_values, late_values)
    same_mean = float(np.nanmean(same_values))

    cross_means = []
    institute_values = np.asarray(early.index.get_level_values(0))
    for _ in range(n_permutations):
        permutation = np.arange(len(late_values))
        for institute_year in sorted(set(institute_values)):
            positions = np.flatnonzero(institute_values == institute_year)
            permutation[positions] = rng.permutation(positions)
        cross_means.append(
            float(np.nanmean(row_cosines(early_values, late_values[permutation])))
        )
    cross_mean = float(np.mean(cross_means))
    p_value = float((1 + np.sum(np.asarray(cross_means) >= same_mean)) / (n_permutations + 1))
    return {
        "test": "profile_cosine",
        "scope": scope,
        "measure": value_column,
        "n_users": int(len(early)),
        "same_user_mean": same_mean,
        "matched_cross_user_mean": cross_mean,
        "delta": same_mean - cross_mean,
        "permutation_p": p_value,
    }


def composite_stability(
    half_composites: pd.DataFrame,
    scope: str,
    measure: str,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> dict[str, float | int | str]:
    subset = half_composites[half_composites["scope"].eq(scope)]
    pivot = subset.pivot_table(
        index=["institute_year", "user_id"], columns="half", values=measure, aggfunc="first"
    ).dropna(subset=["early", "late"])
    r = pearson_r(pivot["early"].to_numpy(), pivot["late"].to_numpy())
    low, high = bootstrap_correlation(
        pivot["early"].to_numpy(), pivot["late"].to_numpy(), rng, n_bootstrap
    )
    return {
        "test": "composite_split_half",
        "scope": scope,
        "measure": measure,
        "n_users": int(len(pivot)),
        "pearson_r": r,
        "bootstrap_ci_low": low,
        "bootstrap_ci_high": high,
    }


def design_matrix(df: pd.DataFrame, numeric: list[str]) -> np.ndarray:
    categorical = pd.get_dummies(
        df[["institute_year", "platform"]].fillna("missing"), drop_first=False
    ).astype(float)
    pieces = [np.ones((len(df), 1)), categorical.to_numpy(dtype=float)]
    for column in numeric:
        values = pd.to_numeric(df[column], errors="coerce")
        values = values.fillna(values.median())
        std = values.std(ddof=0)
        if np.isfinite(std) and std > 1e-12:
            pieces.append(((values - values.mean()) / std).to_numpy()[:, None])
    return np.hstack(pieces)


def residualize(values: np.ndarray, design: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values - design @ (np.linalg.pinv(design) @ values)


def trait_associations(
    person: pd.DataFrame,
    rng: np.random.Generator,
    n_bootstrap: int,
    n_permutations: int,
    control_set: str,
    numeric_covariates: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    cache: dict[str, tuple[pd.DataFrame, np.ndarray, dict[str, np.ndarray]]] = {}
    for target in TRAIT_TARGETS:
        columns = [
            "institute_year",
            "platform",
            target,
            *numeric_covariates,
            *FORECAST_CONSTRUCTS,
        ]
        sample = person[columns].dropna(subset=[target, *FORECAST_CONSTRUCTS]).copy()
        design = design_matrix(sample, numeric_covariates)
        y = pd.to_numeric(sample[target], errors="coerce").to_numpy(dtype=float)
        y_residual = residualize(y, design)
        x_residuals = {}
        for construct in FORECAST_CONSTRUCTS:
            x = pd.to_numeric(sample[construct], errors="coerce").to_numpy(dtype=float)
            x_residual = residualize(x, design)
            x_residuals[construct] = x_residual
            r = pearson_r(x_residual, y_residual)
            low, high = bootstrap_correlation(x_residual, y_residual, rng, n_bootstrap)
            rows.append(
                {
                    "control_set": control_set,
                    "construct": construct,
                    "target": target,
                    "n": int(len(sample)),
                    "partial_r": r,
                    "p_fisher": fisher_p(r, len(sample)),
                    "bootstrap_ci_low": low,
                    "bootstrap_ci_high": high,
                }
            )
        cache[target] = (sample, y_residual, x_residuals)

    associations = pd.DataFrame(rows)
    associations["q_fdr_all"] = bh_fdr(associations["p_fisher"].to_numpy())
    associations["bonferroni_p_all"] = np.minimum(
        associations["p_fisher"] * associations["p_fisher"].notna().sum(), 1.0
    )

    null_rows = []
    max_values = []
    for permutation in range(n_permutations):
        permutation_rs = []
        for target, (sample, y_residual, x_residuals) in cache.items():
            shuffled_y = y_residual.copy()
            strata = sample.groupby(["institute_year", "platform"], dropna=False).indices
            for positions in strata.values():
                positions = np.asarray(positions)
                shuffled_y[positions] = rng.permutation(shuffled_y[positions])
            for construct, x_residual in x_residuals.items():
                value = pearson_r(x_residual, shuffled_y)
                permutation_rs.append(abs(value))
        finite_rs = np.asarray(permutation_rs)[np.isfinite(permutation_rs)]
        if not len(finite_rs):
            raise RuntimeError("No finite trait-permutation correlations were produced")
        max_value = float(np.max(finite_rs))
        max_values.append(max_value)
        null_rows.append(
            {
                "control_set": control_set,
                "permutation": permutation,
                "max_abs_partial_r": max_value,
            }
        )
    max_values_array = np.asarray(max_values)
    associations["max_stat_permutation_p"] = [
        float((1 + np.sum(max_values_array >= abs(value))) / (n_permutations + 1))
        for value in associations["partial_r"]
    ]
    return associations.sort_values("p_fisher"), pd.DataFrame(null_rows)


def plot_results(
    half_composites: pd.DataFrame, associations: pd.DataFrame, out_dir: Path
) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    outputs = []
    core = half_composites[half_composites["scope"].eq("core")]
    pivot = core.pivot_table(
        index=["institute_year", "user_id"],
        columns="half",
        values="forecastability",
        aggfunc="first",
    ).dropna()
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(pivot["early"], pivot["late"], s=12, alpha=0.35, color="#2563eb")
    if len(pivot) >= 2:
        slope, intercept = np.polyfit(pivot["early"], pivot["late"], 1)
        xs = np.linspace(pivot["early"].min(), pivot["early"].max(), 100)
        ax.plot(xs, intercept + slope * xs, color="#b91c1c", lw=2)
    ax.set_xlabel("Early prospective forecastability")
    ax.set_ylabel("Late prospective forecastability")
    ax.set_title("Stability of core routine forecastability")
    fig.tight_layout()
    path = out_dir / "core_forecastability_stability.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    outputs.append(str(path))

    trait = associations[
        associations["construct"].eq("core_forecastability")
        & associations["control_set"].eq("level_variability_controlled")
    ].copy()
    trait = trait.sort_values("partial_r")
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = trait["target"].str.replace("baseline_", "", regex=False)
    colors = ["#0f766e" if value >= 0 else "#b45309" for value in trait["partial_r"]]
    ax.barh(labels, trait["partial_r"], color=colors)
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("Fully controlled partial correlation")
    ax.set_title("Routine forecastability and baseline constructs")
    fig.tight_layout()
    path = out_dir / "core_forecastability_trait_associations.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    outputs.append(str(path))
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--min-half-rows", type=int, default=4)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--permutations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    out_dir = repo_root / "results" / "forecastability"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    prediction_path = repo_root / "results" / "behavioural_weather_forecast" / "predictions.parquet"
    columns = [
        "institute_year",
        "user_id",
        "date",
        "target_name",
        "horizon",
        "issue_time",
        "target_value",
        "reg__calendar_only",
        "reg__history_plus_fingerprint",
    ]
    predictions = pd.read_parquet(prediction_path, columns=columns)
    predictions = predictions[
        predictions["horizon"].eq("tomorrow") & predictions["issue_time"].eq("evening")
    ].copy()
    metrics = make_target_half_metrics(predictions, args.min_half_rows)
    half_composites, person = make_composites(metrics)

    coverage = (
        predictions.groupby(["institute_year", "user_id"], sort=False)
        .agg(
            n_prediction_rows=("date", "size"),
            n_targets_observed=("target_name", "nunique"),
            n_unique_dates=("date", "nunique"),
            first_date=("date", "min"),
            last_date=("date", "max"),
        )
        .reset_index()
    )
    coverage["date_span_days"] = (
        pd.to_datetime(coverage["last_date"]) - pd.to_datetime(coverage["first_date"])
    ).dt.days + 1
    coverage["mean_rows_per_target"] = (
        coverage["n_prediction_rows"] / coverage["n_targets_observed"].clip(lower=1)
    )
    person = person.merge(coverage, on=["institute_year", "user_id"], how="left")
    profile_covariates = behavior_profile_covariates(predictions)
    person = person.merge(
        profile_covariates, on=["institute_year", "user_id"], how="left"
    )
    baseline = load_baseline(repo_root / "data" / "raw" / "globem" / "1.1")
    person = person.merge(baseline, on=["institute_year", "user_id"], how="left")

    stability_rows = []
    for scope in ["core", "rich"]:
        for measure in ["forecastability", "history_skill"]:
            stability_rows.append(
                composite_stability(half_composites, scope, measure, rng, args.bootstrap)
            )
            value_column = (
                "forecastability_raw_z" if measure == "forecastability" else "history_skill_z"
            )
            stability_rows.append(
                profile_similarity(metrics, scope, value_column, rng, args.permutations)
            )
    stability = pd.DataFrame(stability_rows)
    coverage_covariates = [
        "n_prediction_rows",
        "n_targets_observed",
        "n_unique_dates",
        "date_span_days",
        "mean_rows_per_target",
    ]
    level_covariates = [
        *coverage_covariates,
        "behavior_level_pc1",
        "behavior_level_pc2",
        "behavior_level_pc3",
    ]
    strict_covariates = [
        *level_covariates,
        "behavior_variability_pc1",
        "behavior_variability_pc2",
        "behavior_variability_pc3",
    ]
    level_associations, level_null = trait_associations(
        person,
        rng,
        args.bootstrap,
        args.permutations,
        "level_controlled",
        level_covariates,
    )
    strict_associations, strict_null = trait_associations(
        person,
        rng,
        args.bootstrap,
        args.permutations,
        "level_variability_controlled",
        strict_covariates,
    )
    associations = pd.concat([level_associations, strict_associations], ignore_index=True)
    permutation_null = pd.concat([level_null, strict_null], ignore_index=True)

    metrics.to_csv(out_dir / "per_user_target_half_metrics.csv", index=False)
    half_composites.to_csv(out_dir / "half_composites.csv", index=False)
    person.to_csv(out_dir / "per_user_forecastability.csv", index=False)
    stability.to_csv(out_dir / "stability_metrics.csv", index=False)
    associations.to_csv(out_dir / "trait_associations.csv", index=False)
    permutation_null.to_csv(out_dir / "trait_permutation_null.csv", index=False)
    figures = plot_results(half_composites, associations, out_dir)

    h1 = stability[
        stability["test"].eq("composite_split_half")
        & stability["scope"].eq("core")
        & stability["measure"].eq("forecastability")
    ].iloc[0]
    h2 = stability[
        stability["test"].eq("profile_cosine")
        & stability["scope"].eq("core")
        & stability["measure"].eq("forecastability_raw_z")
    ].iloc[0]
    h1_pass = bool(h1["pearson_r"] >= 0.30 and h1["bootstrap_ci_low"] > 0)
    h2_pass = bool(h2["delta"] >= 0.10 and h2["permutation_p"] < 0.05)
    h3_trait_targets = {
        "baseline_conscientiousness",
        "baseline_extraversion",
        "baseline_social_fit",
    }
    h3_rows = associations[
        associations["control_set"].eq("level_variability_controlled")
        & associations["target"].isin(h3_trait_targets)
        & (associations["partial_r"].abs() >= 0.10)
        & (associations["max_stat_permutation_p"] < 0.05)
    ]
    if h1_pass and h2_pass:
        decision = "stable_forecastability_phenotype_supported"
    elif h1_pass:
        decision = "stable_degree_only"
    else:
        decision = "forecastability_phenotype_not_supported"
    summary = {
        "contract": "experiments/README.md",
        "source_predictions": str(prediction_path.relative_to(repo_root)),
        "n_prediction_rows_input": int(len(predictions)),
        "n_users_input": int(predictions["user_id"].nunique()),
        "n_users_with_person_score": int(person["user_id"].nunique()),
        "n_targets": int(predictions["target_name"].nunique()),
        "h1_stable_degree": {"passed": h1_pass, **h1.dropna().to_dict()},
        "h2_stable_profile": {"passed": h2_pass, **h2.dropna().to_dict()},
        "h3_corrected_trait_associations": {
            "passed": bool(len(h3_rows)),
            "n": int(len(h3_rows)),
            "rows": h3_rows.to_dict(orient="records"),
        },
        "decision_scope": "historical_e27_contract_gates",
        "historical_contract_decision": decision,
        "historical_contract_novelty_gate_passed": bool(h1_pass and h2_pass),
        "current_interpretation_document": (
            "experiments/README.md"
        ),
        "figures": [str(Path(path).relative_to(repo_root)) for path in figures],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
