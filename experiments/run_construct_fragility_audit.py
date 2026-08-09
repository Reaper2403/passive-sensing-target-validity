from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


PROCESSED_TARGETS = ["target_dep_weekly", "feel_depressed", "feel_anxious", "phq4"]
EMA_TARGETS = [
    "phq4_EMA",
    "phq4_anxiety_EMA",
    "phq4_depression_EMA",
    "pss4_EMA",
    "positive_affect_EMA",
    "negative_affect_EMA",
]


def zscore(x: pd.Series) -> pd.Series:
    s = pd.to_numeric(x, errors="coerce")
    std = s.std(ddof=0)
    if not np.isfinite(std) or std <= 1e-12:
        return s * np.nan
    return (s - s.mean()) / std


def pearson(x: pd.Series, y: pd.Series) -> tuple[float, int]:
    xx = pd.to_numeric(x, errors="coerce")
    yy = pd.to_numeric(y, errors="coerce")
    mask = xx.notna() & yy.notna()
    n = int(mask.sum())
    if n < 10:
        return np.nan, n
    return float(np.corrcoef(xx[mask], yy[mask])[0, 1]), n


def load_ema(raw_root: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(raw_root.glob("INS-W_*/SurveyData/ema.csv")):
        institute_year = path.parts[-3]
        ema = pd.read_csv(path)
        ema = ema.rename(columns={"pid": "user_id"})
        ema["institute_year"] = institute_year
        ema["date"] = pd.to_datetime(ema["date"])
        keep = ["institute_year", "user_id", "date", *[c for c in EMA_TARGETS if c in ema.columns]]
        frames.append(ema[keep])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_repeated_labels(processed: Path, raw_root: Path) -> pd.DataFrame:
    daily = pd.read_parquet(processed)
    daily["date"] = pd.to_datetime(daily["date"])
    cols = ["institute_year", "user_id", "date", *[c for c in PROCESSED_TARGETS if c in daily.columns]]
    labels = daily[cols].copy()
    if "target_dep_weekly" in labels.columns:
        labels["target_dep_weekly"] = labels["target_dep_weekly"].map(
            {True: 1.0, False: 0.0, "True": 1.0, "False": 0.0}
        )
    ema = load_ema(raw_root)
    if not ema.empty:
        labels = labels.merge(ema, on=["institute_year", "user_id", "date"], how="outer")
    labels = labels.sort_values(["institute_year", "user_id", "date"]).reset_index(drop=True)
    return labels


def same_day_correlations(labels: pd.DataFrame, targets: list[str]) -> pd.DataFrame:
    rows = []
    for i, a in enumerate(targets):
        for b in targets[i + 1 :]:
            r, n = pearson(labels[a], labels[b])
            rows.append({"target_a": a, "target_b": b, "n_overlap": n, "pearson_r": r, "abs_r": abs(r)})
    return pd.DataFrame(rows).sort_values(["abs_r", "n_overlap"], ascending=[False, False])


def add_prior_columns(labels: pd.DataFrame, targets: list[str]) -> pd.DataFrame:
    out = labels[["institute_year", "user_id", "date", *targets]].copy()
    for target in targets:
        out[f"prior__{target}"] = np.nan
        out[f"days_since__{target}"] = np.nan
    for _, idx in out.groupby(["institute_year", "user_id"], sort=False).groups.items():
        idx = list(idx)
        sub = out.loc[idx].sort_values("date")
        for target in targets:
            prior = sub[target].ffill().shift(1)
            observed_dates = sub["date"].where(sub[target].notna()).ffill().shift(1)
            out.loc[sub.index, f"prior__{target}"] = prior.to_numpy()
            out.loc[sub.index, f"days_since__{target}"] = (sub["date"] - observed_dates).dt.days.to_numpy()
    return out


def temporal_split(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    train_idx, test_idx = [], []
    for _, sub in df.sort_values("date").groupby(["institute_year", "user_id"], sort=False):
        idx = sub.index.to_numpy()
        if len(idx) < 4:
            continue
        cut = max(1, int(math.floor(len(idx) * 0.65)))
        cut = min(cut, len(idx) - 1)
        train_idx.extend(idx[:cut])
        test_idx.extend(idx[cut:])
    return np.asarray(train_idx), np.asarray(test_idx)


def fit_predict_ridge(train_x: pd.DataFrame, train_y: pd.Series, test_x: pd.DataFrame, alpha: float = 10.0) -> np.ndarray:
    x_train = train_x.apply(pd.to_numeric, errors="coerce").copy()
    x_test = test_x.apply(pd.to_numeric, errors="coerce").copy()
    med = x_train.median(axis=0)
    x_train = x_train.fillna(med)
    x_test = x_test.fillna(med)
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0, ddof=0).replace(0, 1.0)
    x_train = ((x_train - mean) / std).to_numpy(dtype=float)
    x_test = ((x_test - mean) / std).to_numpy(dtype=float)
    x_train = np.nan_to_num(x_train, nan=0.0, posinf=0.0, neginf=0.0)
    x_test = np.nan_to_num(x_test, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(train_y.to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    design = np.column_stack([np.ones(len(x_train)), x_train])
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    lhs = design.T @ design + penalty
    rhs = design.T @ y
    try:
        beta = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(lhs) @ rhs
    test_design = np.column_stack([np.ones(len(x_test)), x_test])
    return test_design @ beta


def r2_score_np(y_true: pd.Series, y_pred: np.ndarray) -> float:
    y = y_true.to_numpy(dtype=float)
    denom = float(((y - y.mean()) ** 2).sum())
    if denom <= 1e-12:
        return np.nan
    return float(1.0 - (((y - y_pred) ** 2).sum() / denom))


def evaluate_prior_models(prior_df: pd.DataFrame, targets: list[str], min_rows: int) -> pd.DataFrame:
    rows = []
    for target in targets:
        own_cols = [f"prior__{target}", f"days_since__{target}"]
        other_targets = [t for t in targets if t != target]
        other_cols = [f"prior__{t}" for t in other_targets] + [f"days_since__{t}" for t in other_targets]
        candidate = prior_df[["institute_year", "user_id", "date", target, *own_cols, *other_cols]].dropna(
            subset=[target]
        )
        candidate = candidate[candidate[own_cols[0]].notna()].copy()
        if len(candidate) < min_rows:
            continue
        train_idx, test_idx = temporal_split(candidate)
        if len(train_idx) < 50 or len(test_idx) < 50:
            continue
        train = candidate.loc[train_idx]
        test = candidate.loc[test_idx]
        y_train = zscore(train[target])
        y_test = zscore(test[target])
        if y_train.notna().sum() < 50 or y_test.notna().sum() < 50:
            continue
        for model_name, cols in {
            "own_prior_only": own_cols,
            "other_construct_priors": other_cols,
            "own_plus_other_priors": own_cols + other_cols,
        }.items():
            pred = fit_predict_ridge(train[cols], y_train, test[cols])
            rows.append(
                {
                    "target": target,
                    "model": model_name,
                    "n_train": int(len(train)),
                    "n_test": int(len(test)),
                    "n_test_users": int(test["user_id"].nunique()),
                    "r2": r2_score_np(y_test, pred),
                    "pearson_r": float(np.corrcoef(y_test, pred)[0, 1]),
                }
            )
    return pd.DataFrame(rows)


def summarize(corr: pd.DataFrame, metrics: pd.DataFrame) -> dict:
    pivot = metrics.pivot_table(index="target", columns="model", values="r2", aggfunc="first").reset_index()
    if not pivot.empty:
        pivot["lift_other_over_own"] = pivot["other_construct_priors"] - pivot["own_prior_only"]
        pivot["lift_other_added_to_own"] = pivot["own_plus_other_priors"] - pivot["own_prior_only"]
    return {
        "n_construct_pairs": int(len(corr)),
        "median_same_day_abs_r": float(corr["abs_r"].median()),
        "top_same_day_pairs": corr.head(12).to_dict(orient="records"),
        "n_temporal_targets": int(pivot["target"].nunique()) if not pivot.empty else 0,
        "mean_own_prior_r2": float(pivot["own_prior_only"].mean()) if not pivot.empty else None,
        "mean_other_construct_prior_r2": float(pivot["other_construct_priors"].mean()) if not pivot.empty else None,
        "mean_lift_other_added_to_own": float(pivot["lift_other_added_to_own"].mean()) if not pivot.empty else None,
        "temporal_summary": pivot.to_dict(orient="records"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed", default="data/processed/globem_daily_amber_compatible.parquet")
    parser.add_argument("--raw-root", default="data/raw/globem/1.1")
    parser.add_argument("--results", default="results")
    parser.add_argument("--min-rows", type=int, default=200)
    args = parser.parse_args()

    out_dir = Path(args.results) / "construct_fragility_audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = load_repeated_labels(Path(args.processed), Path(args.raw_root))
    targets = [c for c in [*PROCESSED_TARGETS, *EMA_TARGETS] if c in labels.columns and labels[c].notna().sum() >= args.min_rows]
    corr = same_day_correlations(labels, targets)
    prior_df = add_prior_columns(labels, targets)
    metrics = evaluate_prior_models(prior_df, targets, args.min_rows)
    summary = summarize(corr, metrics)
    summary["targets"] = targets
    summary["n_rows_with_any_label"] = int(labels[targets].notna().any(axis=1).sum())
    summary["contract"] = "post-hoc skeptical construct-fragility audit"

    corr.to_csv(out_dir / "same_day_construct_correlations.csv", index=False)
    metrics.to_csv(out_dir / "temporal_prior_model_metrics.csv", index=False)
    pd.DataFrame(summary["temporal_summary"]).to_csv(out_dir / "temporal_prior_summary.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
