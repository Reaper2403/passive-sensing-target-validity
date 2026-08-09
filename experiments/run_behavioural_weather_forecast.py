from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


TARGETS = {
    "screen_unlock_count": "f_screen:phone_screen_rapids_countepisodeunlock",
    "screen_unlock_duration": "f_screen:phone_screen_rapids_sumdurationunlock",
    "steps_sum": "f_steps:fitbit_steps_intraday_rapids_sumsteps",
    "active_duration": "f_steps:fitbit_steps_intraday_rapids_sumdurationactivebout",
    "sedentary_duration": "f_steps:fitbit_steps_intraday_rapids_sumdurationsedentarybout",
    "sleep_duration": "f_slp:fitbit_sleep_summary_rapids_sumdurationasleepmain",
    "sleep_in_bed": "f_slp:fitbit_sleep_summary_rapids_sumdurationinbedmain",
    "sleep_efficiency": "f_slp:fitbit_sleep_summary_rapids_avgefficiencymain",
    "location_home_time": "f_loc:phone_locations_barnett_hometime",
    "location_distance": "f_loc:phone_locations_barnett_disttravelled",
    "location_entropy": "f_loc:phone_locations_doryab_locationentropy",
    "significant_places": "f_loc:phone_locations_doryab_numberofsignificantplaces",
    "incoming_call_count": "f_call:phone_calls_rapids_incoming_count",
    "outgoing_call_count": "f_call:phone_calls_rapids_outgoing_count",
    "bluetooth_unique_devices": "f_blue:phone_bluetooth_rapids_uniquedevices",
    "wifi_unique_devices": "f_wifi:phone_wifi_connected_rapids_uniquedevices",
}

ISSUE_SEGMENTS = {
    "morning": ["morning"],
    "afternoon": ["morning", "afternoon"],
    "evening": ["morning", "afternoon", "evening"],
}


def load_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def load_rapids(raw_root: Path) -> pd.DataFrame:
    needed_bases = list(TARGETS.values())
    needed_cols = ["pid", "date"]
    for base in needed_bases:
        needed_cols.append(f"{base}:allday")
        for segs in ISSUE_SEGMENTS.values():
            for seg in segs:
                needed_cols.append(f"{base}:{seg}")
    needed_cols = sorted(set(needed_cols))
    frames = []
    for p in sorted(raw_root.glob("INS-W_*/FeatureData/rapids.csv")):
        institute_year = p.parts[-3]
        df = pd.read_csv(p, usecols=lambda c: c in needed_cols)
        df["institute_year"] = institute_year
        frames.append(df)
    out = pd.concat(frames, ignore_index=True).rename(columns={"pid": "user_id"})
    out["date"] = pd.to_datetime(out["date"])
    return out


def add_history_features(df: pd.DataFrame, target_name: str, base_col: str) -> pd.DataFrame:
    work = df[["institute_year", "user_id", "date", base_col]].copy()
    work = work.sort_values(["institute_year", "user_id", "date"])
    group = work.groupby(["institute_year", "user_id"])[base_col]
    shifted = group.shift(1)
    work[f"{target_name}__yesterday"] = shifted
    for window in [7, 14, 42]:
        work[f"{target_name}__roll{window}_mean"] = shifted.groupby(
            [work["institute_year"], work["user_id"]]
        ).transform(lambda s: s.rolling(window, min_periods=max(3, min(window, 7))).mean())
    work[f"{target_name}__roll42_std"] = shifted.groupby(
        [work["institute_year"], work["user_id"]]
    ).transform(lambda s: s.rolling(42, min_periods=7).std())
    return work.drop(columns=[base_col])


def make_future_targets(df: pd.DataFrame, target_name: str, base_col: str, horizon: str) -> pd.DataFrame:
    work = df[["institute_year", "user_id", "date", base_col]].copy()
    work = work.sort_values(["institute_year", "user_id", "date"])
    if horizon == "tomorrow":
        out = work.copy()
        out["target_date"] = out["date"] + pd.to_timedelta(1, unit="D")
        future = work.rename(columns={"date": "target_date", base_col: "target_value"})
        out = out.drop(columns=[base_col]).merge(
            future[["institute_year", "user_id", "target_date", "target_value"]],
            on=["institute_year", "user_id", "target_date"],
            how="inner",
        )
    elif horizon == "next_week":
        values = []
        for _, sub in work.groupby(["institute_year", "user_id"], sort=False):
            s = sub[base_col].astype(float)
            future_mean = (
                s.shift(-1)
                .rolling(7, min_periods=4)
                .mean()
                .shift(-6)
            )
            tmp = sub[["institute_year", "user_id", "date"]].copy()
            tmp["target_date"] = tmp["date"] + pd.to_timedelta(7, unit="D")
            tmp["target_value"] = future_mean.to_numpy()
            values.append(tmp)
        out = pd.concat(values, ignore_index=True)
    else:
        raise ValueError(horizon)
    out["target_name"] = target_name
    out["horizon"] = horizon
    return out


def temporal_split(samples: pd.DataFrame, test_frac: float = 0.35) -> tuple[np.ndarray, np.ndarray]:
    train_idx: list[int] = []
    test_idx: list[int] = []
    for _, sub in samples.sort_values("date").groupby(["institute_year", "user_id"]):
        idx = sub.index.to_numpy()
        if len(idx) < 5:
            continue
        cut = max(1, int(np.floor(len(idx) * (1 - test_frac))))
        if cut >= len(idx):
            cut = len(idx) - 1
        train_idx.extend(idx[:cut])
        test_idx.extend(idx[cut:])
    return np.array(train_idx), np.array(test_idx)


def make_regressor() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ]
    )


def make_classifier(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=seed,
                ),
            ),
        ]
    )


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    corr = np.nan if len(y_true) < 3 else float(np.corrcoef(y_true, y_pred)[0, 1])
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae_log": float(mean_absolute_error(y_true, y_pred)),
        "pearson_r": corr,
    }


def classification_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    y_pred = (y_score >= 0.5).astype(int)
    return {
        "auroc": float(roc_auc_score(y_true, y_score)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def plot_summary(out_dir: Path, summary: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    if summary.empty:
        return
    for horizon in summary["horizon"].unique():
        plot = summary[(summary["horizon"] == horizon) & (summary["issue_time"] == "afternoon")].copy()
        if plot.empty:
            continue
        plot = plot.sort_values("history_fingerprint_plus_today", ascending=True)
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.barh(plot["target"], plot["history_fingerprint_plus_today"], label="history + fingerprint + today")
        ax.axvline(0, color="black", lw=1)
        ax.set_xlabel("Held-out R2, log future value")
        ax.set_title(f"Behavioural forecast regression performance: {horizon}, afternoon issue")
        fig.tight_layout()
        fig.savefig(out_dir / f"regression_r2_{horizon}_afternoon.svg")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/local_smoke.yaml")
    parser.add_argument("--min-samples", type=int, default=300)
    args = parser.parse_args()

    config = load_config(Path(args.config))
    seed = int(config.get("run", {}).get("seed", 42))
    paths = config["paths"]
    raw_root = Path(paths["globem_raw"])
    results = Path(paths["results"])
    out_dir = results / "behavioural_weather_forecast"
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = load_rapids(raw_root)
    raw["first_date"] = raw.groupby(["institute_year", "user_id"])["date"].transform("min")
    raw["study_day"] = (raw["date"] - raw["first_date"]).dt.days
    raw["weekday"] = raw["date"].dt.weekday
    raw["is_weekend"] = raw["weekday"].isin([5, 6]).astype(int)

    all_samples = []
    reg_rows = []
    clf_rows = []
    pred_frames = []
    feature_manifest = []

    for target_name, base in TARGETS.items():
        all_day = f"{base}:allday"
        if all_day not in raw.columns:
            continue
        hist = add_history_features(raw, target_name, all_day)
        for horizon in ["tomorrow", "next_week"]:
            future = make_future_targets(raw, target_name, all_day, horizon)
            for issue_time, segments in ISSUE_SEGMENTS.items():
                seg_cols = [f"{base}:{seg}" for seg in segments if f"{base}:{seg}" in raw.columns]
                if not seg_cols:
                    continue
                samples = raw[
                    ["institute_year", "user_id", "date", "study_day", "weekday", "is_weekend", *seg_cols]
                ].copy()
                samples = samples.merge(hist, on=["institute_year", "user_id", "date"], how="left")
                samples = samples.merge(
                    future,
                    on=["institute_year", "user_id", "date"],
                    how="inner",
                )
                samples = samples.dropna(subset=["target_value"]).copy()
                samples = samples[samples["study_day"] >= 42].copy()
                if len(samples) < args.min_samples or samples["user_id"].nunique() < 30:
                    continue

                history_cols = [
                    f"{target_name}__yesterday",
                    f"{target_name}__roll7_mean",
                    f"{target_name}__roll14_mean",
                ]
                fingerprint_cols = [f"{target_name}__roll42_mean", f"{target_name}__roll42_std"]
                calendar_cols = ["weekday", "is_weekend", "study_day"]
                feature_sets = {
                    "calendar_only": calendar_cols,
                    "recent_history_only": calendar_cols + history_cols,
                    "fingerprint_42d_only": calendar_cols + fingerprint_cols,
                    "today_partial_only": calendar_cols + seg_cols,
                    "history_plus_fingerprint": calendar_cols + history_cols + fingerprint_cols,
                    "history_fingerprint_plus_today": calendar_cols + history_cols + fingerprint_cols + seg_cols,
                }
                for model_name, cols in feature_sets.items():
                    for col in cols:
                        feature_manifest.append(
                            {
                                "target": target_name,
                                "horizon": horizon,
                                "issue_time": issue_time,
                                "model": model_name,
                                "feature": col,
                            }
                        )

                train_idx, test_idx = temporal_split(samples)
                if len(train_idx) < 100 or len(test_idx) < 100:
                    continue
                train = samples.loc[train_idx].copy()
                test = samples.loc[test_idx].copy()
                if train["target_value"].notna().sum() < 100 or test["target_value"].notna().sum() < 100:
                    continue

                y_train_reg = np.log1p(train["target_value"].clip(lower=0).astype(float))
                y_test_reg = np.log1p(test["target_value"].clip(lower=0).astype(float))
                threshold = float(train["target_value"].median())
                train["target_high"] = (train["target_value"] > threshold).astype(int)
                test["target_high"] = (test["target_value"] > threshold).astype(int)

                pred_out = test[
                    [
                        "institute_year",
                        "user_id",
                        "date",
                        "target_date",
                        "target_name",
                        "horizon",
                        "target_value",
                        "target_high",
                    ]
                ].copy()
                pred_out["issue_time"] = issue_time
                pred_out["high_threshold_train_median"] = threshold

                for model_name, cols in feature_sets.items():
                    reg = make_regressor()
                    reg.fit(train[cols], y_train_reg)
                    y_pred = reg.predict(test[cols])
                    reg_rows.append(
                        {
                            "target": target_name,
                            "horizon": horizon,
                            "issue_time": issue_time,
                            "model": model_name,
                            "n_train": len(train),
                            "n_test": len(test),
                            "n_train_users": train["user_id"].nunique(),
                            "n_test_users": test["user_id"].nunique(),
                            **regression_metrics(y_test_reg.to_numpy(), y_pred),
                        }
                    )
                    pred_out[f"reg__{model_name}"] = y_pred

                    if train["target_high"].nunique() == 2 and test["target_high"].nunique() == 2:
                        clf = make_classifier(seed)
                        clf.fit(train[cols], train["target_high"])
                        y_score = clf.predict_proba(test[cols])[:, 1]
                        clf_rows.append(
                            {
                                "target": target_name,
                                "horizon": horizon,
                                "issue_time": issue_time,
                                "model": model_name,
                                "n_train": len(train),
                                "n_test": len(test),
                                "n_train_users": train["user_id"].nunique(),
                                "n_test_users": test["user_id"].nunique(),
                                "positive_rate_test": float(test["target_high"].mean()),
                                **classification_metrics(test["target_high"].to_numpy(), y_score),
                            }
                        )
                        pred_out[f"clf__{model_name}"] = y_score
                pred_frames.append(pred_out)
                all_samples.append(
                    samples[
                        [
                            "institute_year",
                            "user_id",
                            "date",
                            "target_date",
                            "target_name",
                            "horizon",
                            "target_value",
                            "study_day",
                            "weekday",
                            "is_weekend",
                            *seg_cols,
                            *history_cols,
                            *fingerprint_cols,
                        ]
                    ].assign(issue_time=issue_time)
                )

    reg_metrics = pd.DataFrame(reg_rows)
    clf_metrics = pd.DataFrame(clf_rows)
    predictions = pd.concat(pred_frames, ignore_index=True) if pred_frames else pd.DataFrame()
    sample_df = pd.concat(all_samples, ignore_index=True) if all_samples else pd.DataFrame()
    reg_metrics.to_csv(out_dir / "regression_metrics.csv", index=False)
    clf_metrics.to_csv(out_dir / "classification_metrics.csv", index=False)
    predictions.to_parquet(out_dir / "predictions.parquet", index=False)
    sample_df.to_parquet(out_dir / "prediction_samples.parquet", index=False)
    pd.DataFrame(feature_manifest).drop_duplicates().to_csv(out_dir / "feature_manifest.csv", index=False)

    reg_summary = reg_metrics.pivot_table(
        index=["target", "horizon", "issue_time"], columns="model", values="r2", aggfunc="first"
    ).reset_index()
    clf_summary = clf_metrics.pivot_table(
        index=["target", "horizon", "issue_time"], columns="model", values="auroc", aggfunc="first"
    ).reset_index()
    for pivot in [reg_summary, clf_summary]:
        if {"history_fingerprint_plus_today", "history_plus_fingerprint"}.issubset(pivot.columns):
            pivot["lift_today_over_history_fingerprint"] = (
                pivot["history_fingerprint_plus_today"] - pivot["history_plus_fingerprint"]
            )
        if {"history_plus_fingerprint", "calendar_only"}.issubset(pivot.columns):
            pivot["lift_history_fingerprint_over_calendar"] = (
                pivot["history_plus_fingerprint"] - pivot["calendar_only"]
            )
    reg_summary.to_csv(out_dir / "regression_summary_r2.csv", index=False)
    clf_summary.to_csv(out_dir / "classification_summary_auroc.csv", index=False)
    plot_summary(out_dir, reg_summary)

    aggregate = {}
    for name, df, metric in [
        ("regression", reg_summary, "r2"),
        ("classification", clf_summary, "auroc"),
    ]:
        if df.empty:
            continue
        aggregate[name] = {}
        for horizon in sorted(df["horizon"].unique()):
            sub = df[df["horizon"].eq(horizon)]
            aggregate[name][horizon] = {}
            for issue_time in sorted(sub["issue_time"].unique()):
                s = sub[sub["issue_time"].eq(issue_time)]
                aggregate[name][horizon][issue_time] = {
                    "n_targets": int(len(s)),
                    "mean_history_fingerprint_plus_today": float(s["history_fingerprint_plus_today"].mean()),
                    "mean_history_plus_fingerprint": float(s["history_plus_fingerprint"].mean()),
                    "mean_lift_today": float(s["lift_today_over_history_fingerprint"].mean()),
                    "median_lift_today": float(s["lift_today_over_history_fingerprint"].median()),
                    "targets_positive_lift_today": int((s["lift_today_over_history_fingerprint"] > 0).sum()),
                }

    primary_reg = reg_summary[
        (reg_summary["horizon"].eq("tomorrow"))
        & (reg_summary["lift_today_over_history_fingerprint"] >= 0.02)
    ]
    primary_clf = clf_summary[
        (clf_summary["horizon"].eq("tomorrow"))
        & (clf_summary["lift_today_over_history_fingerprint"] >= 0.02)
    ]
    summary = {
        "pre_registration": "experiments/README.md",
        "n_targets_attempted": len(TARGETS),
        "n_prediction_samples": int(len(sample_df)),
        "n_regression_rows": int(len(reg_metrics)),
        "n_classification_rows": int(len(clf_metrics)),
        "aggregate": aggregate,
        "primary_success_any_target": bool(len(primary_reg) > 0 or len(primary_clf) > 0),
        "best_regression_lift_today": None,
        "best_classification_lift_today": None,
    }
    if not reg_summary.empty:
        summary["best_regression_lift_today"] = reg_summary.sort_values(
            "lift_today_over_history_fingerprint", ascending=False
        ).head(1).to_dict(orient="records")[0]
    if not clf_summary.empty:
        summary["best_classification_lift_today"] = clf_summary.sort_values(
            "lift_today_over_history_fingerprint", ascending=False
        ).head(1).to_dict(orient="records")[0]
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
