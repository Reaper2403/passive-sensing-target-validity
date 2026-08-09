from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from amber_thesis.config import get_paths, load_config
from amber_thesis.eval.signal import (
    NumericFrameCleaner,
    score_predictions,
    select_feature_columns,
    train_test_within_user_temporal,
)


def make_rf(seed: int = 42, n_estimators: int = 300) -> object:
    return make_pipeline(
        NumericFrameCleaner(),
        SimpleImputer(strategy="median"),
        RandomForestClassifier(
            n_estimators=n_estimators,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
    )


def add_prior_label_rate(
    train: pd.DataFrame,
    test: pd.DataFrame,
    global_fallback: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = train.copy()
    test = test.copy()
    train["target_dep_weekly_numeric"] = train["target_dep_weekly"].astype(float)
    train["prior_label_count"] = (
        train.sort_values("date")
        .groupby(["institute_year", "user_id"])
        .cumcount()
        .reindex(train.index)
    )
    cumulative_sum = (
        train.sort_values("date")
        .groupby(["institute_year", "user_id"])["target_dep_weekly_numeric"]
        .cumsum()
        .reindex(train.index)
    )
    train["prior_dep_weekly_rate"] = (cumulative_sum - train["target_dep_weekly_numeric"]) / train[
        "prior_label_count"
    ].replace(0, np.nan)
    train["prior_dep_weekly_rate"] = train["prior_dep_weekly_rate"].fillna(global_fallback)

    train_rates = (
        train.groupby(["institute_year", "user_id"])["target_dep_weekly_numeric"]
        .mean()
        .reset_index(name="prior_dep_weekly_rate")
    )
    test = test.merge(train_rates, on=["institute_year", "user_id"], how="left")
    test["prior_dep_weekly_rate"] = test["prior_dep_weekly_rate"].fillna(global_fallback)
    return train.drop(columns=["target_dep_weekly_numeric"]), test


def evaluate_model(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    feature_set: str,
    seed: int,
    n_estimators: int,
) -> tuple[dict[str, object], pd.DataFrame]:
    model = make_rf(seed=seed, n_estimators=n_estimators)
    y_train = train["target_dep_weekly"].astype(int)
    y_test = test["target_dep_weekly"].astype(int)
    model.fit(train[features], y_train)
    pred = model.predict(test[features])
    proba = model.predict_proba(test[features])[:, 1]
    scores = score_predictions(y_test, pred, proba)
    predictions = test[["institute_year", "user_id", "date", "target_dep_weekly"]].copy()
    predictions["feature_set"] = feature_set
    predictions["y_pred"] = pred
    predictions["p_positive"] = proba
    return (
        {
            "model_name": "random_forest",
            "feature_set": feature_set,
            "n_features": len(features),
            "n_train": len(train),
            "n_test": len(test),
            "n_train_users": train[["institute_year", "user_id"]].drop_duplicates().shape[0],
            "n_test_users": test[["institute_year", "user_id"]].drop_duplicates().shape[0],
            **scores,
        },
        predictions,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/local_smoke.yaml")
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--test-frac", type=float, default=0.35)
    parser.add_argument("--min-labelled-days", type=int, default=4)
    args = parser.parse_args()

    config = load_config(args.config)
    paths = get_paths(config)
    out_dir = paths.results / "label_baseline_control"
    out_dir.mkdir(parents=True, exist_ok=True)

    daily = pd.read_parquet(paths.processed / "globem_daily_amber_compatible.parquet")
    train, test = train_test_within_user_temporal(
        daily,
        test_frac=args.test_frac,
        min_labelled_days=args.min_labelled_days,
    )
    global_fallback = float(train["target_dep_weekly"].astype(float).mean())
    train_label, test_label = add_prior_label_rate(train, test, global_fallback)

    rows = []
    pred_frames = []
    for feature_mode in ["allday_raw", "history_raw"]:
        base_features = select_feature_columns(daily, feature_mode)
        for suffix, frame_train, frame_test, features in [
            ("", train, test, base_features),
            ("_plus_prior_label_rate", train_label, test_label, base_features + ["prior_dep_weekly_rate"]),
        ]:
            row, preds = evaluate_model(
                frame_train,
                frame_test,
                features,
                feature_set=f"{feature_mode}{suffix}",
                seed=42,
                n_estimators=args.n_estimators,
            )
            rows.append(row)
            pred_frames.append(preds)

    row, preds = evaluate_model(
        train_label,
        test_label,
        ["prior_dep_weekly_rate"],
        feature_set="prior_label_rate_only",
        seed=42,
        n_estimators=args.n_estimators,
    )
    rows.append(row)
    pred_frames.append(preds)

    metrics = pd.DataFrame(rows)
    predictions = pd.concat(pred_frames, ignore_index=True)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    predictions.to_parquet(out_dir / "predictions.parquet", index=False)
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "contract": {
                    "question": "How much of the Chapter 3 history advantage remains after controlling for within-user prior dep_weekly rate?",
                    "label_baseline_feature": "prior_dep_weekly_rate",
                    "train_rows": "expanding prior user label rate; fallback to train global rate if no prior label",
                    "test_rows": "user train-period label rate; fallback to train global rate if unavailable",
                },
                "global_fallback": global_fallback,
                "metrics": metrics.to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
