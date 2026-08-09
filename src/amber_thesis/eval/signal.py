from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit


@dataclass(frozen=True)
class SignalResult:
    model_name: str
    feature_set: str
    n_train: int
    n_test: int
    n_features: int
    macro_f1: float
    balanced_accuracy: float
    accuracy: float
    auroc: float | None


class NumericFrameCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y=None):
        X_num = X.apply(pd.to_numeric, errors="coerce")
        self.columns_ = X_num.columns[X_num.notna().any(axis=0)].tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_num = X.apply(pd.to_numeric, errors="coerce")
        return X_num.reindex(columns=self.columns_)


def select_feature_columns(df: pd.DataFrame, mode: str) -> list[str]:
    prefixes = ("screen__", "steps__", "sleep__")
    candidates = [col for col in df.columns if col.startswith(prefixes)]
    if mode == "allday_raw":
        return [
            col
            for col in candidates
            if col.endswith(":allday") and "_norm:" not in col and "_dis:" not in col
        ]
    if mode == "history_raw":
        return [
            col
            for col in candidates
            if col.endswith((":7dhist", ":14dhist")) and "_norm:" not in col and "_dis:" not in col
        ]
    if mode == "allday_plus_history_raw":
        return [
            col
            for col in candidates
            if col.endswith((":allday", ":7dhist", ":14dhist"))
            and "_norm:" not in col
            and "_dis:" not in col
        ]
    raise ValueError(f"Unknown feature selection mode: {mode}")


def train_test_by_dataset(
    df: pd.DataFrame,
    train_datasets: tuple[str, ...] = ("INS-W_2", "INS-W_3"),
    test_datasets: tuple[str, ...] = ("INS-W_4",),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    labelled = df[df["target_dep_weekly"].notna()].copy()
    train = labelled[labelled["institute_year"].isin(train_datasets)]
    test = labelled[labelled["institute_year"].isin(test_datasets)]
    return train, test


def train_test_within_user_temporal(
    df: pd.DataFrame,
    test_frac: float = 0.35,
    min_labelled_days: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    labelled = df[df["target_dep_weekly"].notna()].copy()
    train_parts = []
    test_parts = []
    for _, group in labelled.sort_values("date").groupby(["institute_year", "user_id"]):
        if len(group) < min_labelled_days:
            continue
        n_test = max(1, int(round(len(group) * test_frac)))
        n_train = len(group) - n_test
        if n_train < 2:
            continue
        train_parts.append(group.iloc[:n_train])
        test_parts.append(group.iloc[n_train:])

    if not train_parts or not test_parts:
        raise ValueError("Temporal split produced no train/test rows.")

    return (
        pd.concat(train_parts, ignore_index=True),
        pd.concat(test_parts, ignore_index=True),
    )


def train_test_held_out_users(
    df: pd.DataFrame,
    test_frac: float = 0.25,
    min_labelled_days: int = 4,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    labelled = df[df["target_dep_weekly"].notna()].copy()
    counts = labelled.groupby(["institute_year", "user_id"]).size().reset_index(name="n")
    keep = counts[counts["n"] >= min_labelled_days][["institute_year", "user_id"]]
    labelled = labelled.merge(keep, on=["institute_year", "user_id"], how="inner")
    labelled["group_id"] = labelled["institute_year"].astype(str) + "::" + labelled["user_id"].astype(str)

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=random_state)
    train_idx, test_idx = next(splitter.split(labelled, labelled["target_dep_weekly"], groups=labelled["group_id"]))
    train = labelled.iloc[train_idx].drop(columns=["group_id"]).reset_index(drop=True)
    test = labelled.iloc[test_idx].drop(columns=["group_id"]).reset_index(drop=True)
    return train, test


def make_models() -> dict[str, object]:
    return {
        "logistic_regression": make_pipeline(
            NumericFrameCleaner(),
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced"),
        ),
        "random_forest": make_pipeline(
            NumericFrameCleaner(),
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
        ),
    }


def score_predictions(y_true: pd.Series, pred: np.ndarray, proba: np.ndarray | None) -> dict[str, float | None]:
    auroc = None
    if proba is not None and len(np.unique(y_true)) == 2:
        auroc = float(roc_auc_score(y_true, proba))
    return {
        "macro_f1": float(f1_score(y_true, pred, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "auroc": auroc,
    }


def evaluate_signal(df: pd.DataFrame, feature_set: str) -> list[SignalResult]:
    features = select_feature_columns(df, feature_set)
    train, test = train_test_by_dataset(df)
    if train.empty or test.empty:
        raise ValueError("Train/test split produced no labelled rows.")

    X_train = train[features]
    y_train = train["target_dep_weekly"].astype(int)
    X_test = test[features]
    y_test = test["target_dep_weekly"].astype(int)

    models = make_models()

    results: list[SignalResult] = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        auroc = None
        if hasattr(model, "predict_proba") and len(np.unique(y_test)) == 2:
            proba = model.predict_proba(X_test)[:, 1]
            auroc = float(roc_auc_score(y_test, proba))
        results.append(
            SignalResult(
                model_name=name,
                feature_set=feature_set,
                n_train=len(train),
                n_test=len(test),
                n_features=len(features),
                macro_f1=float(f1_score(y_test, pred, average="macro")),
                balanced_accuracy=float(balanced_accuracy_score(y_test, pred)),
                accuracy=float(accuracy_score(y_test, pred)),
                auroc=auroc,
            )
        )
    return results


def evaluate_temporal_signal(
    df: pd.DataFrame,
    feature_set: str,
    test_frac: float = 0.35,
    min_labelled_days: int = 4,
) -> tuple[list[dict[str, object]], pd.DataFrame, pd.DataFrame]:
    features = select_feature_columns(df, feature_set)
    train, test = train_test_within_user_temporal(
        df,
        test_frac=test_frac,
        min_labelled_days=min_labelled_days,
    )

    X_train = train[features]
    y_train = train["target_dep_weekly"].astype(int)
    X_test = test[features]
    y_test = test["target_dep_weekly"].astype(int)

    rows = []
    prediction_frames = []
    per_user_frames = []
    for name, model in make_models().items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        scores = score_predictions(y_test, pred, proba)
        predictions = test[["institute_year", "user_id", "date", "target_dep_weekly"]].copy()
        predictions["model_name"] = name
        predictions["feature_set"] = feature_set
        predictions["y_pred"] = pred
        predictions["confidence"] = np.maximum(proba, 1 - proba) if proba is not None else np.nan
        predictions["p_positive"] = proba if proba is not None else np.nan
        prediction_frames.append(predictions)

        per_user = []
        for (year, user_id), group in predictions.groupby(["institute_year", "user_id"]):
            if len(group) < 2 or group["target_dep_weekly"].nunique() < 2:
                continue
            user_scores = score_predictions(
                group["target_dep_weekly"].astype(int),
                group["y_pred"].to_numpy(),
                group["p_positive"].to_numpy() if group["p_positive"].notna().all() else None,
            )
            per_user.append(
                {
                    "model_name": name,
                    "feature_set": feature_set,
                    "institute_year": year,
                    "user_id": user_id,
                    "n_examples": len(group),
                    **user_scores,
                }
            )
        per_user_df = pd.DataFrame(per_user)
        per_user_frames.append(per_user_df)

        rows.append(
            {
                "model_name": name,
                "feature_set": feature_set,
                "split_type": "within_user_temporal",
                "test_frac": test_frac,
                "min_labelled_days": min_labelled_days,
                "n_train": len(train),
                "n_test": len(test),
                "n_users_train": train[["institute_year", "user_id"]].drop_duplicates().shape[0],
                "n_users_test": test[["institute_year", "user_id"]].drop_duplicates().shape[0],
                "n_features": len(features),
                **scores,
                "per_user_macro_f1_mean": float(per_user_df["macro_f1"].mean()) if not per_user_df.empty else np.nan,
                "per_user_macro_f1_std": float(per_user_df["macro_f1"].std()) if not per_user_df.empty else np.nan,
                "per_user_bal_acc_mean": float(per_user_df["balanced_accuracy"].mean()) if not per_user_df.empty else np.nan,
                "per_user_bal_acc_std": float(per_user_df["balanced_accuracy"].std()) if not per_user_df.empty else np.nan,
                "worst_quartile_user_macro_f1": float(per_user_df["macro_f1"].quantile(0.25)) if not per_user_df.empty else np.nan,
            }
        )

    return (
        rows,
        pd.concat(prediction_frames, ignore_index=True),
        pd.concat(per_user_frames, ignore_index=True),
    )


def evaluate_held_out_user_signal(
    df: pd.DataFrame,
    feature_set: str,
    test_frac: float = 0.25,
    min_labelled_days: int = 4,
    random_state: int = 42,
) -> tuple[list[dict[str, object]], pd.DataFrame, pd.DataFrame]:
    features = select_feature_columns(df, feature_set)
    train, test = train_test_held_out_users(
        df,
        test_frac=test_frac,
        min_labelled_days=min_labelled_days,
        random_state=random_state,
    )

    X_train = train[features]
    y_train = train["target_dep_weekly"].astype(int)
    X_test = test[features]
    y_test = test["target_dep_weekly"].astype(int)

    rows = []
    prediction_frames = []
    per_user_frames = []
    for name, model in make_models().items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        scores = score_predictions(y_test, pred, proba)
        predictions = test[["institute_year", "user_id", "date", "target_dep_weekly"]].copy()
        predictions["model_name"] = name
        predictions["feature_set"] = feature_set
        predictions["y_pred"] = pred
        predictions["confidence"] = np.maximum(proba, 1 - proba) if proba is not None else np.nan
        predictions["p_positive"] = proba if proba is not None else np.nan
        prediction_frames.append(predictions)

        per_user = []
        for (year, user_id), group in predictions.groupby(["institute_year", "user_id"]):
            if len(group) < 2 or group["target_dep_weekly"].nunique() < 2:
                continue
            user_scores = score_predictions(
                group["target_dep_weekly"].astype(int),
                group["y_pred"].to_numpy(),
                group["p_positive"].to_numpy() if group["p_positive"].notna().all() else None,
            )
            per_user.append(
                {
                    "model_name": name,
                    "feature_set": feature_set,
                    "institute_year": year,
                    "user_id": user_id,
                    "n_examples": len(group),
                    **user_scores,
                }
            )
        per_user_df = pd.DataFrame(per_user)
        per_user_frames.append(per_user_df)

        rows.append(
            {
                "model_name": name,
                "feature_set": feature_set,
                "split_type": "held_out_users",
                "test_frac": test_frac,
                "min_labelled_days": min_labelled_days,
                "n_train": len(train),
                "n_test": len(test),
                "n_users_train": train[["institute_year", "user_id"]].drop_duplicates().shape[0],
                "n_users_test": test[["institute_year", "user_id"]].drop_duplicates().shape[0],
                "n_features": len(features),
                **scores,
                "per_user_macro_f1_mean": float(per_user_df["macro_f1"].mean()) if not per_user_df.empty else np.nan,
                "per_user_macro_f1_std": float(per_user_df["macro_f1"].std()) if not per_user_df.empty else np.nan,
                "per_user_bal_acc_mean": float(per_user_df["balanced_accuracy"].mean()) if not per_user_df.empty else np.nan,
                "per_user_bal_acc_std": float(per_user_df["balanced_accuracy"].std()) if not per_user_df.empty else np.nan,
                "worst_quartile_user_macro_f1": float(per_user_df["macro_f1"].quantile(0.25)) if not per_user_df.empty else np.nan,
            }
        )

    return (
        rows,
        pd.concat(prediction_frames, ignore_index=True),
        pd.concat(per_user_frames, ignore_index=True),
    )


def per_user_metrics(df: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    merged = predictions.merge(
        df[["user_id", "date", "institute_year", "target_dep_weekly"]],
        on=["user_id", "date", "institute_year"],
        how="left",
    )
    rows = []
    for user_id, group in merged.groupby("user_id"):
        if len(group) < 2:
            continue
        y_true = group["target_dep_weekly"].astype(int)
        y_pred = group["y_pred"].astype(int)
        rows.append(
            {
                "user_id": user_id,
                "n_examples": len(group),
                "macro_f1": f1_score(y_true, y_pred, average="macro"),
                "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
                "accuracy": accuracy_score(y_true, y_pred),
            }
        )
    return pd.DataFrame(rows)
