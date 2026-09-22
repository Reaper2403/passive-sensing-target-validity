from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


KEYS = ["institute_year", "user_id", "date", "target_dep_weekly"]
PRIMARY_PAIRS = [
    ("allday_raw_plus_prior_label_rate", "prior_label_rate_only"),
    ("history_raw_plus_prior_label_rate", "prior_label_rate_only"),
]


def aligned_predictions(path: Path, feature_sets: list[str]) -> pd.DataFrame:
    predictions = pd.read_parquet(path)
    merged: pd.DataFrame | None = None
    join_keys = KEYS + ["row_occurrence"]
    for feature_set in feature_sets:
        frame = predictions[predictions["feature_set"] == feature_set].copy()
        frame["row_occurrence"] = frame.groupby(KEYS, dropna=False).cumcount()
        frame = frame[join_keys + ["p_positive"]].rename(
            columns={"p_positive": feature_set}
        )
        merged = frame if merged is None else merged.merge(
            frame, on=join_keys, how="inner", validate="one_to_one"
        )
    if merged is None or len(merged) == 0:
        raise ValueError("No aligned prediction rows found.")
    counts = predictions.groupby("feature_set").size()
    expected = {feature_set: int(counts[feature_set]) for feature_set in feature_sets}
    if len(set(expected.values())) != 1 or len(merged) != next(iter(expected.values())):
        raise ValueError(f"Prediction alignment lost rows: {expected}, merged={len(merged)}")
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("results/label_baseline_control/predictions.parquet"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/rq1_clustered_inference")
    )
    parser.add_argument("--n-bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260830)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    feature_sets = sorted({value for pair in PRIMARY_PAIRS for value in pair})
    frame = aligned_predictions(args.predictions, feature_sets)
    y = frame["target_dep_weekly"].astype(int).to_numpy()
    groups = [
        group.index.to_numpy()
        for _, group in frame.groupby(["institute_year", "user_id"], sort=False)
    ]
    arrays = {feature_set: frame[feature_set].to_numpy() for feature_set in feature_sets}
    point_aurocs = {
        feature_set: float(roc_auc_score(y, values))
        for feature_set, values in arrays.items()
    }

    rng = np.random.default_rng(args.seed)
    draw_rows = []
    draw_values = {pair: [] for pair in PRIMARY_PAIRS}
    for draw in range(args.n_bootstrap):
        selected = rng.integers(0, len(groups), size=len(groups))
        indices = np.concatenate([groups[index] for index in selected])
        draw_y = y[indices]
        draw_aurocs = {
            feature_set: roc_auc_score(draw_y, values[indices])
            for feature_set, values in arrays.items()
        }
        for model, baseline in PRIMARY_PAIRS:
            delta = float(draw_aurocs[model] - draw_aurocs[baseline])
            draw_values[(model, baseline)].append(delta)
            draw_rows.append(
                {
                    "draw": draw,
                    "model": model,
                    "baseline": baseline,
                    "delta_auroc": delta,
                }
            )

    result_rows = []
    for model, baseline in PRIMARY_PAIRS:
        values = np.asarray(draw_values[(model, baseline)])
        low, high = np.quantile(values, [0.025, 0.975])
        if low > 0:
            decision = "detectable_positive_increment"
        elif high < 0:
            decision = "detectable_negative_difference"
        else:
            decision = "no_detectable_increment"
        result_rows.append(
            {
                "model": model,
                "baseline": baseline,
                "model_auroc": point_aurocs[model],
                "baseline_auroc": point_aurocs[baseline],
                "delta_auroc": point_aurocs[model] - point_aurocs[baseline],
                "ci_low": float(low),
                "ci_high": float(high),
                "bootstrap_probability_positive": float(np.mean(values > 0)),
                "decision": decision,
            }
        )

    results = pd.DataFrame(result_rows)
    results.to_csv(args.output / "paired_clustered_auroc.csv", index=False)
    pd.DataFrame(draw_rows).to_parquet(args.output / "bootstrap_draws.parquet", index=False)
    summary = {
        "contract": "docs/72_rq1_clustered_inference_contract.md",
        "n_test_rows": int(len(frame)),
        "n_trajectory_clusters": int(len(groups)),
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
        "results": results.to_dict(orient="records"),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
