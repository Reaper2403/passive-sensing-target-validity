from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from run_rq1_clustered_inference import KEYS, aligned_predictions


FEATURE_SETS = [
    "allday_raw",
    "history_raw",
    "prior_label_rate_only",
    "allday_raw_plus_prior_label_rate",
    "history_raw_plus_prior_label_rate",
]
PRIMARY_MODEL = "history_raw_plus_prior_label_rate"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("results/label_baseline_control/predictions.parquet"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/rq1_within_person_deviation")
    )
    parser.add_argument("--n-bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260831)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    frame = aligned_predictions(args.predictions, FEATURE_SETS)
    per_user_rows = []
    for (institute_year, user_id), group in frame.groupby(
        ["institute_year", "user_id"], sort=False
    ):
        y = group["target_dep_weekly"].astype(int).to_numpy()
        if len(np.unique(y)) != 2:
            continue
        n_positive = int(y.sum())
        n_negative = int(len(y) - n_positive)
        row = {
            "institute_year": institute_year,
            "user_id": user_id,
            "n_test_rows": int(len(group)),
            "n_positive": n_positive,
            "n_negative": n_negative,
            "n_pairs": n_positive * n_negative,
        }
        for feature_set in FEATURE_SETS:
            row[f"auroc__{feature_set}"] = float(roc_auc_score(y, group[feature_set]))
        per_user_rows.append(row)

    per_user = pd.DataFrame(per_user_rows)
    if per_user.empty:
        raise ValueError("No trajectories contain both target classes in test.")

    rng = np.random.default_rng(args.seed)
    selected = rng.integers(
        0, len(per_user), size=(args.n_bootstrap, len(per_user))
    )
    metric_rows = []
    draw_rows = []
    for feature_set in FEATURE_SETS:
        values = per_user[f"auroc__{feature_set}"].to_numpy()
        weights = per_user["n_pairs"].to_numpy(dtype=float)
        macro = float(values.mean())
        pair_weighted = float(np.average(values, weights=weights))
        draws = values[selected].mean(axis=1)
        low, high = np.quantile(draws, [0.025, 0.975])
        if low > 0.50 and macro >= 0.55:
            decision = "practically_supported_within_person_ranking"
        elif low > 0.50:
            decision = "small_detectable_within_person_ranking"
        else:
            decision = "no_detectable_within_person_ranking"
        metric_rows.append(
            {
                "feature_set": feature_set,
                "n_trajectories": int(len(per_user)),
                "n_test_rows": int(per_user["n_test_rows"].sum()),
                "macro_within_trajectory_auroc": macro,
                "ci_low": float(low),
                "ci_high": float(high),
                "pair_weighted_auroc": pair_weighted,
                "decision": decision,
                "primary": feature_set == PRIMARY_MODEL,
            }
        )
        draw_rows.extend(
            {
                "draw": draw,
                "feature_set": feature_set,
                "macro_within_trajectory_auroc": float(value),
            }
            for draw, value in enumerate(draws)
        )

    metrics = pd.DataFrame(metric_rows)
    primary = metrics.set_index("feature_set").loc[PRIMARY_MODEL]
    summary = {
        "contract": "docs/73_rq1_within_person_deviation_contract.md",
        "eligibility": {
            "n_all_test_trajectories": int(
                frame[["institute_year", "user_id"]].drop_duplicates().shape[0]
            ),
            "n_informative_trajectories": int(len(per_user)),
            "n_informative_test_rows": int(per_user["n_test_rows"].sum()),
        },
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
        "primary_model": PRIMARY_MODEL,
        "primary_decision": primary["decision"],
        "metrics": metrics.to_dict(orient="records"),
    }
    per_user.to_parquet(args.output / "per_trajectory_auroc.parquet", index=False)
    metrics.to_csv(args.output / "metrics.csv", index=False)
    pd.DataFrame(draw_rows).to_parquet(args.output / "bootstrap_draws.parquet", index=False)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
