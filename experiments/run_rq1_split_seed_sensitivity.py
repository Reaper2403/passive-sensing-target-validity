from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from amber_thesis.config import get_paths, load_config  # noqa: E402
from amber_thesis.eval.signal import (  # noqa: E402
    select_feature_columns,
    train_test_within_user_temporal,
)
from run_label_baseline_control import add_prior_label_rate, evaluate_model  # noqa: E402


def parse_values(value: str, cast: type) -> list:
    return [cast(part.strip()) for part in value.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/local_smoke.yaml")
    parser.add_argument("--test-fracs", default="0.25,0.35,0.45")
    parser.add_argument("--seeds", default="11,42,73")
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--min-labelled-days", type=int, default=4)
    parser.add_argument(
        "--output", type=Path, default=Path("results/rq1_split_seed_sensitivity")
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    test_fracs = parse_values(args.test_fracs, float)
    seeds = parse_values(args.seeds, int)
    if test_fracs != [0.25, 0.35, 0.45] or seeds != [11, 42, 73]:
        raise ValueError("Frozen grid is test_fracs=0.25,0.35,0.45 and seeds=11,42,73")

    config = load_config(args.config)
    paths = get_paths(config)
    daily = pd.read_parquet(paths.processed / "globem_daily_amber_compatible.parquet")
    feature_map = {
        mode: select_feature_columns(daily, mode)
        for mode in ["allday_raw", "history_raw"]
    }
    rows = []
    for test_frac in test_fracs:
        train, test = train_test_within_user_temporal(
            daily,
            test_frac=test_frac,
            min_labelled_days=args.min_labelled_days,
        )
        global_fallback = float(train["target_dep_weekly"].astype(float).mean())
        train_label, test_label = add_prior_label_rate(train, test, global_fallback)
        specifications = []
        for mode in ["allday_raw", "history_raw"]:
            specifications.extend(
                [
                    (mode, train, test, feature_map[mode]),
                    (
                        f"{mode}_plus_prior_label_rate",
                        train_label,
                        test_label,
                        feature_map[mode] + ["prior_dep_weekly_rate"],
                    ),
                ]
            )
        specifications.append(
            (
                "prior_label_rate_only",
                train_label,
                test_label,
                ["prior_dep_weekly_rate"],
            )
        )
        for seed in seeds:
            for feature_set, frame_train, frame_test, features in specifications:
                row, _ = evaluate_model(
                    frame_train,
                    frame_test,
                    features,
                    feature_set=feature_set,
                    seed=seed,
                    n_estimators=args.n_estimators,
                )
                row.update({"test_frac": test_frac, "seed": seed})
                rows.append(row)

    metrics = pd.DataFrame(rows)
    wide = metrics.pivot(
        index=["test_frac", "seed"], columns="feature_set", values="auroc"
    ).reset_index()
    contrasts = {
        "allday_plus_prior_minus_prior":
            wide["allday_raw_plus_prior_label_rate"] - wide["prior_label_rate_only"],
        "history_plus_prior_minus_prior":
            wide["history_raw_plus_prior_label_rate"] - wide["prior_label_rate_only"],
    }
    for name, values in contrasts.items():
        wide[name] = values

    summary_rows = []
    for name, values in contrasts.items():
        n_nonpositive = int((values <= 0).sum())
        median = float(values.median())
        maximum = float(values.max())
        passed = median <= 0 and n_nonpositive >= 7 and maximum < 0.01
        summary_rows.append(
            {
                "contrast": name,
                "median_delta": median,
                "minimum_delta": float(values.min()),
                "maximum_delta": maximum,
                "n_nonpositive": n_nonpositive,
                "n_cells": int(len(values)),
                "robust_no_material_increment": passed,
            }
        )
    summary_table = pd.DataFrame(summary_rows)
    summary = {
        "contract": "docs/74_rq1_split_seed_sensitivity_contract.md",
        "test_fracs": test_fracs,
        "seeds": seeds,
        "n_estimators": args.n_estimators,
        "summary": summary_table.to_dict(orient="records"),
    }
    metrics.to_csv(args.output / "metrics.csv", index=False)
    wide.to_csv(args.output / "contrast_grid.csv", index=False)
    summary_table.to_csv(args.output / "contrast_summary.csv", index=False)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(summary_table.to_string(index=False))


if __name__ == "__main__":
    main()
