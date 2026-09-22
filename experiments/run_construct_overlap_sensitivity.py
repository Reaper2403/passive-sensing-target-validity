from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


FAMILIES = {
    "phq4": "phq",
    "phq4_EMA": "phq",
    "phq4_anxiety_EMA": "phq",
    "phq4_depression_EMA": "phq",
    "target_dep_weekly": "weekly_depression",
    "feel_depressed": "depressed_feeling",
    "feel_anxious": "anxious_feeling",
    "pss4_EMA": "stress",
    "positive_affect_EMA": "positive_affect",
    "negative_affect_EMA": "negative_affect",
}


def describe(frame: pd.DataFrame) -> dict[str, float | int | None]:
    if frame.empty:
        return {"n_pairs": 0, "median_abs_r": None, "q1_abs_r": None, "q3_abs_r": None}
    return {
        "n_pairs": int(len(frame)),
        "median_abs_r": float(frame["abs_r"].median()),
        "q1_abs_r": float(frame["abs_r"].quantile(0.25)),
        "q3_abs_r": float(frame["abs_r"].quantile(0.75)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/construct_fragility_audit/same_day_construct_correlations.csv"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/construct_overlap_sensitivity")
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    pairs = pd.read_csv(args.input)
    pairs["family_a"] = pairs["target_a"].map(FAMILIES)
    pairs["family_b"] = pairs["target_b"].map(FAMILIES)
    if pairs[["family_a", "family_b"]].isna().any().any():
        missing = sorted(
            set(pairs.loc[pairs["family_a"].isna(), "target_a"])
            | set(pairs.loc[pairs["family_b"].isna(), "target_b"])
        )
        raise ValueError(f"Unmapped targets: {missing}")
    pairs["same_family"] = pairs["family_a"] == pairs["family_b"]
    finite = pairs[pairs["abs_r"].notna()].copy()
    cross_family = finite[~finite["same_family"]]
    primary = cross_family[cross_family["n_overlap"] >= 100]
    same_phq = finite[
        finite["same_family"] & (finite["family_a"] == "phq")
    ]

    summary = {
        "contract": "docs/75_construct_overlap_sensitivity_contract.md",
        "all_finite": describe(finite),
        "cross_family_all_finite": describe(cross_family),
        "cross_family_n_at_least_100_primary": describe(primary),
        "same_phq_family": describe(same_phq),
        "interpretation": (
            "Cross-family overlap remains substantial but is lower than the mixed "
            "summary containing derived PHQ representations."
        ),
    }
    pairs.to_csv(args.output / "classified_pairs.csv", index=False)
    primary.to_csv(args.output / "primary_cross_family_pairs.csv", index=False)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
