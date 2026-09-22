from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm


def simulate_macro_aurocs(
    counts: pd.DataFrame,
    true_auc: float,
    n_simulations: int,
    rng: np.random.Generator,
    batch_size: int,
) -> np.ndarray:
    """Simulate equal-weight mean trajectory AUROC for the observed sparse design."""
    if not 0.5 <= true_auc < 1.0:
        raise ValueError("true_auc must be in [0.5, 1).")

    delta = np.sqrt(2.0) * norm.ppf(true_auc)
    grouped = (
        counts.groupby(["n_positive", "n_negative"], sort=True)
        .size()
        .rename("n_trajectories")
        .reset_index()
    )
    n_trajectories = int(grouped["n_trajectories"].sum())
    output = np.empty(n_simulations, dtype=float)

    for start in range(0, n_simulations, batch_size):
        stop = min(start + batch_size, n_simulations)
        batch = stop - start
        auc_sum = np.zeros(batch, dtype=float)
        for row in grouped.itertuples(index=False):
            n_positive = int(row.n_positive)
            n_negative = int(row.n_negative)
            n_groups = int(row.n_trajectories)
            positive_scores = rng.normal(loc=delta, size=(batch, n_groups, n_positive))
            negative_scores = rng.normal(loc=0.0, size=(batch, n_groups, n_negative))
            comparisons = positive_scores[..., :, None] > negative_scores[..., None, :]
            trajectory_aurocs = comparisons.mean(axis=(-2, -1))
            auc_sum += trajectory_aurocs.sum(axis=1)
        output[start:stop] = auc_sum / n_trajectories
    return output


def first_power_bracket(
    power_table: pd.DataFrame, column: str, target: float = 0.80
) -> dict[str, float | None]:
    ordered = power_table.sort_values("true_auc").reset_index(drop=True)
    reached = ordered.index[ordered[column] >= target].tolist()
    if not reached:
        return {"lower_exclusive": float(ordered.iloc[-1]["true_auc"]), "upper_inclusive": None}
    upper_index = reached[0]
    if upper_index == 0:
        return {"lower_exclusive": None, "upper_inclusive": float(ordered.iloc[0]["true_auc"])}
    return {
        "lower_exclusive": float(ordered.iloc[upper_index - 1]["true_auc"]),
        "upper_inclusive": float(ordered.iloc[upper_index]["true_auc"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post-hoc power diagnostic for sparse within-trajectory AUROC."
    )
    parser.add_argument(
        "--per-trajectory",
        type=Path,
        default=Path("results/rq1_within_person_deviation/per_trajectory_auroc.parquet"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/rq1_within_person_power_diagnostic"),
    )
    parser.add_argument("--n-simulations", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--batch-size", type=int, default=1_000)
    args = parser.parse_args()

    frame = pd.read_parquet(args.per_trajectory)
    required = {
        "n_test_rows",
        "n_positive",
        "n_negative",
        "n_pairs",
        "auroc__history_raw_plus_prior_label_rate",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if (frame[["n_positive", "n_negative"]] < 1).any().any():
        raise ValueError("Every included trajectory must contain both classes.")

    args.output.mkdir(parents=True, exist_ok=True)
    counts = frame[["n_test_rows", "n_positive", "n_negative", "n_pairs"]].copy()
    count_distribution = (
        counts.groupby(["n_test_rows", "n_positive", "n_negative", "n_pairs"])
        .size()
        .rename("n_trajectories")
        .reset_index()
        .sort_values(["n_test_rows", "n_positive", "n_negative"])
    )

    true_auc_grid = np.round(np.arange(0.50, 0.701, 0.01), 2)
    seed_sequence = np.random.SeedSequence(args.seed)
    child_seeds = seed_sequence.spawn(len(true_auc_grid))
    simulations: dict[float, np.ndarray] = {}
    for true_auc, child_seed in zip(true_auc_grid, child_seeds, strict=True):
        simulations[float(true_auc)] = simulate_macro_aurocs(
            counts=counts,
            true_auc=float(true_auc),
            n_simulations=args.n_simulations,
            rng=np.random.default_rng(child_seed),
            batch_size=args.batch_size,
        )

    # A lower 95% confidence bound above 0.5 corresponds approximately to a
    # one-sided alpha=0.025 test. The null quantile supplies a design-matched cutoff.
    null_draws = simulations[0.5]
    detection_cutoff = float(np.quantile(null_draws, 0.975, method="higher"))
    practical_cutoff = max(0.55, detection_cutoff)

    power_rows = []
    for true_auc in true_auc_grid:
        draws = simulations[float(true_auc)]
        power_rows.append(
            {
                "true_common_auc": float(true_auc),
                "mean_simulated_macro_auc": float(draws.mean()),
                "sd_simulated_macro_auc": float(draws.std(ddof=1)),
                "power_detect_above_chance": float(np.mean(draws > detection_cutoff)),
                "power_pass_full_recorded_rule": float(np.mean(draws >= practical_cutoff)),
            }
        )
    power = pd.DataFrame(power_rows).rename(columns={"true_common_auc": "true_auc"})

    observed_macro_auc = float(frame["auroc__history_raw_plus_prior_label_rate"].mean())
    observed_null_tail = float(
        (np.count_nonzero(null_draws >= observed_macro_auc) + 1) / (len(null_draws) + 1)
    )
    summary = {
        "status": "post_hoc_design_diagnostic",
        "input": str(args.per_trajectory),
        "seed": args.seed,
        "n_simulations_per_grid_point": args.n_simulations,
        "n_trajectories": int(len(frame)),
        "n_test_rows": int(frame["n_test_rows"].sum()),
        "rows_per_trajectory": {
            "mean": float(frame["n_test_rows"].mean()),
            "median": float(frame["n_test_rows"].median()),
            "minimum": int(frame["n_test_rows"].min()),
            "maximum": int(frame["n_test_rows"].max()),
            "q25": float(frame["n_test_rows"].quantile(0.25)),
            "q75": float(frame["n_test_rows"].quantile(0.75)),
        },
        "positive_negative_pairs_per_trajectory": {
            "mean": float(frame["n_pairs"].mean()),
            "median": float(frame["n_pairs"].median()),
            "minimum": int(frame["n_pairs"].min()),
            "maximum": int(frame["n_pairs"].max()),
            "q25": float(frame["n_pairs"].quantile(0.25)),
            "q75": float(frame["n_pairs"].quantile(0.75)),
        },
        "observed_primary_macro_auc": observed_macro_auc,
        "design_matched_detection_cutoff": detection_cutoff,
        "observed_upper_null_tail_probability": observed_null_tail,
        "mde_80_percent_power_bracket_detection": first_power_bracket(
            power, "power_detect_above_chance"
        ),
        "mde_80_percent_power_bracket_full_rule": first_power_bracket(
            power, "power_pass_full_recorded_rule"
        ),
        "assumptions": [
            "Observed positive and negative test-row counts are held fixed.",
            "Every trajectory shares one true AUROC at each grid point.",
            "Negative scores are N(0,1) and positive scores are N(delta,1), with delta chosen for the requested AUROC.",
            "Trajectories and rows are conditionally independent in the simulation.",
            "The fitted prediction model is treated as fixed; training and model-selection uncertainty are excluded.",
            "The null 97.5th percentile approximates the existing lower-95%-bound-above-0.5 decision.",
        ],
        "interpretation_boundary": (
            "This post-hoc simulation describes sensitivity of the aggregate macro-AUROC "
            "under explicit parametric assumptions. It does not estimate power for a "
            "single user and does not turn the observed null into evidence of equivalence."
        ),
    }

    count_distribution.to_csv(args.output / "count_distribution.csv", index=False)
    power.to_csv(args.output / "power_curve.csv", index=False)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print("\nPower curve:\n" + power.to_string(index=False))


if __name__ == "__main__":
    main()
