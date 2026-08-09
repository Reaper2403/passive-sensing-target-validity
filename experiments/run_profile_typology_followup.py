from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.run_referee_robustness_checks import CORE_TARGETS


def normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return np.divide(
        values,
        norms,
        out=np.zeros_like(values, dtype=float),
        where=norms > 1e-12,
    )


def assign_clusters(values: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    return np.argmax(normalize_rows(values) @ normalize_rows(centroids).T, axis=1)


def spherical_kmeans(
    values: np.ndarray,
    k: int,
    rng: np.random.Generator,
    n_init: int = 30,
    max_iter: int = 100,
) -> tuple[np.ndarray, np.ndarray, float]:
    values = normalize_rows(values)
    best_centroids = None
    best_labels = None
    best_objective = -np.inf
    for _ in range(n_init):
        first = int(rng.integers(0, len(values)))
        chosen = [first]
        while len(chosen) < k:
            similarity = values @ values[chosen].T
            distance = 1.0 - np.max(similarity, axis=1)
            distance[chosen] = 0.0
            clipped = np.clip(distance, 0.0, None)
            total = clipped.sum()
            if total <= 1e-12:
                candidates = [index for index in range(len(values)) if index not in chosen]
                chosen.append(int(rng.choice(candidates)))
            else:
                chosen.append(int(rng.choice(len(values), p=clipped / total)))
        centroids = values[chosen].copy()
        labels = np.full(len(values), -1, dtype=int)
        for _ in range(max_iter):
            new_labels = assign_clusters(values, centroids)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for cluster in range(k):
                members = values[labels == cluster]
                if len(members):
                    centroid = members.mean(axis=0)
                    norm = np.linalg.norm(centroid)
                    centroids[cluster] = centroid / norm if norm > 1e-12 else centroid
                else:
                    centroids[cluster] = values[int(rng.integers(0, len(values)))]
        objective = float(np.sum(np.max(values @ centroids.T, axis=1)))
        if objective > best_objective:
            best_objective = objective
            best_centroids = centroids.copy()
            best_labels = labels.copy()
    if best_centroids is None or best_labels is None:
        raise RuntimeError("Spherical k-means did not produce a solution")
    return best_centroids, best_labels, best_objective


def adjusted_rand_index(first: np.ndarray, second: np.ndarray) -> float:
    table = pd.crosstab(first, second).to_numpy(dtype=int)

    def combinations(values: np.ndarray) -> float:
        return float(np.sum(values * (values - 1) / 2))

    n = int(table.sum())
    if n < 2:
        return np.nan
    total = n * (n - 1) / 2
    sum_cells = combinations(table)
    sum_rows = combinations(table.sum(axis=1))
    sum_columns = combinations(table.sum(axis=0))
    expected = sum_rows * sum_columns / total
    maximum = 0.5 * (sum_rows + sum_columns)
    denominator = maximum - expected
    if abs(denominator) <= 1e-12:
        return 1.0 if np.array_equal(first, second) else 0.0
    return float((sum_cells - expected) / denominator)


def cosine_silhouette(values: np.ndarray, labels: np.ndarray) -> float:
    values = normalize_rows(values)
    distance = 1.0 - values @ values.T
    scores = []
    for index, label in enumerate(labels):
        own = np.flatnonzero(labels == label)
        own = own[own != index]
        if not len(own):
            scores.append(0.0)
            continue
        within = float(distance[index, own].mean())
        alternatives = []
        for other in sorted(set(labels) - {label}):
            positions = np.flatnonzero(labels == other)
            alternatives.append(float(distance[index, positions].mean()))
        between = min(alternatives)
        denominator = max(within, between)
        scores.append((between - within) / denominator if denominator > 1e-12 else 0.0)
    return float(np.mean(scores))


def make_profiles(metrics: pd.DataFrame, regime: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    subset = metrics[
        metrics["regime"].eq(regime) & metrics["target_name"].isin(CORE_TARGETS)
    ]
    pivot = subset.pivot_table(
        index=["institute_year", "user_id", "half"],
        columns="target_name",
        values="history_skill_residual_sd_entropy_z",
        aggfunc="first",
    )
    early = pivot.xs("early", level="half", drop_level=True).reindex(columns=CORE_TARGETS)
    late = pivot.xs("late", level="half", drop_level=True).reindex(columns=CORE_TARGETS)
    common = early.index.intersection(late.index)
    early = early.loc[common]
    late = late.loc[common]
    usable = (early.notna() & late.notna()).sum(axis=1) >= 6
    return early.loc[usable].fillna(0.0), late.loc[usable].fillna(0.0)


def validate_typology(
    metrics: pd.DataFrame,
    rng: np.random.Generator,
    n_permutations: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    fold_rows = []
    summaries = []
    profiles_by_regime = {}
    for regime in ["fixed_original", "disjoint_refit"]:
        early, late = make_profiles(metrics, regime)
        profiles_by_regime[regime] = early
        institutes = np.asarray(early.index.get_level_values("institute_year"))
        for k in range(2, 6):
            for held_out in sorted(set(institutes)):
                train_mask = institutes != held_out
                test_mask = institutes == held_out
                centroids, _, _ = spherical_kmeans(
                    early.to_numpy(dtype=float)[train_mask], k, rng
                )
                early_labels = assign_clusters(
                    early.to_numpy(dtype=float)[test_mask], centroids
                )
                late_labels = assign_clusters(
                    late.to_numpy(dtype=float)[test_mask], centroids
                )
                accuracy = float(np.mean(early_labels == late_labels))
                ari = adjusted_rand_index(early_labels, late_labels)
                null = []
                for _ in range(n_permutations):
                    null.append(
                        float(
                            np.mean(
                                early_labels == rng.permutation(late_labels)
                            )
                        )
                    )
                counts = np.bincount(early_labels, minlength=k) / len(early_labels)
                fold_rows.append(
                    {
                        "regime": regime,
                        "k": k,
                        "held_out_cohort": held_out,
                        "n_trajectories": int(len(early_labels)),
                        "assignment_accuracy": accuracy,
                        "adjusted_rand_index": ari,
                        "accuracy_permutation_p": float(
                            (1 + np.sum(np.asarray(null) >= accuracy))
                            / (n_permutations + 1)
                        ),
                        "minimum_cluster_prevalence": float(counts.min()),
                        "early_silhouette": cosine_silhouette(
                            early.to_numpy(dtype=float)[test_mask], early_labels
                        ),
                    }
                )
            current = pd.DataFrame(fold_rows)
            current = current[
                current["regime"].eq(regime) & current["k"].eq(k)
            ]
            successful = (
                current["adjusted_rand_index"].gt(0.20)
                & current["accuracy_permutation_p"].lt(0.05)
                & current["minimum_cluster_prevalence"].ge(0.10)
            )
            summaries.append(
                {
                    "regime": regime,
                    "k": k,
                    "median_adjusted_rand_index": float(
                        current["adjusted_rand_index"].median()
                    ),
                    "median_assignment_accuracy": float(
                        current["assignment_accuracy"].median()
                    ),
                    "median_silhouette": float(current["early_silhouette"].median()),
                    "successful_cohorts": int(successful.sum()),
                    "minimum_prevalence_across_folds": float(
                        current["minimum_cluster_prevalence"].min()
                    ),
                }
            )
    return pd.DataFrame(fold_rows), pd.DataFrame(summaries), profiles_by_regime


def final_centroids(
    profiles: pd.DataFrame,
    k: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    centroids, labels, objective = spherical_kmeans(
        profiles.to_numpy(dtype=float), k, rng, n_init=100
    )
    centroid_frame = pd.DataFrame(centroids, columns=CORE_TARGETS)
    centroid_frame.insert(0, "cluster", np.arange(k))
    centroid_frame["top_positive_domain"] = [
        CORE_TARGETS[int(np.argmax(row))] for row in centroids
    ]
    centroid_frame["top_negative_domain"] = [
        CORE_TARGETS[int(np.argmin(row))] for row in centroids
    ]
    assignment = profiles.reset_index()[["institute_year", "user_id"]]
    assignment["cluster"] = labels
    assignment["objective"] = objective
    return centroid_frame, assignment


def plot_validation(summary: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    fixed = summary[summary["regime"].eq("fixed_original")]
    disjoint = summary[summary["regime"].eq("disjoint_refit")]
    x = np.arange(2, 6)
    width = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(
        x - width / 2,
        fixed["median_adjusted_rand_index"],
        width,
        label="Fixed",
    )
    axes[0].bar(
        x + width / 2,
        disjoint["median_adjusted_rand_index"],
        width,
        label="Disjoint",
    )
    axes[0].axhline(0.20, color="black", linestyle="--", linewidth=1)
    axes[0].set_xticks(x)
    axes[0].set_xlabel("Number of clusters")
    axes[0].set_ylabel("Median held-out adjusted Rand index")
    axes[0].legend()
    axes[1].bar(
        x - width / 2,
        fixed["successful_cohorts"],
        width,
        label="Fixed",
    )
    axes[1].bar(
        x + width / 2,
        disjoint["successful_cohorts"],
        width,
        label="Disjoint",
    )
    axes[1].axhline(3, color="black", linestyle="--", linewidth=1)
    axes[1].set_xticks(x)
    axes[1].set_xlabel("Number of clusters")
    axes[1].set_ylabel("Successful held-out cohorts (of 4)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--permutations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=307)
    args = parser.parse_args()

    root = args.repo_root.resolve()
    output = root / "results/structure_followup"
    output.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(
        root / "results/referee_robustness/target_half_metrics.csv"
    )
    rng = np.random.default_rng(args.seed)
    folds, summary, profiles = validate_typology(metrics, rng, args.permutations)

    supported = []
    for k in range(2, 6):
        fixed = summary[
            summary["regime"].eq("fixed_original") & summary["k"].eq(k)
        ].iloc[0]
        disjoint = summary[
            summary["regime"].eq("disjoint_refit") & summary["k"].eq(k)
        ].iloc[0]
        if (
            fixed["median_adjusted_rand_index"] > 0.20
            and fixed["successful_cohorts"] >= 3
            and disjoint["median_adjusted_rand_index"] > 0.20
            and disjoint["successful_cohorts"] >= 3
        ):
            supported.append(k)

    best_row = summary[summary["regime"].eq("fixed_original")].sort_values(
        ["successful_cohorts", "median_adjusted_rand_index"], ascending=False
    ).iloc[0]
    descriptive_k = supported[0] if supported else int(best_row["k"])
    centroids, assignments = final_centroids(
        profiles["fixed_original"], descriptive_k, rng
    )
    folds.to_csv(output / "typology_leave_one_cohort_out.csv", index=False)
    summary.to_csv(output / "typology_k_summary.csv", index=False)
    centroids.to_csv(output / "typology_centroids.csv", index=False)
    assignments.to_csv(output / "typology_assignments.csv", index=False)
    plot_validation(summary, output / "typology_validation.png")

    payload = {
        "status": "post_hoc_exploratory",
        "contract": "experiments/README.md",
        "supported_typology": bool(supported),
        "supported_k": supported,
        "descriptive_k": descriptive_k,
        "decision": (
            "Discrete profile types pass the cross-cohort gate."
            if supported
            else "No k passes the cross-cohort typology gate; retain continuous profiles."
        ),
        "validation": summary.to_dict(orient="records"),
        "descriptive_centroids": centroids.to_dict(orient="records"),
    }
    (output / "typology_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
