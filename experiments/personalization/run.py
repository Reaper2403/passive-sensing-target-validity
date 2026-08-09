"""Outcome-blind Phase A availability and robust-scale feasibility audit.

This module deliberately does not fit models, generate predictions, calculate
errors, or write identifier-level outputs. It reads target values only to reject
invalid nonmissing cells and to perform the proposal-authorized robust-scale
degeneracy check, retaining only aggregate counts and pass/fail indicators.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = REPO_ROOT / "data/raw/globem/1.1"
STUDY_ROOT = REPO_ROOT / "experiments/personalization"
REPEAT_COMMIT = "4f140fc5290dc97204298cca28b956165aa0a29f"
REPEAT_URL = (
    "https://raw.githubusercontent.com/UW-EXP/GLOBEM/"
    f"{REPEAT_COMMIT}/data/additional_user_setup/overlapping_pids.json"
)
REPEAT_SHA256 = "8320d6b2a012bec53c5cf777dd8f127d5d8162a0e483655b028394899403d532"
YEARS = ("INS-W_1", "INS-W_2", "INS-W_3", "INS-W_4")

# Exact 16 targets imported from experiments/run_behavioural_weather_forecast.py.
TARGETS: dict[str, str] = {
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

TARGET_BOUNDS: dict[str, tuple[float | None, float | None]] = {
    "screen_unlock_count": (0.0, None),
    "screen_unlock_duration": (0.0, 1440.0),
    "steps_sum": (0.0, None),
    "active_duration": (0.0, 1440.0),
    "sedentary_duration": (0.0, 1440.0),
    "sleep_duration": (0.0, 1440.0),
    "sleep_in_bed": (0.0, 1440.0),
    "sleep_efficiency": (0.0, 100.0),
    "location_home_time": (0.0, 1440.0),
    "location_distance": (0.0, None),
    "location_entropy": (0.0, None),
    "significant_places": (0.0, None),
    "incoming_call_count": (0.0, None),
    "outgoing_call_count": (0.0, None),
    "bluetooth_unique_devices": (0.0, None),
    "wifi_unique_devices": (0.0, None),
}

FAMILIES: dict[str, tuple[str, ...]] = {
    "mobility": (
        "location_home_time",
        "location_distance",
        "location_entropy",
        "significant_places",
    ),
    "physical_activity": ("steps_sum", "active_duration", "sedentary_duration"),
    "sleep_daily_rhythm": ("sleep_duration", "sleep_in_bed", "sleep_efficiency"),
    "phone_interaction": (
        "screen_unlock_count",
        "screen_unlock_duration",
        "incoming_call_count",
        "outgoing_call_count",
        "bluetooth_unique_devices",
        "wifi_unique_devices",
    ),
}

BRANCHES = {
    "preferred": {"adaptation": (1, 28, 14), "evaluation": (29, 56, 14)},
    "reduced": {"adaptation": (1, 14, 7), "evaluation": (15, 42, 14)},
}
MIN_ELIGIBLE_PER_YEAR = 30
MIN_ELIGIBLE_TOTAL = 160


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_repeat_candidates(content: bytes) -> set[tuple[str, str]]:
    metadata = json.loads(content)["dep_weekly"]
    candidates: set[tuple[str, str]] = set()
    for reference_year, year_lists in metadata.items():
        if not reference_year.startswith("INS-W_"):
            continue
        for institute_year, identifiers in year_lists.items():
            if not institute_year.startswith("INS-W_"):
                continue
            candidates.update(
                (institute_year, str(identifier).split("#", maxsplit=1)[0])
                for identifier in identifiers
            )
    return candidates


def fetch_repeat_metadata(destination: Path) -> set[tuple[str, str]]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(REPEAT_URL, timeout=30) as response:
        content = response.read()
    observed_sha = hashlib.sha256(content).hexdigest()
    if observed_sha != REPEAT_SHA256:
        raise RuntimeError(f"Repeat metadata SHA-256 mismatch: {observed_sha}")
    destination.write_bytes(content)
    return parse_repeat_candidates(content)


def validate_target_values(data: pd.DataFrame) -> pd.DataFrame:
    """Reject invalid nonmissing target cells before availability is counted."""
    validated = data.copy()
    for target, (lower, upper) in TARGET_BOUNDS.items():
        original = validated[target]
        present = original.notna()
        numeric = pd.to_numeric(original, errors="coerce")
        finite = pd.Series(np.isfinite(numeric.to_numpy(dtype=float)), index=data.index)
        invalid_numeric = present & ~finite
        below = present & finite & numeric.lt(lower) if lower is not None else False
        above = present & finite & numeric.gt(upper) if upper is not None else False
        invalid = invalid_numeric | below | above
        if bool(invalid.any()):
            raise RuntimeError(
                f"Target {target} has {int(invalid.sum())} nonmissing non-finite or "
                "out-of-bounds cells"
            )
        validated[target] = numeric
    return validated


def add_study_day(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    first_dates = result.groupby("user_id", sort=False)["date"].transform("min")
    result["study_day"] = (result["date"] - first_dates).dt.days + 1
    return result


def branch_eligibility(
    data: pd.DataFrame, target_column: str, branch: str
) -> dict[str, set[tuple[str, str]]]:
    spec = BRANCHES[branch]
    result: dict[str, set[tuple[str, str]]] = {}
    for block in ("adaptation", "evaluation"):
        first, last, minimum = spec[block]
        block_rows = data.loc[
            data["study_day"].between(first, last) & data[target_column].notna(),
            ["institute_year", "user_id"],
        ]
        counts = block_rows.value_counts()
        result[block] = set(counts[counts >= minimum].index.tolist())
    result["eligible"] = result["adaptation"] & result["evaluation"]
    return result


def robust_scale_is_degenerate(values: pd.Series) -> bool:
    """Proposal-authorized feasibility check; never return/record value statistics."""
    x = values.dropna().to_numpy(dtype=float)
    if not np.isfinite(x).all():
        raise RuntimeError("Robust-scale input contains a non-finite value")
    if len(x) == 0:
        return True
    median = float(np.median(x))
    iqr = float(np.quantile(x, 0.75) - np.quantile(x, 0.25))
    mad = float(np.median(np.abs(x - median)))
    scale = max(iqr, 1.4826 * mad)
    return not np.isfinite(scale) or scale <= 1e-6 * max(1.0, abs(median))


def read_raw(repeat_candidates: set[tuple[str, str]]) -> tuple[pd.DataFrame, dict[str, Any]]:
    target_columns = {name: f"{base}:allday" for name, base in TARGETS.items()}
    usecols = ["pid", "date", *target_columns.values()]
    frames: list[pd.DataFrame] = []
    year_report: dict[str, Any] = {}
    for year in YEARS:
        path = RAW_ROOT / year / "FeatureData/rapids.csv"
        frame = pd.read_csv(path, usecols=usecols, low_memory=False)
        frame = frame.rename(columns={"pid": "user_id", **{v: k for k, v in target_columns.items()}})
        frame["user_id"] = frame["user_id"].astype(str)
        frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
        frame["institute_year"] = year
        duplicate_rows = int(frame.duplicated(["user_id", "date"]).sum())
        if duplicate_rows:
            raise RuntimeError(f"{year} has {duplicate_rows} duplicate person-date rows")
        raw_trajectories = int(frame["user_id"].nunique())
        raw_rows = int(len(frame))
        repeat_mask = frame.apply(
            lambda row: (year, row["user_id"]) in repeat_candidates, axis=1
        )
        repeat_trajectories = int(frame.loc[repeat_mask, "user_id"].nunique())
        frame = frame.loc[~repeat_mask].copy()
        frame = validate_target_values(frame)
        frame = add_study_day(frame)
        year_report[year] = {
            "raw_person_trajectories": raw_trajectories,
            "excluded_repeat_candidate_trajectories": repeat_trajectories,
            "remaining_person_trajectories": int(frame["user_id"].nunique()),
            "raw_person_day_rows_before_exclusion": raw_rows,
            "retained_person_day_rows": int(len(frame)),
            "duplicate_person_date_rows": duplicate_rows,
        }
        frames.append(frame)
    return pd.concat(frames, ignore_index=True), year_report


def target_counts(data: pd.DataFrame) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for target in TARGETS:
        branches = {branch: branch_eligibility(data, target, branch) for branch in BRANCHES}
        by_branch: dict[str, Any] = {}
        for branch, blocks in branches.items():
            by_year = {
                year: int(sum(pair[0] == year for pair in blocks["eligible"])) for year in YEARS
            }
            by_branch[branch] = {
                "eligible_person_targets_by_institute_year": by_year,
                "eligible_person_targets_total": int(len(blocks["eligible"])),
            }
        report[target] = by_branch
    return report


def select_targets(counts: dict[str, Any]) -> tuple[dict[str, str], dict[str, Any]]:
    selected: dict[str, str] = {}
    scorecards: dict[str, Any] = {}
    for family, candidates in FAMILIES.items():
        rows = []
        for target in candidates:
            record = counts[target]["preferred"]
            by_year = record["eligible_person_targets_by_institute_year"]
            rows.append(
                {
                    "target": target,
                    "minimum_eligible_person_targets_across_institute_years": min(by_year.values()),
                    "total_eligible_person_targets": record["eligible_person_targets_total"],
                }
            )
        rows.sort(
            key=lambda row: (
                -row["minimum_eligible_person_targets_across_institute_years"],
                -row["total_eligible_person_targets"],
                row["target"],
            )
        )
        selected[family] = rows[0]["target"]
        scorecards[family] = rows
    return selected, scorecards


def families_passing(counts: dict[str, Any], selected: dict[str, str], branch: str) -> list[str]:
    passing = []
    for family, target in selected.items():
        record = counts[target][branch]
        by_year = record["eligible_person_targets_by_institute_year"]
        if min(by_year.values()) >= MIN_ELIGIBLE_PER_YEAR and record["eligible_person_targets_total"] >= MIN_ELIGIBLE_TOTAL:
            passing.append(family)
    return passing


def decide_branch(
    counts: dict[str, Any], retained: dict[str, str]
) -> tuple[str | None, list[str], list[str], list[str], str]:
    preferred_passing = families_passing(counts, retained, "preferred")
    reduced_passing = families_passing(counts, retained, "reduced")
    if len(preferred_passing) >= 3:
        return "preferred", preferred_passing, reduced_passing, preferred_passing, "feasible"
    if len(reduced_passing) >= 3:
        return "reduced", preferred_passing, reduced_passing, reduced_passing, "feasible"
    return (
        None,
        preferred_passing,
        reduced_passing,
        [],
        "feasibility_boundary_no_personalization_test",
    )


def file_hashes() -> dict[str, str]:
    paths = [RAW_ROOT / year / "FeatureData/rapids.csv" for year in YEARS]
    paths.extend([REPO_ROOT / "experiments/run_behavioural_weather_forecast.py", Path(__file__)])
    return {str(path.relative_to(REPO_ROOT)): sha256(path) for path in paths}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=STUDY_ROOT / "result.json")
    parser.add_argument(
        "--repeat-metadata", type=Path, default=STUDY_ROOT / "private/overlapping_pids.json"
    )
    args = parser.parse_args()

    repeats = fetch_repeat_metadata(args.repeat_metadata)
    if sha256(args.repeat_metadata) != REPEAT_SHA256:
        raise RuntimeError("Saved repeat metadata failed SHA-256 verification")
    data, cohort_report = read_raw(repeats)
    counts = target_counts(data)
    selected, scorecards = select_targets(counts)

    degenerate_families: list[str] = []
    scale_folds: dict[str, dict[str, bool]] = {}
    for family, target in selected.items():
        scale_folds[family] = {}
        for held_out in YEARS:
            is_degenerate = robust_scale_is_degenerate(
                data.loc[data["institute_year"].ne(held_out), target]
            )
            scale_folds[family][held_out] = is_degenerate
        if any(scale_folds[family].values()):
            degenerate_families.append(family)
    retained = {family: target for family, target in selected.items() if family not in degenerate_families}

    branch, preferred_passing, reduced_passing, retained_for_execution, status = decide_branch(
        counts, retained
    )

    report = {
        "audit_type": "Phase A count-only, outcome-blind feasibility audit",
        "generated_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "outcome_analysis_performed": False,
        "allowed_value_operation": "hard finite/bounds validation plus robust-scale degeneracy pass/fail only; no values or scale statistics emitted",
        "repeat_metadata": {
            "url": REPEAT_URL,
            "commit": REPEAT_COMMIT,
            "expected_sha256": REPEAT_SHA256,
            "verified_sha256": sha256(args.repeat_metadata),
            "candidate_trajectory_key_count": len(repeats),
            "pid_metadata_written_to_git": False,
        },
        "study_day_rule": "calendar day, 1-indexed from each retained person-trajectory's earliest RAPIDS date",
        "cohorts_after_repeat_exclusion": cohort_report,
        "target_columns": {target: f"{base}:allday" for target, base in TARGETS.items()},
        "eligibility_counts": counts,
        "selection_basis": "preferred-branch missingness only; rank min cohort eligibility desc, total desc, target ID asc",
        "selection_scorecards": scorecards,
        "selected_target_by_family": selected,
        "robust_scale_degeneracy": {
            "operation": "per selected target and held-out institute-year using all observed target values from the three outer-training institute-years; pass/fail only",
            "degenerate_by_family_and_held_out_institute_year": scale_folds,
            "globally_removed_families": degenerate_families,
        },
        "branch_decision": {
            "status": status,
            "selected_branch": branch,
            "families_passing_preferred": preferred_passing,
            "families_passing_reduced": reduced_passing,
            "families_retained_for_execution": retained_for_execution,
            "selected_targets_retained_for_execution": {
                family: retained[family] for family in retained_for_execution
            },
            "sample_size_gate": {
                "minimum_eligible_person_targets_per_institute_year": MIN_ELIGIBLE_PER_YEAR,
                "minimum_eligible_person_targets_total": MIN_ELIGIBLE_TOTAL,
            },
        },
        "input_file_sha256": file_hashes(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
