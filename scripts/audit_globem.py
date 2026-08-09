from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from amber_thesis.config import get_paths, load_config
from amber_thesis.data.globem_loader import (
    audit_dataset,
    feature_catalog,
    find_institute_years,
    load_canonical_daily,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/local_smoke.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    paths = get_paths(config)
    paths.ensure_output_dirs()
    audit_dir = paths.results / "data_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    dataset_dirs = find_institute_years(paths.globem_raw)
    structure = [audit_dataset(path) for path in dataset_dirs]
    (audit_dir / "globem_structure.json").write_text(json.dumps(structure, indent=2), encoding="utf-8")

    daily = load_canonical_daily(paths.globem_raw)
    canonical_path = paths.processed / "globem_daily_amber_compatible.parquet"
    daily.to_parquet(canonical_path, index=False)

    catalog = feature_catalog(daily)
    catalog.to_csv(audit_dir / "feature_catalog.csv", index=False)

    row_counts = (
        daily.groupby("institute_year")
        .agg(
            rows=("user_id", "size"),
            users=("user_id", "nunique"),
            labelled_rows=("label_available", "sum"),
            mean_feature_coverage=("feature_coverage", "mean"),
        )
        .reset_index()
    )
    row_counts.to_csv(audit_dir / "row_counts.csv", index=False)

    label_counts = (
        daily[daily["target_dep_weekly"].notna()]
        .groupby(["institute_year", "target_dep_weekly"])
        .size()
        .reset_index(name="rows")
    )
    label_counts.to_csv(audit_dir / "label_counts.csv", index=False)

    user_coverage = (
        daily.groupby(["institute_year", "user_id"])
        .agg(
            days=("date", "nunique"),
            labelled_days=("label_available", "sum"),
            mean_feature_coverage=("feature_coverage", "mean"),
        )
        .reset_index()
    )
    user_coverage.to_csv(audit_dir / "user_coverage.csv", index=False)

    summary = {
        "canonical_daily_path": str(canonical_path),
        "rows": int(len(daily)),
        "columns": int(len(daily.columns)),
        "users": int(daily["user_id"].nunique()),
        "institute_years": sorted(daily["institute_year"].unique().tolist()),
        "labelled_rows": int(daily["label_available"].sum()),
        "feature_columns": int(len(catalog)),
    }
    (audit_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
