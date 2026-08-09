from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


AMBER_COMPATIBLE_FAMILIES = ("screen", "steps", "sleep")
GLOBEM_ID_COLUMNS = ("pid", "date")


def find_institute_years(globem_root: str | Path) -> list[Path]:
    """Return GLOBEM institute-year directories such as INS-W_1."""
    root = Path(globem_root)
    if not root.exists():
        raise FileNotFoundError(f"GLOBEM root does not exist: {root}")

    candidates = sorted(path for path in root.rglob("INS-W_*") if path.is_dir())
    return candidates


def audit_globem_root(globem_root: str | Path) -> dict[str, object]:
    """Small, dependency-light audit used before implementing feature loading."""
    root = Path(globem_root)
    institute_years = find_institute_years(root)
    return {
        "root": str(root),
        "institute_year_count": len(institute_years),
        "institute_years": [path.name for path in institute_years],
    }


def read_csv_clean(path: str | Path, **kwargs) -> pd.DataFrame:
    kwargs.setdefault("low_memory", False)
    df = pd.read_csv(path, **kwargs)
    unnamed = [col for col in df.columns if str(col).startswith("Unnamed:")]
    if unnamed:
        df = df.drop(columns=unnamed)
    return df


def dataset_index(name: str) -> int:
    match = re.search(r"_(\d+)$", name)
    if not match:
        raise ValueError(f"Cannot parse dataset index from {name}")
    return int(match.group(1))


def parse_feature_column(column: str) -> dict[str, str | None]:
    if ":" not in column:
        return {
            "column": column,
            "family": None,
            "feature": None,
            "version": None,
            "segment": None,
        }

    family, rest, segment = column.split(":", 2)
    version = "raw"
    feature = rest
    for suffix, name in (("_norm", "norm"), ("_dis", "dis")):
        if rest.endswith(suffix):
            version = name
            feature = rest[: -len(suffix)]
            break

    return {
        "column": column,
        "family": family,
        "feature": feature,
        "version": version,
        "segment": segment,
    }


def audit_dataset(dataset_dir: str | Path) -> dict[str, object]:
    ds = Path(dataset_dir)
    files = sorted(path for path in ds.rglob("*.csv"))
    file_reports = []
    for path in files:
        header = pd.read_csv(path, nrows=0)
        row_count = sum(1 for _ in path.open("r", encoding="utf-8")) - 1
        file_reports.append(
            {
                "dataset": ds.name,
                "relative_path": str(path.relative_to(ds)),
                "rows": row_count,
                "columns": len(header.columns),
            }
        )
    return {"dataset": ds.name, "files": file_reports}


def load_family_features(dataset_dir: str | Path, family: str) -> pd.DataFrame:
    path = Path(dataset_dir) / "FeatureData" / f"{family}.csv"
    df = read_csv_clean(path)
    df["pid"] = df["pid"].astype(str)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_weekly_labels(dataset_dir: str | Path) -> pd.DataFrame:
    ds = Path(dataset_dir)
    labels = read_csv_clean(ds / "SurveyData" / "dep_weekly.csv")
    labels["pid"] = labels["pid"].astype(str)
    labels["date"] = pd.to_datetime(labels["date"])
    keep = [col for col in ["pid", "date", "dep", "phq4", "BDI2", "feel_anxious", "feel_depressed"] if col in labels]
    labels = labels[keep].rename(columns={"dep": "target_dep_weekly"})
    labels["label_available"] = labels["target_dep_weekly"].notna()
    return labels


def load_platform(dataset_dir: str | Path) -> pd.DataFrame:
    path = Path(dataset_dir) / "ParticipantsInfoData" / "platform.csv"
    platform = read_csv_clean(path)
    platform["pid"] = platform["pid"].astype(str)
    return platform


def load_canonical_daily(
    globem_root: str | Path,
    families: tuple[str, ...] = AMBER_COMPATIBLE_FAMILIES,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for ds in find_institute_years(globem_root):
        merged: pd.DataFrame | None = None
        for family in families:
            family_df = load_family_features(ds, family)
            renamed = family_df.rename(
                columns={
                    col: f"{family}__{col}"
                    for col in family_df.columns
                    if col not in GLOBEM_ID_COLUMNS
                }
            )
            if merged is None:
                merged = renamed
            else:
                merged = merged.merge(renamed, on=["pid", "date"], how="outer")

        if merged is None:
            continue

        labels = load_weekly_labels(ds)
        platform = load_platform(ds)
        merged = merged.merge(labels, on=["pid", "date"], how="left")
        merged = merged.merge(platform, on="pid", how="left")
        merged = merged.rename(columns={"pid": "user_id"})
        merged.insert(2, "institute_year", ds.name)
        merged.insert(3, "dataset_idx", dataset_index(ds.name))
        merged.insert(4, "weekday", merged["date"].dt.weekday)
        feature_cols = [
            col
            for col in merged.columns
            if col.startswith(("screen__", "steps__", "sleep__"))
        ]
        merged["feature_coverage"] = merged[feature_cols].notna().mean(axis=1)
        merged["target_name"] = "dep_weekly"
        frames.append(merged.copy())

    if not frames:
        raise ValueError(f"No institute-year data found under {globem_root}")

    daily = pd.concat(frames, ignore_index=True)
    daily = daily.sort_values(["institute_year", "user_id", "date"]).reset_index(drop=True)
    return daily


def load_canonical_rapids_daily(globem_root: str | Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for ds in find_institute_years(globem_root):
        rapids = read_csv_clean(ds / "FeatureData" / "rapids.csv")
        rapids["pid"] = rapids["pid"].astype(str)
        rapids["date"] = pd.to_datetime(rapids["date"])
        labels = load_weekly_labels(ds)
        platform = load_platform(ds)
        merged = rapids.merge(labels, on=["pid", "date"], how="left")
        merged = merged.merge(platform, on="pid", how="left")
        merged = merged.rename(columns={"pid": "user_id"})
        metadata = pd.DataFrame(
            {
                "institute_year": ds.name,
                "dataset_idx": dataset_index(ds.name),
                "weekday": merged["date"].dt.weekday,
            },
            index=merged.index,
        )
        feature_cols = [
            col
            for col in merged.columns
            if col.startswith(("f_loc:", "f_screen:", "f_call:", "f_blue:", "f_steps:", "f_slp:"))
        ]
        merged["feature_coverage"] = merged[feature_cols].notna().mean(axis=1)
        merged["target_name"] = "dep_weekly"
        frames.append(pd.concat([merged[["user_id", "date"]], metadata, merged.drop(columns=["user_id", "date"])], axis=1).copy())

    if not frames:
        raise ValueError(f"No institute-year data found under {globem_root}")
    daily = pd.concat(frames, ignore_index=True)
    daily = daily.sort_values(["institute_year", "user_id", "date"]).reset_index(drop=True)
    return daily


def feature_catalog(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in daily.columns:
        if "__" not in col:
            continue
        source_family, raw = col.split("__", 1)
        parsed = parse_feature_column(raw)
        rows.append(
            {
                "canonical_column": col,
                "source_family_file": source_family,
                **parsed,
                "missing_rate": float(daily[col].isna().mean()),
                "non_null_count": int(daily[col].notna().sum()),
            }
        )
    return pd.DataFrame(rows)
