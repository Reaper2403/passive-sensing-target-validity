from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
YEARS = ("INS-W_1", "INS-W_2", "INS-W_3", "INS-W_4")
REQUIRED_FILES = (
    "FeatureData/rapids.csv",
    "FeatureData/screen.csv",
    "FeatureData/sleep.csv",
    "FeatureData/steps.csv",
    "ParticipantsInfoData/platform.csv",
    "SurveyData/dep_weekly.csv",
    "SurveyData/ema.csv",
    "SurveyData/pre.csv",
)


def check_layout(data_root: Path) -> dict[str, object]:
    missing: list[str] = []
    empty: list[str] = []
    for year in YEARS:
        for relative in REQUIRED_FILES:
            path = data_root / year / relative
            display = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
            if not path.is_file():
                missing.append(display)
            elif path.stat().st_size == 0:
                empty.append(display)
    return {
        "status": "ok" if not missing and not empty else "failed",
        "data_root": str(data_root),
        "expected_institute_years": list(YEARS),
        "missing_files": missing,
        "empty_files": empty,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the local GLOBEM layout without reading participant rows."
    )
    parser.add_argument(
        "--data-root", type=Path, default=ROOT / "data/raw/globem/1.1"
    )
    report = check_layout(parser.parse_args().data_root.resolve())
    print(json.dumps(report, indent=2))
    if report["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
