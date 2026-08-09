from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ALLOWED_TOP_LEVEL = {
    ".github",
    ".gitignore",
    "CITATION.cff",
    "DATA_ACCESS.md",
    "LICENSE",
    "Makefile",
    "README.md",
    "REPRODUCIBILITY.md",
    "configs",
    "data",
    "experiments",
    "pyproject.toml",
    "report",
    "requirements-lock.txt",
    "results",
    "scripts",
    "src",
    "tests",
}
FORBIDDEN_SUFFIXES = {
    ".db", ".joblib", ".parquet", ".pickle", ".pkl", ".pt", ".pth", ".sqlite"
}
FORBIDDEN_PREFIXES = (
    "data/raw/", "data/interim/", "data/processed/", "data/external/"
)
FORBIDDEN_NAME_PARTS = (
    "common_distress_user_table",
    "half_composites",
    "per_user",
    "prediction_samples",
    "predictions",
    "samples_",
    "sequence_metrics",
    "target_half_metrics",
    "typology_assignments",
    "user_coverage",
    "user_level",
)
IDENTIFIER_COLUMNS = {"pid", "user_id", "participant_id", "person_id"}


def release_files() -> list[Path]:
    if (ROOT / ".git").exists():
        result = subprocess.run(
            ["git", "ls-files", "-z"], cwd=ROOT, check=True, capture_output=True
        )
        names = [value for value in result.stdout.decode().split("\0") if value]
        if names:
            return [ROOT / name for name in names]
    excluded = {".git", ".venv", ".pytest_cache", ".ruff_cache", "__pycache__"}
    return sorted(
        path for path in ROOT.rglob("*")
        if path.is_file()
        and not any(
            part in excluded or part.endswith(".egg-info")
            for part in path.relative_to(ROOT).parts
        )
    )


def csv_identifiers(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle), [])
    return sorted({value.strip().lower() for value in header} & IDENTIFIER_COLUMNS)


def audit(files: list[Path]) -> dict[str, object]:
    failures: list[dict[str, object]] = []
    for path in files:
        if not path.exists():
            continue
        relative = path.relative_to(ROOT).as_posix()
        top_level = Path(relative).parts[0]
        if top_level not in ALLOWED_TOP_LEVEL:
            failures.append({"path": relative, "reason": "unapproved top-level content"})
        if relative.startswith(FORBIDDEN_PREFIXES):
            failures.append({"path": relative, "reason": "private data path"})
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            failures.append({"path": relative, "reason": "private data/model format"})
        if any(part in path.name.lower() for part in FORBIDDEN_NAME_PARTS):
            failures.append({"path": relative, "reason": "participant-derived artifact name"})
        if path.stat().st_size > 20 * 1024 * 1024:
            failures.append({"path": relative, "reason": "file exceeds 20 MiB"})
        if path.suffix.lower() == ".csv":
            identifiers = csv_identifiers(path)
            if identifiers:
                failures.append(
                    {"path": relative, "reason": "identifier-bearing CSV", "columns": identifiers}
                )
    return {
        "status": "ok" if not failures else "failed",
        "files_checked": len(files),
        "failures": failures,
    }


def main() -> None:
    report = audit(release_files())
    print(json.dumps(report, indent=2))
    if report["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
