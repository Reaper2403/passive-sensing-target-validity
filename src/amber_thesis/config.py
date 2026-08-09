from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ProjectPaths:
    globem_raw: Path
    processed: Path
    results: Path

    def ensure_output_dirs(self) -> None:
        self.processed.mkdir(parents=True, exist_ok=True)
        self.results.mkdir(parents=True, exist_ok=True)


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    return config


def get_paths(config: dict[str, Any]) -> ProjectPaths:
    paths = config.get("paths", {})
    required = ("globem_raw", "processed", "results")
    missing = [key for key in required if key not in paths]
    if missing:
        raise KeyError(f"Missing path config keys: {missing}")

    return ProjectPaths(
        globem_raw=Path(paths["globem_raw"]),
        processed=Path(paths["processed"]),
        results=Path(paths["results"]),
    )
