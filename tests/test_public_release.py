from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_release_has_only_approved_public_artifacts() -> None:
    audit = load_script("public_release_audit")
    report = audit.audit(audit.release_files())
    assert report["status"] == "ok", report["failures"]


def test_report_headlines_match_aggregate_results() -> None:
    verify = load_script("verify_public_artifacts")
    assert len(verify.verify_headlines()) == 7
    verify.verify_report()


def test_layout_check_is_outcome_blind(tmp_path: Path) -> None:
    checker = load_script("check_globem_layout")
    report = checker.check_layout(tmp_path)
    assert report["status"] == "failed"
    assert len(report["missing_files"]) == 32
