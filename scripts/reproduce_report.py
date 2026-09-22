from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report"


def run(command: list[str], cwd: Path = ROOT) -> None:
    print(f"\n>>> {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def script(relative: str) -> None:
    run([sys.executable, relative])


def verify() -> None:
    script("scripts/public_release_audit.py")
    script("scripts/check_markdown_links.py")
    script("scripts/verify_public_artifacts.py")
    run([sys.executable, "-m", "pytest"])


def build() -> None:
    verify()
    script("report/make_figures.py")
    tectonic = shutil.which("tectonic")
    if tectonic is None:
        raise SystemExit("Tectonic is required only for build/full, not verify")
    run([tectonic, "main.tex"], cwd=REPORT)
    script("scripts/verify_public_artifacts.py")


def full() -> None:
    for relative in [
        "scripts/check_globem_layout.py",
        "scripts/audit_globem.py",
        "experiments/run_label_baseline_control.py",
        "experiments/run_rq1_clustered_inference.py",
        "experiments/run_rq1_within_person_deviation.py",
        "experiments/run_rq1_within_person_power_diagnostic.py",
        "experiments/run_rq1_split_seed_sensitivity.py",
        "experiments/run_construct_fragility_audit.py",
        "experiments/run_construct_overlap_sensitivity.py",
        "experiments/run_behavioural_weather_forecast.py",
        "experiments/build_report_statistical_supplements.py",
        "experiments/run_forecastability_phenotype.py",
        "experiments/run_referee_robustness_checks.py",
        "experiments/run_structure_mechanism_followup.py",
        "experiments/run_profile_typology_followup.py",
        "experiments/run_common_distress_forecastability.py",
        "experiments/personalization/run.py",
    ]:
        script(relative)
    build()


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify or reproduce the project report")
    parser.add_argument("mode", choices=("verify", "build", "full"))
    args = parser.parse_args()
    {"verify": verify, "build": build, "full": full}[args.mode]()


if __name__ == "__main__":
    main()
