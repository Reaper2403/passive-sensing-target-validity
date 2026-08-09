from __future__ import annotations

import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(relative: str) -> dict[str, object]:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def close(name: str, observed: float, expected: float) -> None:
    if not math.isclose(float(observed), expected, rel_tol=0.0, abs_tol=1e-12):
        raise AssertionError(f"{name}: expected {expected}, found {observed}")


def verify_headlines() -> list[str]:
    checked: list[str] = []
    labels = load("results/label_baseline_control/summary.json")
    by_feature = {row["feature_set"]: row for row in labels["metrics"]}
    close("label-only AUROC", by_feature["prior_label_rate_only"]["auroc"], 0.8444280674172024)
    close(
        "current-day plus label AUROC",
        by_feature["allday_raw_plus_prior_label_rate"]["auroc"],
        0.8403424670353838,
    )
    close(
        "history plus label AUROC",
        by_feature["history_raw_plus_prior_label_rate"]["auroc"],
        0.8413784309872072,
    )
    checked.append("label-history control")

    fragility = load("results/construct_fragility_audit/summary.json")
    close("construct overlap", fragility["median_same_day_abs_r"], 0.6628291199453776)
    checked.append("construct overlap")

    behavior = load("results/behavioural_weather_forecast/summary.json")["aggregate"]
    close(
        "tomorrow R2",
        behavior["regression"]["tomorrow"]["evening"]["mean_history_fingerprint_plus_today"],
        0.2716253862926227,
    )
    close(
        "tomorrow AUROC",
        behavior["classification"]["tomorrow"]["evening"][
            "mean_history_fingerprint_plus_today"
        ],
        0.78754700157869,
    )
    checked.append("behavior forecast")

    robust = load("results/referee_robustness/summary.json")
    close(
        "forecastability/variability",
        robust["fixed_model_core_variability_overlap"]["pearson_r"],
        -0.9121934476195503,
    )
    adjusted = robust["original_h1_after_sd_entropy_control"]
    close("adjusted scalar stability", adjusted["pearson_r"], 0.22999774253581529)
    if adjusted["passed"] is not False:
        raise AssertionError("Adjusted scalar gate must remain failed")
    checked.append("forecastability construct audit")

    structure = load("results/structure_followup/summary.json")
    typology = load("results/structure_followup/typology_summary.json")
    if structure["cross_domain_gate_passed"] is not False:
        raise AssertionError("Cross-domain gate must remain failed")
    if typology["supported_typology"] is not False:
        raise AssertionError("Discrete typology must remain unsupported")
    checked.append("structure and typology")

    distress = load("results/distress_sensitivity/summary.json")
    direct = distress["direct_variability_sensitivity"]["associations"]
    direct_core = next(
        row for row in direct
        if row["control_set"] == "direct_core_sd"
        and row["predictor"] == "core_forecastability"
    )
    close("direct-SD distress sensitivity", direct_core["partial_r"], -0.1400565666429912)
    checked.append("distress sensitivity")

    feasibility = load("experiments/personalization/result.json")
    if feasibility["outcome_analysis_performed"] is not False:
        raise AssertionError("Personalization outcome analysis must remain absent")
    if feasibility["branch_decision"]["status"] != "feasibility_boundary_no_personalization_test":
        raise AssertionError("Personalization feasibility status changed")
    checked.append("personalization stop")
    return checked


def verify_report() -> None:
    required = [
        "report/main.tex",
        "report/main.pdf",
        "report/references.bib",
        "report/CLAIM_SOURCE_MAP.md",
        "report/figures/behavior_forecasting_performance.png",
        "report/figures/construct_overlap_heatmap.png",
        "report/figures/feature_family_decomposition.png",
        "report/figures/forecastability_construct_audit.png",
        "report/figures/typology_validation.png",
        "report/figures/validity_boundary_auroc.png",
    ]
    missing = [path for path in required if not (ROOT / path).is_file()]
    if missing:
        raise AssertionError("Missing report files: " + ", ".join(missing))


def main() -> None:
    checked = verify_headlines()
    verify_report()
    print(json.dumps({"status": "ok", "evidence_blocks": checked}, indent=2))


if __name__ == "__main__":
    main()
