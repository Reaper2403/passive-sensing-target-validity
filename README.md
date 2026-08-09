# Passive Sensing Target Validation in GLOBEM

This is the compact repository submitted with the TU Hamburg Research Project.

**[Read the project report](report/main.pdf)**

## Result in one paragraph

In this GLOBEM analysis, prior depression-label history outperformed passive sensing
for depression classification (AUROC 0.844 versus 0.840 and 0.841 for the two combined
models). Direct behavior was a better-supported target: pooled chronological models
reached mean held-out R2 0.272 and AUROC 0.788 for tomorrow across 16 targets. A
proposed scalar routine-forecastability score did not survive construct validation;
it was strongly coupled to behavioral variability (`r = -0.912`), and its adjusted
stability (`r = 0.230`) failed the recorded 0.30 gate. A narrower continuous profile
result remained post-hoc, while cross-domain coupling, discrete routine types, and
the proposed personalization study were unsupported or stopped before modeling.

## Evidence included

| Analysis | Code | Aggregate results |
|---|---|---|
| Label-history control | [runner](experiments/run_label_baseline_control.py) | [results](results/label_baseline_control/) |
| Psychological construct overlap | [runner](experiments/run_construct_fragility_audit.py) | [results](results/construct_fragility_audit/) |
| Prospective behavior forecast | [runner](experiments/run_behavioural_weather_forecast.py) | [results](results/behavioural_weather_forecast/) |
| Forecastability construct audit | [initial runner](experiments/run_forecastability_phenotype.py), [robustness runner](experiments/run_referee_robustness_checks.py) | [initial results](results/forecastability/), [robustness results](results/referee_robustness/) |
| Profile mechanism and typology | [mechanism runner](experiments/run_structure_mechanism_followup.py), [typology runner](experiments/run_profile_typology_followup.py) | [results](results/structure_followup/) |
| Distress sensitivity analysis | [runner](experiments/run_common_distress_forecastability.py) | [results](results/distress_sensitivity/) |
| Personalization feasibility stop | [method and runner](experiments/personalization/) | [result](experiments/personalization/result.json) |

The frozen methods and interpretation boundaries are summarized in
[`experiments/README.md`](experiments/README.md). Every displayed value is mapped to
an aggregate source in [`report/CLAIM_SOURCE_MAP.md`](report/CLAIM_SOURCE_MAP.md).

## Reproduce

Verify the public aggregate evidence and rebuild the report:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip install -e .
python scripts/reproduce_report.py verify
```

Recomputing the analyses requires authorized GLOBEM v1.1 access:

```bash
python scripts/reproduce_report.py full
```

See [`DATA_ACCESS.md`](DATA_ACCESS.md) and [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).
No GLOBEM source data, participant identifiers, participant-level tables, or
row-level predictions are included in this repository.
