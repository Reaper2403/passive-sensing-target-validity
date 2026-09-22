# Included Experiments

Only analyses used by the project report are included here. All splits are
chronological where prediction is evaluated; all stated thresholds are internal
project gates rather than independently timestamped preregistrations.

## 1. Label-history control

Five random-forest models compare current-day passive features, passive history, a
same-person prior-label rate, and the two combined variants. The primary comparison
is held-out AUROC against the label-only baseline. Neither combined model may be
described as an improvement unless it exceeds that baseline.
The additional `run_rq1_*` runners estimate trajectory-clustered uncertainty,
within-person ranking, a power diagnostic, and split/seed sensitivity. Their public
outputs are aggregate only; full reruns create ignored participant-level
intermediates.

## 2. Psychological construct overlap

Ten repeated psychological columns are audited using finite same-day Pearson
correlations and chronological prior-only ridge models. The analysis tests whether
apparently different prediction targets substantially overlap or predict one another
from their own and related prior values.
`run_construct_overlap_sensitivity.py` removes redundant representations for the
report's primary cross-family overlap summary.

## 3. Prospective behavior forecast

Six pooled models forecast 16 directly observed behavioral targets for tomorrow and
the next seven days. Features use calendar structure, recent history, a 42-day
fingerprint, and information available by the morning, afternoon, or evening issue
time. Future observations are excluded from every feature set.
`build_report_statistical_supplements.py` creates prevalence, robust RQ2 summaries,
and the per-target LaTeX table after the source analyses have run.

## 4. Forecastability construct audit

The original score is normalized negative MAE. Early/late scalar stability must reach
`r >= 0.30`; continuous profile advantage must reach `delta >= 0.10` with corrected
permutation `p < 0.05`. Referee checks refit models in expanding and disjoint windows,
control behavioral standard deviation and marginal entropy, and compare relative
history benefit with raw forecast error. The scalar interpretation fails because the
adjusted score does not pass its gate.

## 5. Profile mechanism and typology

Feature-family analyses separate yesterday, 7/14-day means, recent history, a 42-day
fingerprint, all own-domain history, and cross-domain increment. Additional controls
adjust test-window variability, training-window moments, and short sequence proxies.
Spherical profile clusters for `k = 2..5` are evaluated by leave-one-cohort-out
assignment accuracy and adjusted Rand index. Cross-domain coupling and discrete
routine types fail their gates.

## 6. Distress sensitivity

This post-hoc, between-person analysis tests association with a common-distress
component after coverage, behavioral level, variability, and direct per-target
standard-deviation controls. It is reported as heterogeneous sensitivity evidence,
not distress prediction or within-person change detection.

## 7. Personalization feasibility

The outcome-blind audit requires an allowed multi-family branch with at least 30
eligible person-targets in every institute-year and 160 overall. No branch satisfied
all requirements, so no personalization model, prediction, loss, or outcome
comparison was run. See `personalization/contract.md`, `result.json`, and `review.md`.

## Separate Android pilot

No private Android runner or raw phone log is published here. The report's Appendix C
figures are regenerated from privacy-safe aggregate extracts in
`results/android_pilot/`. This permits arithmetic and visualization checks, not
independent session-level recomputation.
