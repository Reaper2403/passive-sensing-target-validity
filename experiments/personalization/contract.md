# Outcome-Blind Personalization Feasibility Contract

The proposed follow-up would compare a pooled behavioral forecast with a version
having a frozen person-level calibration term. Before any outcome modeling, Phase A
tested whether enough eligible person-targets existed across all four GLOBEM
institute-years.

## Gate

A behavioral family could proceed only if its selected target had:

- at least 30 eligible person-targets in every institute-year;
- at least 160 eligible person-targets in total; and
- finite, non-degenerate scale in every leave-one-cohort-out training fold.

An execution branch also required the specified combination of behavioral families.
Phase A was restricted to missingness counts, hard value validation, and pass/fail
scale checks. It authorized no predictive model, prediction, loss calculation, or
outcome comparison.

## Decision rule

If no allowed multi-family branch survived, the study had to stop and report
`feasibility_boundary_no_personalization_test`. A stop is not a null result for
personalization because no personalization hypothesis was evaluated.
