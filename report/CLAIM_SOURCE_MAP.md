# Claim and Display Source Map

Status: checked against the included aggregate outputs. Values below use report
rounding unless exact precision is needed to preserve a decision.

## Headline Claims

| Report claim | Displayed value/status | Authoritative source |
|---|---:|---|
| Current-day passive depression classification | AUROC 0.579 | [label summary](../results/label_baseline_control/summary.json), `allday_raw` |
| Passive-history depression classification | AUROC 0.691 | Same source, `history_raw` |
| Prior-label-only depression baseline | AUROC 0.844 | Same source, `prior_label_rate_only` |
| Current-day passive plus label history | AUROC 0.840; difference -0.004 | Same source, `allday_raw_plus_prior_label_rate` |
| Passive history plus label history | AUROC 0.841; difference -0.003 | Same source, `history_raw_plus_prior_label_rate` |
| Psychological construct overlap | Median finite same-day absolute r 0.663 | [construct summary](../results/construct_fragility_audit/summary.json); [pair details](../results/construct_fragility_audit/same_day_construct_correlations.csv) |
| Pair-estimate availability | 45 possible rows; 29 finite correlations | [pair details](../results/construct_fragility_audit/same_day_construct_correlations.csv) |
| Own-prior psychological prediction | Mean held-out R2 0.479 | [construct summary](../results/construct_fragility_audit/summary.json) |
| Other-construct-prior prediction | Mean held-out R2 0.426 | Same source |
| Other priors added to own prior | Mean R2 lift 0.044 | Same source |
| Tomorrow behavior forecast | Mean R2 0.272; AUROC 0.788 across 16 targets | [behavior summary](../results/behavioural_weather_forecast/summary.json), evening full model |
| Seven-day behavior forecast | Mean R2 0.506; AUROC 0.871 | Same source |
| Frozen forecastability sample | 120,936 rows; 696 input trajectories; 598 scored; 521 core | [forecastability summary](../results/forecastability/summary.json) |
| Original fixed scalar stability | r 0.412; 95% CI [0.316, 0.501]; n=521 | Same source |
| Original fixed profile advantage | delta 0.287; p=0.002; n=521 | Same source; 500 permutations |
| Expanding-refit scalar | r 0.407; 95% CI [0.307, 0.496]; n=521 | [robustness summary](../results/referee_robustness/summary.json) |
| Disjoint-refit scalar | r 0.322; 95% CI [0.219, 0.414]; n=499 | Same source |
| Raw forecastability versus behavioral SD | Pearson r -0.912; n=521 | Same source, `fixed_model_core_variability_overlap` |
| SD/entropy-adjusted fixed scalar | r 0.230; 95% CI [0.049, 0.409] | Same source, `original_h1_after_sd_entropy_control` |
| Adjusted scalar decision | Failed internally recorded 0.30 gate | Same object, `passed=false` |
| History benefit versus SD | Pearson r -0.268 | Same source, `history_skill.person_average_sd_overlap` |
| History benefit versus marginal entropy | Pearson r -0.518 | Same source, `history_skill.person_average_entropy_overlap` |
| Test-window-adjusted history-benefit profiles | delta 0.362 / 0.360 / 0.254 | Same source; fixed / expanding / disjoint; 10,000 permutations each |
| Training-window-adjusted profiles | delta 0.354 / 0.311 / 0.217 | [aggregate results](../results/structure_followup/training_moment_adjusted_profiles.csv) |
| Sequence-adjusted profiles | delta 0.314 / 0.308 / 0.202 | [aggregate results](../results/structure_followup/sequence_adjusted_profiles.csv) |
| Sequence-adjusted scalar | r 0.240 / 0.219 / 0.246 | [aggregate results](../results/structure_followup/sequence_adjusted_stability.csv); all below 0.30 |
| Cross-domain incremental performance | Mean relative gain -0.0047 / -0.0044 / -0.0042 | [structure summary](../results/structure_followup/summary.json) |
| Cross-domain targets improved | 4/8 / 4/8 / 2/8 | Same source; fixed / expanding / disjoint |
| Discrete typology | No k=2..5 passed; zero successful held-out cohorts | [typology summary](../results/structure_followup/typology_summary.json) |
| Original common-distress sensitivity | Partial r -0.178; n=518 | [distress summary](../results/distress_sensitivity/summary.json), `h2_controlled_association` |
| Direct-SD distress sensitivity | Partial r -0.140; cohort range 0.008 to -0.342 | Same source, `direct_variability_sensitivity` |
| Personalization follow-up | Stopped at feasibility; no model run | [feasibility result](../experiments/personalization/result.json), `outcome_analysis_performed=false` |
| Personalization review | Accepted for the feasibility record only | [review](../experiments/personalization/review.md) |

## Figures

| Figure | Asset | Source data |
|---|---|---|
| Label-history validity boundary | [image](figures/validity_boundary_auroc.png) | [label summary](../results/label_baseline_control/summary.json) |
| Psychological construct overlap | [image](figures/construct_overlap_heatmap.png) | [pair correlations](../results/construct_fragility_audit/same_day_construct_correlations.csv) |
| Behavior forecast performance | [image](figures/behavior_forecasting_performance.png) | [behavior summary](../results/behavioural_weather_forecast/summary.json) |
| Forecastability construct audit | [image](figures/forecastability_construct_audit.png) | [stability](../results/referee_robustness/stability_robustness.csv); [profiles](../results/referee_robustness/profile_robustness.csv) |
| Feature-family decomposition | [image](figures/feature_family_decomposition.png) | [stability](../results/structure_followup/feature_family_stability.csv); [profiles](../results/structure_followup/feature_family_profiles.csv) |
| Typology validation | [image](figures/typology_validation.png) | [typology summary](../results/structure_followup/typology_summary.json) |

All assets are generated by [the figure script](make_figures.py). In the typology
plot, both successful-cohort series are zero and are visibly offset only for display.

## Implementation

| Analysis | Method and code |
|---|---|
| Shared definitions | [experiment definitions](../experiments/README.md); [configuration](../configs/local_smoke.yaml) |
| Label-history control | [runner](../experiments/run_label_baseline_control.py) |
| Construct overlap | [runner](../experiments/run_construct_fragility_audit.py) |
| Behavior forecast | [runner](../experiments/run_behavioural_weather_forecast.py) |
| Forecastability and robustness | [initial runner](../experiments/run_forecastability_phenotype.py); [robustness runner](../experiments/run_referee_robustness_checks.py) |
| Mechanism and typology | [mechanism runner](../experiments/run_structure_mechanism_followup.py); [typology runner](../experiments/run_profile_typology_followup.py) |
| Distress sensitivity | [runner](../experiments/run_common_distress_forecastability.py) |
| Personalization feasibility | [contract](../experiments/personalization/contract.md); [runner](../experiments/personalization/run.py) |

## Interpretation Locks

- Passive features did not improve the label-only model.
- The internally recorded gates are not preregistrations.
- Forecastability and history benefit are not validated phenotypes or traits.
- Pooled-model history benefit is not idiographic learnability.
- The sequence proxies are not an entropy ceiling.
- The profile has no external replication.
- The feasibility stop is not a personalization null.
