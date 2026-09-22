# Passive sensing: target validity and behavioral forecasting

Compact research-project companion to the **16 September 2026** TU Hamburg report by
Ashutosh Chatterjee. The [48-page report](report/main.pdf) is the scientific source
of record. This repository provides its analysis code, privacy-safe aggregate
evidence, figures, and a claim-to-result index. It is not the exploratory scratchpad.

## Result in 30 seconds

- **RQ1, psychological label:** Passive history reached AUROC 0.691 for the released
  depression-screening proxy, but prior same-user label rate reached 0.844. Adding
  passive history did not produce a detectable increment (clustered difference
  -0.003, 95% CI [-0.013, 0.007]). Within-person AUROC was 0.514.
- **RQ2, behavior:** Tomorrow's directly observed behavior was forecastable across
  16 targets (mean R2 0.272 with the full model), but recent same-channel history
  already reached 0.244. The following-week mean R2 was 0.506 with the full model.
- **RQ3, fingerprint:** The original scalar forecastability score mostly tracked
  behavioral steadiness (correlation with own variability -0.912). A corrected
  continuous history-benefit profile is a **post-hoc replication lead**, not a
  validated psychological trait.
- **RQ4, mechanism:** Extra cross-domain history and discrete routine clusters did
  not clear the reported tests. The personalization analysis stopped at a data
  feasibility gate before fitting an outcome model.
- **Android pilot, separate n=1 study:** The first policy lost to keeping current
  volume (MAE 17.69 versus 8.17). A later 110-session follow-up showed an observed
  8% improvement for a conservative hybrid, but its day-clustered interval included
  zero. This is exploratory, not GLOBEM transfer or deployment evidence.

## Start here

1. Read [the report](report/main.pdf), especially the abstract, Chapter 5, and
   Appendix C.
2. Open the [claim-to-evidence map](report/CLAIM_SOURCE_MAP.md) to locate each
   reported number in `results/`.
3. Verify the frozen public evidence:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip install -e .
python scripts/reproduce_report.py verify
```

The verify command needs **no GLOBEM data or TeX installation**. To regenerate
figures and compile the PDF, install Tectonic and run
`python scripts/reproduce_report.py build`. To recompute GLOBEM analyses from
licensed data, see [REPRODUCIBILITY.md](REPRODUCIBILITY.md) and
[DATA_ACCESS.md](DATA_ACCESS.md).

## Small repository map

| Path | Purpose |
|---|---|
| `report/` | Final PDF, LaTeX, figures, and claim map |
| `experiments/` | Report-facing GLOBEM analysis runners |
| `results/` | Frozen aggregate evidence; no row-level predictions |
| `src/` | Shared loader and evaluation utilities |
| `scripts/reproduce_report.py` | Three explicit modes: verify, build, full |
| `tests/` | Smoke and release-boundary checks |

Restricted GLOBEM files, participant identifiers, private Android logs, and
session-level predictions are not included. The Android aggregate summaries can
rebuild the figures, but the private raw-data analysis cannot be independently
rerun from this public repository. This distinction is intentional.
