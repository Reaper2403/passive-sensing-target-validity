# Reproducibility and limits

The final report is `report/main.pdf` (16 September 2026, 48 pages). The public
repository offers three levels of reproducibility.

## 1. Verify the frozen public evidence

From the repository root after installing `requirements-lock.txt` and `pyproject.toml`:

```bash
python scripts/reproduce_report.py verify
```

This checks the tracked-file privacy policy, links, numerical headline assertions,
presence of report assets, and tests. It does **not** need GLOBEM, private phone
data, or a TeX compiler. The [claim map](report/CLAIM_SOURCE_MAP.md) links every
major displayed result to a compact aggregate JSON or CSV file.

## 2. Rebuild the report from aggregate evidence

Install [Tectonic](https://tectonic-typesetting.github.io/) and run:

```bash
python scripts/reproduce_report.py build
```

This also regenerates all report figures and compiles the LaTeX PDF. The Android
figures use privacy-safe aggregate extracts; the raw personal phone logs are not
public. The sample-flow figure displays audited counts recorded in figure code.

## 3. Recompute GLOBEM analyses

Obtain authorized GLOBEM v1.1 data as described in [DATA_ACCESS.md](DATA_ACCESS.md),
place it under the ignored `data/raw/` directory, and run:

```bash
python scripts/reproduce_report.py full
```

This executes the report-facing GLOBEM runners, including the RQ1 label-rate,
clustered inference, within-person, split/seed, construct-overlap, behavioral
forecasting, forecastability, mechanism, typology, and personalization-feasibility
checks. It then builds the report. Full recomputation can take substantial CPU
time. Outputs containing trajectories or predictions are ignored by Git and must
not be published.

## Boundary of the public release

- The public artifacts reproduce numerical checks and displays, not the licensed
  source data.
- The separate Android pilot can be checked and visualized from aggregate values,
  but **cannot** be independently recomputed from private raw logs here.
- The September Android saved-model reconstruction did not independently establish
  an immutable pre-September model freeze. Its MAE confidence interval crosses zero.
- Post-hoc profile structure is not externally replicated or person-specific model
  evidence.
- The released depression target is a cohort-dependent screening proxy, not a
  clinical diagnosis.

The frozen development environment used Python 3.14; the published lock resolves
under Python 3.12, which is used in CI. Exact floating-point output may vary with
library and hardware versions. `requirements-lock.txt` records the tested packages.
The report, aggregate files, code, and provenance map are the
submission unit; older exploratory notebooks and raw artifacts are intentionally
absent.
