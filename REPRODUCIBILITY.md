# Reproducibility

## Claim

The public repository rebuilds the report and verifies every headline value from the
included aggregate outputs. Recomputing those outputs requires authorized access to
GLOBEM v1.1; controlled participant data are not redistributed.

## Reference environment

- Python: 3.14.6 for the frozen release; package metadata supports Python 3.10+
- Dependency versions: `requirements-lock.txt`
- PDF compiler: Tectonic
- Operating system used for the frozen release: macOS

Set up the environment from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
python -m pip install -e .
```

## Level 1: verify the public artifacts

```bash
python scripts/reproduce_report.py verify
```

This command:

1. audits the tracked release for prohibited data and artifact classes;
2. checks frozen headline numbers directly against aggregate JSON and CSV files;
3. runs the automated tests;
4. regenerates the six report figures and compiles `report/main.pdf`.

It does not read `data/` or require GLOBEM.

## Level 2: recompute from controlled source data

Follow [DATA_ACCESS.md](DATA_ACCESS.md), then run:

```bash
python scripts/reproduce_report.py full
```

The full command executes only the analyses represented in the report:

1. local GLOBEM layout validation;
2. canonical data audit;
3. label-history baseline control;
4. construct-fragility audit;
5. prospective behavioral forecasting;
6. forecastability and referee robustness checks;
7. profile mechanism, typology, and distress sensitivity analyses;
8. the outcome-blind personalization feasibility audit;
9. artifact validation, tests, figure regeneration, and report compilation.

The prospective behavioral forecast and permutation-heavy robustness checks are the
longest stages. Runtime depends strongly on CPU, memory, and storage.

## Output policy

Full runs create both aggregate and participant-derived intermediate outputs.
Participant-derived outputs are necessary for computation but are not public release
artifacts. They remain local under ignored paths. Before publication, the release
audit examines the exact tracked file set rather than trusting filename conventions
alone.

The report's quantitative provenance is recorded in
[the claim-source map](report/CLAIM_SOURCE_MAP.md).
No unrelated historical experiments are included in this repository.

## Determinism and remaining limits

Experiment runners set explicit random seeds where stochastic models, bootstrap
intervals, or permutations are used. Exact floating-point equality can still vary
slightly across architectures or library builds. The verifier uses tight numerical
tolerances around the frozen headline values.

The repository reproduces the reported analyses; it does not make controlled GLOBEM
data public, reconstruct identities across cohorts, or turn post-hoc findings into
prospectively registered evidence.
