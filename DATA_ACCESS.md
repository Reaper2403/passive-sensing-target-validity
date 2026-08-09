# GLOBEM Data Access

## Why the data are not in this repository

This project uses the controlled-access GLOBEM v1.1 release on PhysioNet. Its
participant-level source files and participant-derived row-level outputs are not
redistributed. Public reproducibility therefore has two layers:

1. Anyone can inspect the code, contracts, aggregate results, figures, and report,
   and can run the aggregate artifact verification without the dataset.
2. Authorized users can reproduce the numerical analyses from source after obtaining
   GLOBEM under its Data Use Agreement.

Dataset page: <https://physionet.org/content/globem/1.1/>

Data Use Agreement: <https://physionet.org/content/globem/view-dua/1.1/>

Dataset DOI: <https://doi.org/10.13026/r9s1-s711>

Official GLOBEM code: <https://github.com/UW-EXP/GLOBEM>

## Expected local layout

After completing PhysioNet's access process, place the downloaded institute-year
folders at exactly:

```text
data/raw/globem/1.1/
├── INS-W_1/
│   ├── FeatureData/
│   │   ├── rapids.csv
│   │   ├── screen.csv
│   │   ├── sleep.csv
│   │   └── steps.csv
│   ├── ParticipantsInfoData/
│   │   └── platform.csv
│   └── SurveyData/
│       ├── dep_weekly.csv
│       ├── ema.csv
│       └── pre.csv
├── INS-W_2/ ...
├── INS-W_3/ ...
└── INS-W_4/ ...
```

The full archive may contain additional files. The tree above lists the files used by
the report's principal and robustness analyses. Check the installation without
printing participant data:

```bash
python scripts/check_globem_layout.py
```

## Local-data boundary

The following paths and artifact classes are intentionally ignored by Git:

- `data/raw/`, `data/interim/`, `data/processed/`, and `data/external/`;
- Parquet files and serialized model objects;
- row-level predictions, samples, participant-level metrics, and trajectory tables;
- any other private field-study data.

Do not weaken these rules to make a run easier to commit. The release audit in
`scripts/public_release_audit.py` independently rejects restricted artifact names,
identifier-bearing CSVs, raw-data paths, and binary analysis objects.

The scripts never require PhysioNet credentials. Authentication and downloading are
performed outside this repository using the access method documented by PhysioNet.
