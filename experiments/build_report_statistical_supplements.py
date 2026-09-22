from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "behavioural_weather_forecast"
OUTPUT = ROOT / "results" / "report_statistical_supplements"
LABEL_SOURCE = ROOT / "results" / "label_baseline_control"


def robust_summary(frame: pd.DataFrame, metric: str) -> dict[str, float | int]:
    values = frame[metric].dropna()
    return {
        "n_targets": int(len(values)),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "q1": float(values.quantile(0.25)),
        "q3": float(values.quantile(0.75)),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
    }


def latex_escape(value: str) -> str:
    return value.replace("_", r"\_").replace("%", r"\%")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)

    label_summary = json.loads((LABEL_SOURCE / "summary.json").read_text())
    label_metrics = label_summary["metrics"][0]
    n_train = int(label_metrics["n_train"])
    n_train_positive = int(round(float(label_summary["global_fallback"]) * n_train))
    label_predictions = pd.read_parquet(LABEL_SOURCE / "predictions.parquet")
    label_test = label_predictions[
        label_predictions["feature_set"].eq("prior_label_rate_only")
    ].copy()
    prevalence_rows = [
        {
            "split": "train",
            "n": n_train,
            "positive": n_train_positive,
            "negative": n_train - n_train_positive,
            "positive_rate": n_train_positive / n_train,
        },
        {
            "split": "test",
            "n": int(len(label_test)),
            "positive": int(label_test["target_dep_weekly"].sum()),
            "negative": int((~label_test["target_dep_weekly"]).sum()),
            "positive_rate": float(label_test["target_dep_weekly"].mean()),
        },
    ]
    pd.DataFrame(prevalence_rows).to_csv(OUTPUT / "rq1_target_prevalence.csv", index=False)
    (OUTPUT / "rq1_target_prevalence.json").write_text(
        json.dumps(
            {
                "target_column": "dep_weekly.csv:dep",
                "split": "within-trajectory chronological 65/35",
                "rows": prevalence_rows,
            },
            indent=2,
        )
        + "\n"
    )

    regression = pd.read_csv(SOURCE / "regression_metrics.csv")
    classification = pd.read_csv(SOURCE / "classification_metrics.csv")
    selection = {
        "issue_time": "evening",
        "model": "history_fingerprint_plus_today",
    }

    reg = regression[
        (regression["issue_time"] == selection["issue_time"])
        & (regression["model"] == selection["model"])
    ].copy()
    cls = classification[
        (classification["issue_time"] == selection["issue_time"])
        & (classification["model"] == selection["model"])
    ].copy()
    expected = {(target, horizon) for target in reg["target"] for horizon in reg["horizon"]}
    observed_reg = set(zip(reg["target"], reg["horizon"]))
    observed_cls = set(zip(cls["target"], cls["horizon"]))
    if observed_reg != expected or observed_cls != expected:
        raise ValueError("The frozen full-model target-by-horizon grid is incomplete")

    baseline_models = [
        "calendar_only",
        "recent_history_only",
        "fingerprint_42d_only",
        "history_plus_fingerprint",
        "history_fingerprint_plus_today",
    ]
    baseline_rows = []
    for horizon in ["tomorrow", "next_week"]:
        for model in baseline_models:
            for metric, source_frame in [("r2", regression), ("auroc", classification)]:
                frame = source_frame[
                    source_frame["horizon"].eq(horizon)
                    & source_frame["issue_time"].eq("evening")
                    & source_frame["model"].eq(model)
                ]
                baseline_rows.append(
                    {"horizon": horizon, "metric": metric, "model": model, **robust_summary(frame, metric)}
                )
    baseline_frame = pd.DataFrame(baseline_rows)
    baseline_frame.to_csv(OUTPUT / "rq2_baseline_summary.csv", index=False)

    summary_rows = []
    summary_json: dict[str, object] = {
        "selection": selection,
        "horizons": {},
        "baseline_comparison": baseline_rows,
    }
    for horizon in ["tomorrow", "next_week"]:
        horizon_reg = reg[reg["horizon"] == horizon]
        horizon_cls = cls[cls["horizon"] == horizon]
        for metric, frame in [("r2", horizon_reg), ("auroc", horizon_cls)]:
            values = robust_summary(frame, metric)
            summary_rows.append({"horizon": horizon, "metric": metric, **values})
            summary_json["horizons"].setdefault(horizon, {})[metric] = values

    per_target = (
        reg.pivot(index="target", columns="horizon", values="r2")
        .rename(columns={"tomorrow": "tomorrow_r2", "next_week": "next_week_r2"})
        .join(
            cls.pivot(index="target", columns="horizon", values="auroc").rename(
                columns={
                    "tomorrow": "tomorrow_auroc",
                    "next_week": "next_week_auroc",
                }
            )
        )
        .reset_index()
        .sort_values("tomorrow_r2", ascending=False)
    )
    pd.DataFrame(summary_rows).to_csv(OUTPUT / "rq2_robust_summary.csv", index=False)
    per_target.to_csv(OUTPUT / "rq2_per_target.csv", index=False)
    (OUTPUT / "rq2_summary.json").write_text(json.dumps(summary_json, indent=2) + "\n")

    lines = [
        r"\footnotesize",
        r"\begin{longtable}{@{}L{0.28\textwidth}C{0.14\textwidth}C{0.14\textwidth}C{0.14\textwidth}C{0.14\textwidth}@{}}",
        r"\caption[RQ2 per-target results]{Per-target held-out performance for the full evening model. The model combines recent history, the 42-day fingerprint, and the current partial day. Values are descriptive across heterogeneous direct-behavior targets.}\label{tab:rq2-per-target}\\",
        r"\toprule",
        r"Target & \makecell{Tomorrow\\$R^2$} & \makecell{Tomorrow\\AUROC} & \makecell{Seven-day\\$R^2$} & \makecell{Seven-day\\AUROC} \\",
        r"\midrule",
        r"\endfirsthead",
        r"\multicolumn{5}{l}{\textit{Table \thetable\ continued}}\\",
        r"\toprule",
        r"Target & \makecell{Tomorrow\\$R^2$} & \makecell{Tomorrow\\AUROC} & \makecell{Seven-day\\$R^2$} & \makecell{Seven-day\\AUROC} \\",
        r"\midrule",
        r"\endhead",
        r"\midrule",
        r"\multicolumn{5}{r}{\textit{Continued on next page}}\\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    for row in per_target.itertuples(index=False):
        label = latex_escape(row.target.replace("_", " ").title())
        lines.append(
            f"{label} & {row.tomorrow_r2:.3f} & {row.tomorrow_auroc:.3f} & "
            f"{row.next_week_r2:.3f} & {row.next_week_auroc:.3f} \\\\"
        )
    lines.append(r"\end{longtable}")
    lines.append(r"\normalsize")
    (OUTPUT / "rq2_per_target_table.tex").write_text("\n".join(lines) + "\n")
    print(pd.DataFrame(summary_rows).to_string(index=False))


if __name__ == "__main__":
    main()
