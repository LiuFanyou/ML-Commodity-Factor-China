from __future__ import annotations

from pathlib import Path

import pandas as pd

from unified_factor_system.config import load_config
from unified_factor_system.reports import write_screening_report
from unified_factor_system.screening import screen_factors


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config = load_config(project_root / "config" / "project.json")

    raw_factors = pd.read_parquet(
        config.output_root / "intermediate" / "raw_factors_57.parquet"
    )
    labels = pd.read_parquet(
        config.output_root / "intermediate" / "labels.parquet"
    )
    classified = pd.read_csv(
        config.output_root / "screening" / "factor_classification.csv"
    )
    incremental = pd.read_csv(
        config.output_root / "screening" / "incremental_ridge_tests.csv"
    )

    metrics, _, _, _ = screen_factors(
        raw_factors=raw_factors,
        labelled_panel=labels,
        evaluation_start=pd.Timestamp(
            config.raw["screening"]["evaluation_start"]
        ),
        evaluation_end=pd.Timestamp(
            config.raw["screening"]["evaluation_end"]
        ),
        minimum_cross_section=int(
            config.raw["screening"]["minimum_cross_section"]
        ),
        long_fraction=float(config.raw["execution"]["long_fraction"]),
        short_fraction=float(config.raw["execution"]["short_fraction"]),
        cost_scenarios=[
            int(value)
            for value in config.raw["execution"]["cost_scenarios_bps_one_way"]
        ],
    )

    if {"rank_ic_std", "rank_ic_ir"}.issubset(classified.columns):
        expanded = classified.copy()
    else:
        additions = metrics[
            ["factor_id", "rank_ic_std", "rank_ic_ir"]
        ].copy()
        expanded = classified.merge(
            additions,
            on="factor_id",
            how="left",
            validate="one_to_one",
        )

    metric_path = (
        config.output_root / "screening" / "factor_metrics_with_icir.csv"
    )
    report_path = config.report_root / "factor_screening_report_with_icir.md"
    expanded.to_csv(metric_path, index=False, encoding="utf-8")
    write_screening_report(report_path, expanded, incremental)

    print(f"wrote {report_path}")
    print(f"wrote {metric_path}")


if __name__ == "__main__":
    main()
