from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .catalog import CATALOG_BY_ID
from .classification import classify_conceptual_factors, classify_features
from .config import ProjectConfig, load_config
from .data import data_availability_summary, load_curve_market, load_panel
from .dedup import choose_cluster_representatives, cluster_factors, cross_sectional_factor_correlation
from .factor_processing import build_feature_matrix_and_definitions
from .feature_processing import build_ml_features
from .freeze import create_freeze_manifest, verify_freeze_manifest
from .incremental import walk_forward_incremental_tests
from .labels import add_forward_labels, assert_labels_are_forward
from .metrics import compute_screening_metrics, conceptual_catalog_frame
from .point_in_time import run_point_in_time_audit
from .reporting import (
    write_data_audit_report,
    write_factor_family_report,
    write_factor_screening_report,
    write_ml_reports,
)


def _ensure_output_dirs(root: Path) -> None:
    for relative in ["reports", "features", "artifacts/development", "artifacts/sealed"]:
        (root / relative).mkdir(parents=True, exist_ok=True)


def _build_research_frame(config: ProjectConfig, end_date: str, phase: str):
    panel = load_panel(config, end_date)
    point_in_time = run_point_in_time_audit(panel, config, phase)
    panel = add_forward_labels(panel, config.horizons)
    assert_labels_are_forward(panel)
    curve_market = load_curve_market(config, end_date)
    factors, definitions, curve_features = build_feature_matrix_and_definitions(panel, curve_market, config)
    keys = ["trade_date", "product", "sector"]
    research = panel.merge(factors, on=keys, how="left", validate="one_to_one")
    return research, panel, curve_features, definitions, point_in_time


def run_development(config: ProjectConfig) -> dict[str, Path]:
    _ensure_output_dirs(config.root)
    research, panel, curve, definitions, pit = _build_research_frame(
        config, config.raw["development_end"], "development"
    )
    metrics = compute_screening_metrics(
        research,
        definitions,
        config,
        config.raw["development_start"],
        config.raw["development_end"],
    )
    development_sample = research.loc[
        research["trade_date"].between(
            pd.Timestamp(config.raw["development_start"]),
            pd.Timestamp(config.raw["development_end"]),
        )
    ]
    feature_names = [definition.feature_name for definition in definitions]
    correlation = cross_sectional_factor_correlation(development_sample, feature_names)
    clusters = cluster_factors(correlation, float(config.raw["screening"]["correlation_cluster_threshold"]))
    primary_metrics = metrics["summary"].loc[metrics["summary"]["horizon"].eq(config.primary_horizon)]
    cluster_info = choose_cluster_representatives(clusters, primary_metrics)
    incremental = walk_forward_incremental_tests(research, feature_names, config)
    feature_classes = classify_features(
        metrics["summary"],
        metrics["yearly"],
        metrics["product"],
        metrics["sector"],
        definitions,
        incremental,
        cluster_info,
        correlation,
        config,
    )
    conceptual = classify_conceptual_factors(feature_classes)
    ml_features, registry = build_ml_features(research, feature_classes, config)

    dev = config.root / "artifacts" / "development"
    factors_path = dev / "factor_values_development.csv.gz"
    factor_columns = ["trade_date", "product", "sector", *feature_names]
    research[factor_columns].to_csv(factors_path, index=False, compression="gzip")
    label_columns = ["trade_date", "product", *[f"future_return_{h}d" for h in config.horizons]]
    research[label_columns].to_csv(dev / "labels_development.csv.gz", index=False, compression="gzip")
    metrics["summary"].to_csv(dev / "factor_metrics.csv", index=False)
    metrics["daily_ic"].to_csv(dev / "daily_rank_ic.csv.gz", index=False, compression="gzip")
    metrics["yearly"].to_csv(dev / "yearly_metrics.csv", index=False)
    metrics["product"].to_csv(dev / "product_metrics.csv", index=False)
    metrics["sector"].to_csv(dev / "sector_metrics.csv", index=False)
    correlation.to_csv(dev / "factor_correlation.csv")
    cluster_info.to_csv(dev / "factor_clusters.csv", index=False)
    incremental.to_csv(dev / "incremental_tests.csv", index=False)
    feature_classes.to_csv(dev / "feature_classification.csv", index=False)
    conceptual.to_csv(dev / "conceptual_factor_classification.csv", index=False)
    conceptual_catalog_frame().to_csv(dev / "factor_registry.csv", index=False)
    metrics["summary"].assign(
        experiment_phase="development_2018_2024",
        sealed_data_used=False,
    ).to_csv(dev / "experiment_registry.csv", index=False)
    ml_features.to_csv(config.root / "features" / "ml_features_development.csv.gz", index=False, compression="gzip")
    registry.to_csv(config.root / "features" / "ml_feature_registry.csv", index=False)

    availability = data_availability_summary(config)
    availability.to_csv(dev / "data_availability.csv", index=False)
    write_data_audit_report(
        config.root / "reports" / "data_audit_report.md",
        config,
        panel,
        curve,
        availability,
        pit,
        conceptual,
    )
    write_factor_screening_report(
        config.root / "reports" / "factor_screening_report.md",
        conceptual,
        feature_classes,
        config,
    )
    write_factor_family_report(
        config.root / "reports" / "factor_family_report.md",
        conceptual,
        feature_classes,
    )
    write_ml_reports(
        config.root / "reports" / "ml_feature_report.md",
        config.root / "features" / "feature_dictionary.md",
        registry,
        feature_classes,
    )
    run_summary = {
        "phase": "development",
        "development_end": config.raw["development_end"],
        "feature_instance_count": len(definitions),
        "conceptual_factor_count": len(conceptual),
        "ml_feature_count": len(registry),
        "class_counts": conceptual["classification"].value_counts().sort_index().to_dict(),
    }
    (dev / "RUN_SUMMARY.json").write_text(json.dumps(run_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "screening_report": config.root / "reports" / "factor_screening_report.md",
        "feature_registry": config.root / "features" / "ml_feature_registry.csv",
    }


def run_sealed(config: ProjectConfig) -> Path:
    verify_freeze_manifest(config)
    result_path = config.root / config.raw["sealed_test"]["result_file"]
    if result_path.exists() and not config.raw["sealed_test"]["allow_overwrite"]:
        raise FileExistsError("sealed result already exists; overwrite is forbidden")

    research, _, _, definitions, _ = _build_research_frame(config, config.raw["sealed_end"], "sealed")
    metrics = compute_screening_metrics(
        research,
        definitions,
        config,
        config.raw["sealed_start"],
        config.raw["sealed_end"],
    )
    sealed = metrics["summary"].loc[metrics["summary"]["horizon"].eq(config.primary_horizon)].copy()
    feature_classes = pd.read_csv(config.root / "artifacts" / "development" / "feature_classification.csv")
    conceptual = pd.read_csv(config.root / "artifacts" / "development" / "conceptual_factor_classification.csv")
    selected = feature_classes.loc[feature_classes["enter_ml"].astype(bool), ["feature_name", "classification", "expected_direction"]]
    sealed = sealed.merge(selected, on="feature_name", how="inner")
    direction = sealed["expected_direction"].copy()
    unknown = direction.eq(0)
    direction.loc[unknown] = np.where(sealed.loc[unknown, "rank_ic_mean"].ge(0), 1, -1)
    sealed["sealed_direction_consistent"] = (sealed["rank_ic_mean"] * direction).gt(0)
    sealed["sealed_evaluation_note"] = "冻结后一次性评估；不用于重新选因子"
    ml_features, _ = build_ml_features(research, feature_classes, config)
    ml_features.to_csv(config.root / "features" / "ml_features.csv.gz", index=False, compression="gzip")
    sealed_display = sealed[
        ["feature_name", "classification", "coverage", "ic_mean", "rank_ic_mean", "group_spread", "sealed_direction_consistent"]
    ].sort_values(["classification", "feature_name"])
    write_factor_screening_report(
        config.root / "reports" / "factor_screening_report.md",
        conceptual,
        feature_classes,
        config,
        sealed_metrics=sealed_display,
    )
    sealed_report = [
        "# 2025封存测试报告",
        "",
        "本结果在代码、配置、开发期分类及特征注册表冻结后生成，且系统禁止覆盖。",
        "",
        sealed_display.to_markdown(index=False),
    ]
    (config.root / "reports" / "sealed_2025_report.md").write_text("\n".join(sealed_report) + "\n", encoding="utf-8")
    # The immutable result is written last. A report/feature failure therefore does not
    # leave a partial sealed result that would prevent safe recovery.
    sealed.to_csv(result_path, index=False)
    return result_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="中国商品期货Alpha因子科学筛选系统")
    parser.add_argument("command", choices=["development", "seal", "sealed"])
    parser.add_argument("--config", default="config/screening.json")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.command == "development":
        outputs = run_development(config)
        print(json.dumps({key: str(value) for key, value in outputs.items()}, ensure_ascii=False, indent=2))
    elif args.command == "seal":
        print(create_freeze_manifest(config))
    else:
        print(run_sealed(config))
    return 0
