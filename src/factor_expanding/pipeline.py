from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from factor_modeling.common import (
    build_strategy_returns,
    cross_sectional_rank_target,
    purge_last_dates,
)
from factor_modeling.linear_ridge import (
    _daily_correlations,
    _fit_ridge,
    _fold_metrics,
    choose_alpha,
)
from factor_screening.classification import classify_conceptual_factors, classify_features
from factor_screening.config import ProjectConfig, load_config
from factor_screening.data import data_availability_summary, load_curve_market, load_panel
from factor_screening.dedup import choose_cluster_representatives, cluster_factors, cross_sectional_factor_correlation
from factor_screening.factor_processing import build_feature_matrix_and_definitions
from factor_screening.feature_processing import build_ml_features
from factor_screening.labels import add_forward_labels, assert_labels_are_forward
from factor_screening.metrics import compute_screening_metrics, conceptual_catalog_frame
from factor_screening.point_in_time import run_point_in_time_audit
from factor_screening.reporting import (
    write_factor_family_report,
    write_factor_screening_report,
    write_ml_reports,
)

from .incremental import expanding_incremental_tests
from .reporting import write_data_audit, write_walk_forward_report


KEYS = ["trade_date", "product"]


def expanding_splits(train_start_year: int, first_oos_year: int, last_oos_year: int) -> list[tuple[int, int, int]]:
    if first_oos_year <= train_start_year or last_oos_year < first_oos_year:
        raise ValueError("invalid expanding-window years")
    return [(train_start_year, test_year - 1, test_year) for test_year in range(first_oos_year, last_oos_year + 1)]


def mask_training_labels(research: pd.DataFrame, horizons: list[int], test_year: int) -> pd.DataFrame:
    """Remove labels whose realized endpoint reaches the outer test period."""
    cutoff = pd.Timestamp(f"{test_year}-01-01")
    out = research.loc[research.trade_date.lt(cutoff)].copy()
    for horizon in horizons:
        label = f"future_return_{horizon}d"
        end_col = f"label_end_date_{horizon}d"
        invalid = out[end_col].isna() | out[end_col].ge(cutoff)
        out.loc[invalid, label] = np.nan
    return out


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _paths(config: ProjectConfig) -> dict[str, Path]:
    base = config.root / config.raw["output_root"]
    return {
        "base": base,
        "reports": base / "reports",
        "features": base / "features",
        "artifacts": base / "artifacts",
        "folds": base / "artifacts" / "folds",
        "handoff": base / "training_handoff",
    }


def _prepare_output(paths: dict[str, Path]) -> None:
    sentinel = paths["base"] / "RUN_COMPLETE.json"
    if sentinel.exists():
        raise FileExistsError(f"expanding run already completed; overwrite forbidden: {sentinel}")
    if paths["base"].exists() and any(paths["base"].iterdir()):
        raise FileExistsError(f"expanding output directory is not empty: {paths['base']}")
    for key in ["reports", "features", "artifacts", "folds", "handoff"]:
        paths[key].mkdir(parents=True, exist_ok=True)


def _build_research(config: ProjectConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[Any], dict[str, Any]]:
    panel = load_panel(config, config.raw["development_end"])
    pit = run_point_in_time_audit(panel, config, "development")
    panel = add_forward_labels(panel, config.horizons)
    assert_labels_are_forward(panel)
    curve_market = load_curve_market(config, config.raw["development_end"])
    factors, definitions, curve_features = build_feature_matrix_and_definitions(panel, curve_market, config)
    research = panel.merge(factors, on=["trade_date", "product", "sector"], how="left", validate="one_to_one")
    return research, panel, curve_features, definitions, pit


def _screen_training_fold(
    research: pd.DataFrame,
    definitions: list[Any],
    config: ProjectConfig,
    train_start_year: int,
    train_end_year: int,
    test_year: int,
) -> dict[str, pd.DataFrame]:
    training = mask_training_labels(research, config.horizons, test_year)
    start_date = f"{train_start_year}-01-01"
    end_date = f"{train_end_year}-12-31"
    metrics = compute_screening_metrics(training, definitions, config, start_date, end_date)
    feature_names = [definition.feature_name for definition in definitions]
    correlation = cross_sectional_factor_correlation(training, feature_names)
    clusters = cluster_factors(correlation, float(config.raw["screening"]["correlation_cluster_threshold"]))
    primary = metrics["summary"].loc[metrics["summary"].horizon.eq(config.primary_horizon)]
    cluster_info = choose_cluster_representatives(clusters, primary)
    first_inner = train_start_year + int(config.raw["walk_forward"]["minimum_inner_train_years"])
    incremental = expanding_incremental_tests(
        training,
        feature_names,
        config.primary_horizon,
        train_start_year,
        first_inner,
        train_end_year,
        int(config.raw["screening"]["minimum_cross_section_assets"]),
    )
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
    return {
        **metrics,
        "correlation": correlation,
        "clusters": cluster_info,
        "incremental": incremental,
        "feature_classes": feature_classes,
        "conceptual": conceptual,
    }


def _save_fold_screening(fold_dir: Path, result: dict[str, pd.DataFrame], registry: pd.DataFrame) -> None:
    fold_dir.mkdir(parents=True, exist_ok=True)
    mappings = {
        "factor_metrics.csv": "summary",
        "yearly_metrics.csv": "yearly",
        "product_metrics.csv": "product",
        "sector_metrics.csv": "sector",
        "factor_correlation.csv": "correlation",
        "factor_clusters.csv": "clusters",
        "incremental_tests.csv": "incremental",
        "feature_classification.csv": "feature_classes",
        "conceptual_factor_classification.csv": "conceptual",
    }
    for filename, key in mappings.items():
        result[key].to_csv(fold_dir / filename, index=key == "correlation")
    result["daily_ic"].to_csv(fold_dir / "daily_rank_ic.csv.gz", index=False, compression="gzip")
    registry.to_csv(fold_dir / "ml_feature_registry.csv", index=False)


def _model_fold(
    research: pd.DataFrame,
    feature_classes: pd.DataFrame,
    config: ProjectConfig,
    train_start_year: int,
    train_end_year: int,
    test_year: int,
) -> dict[str, Any]:
    available = research.loc[research.trade_date.dt.year.between(train_start_year, test_year)].copy()
    ml_features, registry = build_ml_features(available, feature_classes, config)
    model_config = dict(config.raw["model"])
    feature_names = registry.feature_name.tolist()
    labels = available[KEYS + [model_config["raw_return_column"]]].copy()
    data = ml_features.merge(labels, on=KEYS, how="inner", validate="one_to_one")
    data[model_config["model_target_column"]] = cross_sectional_rank_target(data, model_config["raw_return_column"])
    data = data.dropna(subset=[model_config["model_target_column"], model_config["raw_return_column"]]).copy()
    train = data.loc[data.trade_date.dt.year.between(train_start_year, train_end_year)].copy()
    train = purge_last_dates(train, int(model_config["purge_trading_days"]))
    test = data.loc[data.trade_date.dt.year.eq(test_year)].copy()
    if not feature_names or train.empty or test.empty:
        raise ValueError(f"fold {test_year} has empty features, train, or test")
    alpha, alpha_search = choose_alpha(train, feature_names, model_config)
    prediction, coefficients, intercept = _fit_ridge(
        train, test, feature_names, model_config["model_target_column"], alpha
    )
    predictions = test[["trade_date", "product", "sector", model_config["raw_return_column"], model_config["model_target_column"]]].copy()
    predictions = predictions.rename(columns={model_config["model_target_column"]: "target_cs_rank_1d"})
    predictions["prediction"] = prediction
    predictions["test_year"] = test_year
    predictions["selected_alpha"] = alpha
    predictions["model_intercept"] = intercept
    strategy = build_strategy_returns(predictions, model_config)
    strategy["test_year"] = test_year
    metrics = _fold_metrics(test_year, alpha, train, predictions, strategy, model_config)
    metrics["train_start_year"] = train_start_year
    metrics["train_end_year"] = train_end_year
    metrics["ml_feature_count"] = len(feature_names)
    coefficients_frame = pd.DataFrame({
        "feature_name": feature_names,
        "test_year": test_year,
        "standardized_coefficient": coefficients,
    }).merge(
        registry[["feature_name", "source_factor", "usage_type", "classification"]],
        on="feature_name",
        how="left",
        validate="one_to_one",
    )
    return {
        "ml_features": ml_features,
        "registry": registry,
        "predictions": predictions,
        "strategy": strategy,
        "metrics": metrics,
        "alpha_search": alpha_search,
        "coefficients": coefficients_frame,
    }


def _write_final_factor_reports(
    paths: dict[str, Path],
    config: ProjectConfig,
    panel: pd.DataFrame,
    curve: pd.DataFrame,
    pit: dict[str, Any],
    final_screen: dict[str, pd.DataFrame],
    final_registry: pd.DataFrame,
) -> None:
    availability = data_availability_summary(config)
    availability.to_csv(paths["artifacts"] / "data_availability.csv", index=False)
    write_data_audit(paths["reports"] / "data_audit_report.md", config.raw["dataset_version"], panel, curve, availability, pit)

    # Reuse the detailed report writers, then correct the method statement for this version.
    final_train_end = int(config.raw["walk_forward"]["last_oos_year"]) - 1
    report_raw = dict(config.raw)
    report_raw["development_end"] = f"{final_train_end}-12-31"
    report_config = ProjectConfig(root=config.root, raw=report_raw)
    factor_report = paths["reports"] / "factor_screening_report.md"
    write_factor_screening_report(
        factor_report,
        final_screen["conceptual"],
        final_screen["feature_classes"],
        report_config,
    )
    text = factor_report.read_text(encoding="utf-8").replace(
        "- 三年滚动训练，随后一年样本外验证；2025 在冻结前不参与选择。",
        "- 本报告对应最终外层折：2015—2024训练期筛选，随后2025外层测试；早期各折均独立重筛选。",
    )
    factor_report.write_text(text, encoding="utf-8")
    write_factor_family_report(
        paths["reports"] / "factor_family_report.md",
        final_screen["conceptual"],
        final_screen["feature_classes"],
    )
    write_ml_reports(
        paths["reports"] / "ml_feature_report.md",
        paths["features"] / "feature_dictionary.md",
        final_registry,
        final_screen["feature_classes"],
    )


def _write_manifest(config_path: Path, paths: dict[str, Path]) -> None:
    root = config_path.parent.parent
    source_files = sorted((root / "src" / "factor_expanding").glob("*.py"))
    tracked = [config_path, *source_files]
    output_files = sorted(path for path in paths["base"].rglob("*") if path.is_file() and path.name != "RUN_COMPLETE.json")
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_version": "expanding_2015_v1",
        "source_hashes": {str(path.relative_to(root)): _sha256(path) for path in tracked},
        "output_hashes": {str(path.relative_to(paths["base"])): _sha256(path) for path in output_files},
    }
    temporary = paths["base"] / ".RUN_COMPLETE.json.tmp"
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(paths["base"] / "RUN_COMPLETE.json")


def _write_handoff(paths: dict[str, Path], config: ProjectConfig) -> None:
    for source, name in [
        (paths["features"] / "ml_feature_registry.csv", "ml_feature_registry.csv"),
        (paths["features"] / "ml_features.csv.gz", "ml_features.csv.gz"),
        (paths["features"] / "labels.csv.gz", "labels.csv.gz"),
        (paths["features"] / "feature_dictionary.md", "feature_dictionary.md"),
        (paths["reports"] / "data_audit_report.md", "data_audit_report.md"),
        (paths["reports"] / "factor_screening_report.md", "factor_screening_report.md"),
        (paths["reports"] / "ml_feature_report.md", "ml_feature_report.md"),
        (paths["reports"] / "expanding_walk_forward_report.md", "expanding_walk_forward_report.md"),
    ]:
        shutil.copy2(source, paths["handoff"] / name)
    handoff_contract = {
        "run_version": config.raw["run_version"],
        "schema_basis": "2015-2024 training-only factor screening for the 2025 outer fold",
        "feature_file": "ml_features.csv.gz",
        "registry_file": "ml_feature_registry.csv",
        "label_file": "labels.csv.gz",
        "keys": ["trade_date", "product"],
        "feature_availability": "T close computed; earliest use T+1 open",
        "warning": "2025 is historical supplementary evidence, not a pristine sealed test",
    }
    (paths["handoff"] / "training_contract.json").write_text(
        json.dumps(handoff_contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (paths["handoff"] / "README.md").write_text(
        "# 扩展窗口训练交接包\n\n"
        "本目录可独立交给建模人员；特征矩阵、标签、字段注册表和研究报告均已包含。\n\n"
        "- `ml_features.csv.gz`：按最终2025外层折训练期（2015—2024）筛出的字段加工后的全量矩阵。\n"
        "- `labels.csv.gz`：1/5/20日未来开盘至收盘复合收益标签。\n"
        "- `ml_feature_registry.csv`：模型输入字段定义。\n"
        "- `training_contract.json`：时间、键和使用约束。\n\n"
        "必须按时间切分训练与测试，不得随机打乱；特征T日收盘形成，最早T+1开盘使用。\n",
        encoding="utf-8",
    )


def finalize_existing(config_path: str | Path) -> dict[str, Any]:
    """Finish reports and manifests after all fold computations have been persisted."""
    config_path = Path(config_path).resolve()
    config = load_config(config_path)
    paths = _paths(config)
    if (paths["base"] / "RUN_COMPLETE.json").exists():
        raise FileExistsError("expanding run is already finalized")
    required = [
        paths["artifacts"] / "oos_predictions.csv.gz",
        paths["artifacts"] / "strategy_daily_returns.csv",
        paths["artifacts"] / "fold_metrics.csv",
        paths["artifacts"] / "feature_selection_history.csv",
        paths["artifacts"] / "conceptual_class_history.csv",
        paths["features"] / "ml_feature_registry.csv",
        paths["features"] / "ml_features.csv.gz",
        paths["features"] / "labels.csv.gz",
        paths["folds"] / "2025" / "feature_classification.csv",
        paths["folds"] / "2025" / "conceptual_factor_classification.csv",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"cannot finalize incomplete run: {missing}")

    _, panel, curve, _, pit = _build_research(config)
    final_screen = {
        "feature_classes": pd.read_csv(paths["folds"] / "2025" / "feature_classification.csv"),
        "conceptual": pd.read_csv(paths["folds"] / "2025" / "conceptual_factor_classification.csv"),
    }
    final_registry = pd.read_csv(paths["features"] / "ml_feature_registry.csv")
    predictions = pd.read_csv(paths["artifacts"] / "oos_predictions.csv.gz", parse_dates=["trade_date"])
    strategy = pd.read_csv(paths["artifacts"] / "strategy_daily_returns.csv", parse_dates=["trade_date"])
    fold_metrics = pd.read_csv(paths["artifacts"] / "fold_metrics.csv")
    selection_history = pd.read_csv(paths["artifacts"] / "feature_selection_history.csv")
    class_history = pd.read_csv(paths["artifacts"] / "conceptual_class_history.csv")

    _write_final_factor_reports(paths, config, panel, curve, pit, final_screen, final_registry)
    write_walk_forward_report(
        paths["reports"] / "expanding_walk_forward_report.md",
        fold_metrics,
        predictions,
        strategy,
        selection_history,
        class_history,
        config.raw,
    )
    _write_handoff(paths, config)
    _write_manifest(config_path, paths)
    daily_ic = _daily_correlations(predictions)
    return {
        "run_version": config.raw["run_version"],
        "status": "finalized_from_persisted_folds",
        "folds": int(fold_metrics.test_year.nunique()),
        "oos_years": sorted(fold_metrics.test_year.astype(int).unique().tolist()),
        "oos_rows": len(predictions),
        "mean_rank_ic": float(daily_ic.rank_ic.mean()),
        "final_feature_count": len(final_registry),
        "report": str(paths["reports"] / "expanding_walk_forward_report.md"),
    }


def run(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    config = load_config(config_path)
    paths = _paths(config)
    _prepare_output(paths)
    research, panel, curve, definitions, pit = _build_research(config)
    splits = expanding_splits(
        int(config.raw["walk_forward"]["train_start_year"]),
        int(config.raw["walk_forward"]["first_oos_year"]),
        int(config.raw["walk_forward"]["last_oos_year"]),
    )

    prediction_parts: list[pd.DataFrame] = []
    strategy_parts: list[pd.DataFrame] = []
    fold_metric_rows: list[dict[str, Any]] = []
    coefficient_parts: list[pd.DataFrame] = []
    selection_rows: list[pd.DataFrame] = []
    class_rows: list[dict[str, Any]] = []
    final_screen: dict[str, pd.DataFrame] | None = None
    final_registry: pd.DataFrame | None = None
    final_matrix: pd.DataFrame | None = None

    for train_start, train_end, test_year in splits:
        print(f"[expanding] screening {train_start}-{train_end}, testing {test_year}", flush=True)
        screen = _screen_training_fold(research, definitions, config, train_start, train_end, test_year)
        model = _model_fold(research, screen["feature_classes"], config, train_start, train_end, test_year)
        fold_dir = paths["folds"] / str(test_year)
        _save_fold_screening(fold_dir, screen, model["registry"])
        model["alpha_search"].to_csv(fold_dir / "alpha_search.csv", index=False)
        model["coefficients"].to_csv(fold_dir / "coefficients.csv", index=False)
        model["predictions"].to_csv(fold_dir / "predictions.csv.gz", index=False, compression="gzip")
        model["strategy"].to_csv(fold_dir / "strategy_daily_returns.csv", index=False)
        pd.DataFrame([model["metrics"]]).to_csv(fold_dir / "model_metrics.csv", index=False)

        selected = model["registry"][["feature_name", "source_factor", "usage_type", "classification"]].copy()
        selected["test_year"] = test_year
        selection_rows.append(selected)
        counts = screen["conceptual"].classification.value_counts().sort_index()
        class_rows.extend({"test_year": test_year, "classification": key, "count": int(value)} for key, value in counts.items())
        prediction_parts.append(model["predictions"])
        strategy_parts.append(model["strategy"])
        fold_metric_rows.append(model["metrics"])
        coefficient_parts.append(model["coefficients"])
        final_screen = screen
        final_registry = model["registry"]
        final_matrix = model["ml_features"]

    assert final_screen is not None and final_registry is not None and final_matrix is not None
    predictions = pd.concat(prediction_parts, ignore_index=True)
    strategy = pd.concat(strategy_parts, ignore_index=True)
    fold_metrics = pd.DataFrame(fold_metric_rows)
    coefficients = pd.concat(coefficient_parts, ignore_index=True)
    selected_all = pd.concat(selection_rows, ignore_index=True)
    total_folds = len(splits)
    selection_history = (
        selected_all.groupby(["feature_name", "source_factor", "usage_type"], as_index=False)
        .agg(
            selected_folds=("test_year", "nunique"),
            first_test_year=("test_year", "min"),
            last_test_year=("test_year", "max"),
        )
    )
    selection_history["selected_fold_share"] = selection_history.selected_folds / total_folds
    class_history = pd.DataFrame(class_rows)

    predictions.to_csv(paths["artifacts"] / "oos_predictions.csv.gz", index=False, compression="gzip")
    strategy.to_csv(paths["artifacts"] / "strategy_daily_returns.csv", index=False)
    fold_metrics.to_csv(paths["artifacts"] / "fold_metrics.csv", index=False)
    coefficients.to_csv(paths["artifacts"] / "coefficients_by_fold.csv", index=False)
    selection_history.to_csv(paths["artifacts"] / "feature_selection_history.csv", index=False)
    class_history.to_csv(paths["artifacts"] / "conceptual_class_history.csv", index=False)
    conceptual_catalog_frame().to_csv(paths["artifacts"] / "factor_registry.csv", index=False)
    final_registry.to_csv(paths["features"] / "ml_feature_registry.csv", index=False)
    final_matrix.to_csv(paths["features"] / "ml_features.csv.gz", index=False, compression="gzip")
    label_columns = ["trade_date", "product", *[f"future_return_{h}d" for h in config.horizons]]
    research[label_columns].to_csv(paths["features"] / "labels.csv.gz", index=False, compression="gzip")

    _write_final_factor_reports(paths, config, panel, curve, pit, final_screen, final_registry)
    write_walk_forward_report(
        paths["reports"] / "expanding_walk_forward_report.md",
        fold_metrics,
        predictions,
        strategy,
        selection_history,
        class_history,
        config.raw,
    )
    _write_handoff(paths, config)
    _write_manifest(config_path, paths)
    daily_ic = _daily_correlations(predictions)
    return {
        "run_version": config.raw["run_version"],
        "folds": len(splits),
        "oos_years": [split[2] for split in splits],
        "oos_rows": len(predictions),
        "mean_rank_ic": float(daily_ic.rank_ic.mean()),
        "final_feature_count": len(final_registry),
        "report": str(paths["reports"] / "expanding_walk_forward_report.md"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="2015固定起点扩展窗口端到端因子筛选与回测")
    parser.add_argument("--config", default="config/screening_expanding_2015_v1.json")
    parser.add_argument("--finalize-existing", action="store_true", help="从已持久化的完整折结果恢复生成报告")
    args = parser.parse_args(argv)
    result = finalize_existing(args.config) if args.finalize_existing else run(args.config)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
