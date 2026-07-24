from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from .config import ProjectConfig, load_config
from .data import data_audit, load_data_final
from .factors import build_factors
from .features import build_ml_features
from .handoff import build_sealed_evaluation, build_training_handoff
from .incremental import incremental_ridge_tests
from .labels import add_forward_labels, assert_forward_labels
from .reports import (
    write_data_audit_report,
    write_family_report,
    write_feature_report,
    write_json,
    write_sealed_report,
    write_screening_report,
)
from .screening import (
    classify_factors,
    parameter_sensitivity_diagnostics,
    robustness_diagnostics,
    screen_factors,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prepare_output_directories(config: ProjectConfig) -> tuple[Path, Path]:
    protected = [
        config.output_root,
        config.handoff_root,
        config.sealed_root,
        config.feature_root,
        config.report_root,
    ]
    existing = [path for path in protected if path.exists()]
    if existing:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"refusing to overwrite existing generated outputs: {names}"
        )
    report_root = config.report_root
    screening_root = config.output_root / "screening"
    report_root.mkdir(parents=True)
    screening_root.mkdir(parents=True)
    (config.output_root / "intermediate").mkdir(parents=True)
    config.feature_root.mkdir(parents=True)
    return report_root, screening_root


def _protocol_compliance_report(config: ProjectConfig) -> str:
    return """# 统一实验协议合规说明

## 已实现

- 30 个中国商品期货品种，2015-01-05 至 2025-12-31。
- 开发期截止 2024-12-31；2025 输出到独立评价目录。
- 单元与主键：`trade_date × product`；附加 `sector`。
- 主标签：T 收盘形成信号，固定真实合约从 T+1 开盘持有至 T+6 开盘的五日收益。
- 1/5/20 日标签；5 日为主标签。
- 2015 起点不变的六个扩展训练折：2019—2024 逐年样本外验证。
- 2025 仅在开发口径冻结后执行一次最终测试。
- 57 个概念因子全部登记；所有可计算、通过数据质量硬门的因子进入全量输入端。
- V2.1 因子库57个概念因子全部登记；缺少必要数据者保留为F类并写明原因。
- 特征注册表是唯一输入字段接口；不使用目标值决定交接字段。
- 5 日调仓、前后 20%、组内等权、单边 0/3/5/10 bps 筛选诊断。
- 连续变量使用点时安全的同日横截面稳健标准化；市场级变量使用仅含历史的滚动标准化。

## 用户批准的协议修订

- 取消“用 T−1 可见持仓量重新选择主力”。本轮读取 `data_final_raw` 的 TuShare 连续代码，
  再按同日连续行情字段映射到真实合约；信号在 T 收盘后形成，交易发生在 T+1 开盘。
- 原协议的三年滚动折改为用户最终确认的扩展窗口。
- 原协议仅三类核心特征的限制改为筛选全部57个概念因子；相关簇只作诊断，不物理删除字段。
- 基本面只有观察日期而没有日内发布时间，统一延迟一交易日后才允许进入信号。

## 暂不能完整核验

- 源数据只有 OHLC 锁价代理，没有交易所逐日精确涨跌停价及开平仓能否成交的完整明细。
- 缺少盘口、买卖价差、逐笔冲击和容量数据。
- 2021 年以前真实手续费覆盖不足；5bps 是统一研究成本情景，不是实盘成本承诺。
- 现货源截至 2023 年；超过7个自然日的旧现货不进入基差，2024—2025 基差自然缺失。
"""


def run_pipeline(config_path: str | Path) -> dict[str, object]:
    config = load_config(config_path)
    report_root, screening_root = _prepare_output_directories(config)
    bundle = load_data_final(config)
    panel = bundle.panel
    audit = data_audit(bundle, config)
    horizons = [int(item) for item in config.raw["labels"]["horizons"]]
    labelled = add_forward_labels(
        panel,
        bundle.contracts,
        horizons,
        exclude_suspected_locks=bool(
            config.raw["labels"]["exclude_suspected_locked_entry_or_exit"]
        ),
    )
    assert_forward_labels(labelled, horizons)
    for horizon in horizons:
        audit[f"label_{horizon}d_coverage"] = float(
            labelled[f"future_return_{horizon}d"].notna().mean()
        )
    raw_factors = build_factors(panel, config)

    intermediate_root = config.output_root / "intermediate"
    panel.to_parquet(intermediate_root / "mapped_product_panel.parquet", index=False)
    raw_factors.to_parquet(intermediate_root / "raw_factors_57.parquet", index=False)
    labelled[
        [
            "trade_date",
            "product",
            "mapped_actual_ts_code",
            "entry_date",
            *[
                column
                for horizon in horizons
                for column in [
                    f"future_return_{horizon}d",
                    f"label_end_date_{horizon}d",
                ]
            ],
        ]
    ].to_parquet(intermediate_root / "labels.parquet", index=False)

    processing = config.raw["feature_processing"]
    screening_config = config.raw["screening"]
    features, factor_registry, feature_registry, correlation = build_ml_features(
        raw_factors=raw_factors,
        development_end=pd.Timestamp(config.raw["sample"]["development_end"]),
        minimum_dense_coverage=float(
            screening_config["minimum_dense_coverage"]
        ),
        minimum_event_observations=int(
            screening_config["minimum_event_observations"]
        ),
        minimum_event_instruments=int(
            screening_config["minimum_event_instruments"]
        ),
        missing_indicator_threshold=float(processing["missing_indicator_threshold"]),
        mad_clip=float(processing["mad_clip"]),
        rolling_z_window=int(processing["rolling_z_window"]),
        rolling_z_minimum=int(processing["rolling_z_minimum"]),
        correlation_threshold=float(screening_config["correlation_threshold"]),
    )
    features.to_parquet(
        intermediate_root / "ml_features_all_years.parquet",
        index=False,
    )

    execution = config.raw["execution"]
    metrics, yearly, portfolio, quantiles = screen_factors(
        raw_factors=raw_factors,
        labelled_panel=labelled,
        evaluation_start=pd.Timestamp(screening_config["evaluation_start"]),
        evaluation_end=pd.Timestamp(screening_config["evaluation_end"]),
        minimum_cross_section=int(screening_config["minimum_cross_section"]),
        long_fraction=float(execution["long_fraction"]),
        short_fraction=float(execution["short_fraction"]),
        cost_scenarios=[int(item) for item in execution["cost_scenarios_bps_one_way"]],
    )
    incremental = incremental_ridge_tests(
        features=features,
        labels=labelled,
        feature_registry=feature_registry,
        folds=config.raw["folds"],
    )
    classified = classify_factors(factor_registry, metrics)
    horizon_metrics, product_metrics, sector_metrics = robustness_diagnostics(
        raw_factors=raw_factors,
        labelled_panel=labelled,
        evaluation_start=pd.Timestamp(screening_config["evaluation_start"]),
        evaluation_end=pd.Timestamp(screening_config["evaluation_end"]),
        minimum_cross_section=int(screening_config["minimum_cross_section"]),
    )
    parameter_sensitivity = parameter_sensitivity_diagnostics(
        raw_factors=raw_factors,
        labelled_panel=labelled,
        evaluation_start=pd.Timestamp(screening_config["evaluation_start"]),
        evaluation_end=pd.Timestamp(screening_config["evaluation_end"]),
        minimum_cross_section=int(screening_config["minimum_cross_section"]),
    )
    product_summary_rows: list[dict[str, object]] = []
    for factor_id, subset in product_metrics.groupby("factor_id", sort=False):
        valid = subset.loc[subset["time_series_rank_ic"].notna()]
        product_summary_rows.append(
            {
                "factor_id": factor_id,
                "evaluated_products": int(len(valid)),
                "positive_product_rate": (
                    float(valid["time_series_rank_ic"].gt(0).mean())
                    if not valid.empty
                    else float("nan")
                ),
                "median_product_rank_ic": (
                    float(valid["time_series_rank_ic"].median())
                    if not valid.empty
                    else float("nan")
                ),
            }
        )
    sector_summary_rows: list[dict[str, object]] = []
    for factor_id, subset in sector_metrics.groupby("factor_id", sort=False):
        valid = subset.loc[subset["rank_ic"].notna()]
        sector_summary_rows.append(
            {
                "factor_id": factor_id,
                "evaluated_sectors": int(len(valid)),
                "positive_sector_rate": (
                    float(valid["rank_ic"].gt(0).mean())
                    if not valid.empty
                    else float("nan")
                ),
                "median_sector_rank_ic": (
                    float(valid["rank_ic"].median())
                    if not valid.empty
                    else float("nan")
                ),
            }
        )
    yearly_stability = (
        yearly.groupby("factor_id", as_index=False)
        .agg(
            positive_year_rate=("rank_ic", lambda values: float(values.gt(0).mean())),
            yearly_rank_ic_std=("rank_ic", "std"),
        )
    )
    delayed = horizon_metrics.loc[
        horizon_metrics["horizon_days"].eq("5d_signal_delayed_1d"),
        ["factor_id", "rank_ic"],
    ].rename(columns={"rank_ic": "delayed_1d_rank_ic"})
    primary_feature_map = (
        feature_registry.loc[
            ~feature_registry["is_missing_indicator"].astype(bool)
            & ~feature_registry["is_interaction"].astype(bool),
            ["source_factor", "feature_name", "correlation_cluster", "cluster_representative"],
        ]
        .drop_duplicates("source_factor")
        .rename(columns={"source_factor": "factor_id"})
    )
    maximum_correlations = {}
    for feature in correlation.columns:
        others = correlation[feature].drop(index=feature).abs().dropna()
        maximum_correlations[feature] = float(others.max()) if not others.empty else float("nan")
    primary_feature_map["max_abs_feature_correlation"] = primary_feature_map[
        "feature_name"
    ].map(maximum_correlations)
    classified = (
        classified.merge(yearly_stability, on="factor_id", how="left")
        .merge(delayed, on="factor_id", how="left")
        .merge(primary_feature_map, on="factor_id", how="left")
        .merge(pd.DataFrame(product_summary_rows), on="factor_id", how="left")
        .merge(pd.DataFrame(sector_summary_rows), on="factor_id", how="left")
    )

    factor_registry.to_csv(
        screening_root / "factor_registry.csv",
        index=False,
        encoding="utf-8",
    )
    feature_registry.to_csv(
        screening_root / "ml_feature_registry.csv",
        index=False,
        encoding="utf-8",
    )
    metrics.to_csv(screening_root / "factor_metrics.csv", index=False, encoding="utf-8")
    yearly.to_csv(screening_root / "yearly_metrics.csv", index=False, encoding="utf-8")
    horizon_metrics.to_csv(
        screening_root / "horizon_and_delay_metrics.csv",
        index=False,
        encoding="utf-8",
    )
    product_metrics.to_csv(
        screening_root / "product_metrics.csv",
        index=False,
        encoding="utf-8",
    )
    sector_metrics.to_csv(
        screening_root / "sector_metrics.csv",
        index=False,
        encoding="utf-8",
    )
    parameter_sensitivity.to_csv(
        screening_root / "parameter_sensitivity_metrics.csv",
        index=False,
        encoding="utf-8",
    )
    portfolio.to_csv(
        screening_root / "factor_portfolio_returns.csv",
        index=False,
        encoding="utf-8",
    )
    quantiles.to_csv(
        screening_root / "quantile_returns.csv",
        index=False,
        encoding="utf-8",
    )
    incremental.to_csv(
        screening_root / "incremental_ridge_tests.csv",
        index=False,
        encoding="utf-8",
    )
    classified.to_csv(
        screening_root / "factor_classification.csv",
        index=False,
        encoding="utf-8",
    )
    correlation.to_csv(
        screening_root / "feature_correlation.csv",
        index=True,
        encoding="utf-8",
    )
    factor_registry.to_csv(
        config.feature_root / "factor_registry.csv",
        index=False,
        encoding="utf-8",
    )
    feature_registry.to_csv(
        config.feature_root / "ml_feature_registry.csv",
        index=False,
        encoding="utf-8",
    )

    write_data_audit_report(
        report_root / "data_audit_report.md",
        audit,
        factor_registry,
    )
    write_screening_report(
        report_root / "factor_screening_report.md",
        classified,
        incremental,
    )
    write_family_report(
        report_root / "factor_family_report.md",
        classified,
    )
    write_feature_report(
        report_root / "ml_feature_report.md",
        feature_registry,
    )
    (config.feature_root / "feature_dictionary.md").write_text(
        (report_root / "ml_feature_report.md").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (report_root / "protocol_compliance_report.md").write_text(
        _protocol_compliance_report(config),
        encoding="utf-8",
    )

    handoff_summary = build_training_handoff(
        config,
        features,
        labelled,
        feature_registry,
        classified,
    )
    sealed_summary = build_sealed_evaluation(
        config,
        features,
        labelled,
        feature_registry,
    )
    sealed_metrics, sealed_yearly, sealed_portfolio, sealed_quantiles = screen_factors(
        raw_factors=raw_factors,
        labelled_panel=labelled,
        evaluation_start=pd.Timestamp(config.raw["sealed_fold"]["valid_start"]),
        evaluation_end=pd.Timestamp(config.raw["sealed_fold"]["valid_end"]),
        minimum_cross_section=int(screening_config["minimum_cross_section"]),
        long_fraction=float(execution["long_fraction"]),
        short_fraction=float(execution["short_fraction"]),
        cost_scenarios=[
            int(item) for item in execution["cost_scenarios_bps_one_way"]
        ],
    )
    sealed_metrics.to_csv(
        config.sealed_root / "factor_metrics_2025.csv",
        index=False,
        encoding="utf-8",
    )
    sealed_yearly.to_csv(
        config.sealed_root / "yearly_metrics_2025.csv",
        index=False,
        encoding="utf-8",
    )
    sealed_portfolio.to_csv(
        config.sealed_root / "factor_portfolio_returns_2025.csv",
        index=False,
        encoding="utf-8",
    )
    sealed_quantiles.to_csv(
        config.sealed_root / "quantile_returns_2025.csv",
        index=False,
        encoding="utf-8",
    )
    write_sealed_report(
        report_root / "sealed_2025_report.md",
        classified,
        sealed_metrics,
    )
    (config.sealed_root / "sealed_2025_report.md").write_text(
        (report_root / "sealed_2025_report.md").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    source_root = config.project_root / "src"
    source_files = sorted(source_root.rglob("*.py"))
    manifest = {
        "project_version": config.raw["project_version"],
        "data_sha256": _sha256(config.data_file),
        "continuous_source_sha256": {
            str(path.relative_to(config.repo_root)): _sha256(path)
            for path in config.continuous_daily_files
        },
        "protocol_markdown_sha256": _sha256(
            config.resolve_repo_path(config.raw["protocol_markdown"])
        ),
        "factor_library_sha256": _sha256(
            config.resolve_repo_path(config.raw["factor_library"])
        ),
        "config_sha256": _sha256(Path(config_path).resolve()),
        "source_sha256": {
            str(path.relative_to(config.project_root)): _sha256(path)
            for path in source_files
        },
        "audit": audit,
        "handoff": handoff_summary,
        "sealed": sealed_summary,
        "t_minus_1_main_selection_required": False,
        "target_based_feature_selection": False,
        "expanding_folds": True,
    }
    write_json(config.output_root / "RUN_MANIFEST.json", manifest)
    return manifest
