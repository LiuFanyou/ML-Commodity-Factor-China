from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from factor_screening.config import load_config
from factor_screening.data import load_panel
from factor_screening.labels import add_forward_labels

from .common import (
    KEYS,
    _daily_correlations,
    build_strategy_returns,
    cross_sectional_rank_target,
    mean_daily_rank_ic,
    purge_last_dates,
    return_statistics,
)
from .linear_ridge import _fit_ridge, _fold_metrics
from .model_freeze import verify_model_freeze_manifest


def _format(value: Any, digits: int = 4) -> str:
    if value is None or (isinstance(value, (float, np.floating)) and np.isnan(value)):
        return "—"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def _markdown(frame: pd.DataFrame) -> str:
    return frame.to_markdown(index=False) if not frame.empty else "暂无记录。"


def _return_table(strategy: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    rows = [{"成本情景": "毛收益", **return_statistics(strategy["gross_return"], int(config["annualization"]))}]
    for bps in config["one_way_cost_bps"]:
        bps = float(bps)
        label = str(int(bps)) if bps.is_integer() else str(bps).replace(".", "p")
        rows.append({
            "成本情景": f"单边{bps:g}bp",
            **return_statistics(strategy[f"net_{label}bps_return"], int(config["annualization"])),
        })
    return pd.DataFrame(rows)


def _monthly_metrics(predictions: pd.DataFrame, strategy: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    daily_ic = _daily_correlations(predictions)
    daily_ic["month"] = pd.to_datetime(daily_ic["trade_date"]).dt.to_period("M").astype(str)
    ic_monthly = daily_ic.groupby("month", as_index=False).agg(
        rank_ic_mean=("rank_ic", "mean"),
        rank_ic_hit_rate=("rank_ic", lambda values: float(values.gt(0).mean())),
        ic_days=("rank_ic", "size"),
    )
    work = strategy.copy()
    work["month"] = pd.to_datetime(work["trade_date"]).dt.to_period("M").astype(str)
    return_columns = ["gross_return"]
    for bps in config["one_way_cost_bps"]:
        bps = float(bps)
        label = str(int(bps)) if bps.is_integer() else str(bps).replace(".", "p")
        return_columns.append(f"net_{label}bps_return")
    rows: list[dict[str, Any]] = []
    for month, part in work.groupby("month", sort=True):
        row: dict[str, Any] = {"month": month, "strategy_days": len(part)}
        for column in return_columns:
            row[f"{column}_total"] = float((1.0 + part[column]).prod() - 1.0)
            row[f"{column}_hit_rate"] = float(part[column].gt(0).mean())
        rows.append(row)
    return ic_monthly.merge(pd.DataFrame(rows), on="month", how="outer").sort_values("month")


def _comparison_table(root: Path, metrics_2025: dict[str, Any]) -> pd.DataFrame:
    development = pd.read_csv(root / "modeling" / "linear_ridge_v1" / "outputs" / "fold_metrics.csv")
    development_row = {
        "阶段": "2018—2024开发期滚动样本外",
        "平均RankIC": float(development["rank_ic_mean"].mean()),
        "RankIC胜率": float(development["rank_ic_hit_rate"].mean()),
        "毛收益年化": float(development["gross_annual_return"].mean()),
        "毛收益Sharpe": float(development["gross_sharpe"].mean()),
        "单边1bp净收益年化": float(development["net_1bps_annual_return"].mean())
        if "net_1bps_annual_return" in development.columns
        else np.nan,
        "单边3bp净收益年化": float(development["net_3bps_annual_return"].mean()),
    }
    supplemental_row = {
        "阶段": "2025补充样本外",
        "平均RankIC": metrics_2025["rank_ic_mean"],
        "RankIC胜率": metrics_2025["rank_ic_hit_rate"],
        "毛收益年化": metrics_2025["gross_annual_return"],
        "毛收益Sharpe": metrics_2025["gross_sharpe"],
        "单边1bp净收益年化": metrics_2025.get("net_1bps_annual_return", np.nan),
        "单边3bp净收益年化": metrics_2025["net_3bps_annual_return"],
    }
    return pd.DataFrame([development_row, supplemental_row])


def _conclusion(metrics: dict[str, Any]) -> tuple[str, list[str]]:
    rank_positive = float(metrics["rank_ic_mean"]) > 0
    net_1 = float(metrics.get("net_1bps_annual_return", np.nan))
    net_3 = float(metrics["net_3bps_annual_return"])
    net_1_positive = bool(np.isfinite(net_1) and net_1 > 0)
    net_3_positive = bool(np.isfinite(net_3) and net_3 > 0)
    if rank_positive and net_3_positive:
        grade = "补充样本外证据较强"
    elif rank_positive and net_1_positive:
        grade = "存在弱预测证据，但成本承受力有限"
    elif rank_positive:
        grade = "预测方向为正，但尚不具备成本后交易证据"
    else:
        grade = "2025未提供正向预测证据"
    findings = [
        f"2025平均RankIC为{metrics['rank_ic_mean']:.4f}，方向{'为正' if rank_positive else '不为正'}。",
        f"单边1bp净年化收益为{net_1:.2%}。" if np.isfinite(net_1) else "单边1bp净年化收益未记录。",
        f"单边3bp净年化收益为{net_3:.2%}，{'通过' if net_3_positive else '未通过'}中等成本情景。",
    ]
    return grade, findings


def write_report(
    path: Path,
    config: dict[str, Any],
    metrics: dict[str, Any],
    predictions: pd.DataFrame,
    strategy: pd.DataFrame,
    monthly: pd.DataFrame,
    coefficients: pd.DataFrame,
    alpha_search: pd.DataFrame,
    comparison: pd.DataFrame,
    feature_count: int,
) -> None:
    grade, findings = _conclusion(metrics)
    metric_rows = [
        {"指标": "训练区间", "结果": f"{metrics['train_start']} 至 {metrics['train_end']}"},
        {"指标": "2025测试记录", "结果": metrics["test_rows"]},
        {"指标": "2025测试交易日", "结果": metrics["test_days"]},
        {"指标": "选定Ridge alpha", "结果": metrics["selected_alpha"]},
        {"指标": "平均Pearson IC", "结果": metrics["pearson_ic_mean"]},
        {"指标": "平均RankIC", "结果": metrics["rank_ic_mean"]},
        {"指标": "RankIC IR", "结果": metrics["rank_icir"]},
        {"指标": "RankIC胜率", "结果": metrics["rank_ic_hit_rate"]},
        {"指标": "横截面排名目标R²", "结果": metrics["target_r2"]},
        {"指标": "横截面排名目标MAE", "结果": metrics["target_mae"]},
    ]
    metric_table = pd.DataFrame(metric_rows)
    metric_table["结果"] = metric_table["结果"].map(_format)
    return_table = _return_table(strategy, config)
    for column in return_table.columns[1:]:
        return_table[column] = return_table[column].map(_format)
    comparison_display = comparison.copy()
    for column in comparison_display.columns[1:]:
        comparison_display[column] = comparison_display[column].map(_format)
    monthly_display = monthly.copy()
    for column in monthly_display.columns:
        if pd.api.types.is_float_dtype(monthly_display[column]):
            monthly_display[column] = monthly_display[column].map(_format)
    alpha_display = alpha_search.copy()
    for column in ["alpha", "mean_rank_ic"]:
        if column in alpha_display.columns:
            alpha_display[column] = alpha_display[column].map(_format)
    positive = coefficients.nlargest(min(10, len(coefficients)), "standardized_coefficient").copy()
    negative = coefficients.nsmallest(min(10, len(coefficients)), "standardized_coefficient").copy()
    for frame in [positive, negative]:
        frame["standardized_coefficient"] = frame["standardized_coefficient"].map(_format)
    coef_cols = [
        c
        for c in ["feature_name", "source_factor", "usage_type", "standardized_coefficient"]
        if c in coefficients.columns
    ]
    prediction_rows = int(metrics.get("test_rows", len(predictions)))

    lines = [
        "# 2025年Ridge线性因子模型补充回测报告",
        "",
        f"> 结论：**{grade}**。这是补充样本外测试，不是纯粹未查看的模型级封存测试。",
        "",
        "## 测试边界",
        "",
        "- 训练期：2022—2024；测试期：2025；训练边界净化1个交易日。",
        f"- 模型输入：冻结注册表中的全部{feature_count}个字段；没有按2025结果增删特征。",
        "- 标签：信号形成后的下一交易日开盘至收盘收益，并转为当日横截面中心化排名。",
        "- 策略：预测最高20%做多、最低20%做空，多空各0.5，每日开盘建仓、收盘平仓。",
        "- 成本：单边1/3/5bp敏感性；每日往返换手固定按2计。",
        "- 限制：2025年的因子级筛选结果此前已被查看，因此本次只能检验冻结模型在该年度的补充表现；2026及以后才是新的模型级封存期。",
        "",
        "## 核心发现",
        "",
        *[f"- {item}" for item in findings],
        "",
        "## 预测指标",
        "",
        _markdown(metric_table),
        "",
        "## 策略收益与成本敏感性",
        "",
        _markdown(return_table),
        "",
        "## 与开发期结果比较",
        "",
        "开发期一行是七个年度折指标的简单平均，用于方向比较，不等同于拼接日收益后的总体统计。",
        "",
        _markdown(comparison_display),
        "",
        "## 2025分月表现",
        "",
        _markdown(monthly_display),
        "",
        "## 训练窗口内部alpha选择",
        "",
        _markdown(alpha_display),
        "",
        "## 标准化系数最大的正向字段",
        "",
        _markdown(positive[coef_cols]),
        "",
        "## 标准化系数最大的负向字段",
        "",
        _markdown(negative[coef_cols]),
        "",
        "## 研究结论与使用限制",
        "",
        "1. RankIC衡量横截面排序能力；即使为正，也必须同时检查成本后收益。",
        "2. 本回测未核验合约乘数、品种真实手续费、盘口冲击、涨跌停成交和容量，不能直接解释为可实盘收益。",
        "3. 2025结果不得用于反向修改本版本特征、标签、alpha网格或持仓规则；若修改，应登记为新模型版本，并等待新数据验证。",
        "4. 线性系数受特征相关性影响，只表示冻结模型中的条件关联，不是因果结论。",
        "",
        f"自动核验：预测记录{prediction_rows}条、策略日{len(strategy)}天，2025之外记录为0。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _choose_alpha_legacy(
    outer_train: pd.DataFrame,
    feature_names: list[str],
    config: dict[str, Any],
) -> tuple[float, pd.DataFrame]:
    """Inner-year alpha search using calendar purge (legacy 1d supplemental path)."""
    raw_return = config["raw_return_column"]
    target_column = config["model_target_column"]
    inner_year = int(outer_train["trade_date"].dt.year.max())
    inner_train = outer_train.loc[outer_train["trade_date"].dt.year.lt(inner_year)].copy()
    inner_train = purge_last_dates(inner_train, int(config.get("purge_trading_days", 1)))
    inner_valid = outer_train.loc[outer_train["trade_date"].dt.year.eq(inner_year)].copy()
    rows: list[dict[str, Any]] = []
    for alpha in config["alpha_grid"]:
        if inner_train.empty or inner_valid.empty:
            break
        prediction, _, _ = _fit_ridge(
            inner_train,
            inner_valid,
            feature_names,
            target_column,
            float(alpha),
        )
        evaluated = inner_valid[["trade_date", "product", raw_return]].copy()
        evaluated["prediction"] = prediction
        rows.append({
            "inner_validation_year": inner_year,
            "alpha": float(alpha),
            "mean_rank_ic": mean_daily_rank_ic(evaluated, return_column=raw_return),
            "rows": len(inner_valid),
        })
    scores = pd.DataFrame(rows).sort_values(["mean_rank_ic", "alpha"], ascending=[False, True])
    if scores.empty or scores["mean_rank_ic"].isna().all():
        return float(config["fallback_alpha"]), scores
    return float(scores.iloc[0]["alpha"]), scores


def regenerate_report_from_artifacts(config_path: str | Path) -> dict[str, Any]:
    """Rebuild supplemental_2025_report.md from saved CSVs (no feature matrix needed)."""
    config_path = Path(config_path).resolve()
    root = config_path.parents[2]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_dir = root / config["output_directory"]
    required = {
        "metrics": output_dir / "metrics_2025.csv",
        "strategy": output_dir / "strategy_2025_daily_returns.csv",
        "monthly": output_dir / "monthly_metrics_2025.csv",
        "coefficients": output_dir / "coefficients_2025.csv",
        "alpha_search": output_dir / "alpha_search_2025.csv",
        "comparison": output_dir / "development_comparison.csv",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "cannot regenerate report; missing artifacts: "
            + ", ".join(missing)
            + f" under {output_dir}"
        )

    metrics_frame = pd.read_csv(required["metrics"])
    metrics = metrics_frame.iloc[0].to_dict()
    strategy = pd.read_csv(required["strategy"], parse_dates=["trade_date"])
    monthly = pd.read_csv(required["monthly"])
    coefficients = pd.read_csv(required["coefficients"])
    alpha_search = pd.read_csv(required["alpha_search"])
    comparison = pd.read_csv(required["comparison"])

    predictions_path = output_dir / "oos_2025_predictions.csv.gz"
    if predictions_path.is_file():
        predictions = pd.read_csv(predictions_path, parse_dates=["trade_date"])
    else:
        predictions = pd.DataFrame()

    report_path = output_dir / "supplemental_2025_report.md"
    write_report(
        report_path,
        config,
        metrics,
        predictions,
        strategy,
        monthly,
        coefficients,
        alpha_search,
        comparison,
        len(coefficients),
    )
    return {
        "status": "report_regenerated_from_artifacts",
        "test_year": int(metrics.get("test_year", config["test_year"])),
        "features": int(len(coefficients)),
        "selected_alpha": metrics.get("selected_alpha"),
        "rank_ic_mean": metrics.get("rank_ic_mean"),
        "gross_annual_return": metrics.get("gross_annual_return"),
        "net_1bps_annual_return": metrics.get("net_1bps_annual_return"),
        "net_3bps_annual_return": metrics.get("net_3bps_annual_return"),
        "report": str(report_path),
    }


def run(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    root = config_path.parents[2]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_dir = root / config["output_directory"]
    sentinel = output_dir / "metrics_2025.csv"
    if sentinel.exists():
        raise FileExistsError(
            "2025 supplemental test already completed; overwrite is forbidden. "
            "Use --from-artifacts to rebuild the markdown report from saved CSVs."
        )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("2025 supplemental output directory is not empty")

    feature_path = root / config["feature_file"]
    if not feature_path.is_file():
        raise FileNotFoundError(
            f"missing sealed feature matrix: {feature_path}. "
            "This file is gitignored and must be restored from a sealed pipeline run "
            "(features/ml_features.csv.gz). If you only need the report text, use --from-artifacts."
        )

    verify_model_freeze_manifest(root)
    output_dir.mkdir(parents=True, exist_ok=True)

    registry = pd.read_csv(root / config["registry_file"])
    # Old supplemental freeze used the 59-field features registry schema.
    # Prefer features/ml_feature_registry.csv when the configured registry has
    # a mismatched schema relative to the sealed matrix.
    feature_names = registry["feature_name"].tolist()
    features = pd.read_csv(feature_path, parse_dates=["trade_date"])
    if not set(feature_names).issubset(features.columns):
        alt_registry = root / "features" / "ml_feature_registry.csv"
        if alt_registry.is_file():
            registry = pd.read_csv(alt_registry)
            feature_names = registry["feature_name"].tolist()
    if not set(feature_names).issubset(features.columns):
        missing = sorted(set(feature_names) - set(features.columns))
        raise ValueError(f"registered features missing from matrix: {missing[:20]}")

    factor_config = load_config(root / "config" / "screening.json")
    panel = load_panel(factor_config, factor_config.raw["sealed_end"])
    labels = add_forward_labels(panel, [1])[KEYS + ["future_return_1d"]]
    data = features.merge(labels, on=KEYS, how="inner", validate="one_to_one")
    data[config["model_target_column"]] = cross_sectional_rank_target(data, config["raw_return_column"])
    data = data.dropna(subset=[config["model_target_column"], config["raw_return_column"]]).copy()

    train = data.loc[
        data["trade_date"].dt.year.between(int(config["train_start_year"]), int(config["train_end_year"]))
    ].copy()
    train = purge_last_dates(train, int(config["purge_trading_days"]))
    test = data.loc[data["trade_date"].dt.year.eq(int(config["test_year"]))].copy()
    if train.empty or test.empty:
        raise ValueError("empty train or 2025 test data")
    if (
        train["trade_date"].dt.year.min() != int(config["train_start_year"])
        or train["trade_date"].dt.year.max() != int(config["train_end_year"])
    ):
        raise ValueError("training period does not match frozen supplemental configuration")
    if set(test["trade_date"].dt.year.unique()) != {int(config["test_year"])}:
        raise ValueError("test data contains records outside the supplemental year")

    # Fill NaNs like the original supplemental Ridge path.
    train_fit = train.copy()
    test_fit = test.copy()
    train_fit[feature_names] = train_fit[feature_names].fillna(0.0)
    test_fit[feature_names] = test_fit[feature_names].fillna(0.0)

    alpha, alpha_search = _choose_alpha_legacy(train_fit, feature_names, config)
    prediction, coefficients, intercept = _fit_ridge(
        train_fit,
        test_fit,
        feature_names,
        config["model_target_column"],
        alpha,
    )
    predictions = test[
        ["trade_date", "product", "sector", config["raw_return_column"], config["model_target_column"]]
    ].copy()
    predictions["prediction"] = prediction
    predictions["test_year"] = int(config["test_year"])
    predictions["selected_alpha"] = alpha
    predictions["model_intercept"] = intercept
    strategy = build_strategy_returns(predictions, config)
    strategy["test_year"] = int(config["test_year"])
    metrics = _fold_metrics(
        "fold_2025",
        int(config["test_year"]),
        alpha,
        train_fit,
        predictions,
        strategy,
        config,
        prior_universe_size=len(feature_names),
        selected_universe_size=len(feature_names),
    )
    coef_meta_cols = [
        c for c in ["feature_name", "source_factor", "usage_type", "classification"] if c in registry.columns
    ]
    coefficients_frame = pd.DataFrame({
        "feature_name": feature_names,
        "standardized_coefficient": coefficients,
    }).merge(
        registry[coef_meta_cols],
        on="feature_name",
        how="left",
        validate="one_to_one",
    )
    monthly = _monthly_metrics(predictions, strategy, config)
    comparison = _comparison_table(root, metrics)

    predictions.to_csv(output_dir / "oos_2025_predictions.csv.gz", index=False, compression="gzip")
    strategy.to_csv(output_dir / "strategy_2025_daily_returns.csv", index=False)
    alpha_search.to_csv(output_dir / "alpha_search_2025.csv", index=False)
    coefficients_frame.to_csv(output_dir / "coefficients_2025.csv", index=False)
    monthly.to_csv(output_dir / "monthly_metrics_2025.csv", index=False)
    comparison.to_csv(output_dir / "development_comparison.csv", index=False)
    report_path = output_dir / "supplemental_2025_report.md"
    write_report(
        report_path,
        config,
        metrics,
        predictions,
        strategy,
        monthly,
        coefficients_frame,
        alpha_search,
        comparison,
        len(feature_names),
    )

    metrics_frame = pd.DataFrame([metrics])
    temporary = output_dir / ".metrics_2025.csv.tmp"
    metrics_frame.to_csv(temporary, index=False)
    temporary.replace(sentinel)
    return {
        "status": "completed_once",
        "test_year": int(config["test_year"]),
        "features": len(feature_names),
        "selected_alpha": alpha,
        "rank_ic_mean": metrics["rank_ic_mean"],
        "gross_annual_return": metrics["gross_annual_return"],
        "net_1bps_annual_return": metrics.get("net_1bps_annual_return"),
        "net_3bps_annual_return": metrics["net_3bps_annual_return"],
        "report": str(report_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ridge V1的2025补充样本外回测（禁止覆盖）")
    parser.add_argument("--config", default="modeling/linear_ridge_v1/supplemental_2025_config.json")
    parser.add_argument(
        "--from-artifacts",
        action="store_true",
        help="只根据已保存的 CSV 重写 supplemental_2025_report.md，不重跑回测",
    )
    args = parser.parse_args(argv)
    if args.from_artifacts:
        result = regenerate_report_from_artifacts(args.config)
    else:
        result = run(args.config)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
