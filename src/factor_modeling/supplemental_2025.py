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
    purge_last_dates,
    return_statistics,
)
from .linear_ridge import (
    _fit_ridge,
    _fold_metrics,
    choose_alpha,
)
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
        "单边1bp净收益年化": float(development["net_1bps_annual_return"].mean()),
        "单边3bp净收益年化": float(development["net_3bps_annual_return"].mean()),
    }
    supplemental_row = {
        "阶段": "2025补充样本外",
        "平均RankIC": metrics_2025["rank_ic_mean"],
        "RankIC胜率": metrics_2025["rank_ic_hit_rate"],
        "毛收益年化": metrics_2025["gross_annual_return"],
        "毛收益Sharpe": metrics_2025["gross_sharpe"],
        "单边1bp净收益年化": metrics_2025["net_1bps_annual_return"],
        "单边3bp净收益年化": metrics_2025["net_3bps_annual_return"],
    }
    return pd.DataFrame([development_row, supplemental_row])


def _conclusion(metrics: dict[str, Any]) -> tuple[str, list[str]]:
    rank_positive = float(metrics["rank_ic_mean"]) > 0
    net_1_positive = float(metrics["net_1bps_annual_return"]) > 0
    net_3_positive = float(metrics["net_3bps_annual_return"]) > 0
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
        f"单边1bp净年化收益为{metrics['net_1bps_annual_return']:.2%}。",
        f"单边3bp净年化收益为{metrics['net_3bps_annual_return']:.2%}，{'通过' if net_3_positive else '未通过'}中等成本情景。",
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
        alpha_display[column] = alpha_display[column].map(_format)
    positive = coefficients.nlargest(10, "standardized_coefficient").copy()
    negative = coefficients.nsmallest(10, "standardized_coefficient").copy()
    for frame in [positive, negative]:
        frame["standardized_coefficient"] = frame["standardized_coefficient"].map(_format)

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
        _markdown(positive[["feature_name", "source_factor", "usage_type", "standardized_coefficient"]]),
        "",
        "## 标准化系数最大的负向字段",
        "",
        _markdown(negative[["feature_name", "source_factor", "usage_type", "standardized_coefficient"]]),
        "",
        "## 研究结论与使用限制",
        "",
        "1. RankIC衡量横截面排序能力；即使为正，也必须同时检查成本后收益。",
        "2. 本回测未核验合约乘数、品种真实手续费、盘口冲击、涨跌停成交和容量，不能直接解释为可实盘收益。",
        "3. 2025结果不得用于反向修改本版本特征、标签、alpha网格或持仓规则；若修改，应登记为新模型版本，并等待新数据验证。",
        "4. 线性系数受特征相关性影响，只表示冻结模型中的条件关联，不是因果结论。",
        "",
        f"自动核验：预测记录{len(predictions)}条、策略日{len(strategy)}天，2025之外记录为0。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    root = config_path.parents[2]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_dir = root / config["output_directory"]
    sentinel = output_dir / "metrics_2025.csv"
    if sentinel.exists():
        raise FileExistsError("2025 supplemental test already completed; overwrite is forbidden")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("2025 supplemental output directory is not empty")
    verify_model_freeze_manifest(root)
    output_dir.mkdir(parents=True, exist_ok=True)

    registry = pd.read_csv(root / config["registry_file"])
    feature_names = registry["feature_name"].tolist()
    features = pd.read_csv(root / config["feature_file"], parse_dates=["trade_date"])
    if not set(feature_names).issubset(features.columns):
        missing = sorted(set(feature_names) - set(features.columns))
        raise ValueError(f"registered features missing from matrix: {missing}")

    factor_config = load_config(root / "config" / "screening.json")
    panel = load_panel(factor_config, factor_config.raw["sealed_end"])
    labels = add_forward_labels(panel, [1])[KEYS + ["future_return_1d"]]
    data = features.merge(labels, on=KEYS, how="inner", validate="one_to_one")
    data[config["model_target_column"]] = cross_sectional_rank_target(data, config["raw_return_column"])
    data = data.dropna(subset=[config["model_target_column"], config["raw_return_column"]]).copy()

    train = data.loc[data["trade_date"].dt.year.between(int(config["train_start_year"]), int(config["train_end_year"]))].copy()
    train = purge_last_dates(train, int(config["purge_trading_days"]))
    test = data.loc[data["trade_date"].dt.year.eq(int(config["test_year"]))].copy()
    if train.empty or test.empty:
        raise ValueError("empty train or 2025 test data")
    if train["trade_date"].dt.year.min() != int(config["train_start_year"]) or train["trade_date"].dt.year.max() != int(config["train_end_year"]):
        raise ValueError("training period does not match frozen supplemental configuration")
    if set(test["trade_date"].dt.year.unique()) != {int(config["test_year"])}:
        raise ValueError("test data contains records outside the supplemental year")

    alpha, alpha_search = choose_alpha(train, feature_names, config)
    prediction, coefficients, intercept = _fit_ridge(
        train,
        test,
        feature_names,
        config["model_target_column"],
        alpha,
    )
    predictions = test[["trade_date", "product", "sector", config["raw_return_column"], config["model_target_column"]]].copy()
    predictions = predictions.rename(columns={config["model_target_column"]: "target_cs_rank_1d"})
    predictions["prediction"] = prediction
    predictions["test_year"] = int(config["test_year"])
    predictions["selected_alpha"] = alpha
    predictions["model_intercept"] = intercept
    strategy = build_strategy_returns(predictions, config)
    strategy["test_year"] = int(config["test_year"])
    metrics = _fold_metrics(int(config["test_year"]), alpha, train, predictions, strategy, config)
    coefficients_frame = pd.DataFrame({
        "feature_name": feature_names,
        "standardized_coefficient": coefficients,
    }).merge(
        registry[["feature_name", "source_factor", "usage_type", "classification"]],
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
        "net_1bps_annual_return": metrics["net_1bps_annual_return"],
        "net_3bps_annual_return": metrics["net_3bps_annual_return"],
        "report": str(report_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ridge V1的2025补充样本外回测（禁止覆盖）")
    parser.add_argument("--config", default="modeling/linear_ridge_v1/supplemental_2025_config.json")
    args = parser.parse_args(argv)
    print(json.dumps(run(args.config), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
