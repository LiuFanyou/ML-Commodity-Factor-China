"""2025 protocol supplemental backtest for equal-weight multi-factor v1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from factor_modeling.common import cross_sectional_rank_target, return_statistics
from factor_modeling.equal_weight_multi import evaluate_fold
from factor_modeling.handoff import load_handoff_bundle
from factor_modeling.pack_2025 import (
    ensure_empty_output,
    load_2025_pack,
    load_development_train,
)
from factor_modeling.single_factor_rank import select_candidates


def _format(value: Any, digits: int = 4) -> str:
    if value is None or (isinstance(value, (float, np.floating)) and np.isnan(value)):
        return "—"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def _markdown(frame: pd.DataFrame) -> str:
    return frame.to_markdown(index=False) if not frame.empty else "暂无记录。"


def _cost_table(strategy: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    rows = [{"成本情景": "毛收益", **return_statistics(strategy["gross_return"], float(config["annualization"]))}]
    for bps in config["one_way_cost_bps"]:
        bps = float(bps)
        label = str(int(bps)) if bps.is_integer() else str(bps).replace(".", "p")
        rows.append({
            "成本情景": f"单边{bps:g}bp",
            **return_statistics(strategy[f"net_{label}bps_return"], float(config["annualization"])),
        })
    return pd.DataFrame(rows)


def _monthly_table(strategy: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    work = strategy.copy()
    work["month"] = pd.to_datetime(work["trade_date"]).dt.to_period("M").astype(str)
    columns = ["gross_return"]
    for bps in config["one_way_cost_bps"]:
        bps = float(bps)
        label = str(int(bps)) if bps.is_integer() else str(bps).replace(".", "p")
        columns.append(f"net_{label}bps_return")
    rows: list[dict[str, Any]] = []
    for month, part in work.groupby("month", sort=True):
        row: dict[str, Any] = {"month": month, "rebalance_days": int(len(part))}
        for column in columns:
            row[f"{column}_total"] = float((1.0 + part[column]).prod() - 1.0)
            row[f"{column}_hit_rate"] = float(part[column].gt(0).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def _conclusion(metrics: dict[str, Any]) -> tuple[str, list[str]]:
    rank_positive = float(metrics["rank_ic_mean"]) > 0
    net5 = float(metrics.get("net_5bps_annual_return", np.nan))
    net3 = float(metrics.get("net_3bps_annual_return", np.nan))
    if rank_positive and np.isfinite(net5) and net5 > 0:
        grade = "补充样本外证据较强（协议主口径5bp仍为正）"
    elif rank_positive and np.isfinite(net3) and net3 > 0:
        grade = "存在弱交易证据，但主口径5bp未过"
    elif rank_positive:
        grade = "预测方向为正，但尚不具备成本后交易证据"
    else:
        grade = "2025未提供正向预测证据"
    findings = [
        f"2025平均RankIC为{metrics['rank_ic_mean']:.4f}。",
        f"毛收益年化为{metrics['gross_annual_return']:.2%}，Sharpe为{metrics['gross_sharpe']:.3f}。",
        f"单边3bp净年化为{net3:.2%}；单边5bp净年化为{net5:.2%}。",
    ]
    return grade, findings


def write_report(
    path: Path,
    config: dict[str, Any],
    metrics: dict[str, Any],
    strategy: pd.DataFrame,
    directions: pd.DataFrame,
    monthly: pd.DataFrame,
    n_features: int,
) -> None:
    grade, findings = _conclusion(metrics)
    metric_table = pd.DataFrame([
        {"指标": "训练数据", "结果": "training_handoff（2015—2024）"},
        {"指标": "测试数据包", "结果": config.get("pack_2025_directory", "ml_2025_backtest_pack")},
        {"指标": "训练区间", "结果": f"{metrics['train_start']} 至 {metrics['train_end']}"},
        {"指标": "2025测试记录", "结果": metrics["test_rows"]},
        {"指标": "调仓日", "结果": metrics.get("strategy_days", metrics.get("rebalance_days", "—"))},
        {"指标": "入选因子数", "结果": metrics.get("universe_size", n_features)},
        {"指标": "平均RankIC", "结果": metrics["rank_ic_mean"]},
        {"指标": "RankIC胜率", "结果": metrics["rank_ic_hit_rate"]},
        {"指标": "毛Sharpe", "结果": metrics["gross_sharpe"]},
        {"指标": "净5bp Sharpe", "结果": metrics.get("net_5bps_sharpe", np.nan)},
    ])
    metric_table["结果"] = metric_table["结果"].map(_format)
    returns = _cost_table(strategy, config)
    for column in returns.columns[1:]:
        returns[column] = returns[column].map(_format)
    monthly_display = monthly.copy()
    for column in monthly_display.columns:
        if pd.api.types.is_float_dtype(monthly_display[column]):
            monthly_display[column] = monthly_display[column].map(_format)
    direction_display = directions[
        [c for c in ["feature_name", "catalog_direction", "resolved_direction", "direction_source", "train_rank_ic"] if c in directions.columns]
    ].copy()
    if "train_rank_ic" in direction_display.columns:
        direction_display["train_rank_ic"] = direction_display["train_rank_ic"].map(_format)

    lines = [
        "# 2025年等权多因子模型补充回测报告（协议口径）",
        "",
        f"> 结论：**{grade}**。方向仅用开发期训练窗冻结；测试只用 `ml_2025_backtest_pack`。",
        "",
        "## 测试边界",
        "",
        "- 训练：`training_handoff` 2015—2024；不在 2025 上重新选方向/筛因子。",
        f"- 测试：`{config.get('pack_2025_directory', 'ml_2025_backtest_pack')}`。",
        f"- 合成：direction-align → `{config.get('score_transform', 'cs_rank_centered')}` → 等权平均（`{config.get('missing_policy', 'mean_available')}`）。",
        f"- 执行：`{config['raw_return_column']}`，持有/调仓 {config['holding_days']}/{config['rebalance_frequency_days']} 日，多空各 20%。",
        "- 成本：0/3/5/10 bp；主口径 5 bp。",
        "",
        "## 核心发现",
        "",
        *[f"- {item}" for item in findings],
        "",
        "## 预测与回测指标",
        "",
        _markdown(metric_table),
        "",
        "## 成本敏感性",
        "",
        _markdown(returns),
        "",
        "## 2025分月表现",
        "",
        _markdown(monthly_display),
        "",
        "## 训练窗冻结方向（摘要）",
        "",
        _markdown(direction_display.head(30)),
        "",
        "## 使用限制",
        "",
        "1. 不得用本次 2025 结果回头修改方向规则、特征清单或持仓规则。",
        "2. 若修改模型，应登记新版本并用更新年份验证。",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    root = config_path.parents[2]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("combination_scheme", "all_equal") != "all_equal":
        raise ValueError("equal-weight supplemental requires combination_scheme=all_equal")

    output_dir = root / config["output_directory"]
    report_path = root / config["report_file"]
    sentinel = ensure_empty_output(output_dir)

    bundle = load_handoff_bundle(root, config)
    candidates = select_candidates(bundle.registry, config)
    feature_names = candidates["feature_name"].tolist()

    train = load_development_train(root, config, feature_names)
    test = load_2025_pack(root, config, feature_names)
    target_column = config.get("model_target_column", "target_cs_rank_5d")
    raw_return = config["raw_return_column"]
    train[target_column] = cross_sectional_rank_target(train, raw_return)
    test[target_column] = cross_sectional_rank_target(test, raw_return)
    train = train.dropna(subset=[raw_return, target_column]).copy()
    test = test.dropna(subset=[raw_return, target_column]).copy()

    predicted, strategy, metrics, daily_ic, directions, selection_audit = evaluate_fold(
        train,
        test,
        candidates,
        feature_names,
        "fold_2025",
        int(config.get("test_year", 2025)),
        config,
    )
    monthly = _monthly_table(strategy, config) if not strategy.empty else pd.DataFrame()

    predicted.to_csv(output_dir / "oos_2025_predictions.csv.gz", index=False, compression="gzip")
    strategy.to_csv(output_dir / "strategy_2025_daily_returns.csv", index=False)
    directions.to_csv(output_dir / "directions_2025.csv", index=False)
    selection_audit.to_csv(output_dir / "selection_audit_2025.csv", index=False)
    if not daily_ic.empty:
        daily_ic.to_csv(output_dir / "daily_prediction_ic_2025.csv", index=False)
    monthly.to_csv(output_dir / "monthly_metrics_2025.csv", index=False)
    write_report(report_path, config, metrics, strategy, directions, monthly, len(feature_names))

    temporary = output_dir / ".metrics_2025.csv.tmp"
    pd.DataFrame([metrics]).to_csv(temporary, index=False)
    temporary.replace(sentinel)

    summary = {
        "status": "completed_once",
        "model_version": config.get("model_version"),
        "test_year": int(config.get("test_year", 2025)),
        "features": len(feature_names),
        "universe_size": metrics.get("universe_size"),
        "rank_ic_mean": metrics.get("rank_ic_mean"),
        "gross_sharpe": metrics.get("gross_sharpe"),
        "net_5bps_sharpe": metrics.get("net_5bps_sharpe"),
        "gross_annual_return": metrics.get("gross_annual_return"),
        "net_5bps_annual_return": metrics.get("net_5bps_annual_return"),
        "report": str(report_path),
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="等权多因子 v1 的 2025 协议补充回测")
    parser.add_argument(
        "--config",
        default="modeling/equal_weight_multi_v1/supplemental_2025_protocol_config.json",
    )
    args = parser.parse_args(argv)
    print(json.dumps(run(args.config), ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
