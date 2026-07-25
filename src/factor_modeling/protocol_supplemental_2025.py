"""Protocol-aligned 2025 supplemental test for the current handoff Ridge (5d).

Train on ``training_handoff`` (2015–2024 only). Evaluate once on
``ml_2025_backtest_pack`` under the unified execution protocol.

Do not use 2025 to re-select features or retune hyperparameters.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from factor_modeling.common import (
    KEYS,
    build_strategy_returns,
    cross_sectional_rank_target,
    return_statistics,
)
from factor_modeling.handoff import (
    apply_label_end_purge,
    handoff_summary,
    load_handoff_bundle,
    merge_features_and_labels,
)
from factor_modeling.linear_ridge import _fit_ridge, _fold_metrics, choose_alpha
from factor_modeling.single_factor_rank import select_candidates


def _format(value: Any, digits: int = 4) -> str:
    if value is None or (isinstance(value, (float, np.floating)) and np.isnan(value)):
        return "—"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def _markdown(frame: pd.DataFrame) -> str:
    return frame.to_markdown(index=False) if not frame.empty else "暂无记录。"


def _cost_return_table(strategy: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
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
    alpha_search: pd.DataFrame,
    coefficients: pd.DataFrame,
    monthly: pd.DataFrame,
    n_features: int,
    pack_dir: str,
) -> None:
    grade, findings = _conclusion(metrics)
    metric_table = pd.DataFrame([
        {"指标": "训练数据", "结果": "training_handoff（2015—2024）"},
        {"指标": "测试数据包", "结果": pack_dir},
        {"指标": "训练区间", "结果": f"{metrics['train_start']} 至 {metrics['train_end']}"},
        {"指标": "2025测试记录", "结果": metrics["test_rows"]},
        {"指标": "2025测试交易日", "结果": metrics["test_days"]},
        {"指标": "调仓日", "结果": metrics.get("rebalance_days", "—")},
        {"指标": "特征数", "结果": n_features},
        {"指标": "选定Ridge alpha", "结果": metrics["selected_alpha"]},
        {"指标": "平均RankIC", "结果": metrics["rank_ic_mean"]},
        {"指标": "RankIC胜率", "结果": metrics["rank_ic_hit_rate"]},
        {"指标": "毛Sharpe", "结果": metrics["gross_sharpe"]},
        {"指标": "净5bp Sharpe", "结果": metrics.get("net_5bps_sharpe", np.nan)},
    ])
    metric_table["结果"] = metric_table["结果"].map(_format)
    returns = _cost_return_table(strategy, config)
    for column in returns.columns[1:]:
        returns[column] = returns[column].map(_format)
    alpha_display = alpha_search.copy()
    for column in alpha_display.columns:
        if pd.api.types.is_float_dtype(alpha_display[column]):
            alpha_display[column] = alpha_display[column].map(_format)
    monthly_display = monthly.copy()
    for column in monthly_display.columns:
        if pd.api.types.is_float_dtype(monthly_display[column]):
            monthly_display[column] = monthly_display[column].map(_format)
    top = coefficients.nlargest(min(10, len(coefficients)), "standardized_coefficient").copy()
    bottom = coefficients.nsmallest(min(10, len(coefficients)), "standardized_coefficient").copy()
    for frame in [top, bottom]:
        frame["standardized_coefficient"] = frame["standardized_coefficient"].map(_format)

    lines = [
        "# 2025年Ridge线性模型补充回测报告（协议口径）",
        "",
        f"> 结论：**{grade}**。训练只用开发交接包；测试只用 `ml_2025_backtest_pack`。这是补充样本外，不是可反复窥视的调参窗口。",
        "",
        "## 测试边界",
        "",
        f"- 训练：`training_handoff`，{config['train_start_year']}—{config['train_end_year']}；alpha 仅在训练窗内选择。",
        f"- 测试：`{pack_dir}` 的 2025 特征与评价标签。",
        f"- 标签 / 执行：`{config['raw_return_column']}`，持有/调仓 {config['holding_days']}/{config['rebalance_frequency_days']} 日，多空各 20%。",
        f"- 输入：注册表全部 {n_features} 个字段；不按 2025 结果增删特征或改超参。",
        "- 成本：单边 0/3/5/10 bp；协议主口径 5 bp。",
        "- 可成交过滤：剔除 `future_return_5d` 缺失，以及入场/退出执行标记不允许的样本。",
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
        "## 训练窗内部 alpha 选择",
        "",
        _markdown(alpha_display),
        "",
        "## 标准化系数最大的正向字段",
        "",
        _markdown(top[["feature_name", "standardized_coefficient"]]),
        "",
        "## 标准化系数最大的负向字段",
        "",
        _markdown(bottom[["feature_name", "standardized_coefficient"]]),
        "",
        "## 使用限制",
        "",
        "1. 不得用本次 2025 结果回头修改特征清单、筛选阈值、alpha 网格或持仓规则。",
        "2. 若修改模型，应登记新版本，并用更新年份重新做封存/补充测试。",
        "3. 本结果未核验真实手续费、冲击成本与涨跌停成交约束。",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _apply_execution_filters(panel: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    raw_return = config["raw_return_column"]
    usable = panel[raw_return].notna()
    if not bool(config.get("apply_execution_filters", True)):
        return panel.loc[usable].copy()
    if "entry_execution_available" in panel.columns:
        usable &= panel["entry_execution_available"].fillna(0).astype(int).eq(1)
    if "exit_execution_available_5d" in panel.columns:
        usable &= panel["exit_execution_available_5d"].fillna(0).astype(int).eq(1)
    return panel.loc[usable].copy()


def _load_2025_pack(root: Path, config: dict[str, Any], feature_names: list[str]) -> pd.DataFrame:
    pack_dir = root / config.get("pack_2025_directory", "ml_2025_backtest_pack")
    feature_path = pack_dir / config.get("sealed_feature_file", "ml_features_2025.csv.gz")
    label_path = pack_dir / config.get("sealed_label_file", "labels_2025_evaluator_only.csv.gz")
    registry_path = pack_dir / config.get("pack_registry_file", "ml_feature_registry.csv")
    for path in (feature_path, label_path, registry_path):
        if not path.is_file():
            raise FileNotFoundError(f"2025 backtest pack file missing: {path}")

    pack_registry = pd.read_csv(registry_path)
    pack_names = pack_registry["feature_name"].tolist()
    if pack_names != feature_names:
        missing = sorted(set(feature_names) - set(pack_names))
        extra = sorted(set(pack_names) - set(feature_names))
        raise ValueError(
            "2025 pack registry does not match development handoff registry; "
            f"missing={missing[:10]} extra={extra[:10]}"
        )

    features = pd.read_csv(feature_path, parse_dates=["trade_date"])
    labels = pd.read_csv(label_path, parse_dates=["trade_date"])
    for column in ["entry_date", config.get("label_end_column", "label_end_date_5d")]:
        if column in labels.columns:
            labels[column] = pd.to_datetime(labels[column], errors="coerce")

    missing_features = sorted(set(feature_names) - set(features.columns))
    if missing_features:
        raise ValueError(f"2025 feature matrix missing columns: {missing_features[:20]}")

    raw_return = config["raw_return_column"]
    if raw_return not in labels.columns:
        raise ValueError(f"2025 label matrix missing {raw_return}")

    keep_feat = [c for c in KEYS + ["sector"] + feature_names if c in features.columns]
    keep_lab = [
        c
        for c in KEYS
        + [
            raw_return,
            config.get("label_end_column", "label_end_date_5d"),
            "entry_execution_available",
            "exit_execution_available_5d",
            "entry_suspected_limit_lock",
            "exit_suspected_limit_lock_5d",
        ]
        if c in labels.columns
    ]
    panel = features[keep_feat].merge(labels[keep_lab], on=KEYS, how="inner", validate="one_to_one")
    if set(panel["trade_date"].dt.year.unique()) != {int(config["test_year"])}:
        years = sorted(panel["trade_date"].dt.year.unique().tolist())
        raise ValueError(f"2025 pack contains unexpected years: {years}")
    return _apply_execution_filters(panel, config)


def _load_train_panel(root: Path, config: dict[str, Any], feature_names: list[str]) -> pd.DataFrame:
    bundle = load_handoff_bundle(root, config)
    _ = handoff_summary(bundle)
    raw_return = config["raw_return_column"]
    data = merge_features_and_labels(bundle, raw_return_column=raw_return)
    keep = [c for c in KEYS + ["sector"] + feature_names if c in data.columns]
    label_meta = [
        c
        for c in data.columns
        if c == raw_return or c == config.get("label_end_column", "label_end_date_5d")
    ]
    data = data[keep + [c for c in label_meta if c not in keep]].copy()
    data = data.loc[
        data["trade_date"].dt.year.between(int(config["train_start_year"]), int(config["train_end_year"]))
    ].copy()
    valid_start = pd.Timestamp(f"{int(config['test_year'])}-01-01")
    data = apply_label_end_purge(
        data,
        valid_start,
        label_end_column=config.get("label_end_column", "label_end_date_5d"),
        purge_trading_days=int(config.get("purge_trading_days", 5)),
    )
    return data


def run(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    root = config_path.parents[2]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_dir = root / config["output_directory"]
    report_path = root / config["report_file"]
    sentinel = output_dir / "metrics_2025.csv"
    if sentinel.exists():
        raise FileExistsError("protocol 2025 supplemental already completed; overwrite is forbidden")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory not empty: {output_dir}")

    bundle = load_handoff_bundle(root, config)
    candidates = select_candidates(bundle.registry, config)
    feature_names = candidates["feature_name"].tolist()

    train = _load_train_panel(root, config, feature_names)
    test = _load_2025_pack(root, config, feature_names)
    if train.empty or test.empty:
        raise ValueError("empty train or 2025 test after filters")

    target_column = config["model_target_column"]
    raw_return = config["raw_return_column"]
    train[target_column] = cross_sectional_rank_target(train, raw_return)
    test[target_column] = cross_sectional_rank_target(test, raw_return)
    train = train.dropna(subset=[target_column, raw_return]).copy()
    test = test.dropna(subset=[target_column, raw_return]).copy()

    train_fit = train.copy()
    test_fit = test.copy()
    train_fit[feature_names] = train_fit[feature_names].fillna(0.0)
    test_fit[feature_names] = test_fit[feature_names].fillna(0.0)

    alpha, alpha_search = choose_alpha(train_fit, feature_names, config)
    prediction, coefficients, intercept = _fit_ridge(
        train_fit,
        test_fit,
        feature_names,
        target_column,
        alpha,
    )
    predictions = test[["trade_date", "product", "sector", raw_return, target_column]].copy()
    predictions["prediction"] = prediction
    predictions["selected_alpha"] = alpha
    predictions["model_intercept"] = intercept
    predictions["test_year"] = int(config["test_year"])

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
    coefficients_frame = pd.DataFrame({
        "feature_name": feature_names,
        "standardized_coefficient": coefficients,
    })
    monthly = _monthly_table(strategy, config)
    pack_dir = str(config.get("pack_2025_directory", "ml_2025_backtest_pack"))

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / "oos_2025_predictions.csv.gz", index=False, compression="gzip")
    strategy.to_csv(output_dir / "strategy_2025_daily_returns.csv", index=False)
    alpha_search.to_csv(output_dir / "alpha_search_2025.csv", index=False)
    coefficients_frame.to_csv(output_dir / "coefficients_2025.csv", index=False)
    monthly.to_csv(output_dir / "monthly_metrics_2025.csv", index=False)
    write_report(
        report_path,
        config,
        metrics,
        strategy,
        alpha_search,
        coefficients_frame,
        monthly,
        len(feature_names),
        pack_dir,
    )
    temporary = output_dir / ".metrics_2025.csv.tmp"
    pd.DataFrame([metrics]).to_csv(temporary, index=False)
    temporary.replace(sentinel)

    summary = {
        "status": "completed_once",
        "test_year": int(config["test_year"]),
        "features": len(feature_names),
        "pack_2025_directory": pack_dir,
        "selected_alpha": alpha,
        "rank_ic_mean": metrics["rank_ic_mean"],
        "gross_annual_return": metrics["gross_annual_return"],
        "gross_sharpe": metrics["gross_sharpe"],
        "net_3bps_annual_return": metrics.get("net_3bps_annual_return"),
        "net_5bps_annual_return": metrics.get("net_5bps_annual_return"),
        "net_5bps_sharpe": metrics.get("net_5bps_sharpe"),
        "report": str(report_path),
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="当前协议口径 Ridge 的2025补充样本外回测")
    parser.add_argument(
        "--config",
        default="modeling/linear_ridge_v1/supplemental_2025_protocol_config.json",
    )
    args = parser.parse_args(argv)
    print(json.dumps(run(args.config), ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
