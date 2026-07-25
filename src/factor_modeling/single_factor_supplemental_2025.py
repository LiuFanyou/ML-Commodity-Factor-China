"""2025 protocol supplemental backtest for single-factor rank v1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from factor_modeling.common import cross_sectional_rank_target
from factor_modeling.handoff import load_handoff_bundle
from factor_modeling.pack_2025 import (
    ensure_empty_output,
    load_2025_pack,
    load_development_train,
)
from factor_modeling.single_factor_rank import (
    build_leaderboard,
    evaluate_factor_fold,
    select_candidates,
)


def _format(value: Any, digits: int = 4) -> str:
    if value is None or (isinstance(value, (float, np.floating)) and np.isnan(value)):
        return "—"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def _markdown(frame: pd.DataFrame) -> str:
    return frame.to_markdown(index=False) if not frame.empty else "暂无记录。"


def write_report(
    path: Path,
    config: dict[str, Any],
    leaderboard: pd.DataFrame,
    summary: dict[str, Any],
) -> None:
    top = leaderboard.head(int(config.get("leaderboard_top_n", 20))).copy()
    display_cols = [
        c
        for c in [
            "rank",
            "feature_name",
            "usage_type",
            "rank_ic_mean",
            "rank_ic_hit_rate",
            "gross_sharpe",
            "net_5bps_sharpe",
            "gross_annual_return",
            "net_5bps_annual_return",
            "applied_direction",
            "direction_source",
        ]
        if c in top.columns
    ]
    top_display = top[display_cols].copy()
    for column in top_display.columns:
        if pd.api.types.is_float_dtype(top_display[column]):
            top_display[column] = top_display[column].map(_format)

    if "usage_type" in leaderboard.columns:
        predictors = leaderboard.loc[leaderboard["usage_type"].eq("predictor")].copy()
    else:
        predictors = leaderboard.copy()
    if predictors.empty:
        predictors = leaderboard.copy()
    mean_rank = float(predictors["rank_ic_mean"].mean()) if not predictors.empty else float("nan")
    mean_net5 = (
        float(predictors["net_5bps_sharpe"].mean())
        if "net_5bps_sharpe" in predictors.columns and not predictors.empty
        else float("nan")
    )
    positive_share = float(predictors["rank_ic_mean"].gt(0).mean()) if not predictors.empty else float("nan")

    lines = [
        "# 2025年单因子排序模型补充回测报告（协议口径）",
        "",
        "> 每个因子的方向仅用开发期训练窗冻结；2025 只做一次评价，不参与选因子。",
        "",
        "## 测试边界",
        "",
        "- 训练：`training_handoff` 2015—2024。",
        f"- 测试：`{config.get('pack_2025_directory', 'ml_2025_backtest_pack')}`。",
        f"- 字段数：{summary['features']}；评价因子数：{summary['factors_evaluated']}。",
        f"- 执行：`{config['raw_return_column']}`，持有/调仓 {config['holding_days']}/{config['rebalance_frequency_days']} 日。",
        "- 成本：0/3/5/10 bp；主口径 5 bp。",
        "",
        "## 核心发现",
        "",
        f"- predictor（或全体）平均 RankIC 为 {_format(mean_rank)}，RankIC>0 占比 {_format(positive_share)}。",
        f"- predictor（或全体）平均净 5bp Sharpe 为 {_format(mean_net5)}。",
        f"- 按 RankIC 排名第一：`{summary.get('top_feature', '—')}`（RankIC={_format(summary.get('top_rank_ic'))}）。",
        "",
        "## Top 排行榜（按 RankIC）",
        "",
        _markdown(top_display),
        "",
        "## 使用限制",
        "",
        "1. 不得用 2025 结果回头增删因子或改方向规则后再把同一年当最终成绩。",
        "2. quality_flag / `__missing` 字段可出现在全量表中，解读时需与 predictor 区分。",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(config_path: str | Path, feature_limit: int | None = None) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    root = config_path.parents[2]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_dir = root / config["output_directory"]
    report_path = root / config["report_file"]
    sentinel = ensure_empty_output(output_dir)

    bundle = load_handoff_bundle(root, config)
    candidates = select_candidates(bundle.registry, config)
    if feature_limit is not None:
        candidates = candidates.head(int(feature_limit)).copy()
    feature_names = candidates["feature_name"].tolist()

    train = load_development_train(root, config, feature_names)
    test = load_2025_pack(root, config, feature_names)
    target_column = config.get("model_target_column", "target_cs_rank_5d")
    raw_return = config["raw_return_column"]
    train[target_column] = cross_sectional_rank_target(train, raw_return)
    test[target_column] = cross_sectional_rank_target(test, raw_return)
    train = train.dropna(subset=[raw_return, target_column]).copy()
    test = test.dropna(subset=[raw_return, target_column]).copy()
    train[feature_names] = train[feature_names].fillna(0.0)
    test[feature_names] = test[feature_names].fillna(0.0)

    direction_lookup = {
        row.feature_name: int(row.expected_direction) for row in candidates.itertuples()
    }

    all_predictions: list[pd.DataFrame] = []
    all_strategy: list[pd.DataFrame] = []
    all_fold_metrics: list[dict[str, Any]] = []
    all_daily_ic: list[pd.DataFrame] = []

    for idx, feature_name in enumerate(feature_names, start=1):
        predicted, strategy, fold_metric, daily_ic = evaluate_factor_fold(
            train,
            test,
            feature_name,
            direction_lookup[feature_name],
            "fold_2025",
            int(config.get("test_year", 2025)),
            config,
        )
        all_predictions.append(predicted)
        if not strategy.empty:
            all_strategy.append(strategy)
        all_fold_metrics.append(fold_metric)
        if not daily_ic.empty:
            all_daily_ic.append(daily_ic)
        if idx == 1 or idx % 20 == 0 or idx == len(feature_names):
            print(
                f"[{idx}/{len(feature_names)}] {feature_name}: "
                f"rank_ic={fold_metric.get('rank_ic_mean'):.4f} "
                f"net_5bps_sharpe={fold_metric.get('net_5bps_sharpe'):.4f}"
            )

    predictions = pd.concat(all_predictions, ignore_index=True)
    strategy = pd.concat(all_strategy, ignore_index=True) if all_strategy else pd.DataFrame()
    fold_metrics = pd.DataFrame(all_fold_metrics)
    daily_ic = pd.concat(all_daily_ic, ignore_index=True) if all_daily_ic else pd.DataFrame()

    factor_rows: list[dict[str, Any]] = []
    for feature_name, fold_part in fold_metrics.groupby("feature_name", sort=True):
        fold_row = fold_part.iloc[0]
        meta = candidates.loc[candidates["feature_name"].eq(feature_name)].iloc[0]
        row = {
            "feature_name": feature_name,
            "source_factor": meta.get("source_factor", feature_name),
            "source_factor_id": meta.get("source_factor_id", meta.get("factor_no", np.nan)),
            "usage_type": meta.get("usage_type", ""),
            "category": meta.get("category", ""),
            "catalog_direction": int(fold_row["catalog_direction"]),
            "applied_direction": int(fold_row["applied_direction"]),
            "direction_source": str(fold_row["direction_source"]),
            "direction_source_modes": str(fold_row["direction_source"]),
            "applied_direction_modes": str(int(fold_row["applied_direction"])),
            "years_evaluated": 1,
            "years_rank_ic_positive": int(float(fold_row["rank_ic_mean"]) > 0)
            if pd.notna(fold_row["rank_ic_mean"])
            else 0,
            "years_gross_sharpe_positive": int(float(fold_row["gross_sharpe"]) > 0)
            if pd.notna(fold_row.get("gross_sharpe", np.nan))
            else 0,
        }
        for key, value in fold_row.items():
            if key not in row:
                row[key] = value
        factor_rows.append(row)

    factor_metrics = pd.DataFrame(factor_rows)
    leaderboard = build_leaderboard(factor_metrics)

    predictions.to_csv(output_dir / "oos_2025_predictions.csv.gz", index=False, compression="gzip")
    if not strategy.empty:
        strategy.to_csv(output_dir / "strategy_2025_daily_returns.csv", index=False)
    fold_metrics.to_csv(output_dir / "fold_metrics_2025.csv", index=False)
    factor_metrics.to_csv(output_dir / "factor_metrics_2025.csv", index=False)
    leaderboard.to_csv(output_dir / "leaderboard_2025.csv", index=False)
    if not daily_ic.empty:
        daily_ic.to_csv(output_dir / "daily_prediction_ic_2025.csv", index=False)

    top_feature = str(leaderboard.iloc[0]["feature_name"]) if not leaderboard.empty else None
    top_rank_ic = float(leaderboard.iloc[0]["rank_ic_mean"]) if not leaderboard.empty else np.nan
    summary = {
        "status": "completed_once",
        "model_version": config.get("model_version"),
        "test_year": int(config.get("test_year", 2025)),
        "features": len(feature_names),
        "factors_evaluated": int(len(factor_metrics)),
        "top_feature": top_feature,
        "top_rank_ic": top_rank_ic,
        "mean_rank_ic": float(factor_metrics["rank_ic_mean"].mean()) if not factor_metrics.empty else np.nan,
        "mean_net_5bps_sharpe": float(factor_metrics["net_5bps_sharpe"].mean())
        if not factor_metrics.empty
        else np.nan,
        "report": str(report_path),
    }
    write_report(path=report_path, config=config, leaderboard=leaderboard, summary=summary)

    sentinel_row = {
        "test_year": int(config.get("test_year", 2025)),
        "factors_evaluated": int(len(factor_metrics)),
        "mean_rank_ic": summary["mean_rank_ic"],
        "mean_gross_sharpe": float(factor_metrics["gross_sharpe"].mean()) if not factor_metrics.empty else np.nan,
        "mean_net_5bps_sharpe": summary["mean_net_5bps_sharpe"],
        "top_feature": top_feature,
        "top_rank_ic": top_rank_ic,
    }
    temporary = output_dir / ".metrics_2025.csv.tmp"
    pd.DataFrame([sentinel_row]).to_csv(temporary, index=False)
    temporary.replace(sentinel)

    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="单因子排序 v1 的 2025 协议补充回测")
    parser.add_argument(
        "--config",
        default="modeling/single_factor_rank_v1/supplemental_2025_protocol_config.json",
    )
    parser.add_argument("--feature-limit", type=int, default=None)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            run(args.config, feature_limit=args.feature_limit),
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
