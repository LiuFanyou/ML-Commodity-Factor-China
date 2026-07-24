"""Single-factor cross-sectional ranking on the training-handoff interface.

Aligned with ``training_handoff/training_contract.json`` and the unified protocol:

- Features / registry / labels / folds loaded via ``factor_modeling.handoff``.
- Primary label: ``future_return_5d`` (open→open on mapped real contract).
- Expanding windows from ``fold_definitions.csv`` (2019–2024 OOS).
- Train purge: ``label_end_date_5d < valid_start``.
- Execution: T close signal, T+1 open trade, 5-day hold / 5-day rebalance.
- Cost: base 5bp one-way; sensitivity 0 / 3 / 5 / 10 bp.
- No multi-factor weights or regression coefficients.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from factor_modeling.common import (
    KEYS,
    build_strategy_returns,
    cross_sectional_rank_target,
    return_statistics,
)
from factor_modeling.handoff import (
    handoff_summary,
    load_fold_definitions,
    load_handoff_bundle,
    split_fold,
)
from factor_screening.catalog import FACTOR_CATALOG

__all__ = [
    "apply_score_transform",
    "attach_expected_direction",
    "build_signed_prediction",
    "choose_unknown_direction",
    "cross_sectional_centered_rank",
    "main",
    "mean_daily_rank_ic",
    "run",
    "select_candidates",
]


def cross_sectional_centered_rank(values: pd.Series) -> pd.Series:
    """Map same-date valid values to centered ranks in [-1, 1]."""
    rank = values.rank(method="average")
    count = values.notna().sum()
    if count <= 1:
        return pd.Series(np.nan, index=values.index, dtype=float)
    return 2.0 * (rank - (count + 1.0) / 2.0) / (count - 1.0)


def apply_score_transform(
    frame: pd.DataFrame,
    signal_column: str,
    score_transform: str,
) -> pd.Series:
    if score_transform == "raw_signed":
        return frame[signal_column].astype(float)
    if score_transform != "cs_rank_centered":
        raise ValueError(f"unsupported score_transform: {score_transform}")
    # Prefer vectorized group ranks: more stable across pandas versions than
    # transform(callable) which can fail when concatenating Series results.
    grouped = frame.groupby("trade_date", sort=False)[signal_column]
    rank = grouped.rank(method="average")
    count = grouped.transform("count")
    denominator = (count - 1).replace(0, np.nan)
    return 2.0 * (rank - (count + 1.0) / 2.0) / denominator


def catalog_direction_map() -> dict[int, int]:
    return {int(spec.factor_id): int(spec.expected_direction) for spec in FACTOR_CATALOG}


def attach_expected_direction(registry: pd.DataFrame) -> pd.DataFrame:
    """Attach catalog expected_direction using factor_no / source_factor_id."""
    directions = catalog_direction_map()
    out = registry.copy()
    if "source_factor_id" not in out.columns:
        if "factor_no" in out.columns:
            out["source_factor_id"] = out["factor_no"]
        else:
            out["source_factor_id"] = 0
    out["expected_direction"] = out["source_factor_id"].map(
        lambda value: directions.get(int(value), 0) if pd.notna(value) else 0
    ).astype(int)
    return out


def select_candidates(registry: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Select ranking candidates from the handoff registry.

    Default (aligned with Ridge): use **every** ``feature_name`` in
    ``ml_feature_registry.csv`` when ``use_all_registered_features`` is true
    or ``candidate_usage_types`` is null/empty/contains ``*`` / ``all``.
    """
    use_all = bool(config.get("use_all_registered_features", False))
    usage_types = config.get("candidate_usage_types", None)
    if usage_types is None or usage_types == [] or usage_types == ["*"] or usage_types == ["all"]:
        use_all = True

    if use_all:
        selected = registry.copy()
    else:
        selected = registry.loc[registry["usage_type"].isin(set(usage_types))].copy()

    explicit = config.get("candidate_feature_names")
    if explicit:
        selected = selected.loc[selected["feature_name"].isin(explicit)].copy()
    if bool(config.get("recommended_default_only", False)) and "recommended_default" in selected.columns:
        selected = selected.loc[selected["recommended_default"].astype(bool)].copy()
    if selected.empty:
        raise ValueError("no candidate features selected by config")
    return attach_expected_direction(selected).reset_index(drop=True)


def _daily_ic_frame(
    frame: pd.DataFrame,
    prediction_column: str,
    return_column: str,
    min_samples: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for date, part in frame.groupby("trade_date", sort=True):
        valid = part[[prediction_column, return_column]].dropna()
        if (
            len(valid) < min_samples
            or valid[prediction_column].nunique() < 2
            or valid[return_column].nunique() < 2
        ):
            continue
        rows.append({
            "trade_date": date,
            "pearson_ic": valid[prediction_column].corr(valid[return_column], method="pearson"),
            "rank_ic": valid[prediction_column].corr(valid[return_column], method="spearman"),
            "n_samples": int(len(valid)),
        })
    return pd.DataFrame(rows)


def mean_daily_rank_ic(
    frame: pd.DataFrame,
    prediction_column: str,
    return_column: str,
    min_samples: int,
) -> float:
    daily = _daily_ic_frame(frame, prediction_column, return_column, min_samples)
    return float(daily["rank_ic"].mean()) if not daily.empty else float("nan")


def choose_unknown_direction(
    train: pd.DataFrame,
    feature_name: str,
    return_column: str,
    min_samples: int,
) -> int:
    """Pick +1/-1 from training-window mean RankIC only; never use the test fold."""
    evaluated = train[["trade_date", feature_name, return_column]].dropna().copy()
    evaluated["signed_probe"] = evaluated[feature_name].astype(float)
    rank_ic = mean_daily_rank_ic(evaluated, "signed_probe", return_column, min_samples)
    if np.isnan(rank_ic) or rank_ic >= 0:
        return 1
    return -1


def build_signed_prediction(
    frame: pd.DataFrame,
    feature_name: str,
    direction: int,
    score_transform: str,
) -> pd.Series:
    signed = frame[feature_name].astype(float) * float(direction)
    temp = frame[["trade_date"]].copy()
    temp["signed_signal"] = signed
    return apply_score_transform(temp, "signed_signal", score_transform)


def _cost_labels(config: dict[str, Any]) -> list[str]:
    labels: list[str] = []
    for bps in config["one_way_cost_bps"]:
        bps = float(bps)
        labels.append(str(int(bps)) if bps.is_integer() else str(bps).replace(".", "p"))
    return labels


def summarize_prediction_quality(
    predictions: pd.DataFrame,
    strategy: pd.DataFrame,
    config: dict[str, Any],
    daily_ic: pd.DataFrame | None = None,
) -> dict[str, Any]:
    min_samples = int(config.get("min_ic_samples", 5))
    raw_return = config["raw_return_column"]
    target_column = config.get("model_target_column", "target_cs_rank_5d")
    annualization = float(config["annualization"])
    if daily_ic is None:
        daily_ic = _daily_ic_frame(predictions, "prediction", raw_return, min_samples)
    rank_std = float(daily_ic["rank_ic"].std(ddof=1)) if len(daily_ic) > 1 else np.nan
    rank_mean = float(daily_ic["rank_ic"].mean()) if not daily_ic.empty else np.nan
    result: dict[str, Any] = {
        "pearson_ic_mean": float(daily_ic["pearson_ic"].mean()) if not daily_ic.empty else np.nan,
        "rank_ic_mean": rank_mean,
        "rank_ic_hit_rate": float(daily_ic["rank_ic"].gt(0).mean()) if not daily_ic.empty else np.nan,
        "rank_icir": float(rank_mean / rank_std) if rank_std and rank_std > 0 else np.nan,
        "ic_days": int(len(daily_ic)),
        "prediction_rows": int(len(predictions)),
        "prediction_days": int(predictions["trade_date"].nunique()),
        "strategy_days": int(strategy["trade_date"].nunique()) if not strategy.empty else 0,
    }
    if not predictions.empty and target_column in predictions.columns:
        valid = predictions[["prediction", target_column]].dropna()
        if len(valid) >= 2 and valid["prediction"].nunique() > 1:
            result["target_r2"] = float(r2_score(valid[target_column], valid["prediction"]))
            result["target_mae"] = float(mean_absolute_error(valid[target_column], valid["prediction"]))
        else:
            result["target_r2"] = np.nan
            result["target_mae"] = np.nan
    coverage_den = max(int(result["prediction_days"]), 1)
    result["strategy_coverage"] = float(result["strategy_days"] / coverage_den)
    stats = (
        return_statistics(strategy["gross_return"], annualization)
        if not strategy.empty
        else return_statistics(pd.Series(dtype=float), annualization)
    )
    for key, value in stats.items():
        result[f"gross_{key}"] = value
    for label in _cost_labels(config):
        column = f"net_{label}bps_return"
        stats = (
            return_statistics(strategy[column], annualization)
            if not strategy.empty and column in strategy.columns
            else return_statistics(pd.Series(dtype=float), annualization)
        )
        for key, value in stats.items():
            result[f"net_{label}bps_{key}"] = value
    return result


def optional_horizon_rank_ics(
    predictions: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, float]:
    min_samples = int(config.get("min_ic_samples", 5))
    out: dict[str, float] = {}
    for column in config.get("optional_horizon_columns", []):
        if column not in predictions.columns:
            continue
        out[f"rank_ic_mean_{column}"] = mean_daily_rank_ic(
            predictions, "prediction", column, min_samples
        )
    return out


def evaluate_factor_fold(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_name: str,
    catalog_direction: int,
    fold_id: str,
    test_year: int,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame]:
    min_samples = int(config.get("min_ic_samples", 5))
    raw_return = config["raw_return_column"]
    target_column = config.get("model_target_column", "target_cs_rank_5d")
    if int(catalog_direction) == 0:
        applied_direction = choose_unknown_direction(
            train, feature_name, raw_return, min_samples
        )
        direction_source = "train_rank_ic"
    else:
        applied_direction = int(catalog_direction)
        direction_source = "catalog"

    keep = ["trade_date", "product", "sector", raw_return, target_column]
    keep += [c for c in config.get("optional_horizon_columns", []) if c in test.columns]
    predicted = test[keep].copy()
    predicted["feature_name"] = feature_name
    predicted["prediction"] = build_signed_prediction(
        test, feature_name, applied_direction, config["score_transform"]
    ).to_numpy()
    predicted["fold_id"] = fold_id
    predicted["test_year"] = int(test_year)
    predicted["applied_direction"] = int(applied_direction)
    predicted["catalog_direction"] = int(catalog_direction)
    predicted["direction_source"] = direction_source

    strategy = build_strategy_returns(predicted, config)
    if not strategy.empty:
        strategy = strategy.copy()
        strategy["feature_name"] = feature_name
        strategy["fold_id"] = fold_id
        strategy["test_year"] = int(test_year)

    daily_ic = _daily_ic_frame(predicted, "prediction", raw_return, min_samples)
    if not daily_ic.empty:
        daily_ic = daily_ic.copy()
        daily_ic["feature_name"] = feature_name
        daily_ic["fold_id"] = fold_id
        daily_ic["test_year"] = int(test_year)

    fold = {
        "feature_name": feature_name,
        "fold_id": fold_id,
        "test_year": int(test_year),
        "catalog_direction": int(catalog_direction),
        "applied_direction": int(applied_direction),
        "direction_source": direction_source,
        "train_start": train["trade_date"].min().date().isoformat(),
        "train_end": train["trade_date"].max().date().isoformat(),
        "train_rows": int(len(train)),
        "test_rows": int(len(predicted)),
        "score_transform": config["score_transform"],
        **summarize_prediction_quality(predicted, strategy, config, daily_ic),
        **optional_horizon_rank_ics(predicted, config),
    }
    return predicted, strategy, fold, daily_ic


def build_yearly_factor_table(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "feature_name",
        "fold_id",
        "test_year",
        "rank_ic_mean",
        "rank_ic_hit_rate",
        "gross_sharpe",
        "net_5bps_sharpe",
        "gross_annual_return",
        "strategy_days",
        "applied_direction",
        "direction_source",
    ]
    available = [column for column in columns if column in fold_metrics.columns]
    return fold_metrics[available].sort_values(["feature_name", "test_year"]).reset_index(drop=True)


def build_leaderboard(factor_metrics: pd.DataFrame) -> pd.DataFrame:
    board = factor_metrics.copy()
    board = board.sort_values(
        ["rank_ic_mean", "rank_ic_hit_rate", "gross_sharpe", "strategy_coverage", "strategy_days"],
        ascending=[False, False, False, False, False],
        na_position="last",
    ).reset_index(drop=True)
    board.insert(0, "rank", np.arange(1, len(board) + 1))
    return board


def select_top_factors(
    leaderboard: pd.DataFrame,
    top_k: int | None = None,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Select a shortlist of best factors for discussion / follow-up models.

    Important: evaluating all registered fields (e.g. 92) is not the same as
    shortlisting. Quality-flag / missing-indicator columns can dominate RankIC
    without being economically meaningful single-factor signals.

    Default shortlist universe (configurable via ``top_k_selection``):
    - ``usage_type`` in ``predictor`` (and optional orthogonalized predictors)
    - exclude ``*__missing`` / missing-indicator fields
    - then rank by RankIC / hit-rate / gross Sharpe and take top K
    """
    config = config or {}
    if not bool(config.get("select_top_factors", True)):
        return leaderboard.iloc[0:0].copy()

    k = int(top_k if top_k is not None else config.get("top_k_factors", 3))
    if k <= 0 or leaderboard.empty:
        return leaderboard.iloc[0:0].copy()

    selection = config.get("top_k_selection") or {}
    eligible = leaderboard.copy()

    usage_types = selection.get("usage_types", ["predictor", "orthogonalized_predictor"])
    if usage_types:
        if "usage_type" not in eligible.columns:
            raise ValueError("leaderboard missing usage_type required for top-k selection")
        eligible = eligible.loc[eligible["usage_type"].isin(set(usage_types))].copy()

    if bool(selection.get("exclude_missing_indicators", True)):
        name = eligible["feature_name"].astype(str)
        eligible = eligible.loc[~name.str.endswith("__missing")].copy()
        if "is_missing_indicator" in eligible.columns:
            eligible = eligible.loc[~eligible["is_missing_indicator"].fillna(False).astype(bool)].copy()

    if bool(selection.get("require_positive_gross_sharpe", False)) and "gross_sharpe" in eligible.columns:
        eligible = eligible.loc[eligible["gross_sharpe"].gt(0)].copy()

    if eligible.empty:
        return eligible

    # Re-rank inside the eligible shortlist universe (keep overall board rank too).
    eligible = eligible.sort_values(
        ["rank_ic_mean", "rank_ic_hit_rate", "gross_sharpe", "strategy_coverage", "strategy_days"],
        ascending=[False, False, False, False, False],
        na_position="last",
    ).head(k).copy()
    if "rank" in eligible.columns:
        eligible = eligible.rename(columns={"rank": "overall_rank"})
    eligible.insert(0, "shortlist_rank", np.arange(1, len(eligible) + 1))
    # Keep a convenient ``rank`` alias for report/summary serializers.
    eligible.insert(0, "rank", eligible["shortlist_rank"].to_numpy())
    return eligible.reset_index(drop=True)


def factor_metric_record(row: pd.Series, config: dict[str, Any]) -> dict[str, Any]:
    """Serialize one factor's metrics (not cross-factor aggregates)."""
    record: dict[str, Any] = {
        "rank": int(row["rank"]) if "rank" in row and pd.notna(row["rank"]) else None,
        "shortlist_rank": int(row["shortlist_rank"]) if "shortlist_rank" in row and pd.notna(row.get("shortlist_rank")) else None,
        "overall_rank": int(row["overall_rank"]) if "overall_rank" in row and pd.notna(row.get("overall_rank")) else None,
        "feature_name": str(row["feature_name"]),
        "source_factor": str(row.get("source_factor", "")),
        "usage_type": str(row.get("usage_type", "")),
        "catalog_direction": int(row["catalog_direction"]) if pd.notna(row.get("catalog_direction")) else None,
        "rank_ic_mean": float(row["rank_ic_mean"]) if pd.notna(row.get("rank_ic_mean")) else None,
        "rank_ic_hit_rate": float(row["rank_ic_hit_rate"]) if pd.notna(row.get("rank_ic_hit_rate")) else None,
        "rank_icir": float(row["rank_icir"]) if pd.notna(row.get("rank_icir")) else None,
        "pearson_ic_mean": float(row["pearson_ic_mean"]) if pd.notna(row.get("pearson_ic_mean")) else None,
        "gross_annual_return": float(row["gross_annual_return"]) if pd.notna(row.get("gross_annual_return")) else None,
        "gross_annual_volatility": float(row["gross_annual_volatility"]) if pd.notna(row.get("gross_annual_volatility")) else None,
        "gross_sharpe": float(row["gross_sharpe"]) if pd.notna(row.get("gross_sharpe")) else None,
        "gross_max_drawdown": float(row["gross_max_drawdown"]) if pd.notna(row.get("gross_max_drawdown")) else None,
        "gross_total_return": float(row["gross_total_return"]) if pd.notna(row.get("gross_total_return")) else None,
        "gross_hit_rate": float(row["gross_hit_rate"]) if pd.notna(row.get("gross_hit_rate")) else None,
        "strategy_days": int(row["strategy_days"]) if pd.notna(row.get("strategy_days")) else None,
        "years_evaluated": int(row["years_evaluated"]) if pd.notna(row.get("years_evaluated")) else None,
        "years_rank_ic_positive": int(row["years_rank_ic_positive"]) if pd.notna(row.get("years_rank_ic_positive")) else None,
        "years_gross_sharpe_positive": int(row["years_gross_sharpe_positive"]) if pd.notna(row.get("years_gross_sharpe_positive")) else None,
    }
    for label in _cost_labels(config):
        for suffix in ("annual_return", "sharpe", "max_drawdown", "total_return", "hit_rate"):
            key = f"net_{label}bps_{suffix}"
            if key in row.index and pd.notna(row[key]):
                record[key] = float(row[key])
            elif key in row.index:
                record[key] = None
    return record


def _format(value: Any, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def _markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "暂无记录。"
    columns = [str(column) for column in frame.columns]
    rows = frame.fillna("—").astype(str)
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(rows.iloc[i].tolist()) + " |"
        for i in range(len(rows))
    ]
    return "\n".join([header, sep, *body])


def write_report(
    path: Path,
    config: dict[str, Any],
    candidates: pd.DataFrame,
    factor_metrics: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    leaderboard: pd.DataFrame,
    top_factors: pd.DataFrame,
    bundle_info: dict[str, Any],
) -> None:
    raw_return = config["raw_return_column"]
    target_column = config.get("model_target_column", "target_cs_rank_5d")
    cost_text = "/".join(_cost_labels(config))
    years = sorted(fold_metrics["test_year"].unique().tolist()) if not fold_metrics.empty else []
    select_top = bool(config.get("select_top_factors", False)) and not top_factors.empty

    display_cols = [
        "rank",
        "feature_name",
        "source_factor",
        "usage_type",
        "category",
        "rank_ic_mean",
        "rank_ic_hit_rate",
        "rank_icir",
        "gross_annual_return",
        "gross_sharpe",
        "net_0bps_sharpe",
        "net_3bps_sharpe",
        "net_5bps_sharpe",
        "net_10bps_sharpe",
        "strategy_days",
        "years_evaluated",
    ]

    def _display(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame[[c for c in display_cols if c in frame.columns]].copy()
        for column in out.columns:
            if pd.api.types.is_float_dtype(out[column]):
                out[column] = out[column].map(_format)
        return out

    all_display = _display(leaderboard)
    predictor = (
        leaderboard.loc[leaderboard["usage_type"].eq("predictor")].copy()
        if "usage_type" in leaderboard.columns
        else leaderboard.iloc[0:0]
    )
    predictor_display = _display(predictor)
    by_sharpe = leaderboard.sort_values(
        ["gross_sharpe", "rank_ic_mean"], ascending=[False, False], na_position="last"
    )
    sharpe_display = _display(by_sharpe)

    usage_counts = (
        candidates.groupby(["usage_type"], as_index=False)
        .size()
        .rename(columns={"size": "字段数"})
    )

    setup = pd.DataFrame([
        {"项目": "样本外年份", "值": f"{min(years)}—{max(years)}" if years else "—"},
        {"项目": "handoff目录", "值": str(bundle_info.get("handoff_directory", ""))},
        {"项目": "评测字段数", "值": str(len(candidates))},
        {"项目": "短名单TopK", "值": "关闭" if not select_top else str(config.get("top_k_factors", 0))},
        {"项目": "score_transform", "值": config["score_transform"]},
        {"项目": "主标签", "值": raw_return},
        {"项目": "图表目录", "值": config.get("figures_directory", "outputs/figures")},
        {"项目": "有序表目录", "值": config.get("tables_directory", "outputs/tables")},
    ])

    cost_metric_cols: list[str] = []
    for label in _cost_labels(config):
        cost_metric_cols.extend([f"net_{label}bps_annual_return", f"net_{label}bps_sharpe"])
    per_factor_cols = [
        "rank",
        "feature_name",
        "source_factor",
        "usage_type",
        "rank_ic_mean",
        "rank_ic_hit_rate",
        "gross_annual_return",
        "gross_annual_volatility",
        "gross_sharpe",
        "gross_max_drawdown",
        "gross_total_return",
        "gross_hit_rate",
        *cost_metric_cols,
        "strategy_days",
        "years_evaluated",
        "years_gross_sharpe_positive",
    ]
    per_factor = leaderboard[[c for c in per_factor_cols if c in leaderboard.columns]].copy()
    per_factor_display = per_factor.copy()
    for column in per_factor_display.columns:
        if pd.api.types.is_float_dtype(per_factor_display[column]):
            per_factor_display[column] = per_factor_display[column].map(_format)

    lines = [
        "# 单因子排序模型测试报告",
        "",
        "> 本报告由 handoff 矩阵接口自动生成；模型为**单因子横截面排序**，不含多因子加权或回归系数。",
        "> **主结果是每一个因子各自的指标**；不再输出 TopK 短名单。",
        "> 美化图表与有序表见 `outputs/figures/`、`outputs/tables/`。",
        "> 2025 不在本 handoff 包内，不参与因子选择、方向选择或主评价。",
        "",
        "## 运行设置",
        "",
        _markdown(setup),
        "",
        "## 字段组成",
        "",
        _markdown(usage_counts),
        "",
        "## 全部字段：按 RankIC 排序",
        "",
        _markdown(all_display),
        "",
        "## 预测型因子（predictor）：按 RankIC 排序",
        "",
        _markdown(predictor_display),
        "",
        "## 全部字段：按毛 Sharpe 排序",
        "",
        _markdown(sharpe_display),
        "",
        "## 完整策略绩效宽表",
        "",
        "下表对**每一个**候选因子单独给出样本外指标。"
        "CSV 见 `outputs/factor_strategy_metrics.csv` 与 `outputs/tables/`。",
        "",
        _markdown(per_factor_display),
        "",
        "## 模型与策略口径",
        "",
        "- 模型：每个注册字段单独作为横截面排序信号；不做多因子回归。",
        f"- 标签：`{raw_return}`；对照目标 `{target_column}` 不参与选因子。",
        "- 切分：`fold_definitions.csv` 扩展窗口；"
        f"净化 `{config.get('label_end_column', 'label_end_date_5d')} < valid_start`。",
        f"- 持有/调仓 {config.get('holding_days', 5)} 日；多空各 {float(config['long_short_fraction'])*100:.0f}%。",
        f"- 成本敏感性：单边 {cost_text} bp（主口径 {config.get('base_cost_bps_one_way', 5)} bp）。",
        "",
        "## 使用边界",
        "",
        "1. 点时：因子在 T 日收盘后形成，信号仅可用于 T+1 开盘交易。",
        "2. 本模型是单字段排序，**不是**多因子加权或回归拟合结果。",
        f"3. 成本仅为单边 {cost_text} 基点敏感性，不得解读为已核验真实手续费。",
        "4. 2025 未参与主评价。",
        "5. quality_flag / `__missing` 字段可出现在全量表中，解读时需与 predictor 区分。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(config_path: str | Path, feature_limit: int | None = None) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    root = config_path.parents[2]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_dir = root / config["output_directory"]
    report_path = root / config["report_file"]
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    # Accept legacy ``data_directory`` as alias of ``handoff_directory``.
    if "handoff_directory" not in config and "data_directory" in config:
        config = {**config, "handoff_directory": config["data_directory"]}

    bundle = load_handoff_bundle(root, config)
    bundle_info = handoff_summary(bundle)
    folds = load_fold_definitions(bundle.directory, config)
    candidates = select_candidates(bundle.registry, config)
    if feature_limit is not None:
        candidates = candidates.head(int(feature_limit)).copy()
    feature_names = candidates["feature_name"].tolist()

    raw_return = config["raw_return_column"]
    target_column = config.get("model_target_column", "target_cs_rank_5d")
    label_end_column = config.get("label_end_column", "label_end_date_5d")

    # Merge all candidate features + optional horizon labels in one join.
    # load_handoff_bundle already validated the full registry; here we only keep candidates.
    missing = sorted(set(feature_names) - set(bundle.features.columns))
    if missing:
        raise ValueError(f"registered features missing from matrix: {missing}")

    label_cols = list(KEYS) + [raw_return]
    for column in config.get("optional_horizon_columns", []):
        if column in bundle.labels.columns:
            label_cols.append(column)
    for optional in (label_end_column, "label_end_date_1d", "label_end_date_20d"):
        if optional in bundle.labels.columns and optional not in label_cols:
            label_cols.append(optional)

    labels = bundle.labels[label_cols].copy()
    for column in labels.columns:
        if column.startswith("label_end_date_"):
            labels[column] = pd.to_datetime(labels[column], errors="coerce")

    data = bundle.features[KEYS + ["sector"] + feature_names].merge(
        labels,
        on=KEYS,
        how="inner",
        validate="one_to_one",
    )
    development_end = pd.Timestamp(config.get("development_end", "2024-12-31"))
    data = data.loc[data["trade_date"].le(development_end)].copy()
    data[target_column] = cross_sectional_rank_target(data, raw_return)
    data = data.dropna(subset=[raw_return, target_column]).copy()
    data[feature_names] = data[feature_names].fillna(0.0)

    all_predictions: list[pd.DataFrame] = []
    all_strategy: list[pd.DataFrame] = []
    all_fold_metrics: list[dict[str, Any]] = []
    all_daily_ic: list[pd.DataFrame] = []

    direction_lookup = {
        row.feature_name: int(row.expected_direction) for row in candidates.itertuples()
    }

    for feature_name in feature_names:
        for fold in folds.itertuples(index=False):
            outer_train, outer_test = split_fold(
                data,
                fold,
                target_column=target_column,
                label_end_column=label_end_column,
            )
            if outer_train.empty or outer_test.empty:
                continue
            test_year = int(pd.Timestamp(fold.valid_start).year)
            predicted, strategy, fold_metric, daily_ic = evaluate_factor_fold(
                outer_train,
                outer_test,
                feature_name,
                direction_lookup[feature_name],
                str(fold.fold_id),
                test_year,
                config,
            )
            all_predictions.append(predicted)
            if not strategy.empty:
                all_strategy.append(strategy)
            all_fold_metrics.append(fold_metric)
            if not daily_ic.empty:
                all_daily_ic.append(daily_ic)

    if not all_predictions:
        raise RuntimeError("no OOS predictions were produced")

    predictions = pd.concat(all_predictions, ignore_index=True)
    strategy = pd.concat(all_strategy, ignore_index=True) if all_strategy else pd.DataFrame()
    fold_metrics = pd.DataFrame(all_fold_metrics)
    daily_ic = pd.concat(all_daily_ic, ignore_index=True) if all_daily_ic else pd.DataFrame()

    factor_rows: list[dict[str, Any]] = []
    for feature_name, pred_part in predictions.groupby("feature_name", sort=True):
        strat_part = (
            strategy.loc[strategy["feature_name"].eq(feature_name)].copy()
            if not strategy.empty
            else pd.DataFrame()
        )
        meta = candidates.loc[candidates["feature_name"].eq(feature_name)].iloc[0]
        directions = fold_metrics.loc[
            fold_metrics["feature_name"].eq(feature_name),
            ["applied_direction", "direction_source", "catalog_direction"],
        ]
        row = {
            "feature_name": feature_name,
            "source_factor": meta.get("source_factor", ""),
            "source_factor_id": int(meta["source_factor_id"]),
            "usage_type": meta["usage_type"],
            "category": meta.get("category", ""),
            "catalog_direction": (
                int(directions["catalog_direction"].iloc[0])
                if not directions.empty
                else int(meta["expected_direction"])
            ),
            "direction_source_modes": (
                ",".join(sorted(directions["direction_source"].unique()))
                if not directions.empty else ""
            ),
            "applied_direction_modes": (
                ",".join(str(int(v)) for v in sorted(directions["applied_direction"].unique()))
                if not directions.empty else ""
            ),
            **summarize_prediction_quality(pred_part, strat_part, config),
            **optional_horizon_rank_ics(pred_part, config),
        }
        yearly = fold_metrics.loc[fold_metrics["feature_name"].eq(feature_name)]
        row["years_evaluated"] = int(yearly["test_year"].nunique())
        row["years_rank_ic_positive"] = int(yearly["rank_ic_mean"].gt(0).sum())
        row["years_gross_sharpe_positive"] = int(yearly["gross_sharpe"].gt(0).sum())
        factor_rows.append(row)

    factor_metrics = pd.DataFrame(factor_rows)
    leaderboard = build_leaderboard(factor_metrics)
    top_factors = select_top_factors(leaderboard, config=config)

    predictions.to_csv(output_dir / "oos_predictions.csv.gz", index=False, compression="gzip")
    strategy.to_csv(output_dir / "strategy_daily_returns.csv", index=False)
    factor_metrics.to_csv(output_dir / "factor_metrics.csv", index=False)
    fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    daily_ic.to_csv(output_dir / "daily_prediction_ic.csv", index=False)
    leaderboard.to_csv(output_dir / "leaderboard.csv", index=False)
    build_yearly_factor_table(fold_metrics).to_csv(output_dir / "yearly_factor_metrics.csv", index=False)
    folds.to_csv(output_dir / "fold_definitions_used.csv", index=False)
    (output_dir / "handoff_summary.json").write_text(
        json.dumps(bundle_info, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    cost_metric_cols = []
    for label in _cost_labels(config):
        cost_metric_cols.extend([f"net_{label}bps_annual_return", f"net_{label}bps_sharpe"])
    strategy_metric_cols = [
        "rank",
        "feature_name",
        "source_factor",
        "usage_type",
        "rank_ic_mean",
        "rank_ic_hit_rate",
        "rank_icir",
        "gross_annual_return",
        "gross_annual_volatility",
        "gross_sharpe",
        "gross_max_drawdown",
        "gross_total_return",
        "gross_hit_rate",
        *cost_metric_cols,
        "strategy_days",
        "strategy_coverage",
        "years_evaluated",
        "years_rank_ic_positive",
        "years_gross_sharpe_positive",
    ]
    per_factor_table = leaderboard[[c for c in strategy_metric_cols if c in leaderboard.columns]].copy()
    per_factor_table.to_csv(output_dir / "factor_strategy_metrics.csv", index=False)

    per_factor_records = [factor_metric_record(row, config) for _, row in leaderboard.iterrows()]
    (output_dir / "per_factor_metrics.json").write_text(
        json.dumps(per_factor_records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    for obsolete in ("top3_factors.csv", "top3_factors.json"):
        obsolete_path = output_dir / obsolete
        if obsolete_path.exists():
            obsolete_path.unlink()

    write_report(
        report_path,
        config,
        candidates,
        factor_metrics,
        fold_metrics,
        leaderboard,
        top_factors,
        bundle_info,
    )

    summary = {
        "model_version": config["model_version"],
        "protocol": {
            "primary_target": raw_return,
            "split": "expanding_window_from_fold_definitions",
            "holding_days": config.get("holding_days", 5),
            "rebalance_frequency_days": config.get("rebalance_frequency_days", 5),
            "base_cost_bps_one_way": config.get("base_cost_bps_one_way", 5),
            "cost_scenarios_bps_one_way": config["one_way_cost_bps"],
        },
        "score_transform": config["score_transform"],
        "candidate_count": len(candidates),
        "folds": [str(x) for x in folds["fold_id"].tolist()],
        "handoff": bundle_info,
        "oos_rows": int(len(predictions)),
        "select_top_factors": bool(config.get("select_top_factors", False)),
        "per_factor_metrics": per_factor_records,
        "report": str(report_path),
        "note": (
            "Primary outputs are per-factor metrics for all registered fields; "
            "top-k shortlist is disabled. See figures/ and tables/ for presentation assets."
        ),
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="单因子横截面排序（协议主策略 / handoff 接口）")
    parser.add_argument("--config", default="modeling/single_factor_rank_v1/config.json")
    parser.add_argument("--limit", type=int, default=None, help="仅跑前 N 个候选字段（冒烟）")
    args = parser.parse_args(argv)
    print(json.dumps(run(args.config, feature_limit=args.limit), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
