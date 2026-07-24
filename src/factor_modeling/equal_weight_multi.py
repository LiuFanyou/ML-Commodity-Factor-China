"""Equal-weight multi-factor composite on the training-handoff interface.

Aligned with ``training_handoff/training_contract.json`` and the unified protocol:

- Features / registry / labels / folds loaded via ``factor_modeling.handoff``.
- Primary label: ``future_return_5d`` (open→open on mapped real contract).
- Expanding windows from ``fold_definitions.csv`` (2019–2024 OOS).
- Train purge: ``label_end_date_5d < valid_start``.
- Execution: T close signal, T+1 open trade, 5-day hold / 5-day rebalance.
- Cost: base 5bp one-way; sensitivity 0 / 3 / 5 / 10 bp.
- Combination: direction-align → cs_rank_centered → equal-weight average.
- v1 default universe: **all** registered fields (92; floor baseline).
- v2 optional ``selection_mode=train_ic_corr``: predictor prior + train-window
  RankIC filter + greedy correlation dedup (no OOS peeking).
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
    merge_features_and_labels,
    split_fold,
)
from factor_modeling.single_factor_rank import (
    apply_score_transform,
    choose_unknown_direction,
    select_candidates,
    _daily_ic_frame,
    mean_daily_rank_ic,
)

__all__ = [
    "build_equal_weight_prediction",
    "evaluate_fold",
    "main",
    "mean_daily_abs_corr_matrix",
    "resolve_directions",
    "run",
    "select_fold_universe",
    "write_report",
]


def resolve_directions(
    train: pd.DataFrame,
    candidates: pd.DataFrame,
    return_column: str,
    min_samples: int,
) -> pd.DataFrame:
    """Resolve +1/-1 per universe factor; catalog directions are frozen as-is.

    Always records signed training-window RankIC after the resolved direction
    (used by v2 selection; v1 may leave it unused).
    """
    rows: list[dict[str, Any]] = []
    for row in candidates.itertuples():
        catalog_direction = int(row.expected_direction)
        feature_name = str(row.feature_name)
        if catalog_direction == 0:
            resolved = choose_unknown_direction(
                train, feature_name, return_column, min_samples
            )
            source = "train_rank_ic"
        else:
            resolved = catalog_direction
            source = "catalog"

        evaluated = train[["trade_date", feature_name, return_column]].dropna().copy()
        evaluated["signed_probe"] = evaluated[feature_name].astype(float) * float(resolved)
        train_ic = mean_daily_rank_ic(
            evaluated, "signed_probe", return_column, min_samples
        )
        rows.append({
            "feature_name": feature_name,
            "source_factor": row.source_factor,
            "usage_type": row.usage_type,
            "catalog_direction": catalog_direction,
            "resolved_direction": int(resolved),
            "direction_source": source,
            "train_rank_ic": float(train_ic) if pd.notna(train_ic) else np.nan,
        })
    return pd.DataFrame(rows)


def mean_daily_abs_corr_matrix(
    frame: pd.DataFrame,
    feature_names: list[str],
    min_samples: int,
) -> pd.DataFrame:
    """Mean absolute cross-sectional Pearson corr across train days."""
    n = len(feature_names)
    accum = np.zeros((n, n), dtype=float)
    counts = np.zeros((n, n), dtype=float)
    index = {name: i for i, name in enumerate(feature_names)}

    for _, part in frame.groupby("trade_date", sort=False):
        valid = part[feature_names]
        usable = [c for c in feature_names if valid[c].notna().sum() >= min_samples]
        if len(usable) < 2:
            continue
        sub = valid[usable]
        usable = [c for c in usable if sub[c].nunique(dropna=True) >= 2]
        if len(usable) < 2:
            continue
        corr = sub[usable].corr(method="pearson")
        for a in usable:
            ia = index[a]
            for b in usable:
                ib = index[b]
                value = corr.loc[a, b]
                if pd.isna(value):
                    continue
                accum[ia, ib] += abs(float(value))
                counts[ia, ib] += 1.0

    with np.errstate(invalid="ignore", divide="ignore"):
        mean_abs = np.where(counts > 0, accum / counts, np.nan)
    return pd.DataFrame(mean_abs, index=feature_names, columns=feature_names)


def select_fold_universe(
    train: pd.DataFrame,
    candidates: pd.DataFrame,
    directions_frame: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[list[str], pd.DataFrame]:
    """Train-window RankIC filter + greedy correlation dedup.

    Returns selected feature names (stable order by descending train RankIC)
    and a per-candidate audit table.
    """
    selection_mode = str(config.get("selection_mode", "none") or "none")
    prior_names = [str(n) for n in candidates["feature_name"].tolist()]
    if selection_mode in {"none", "", "all_equal"}:
        audit = directions_frame.copy()
        audit["selected"] = True
        audit["selection_reason"] = "prior_universe"
        audit["max_abs_corr_vs_selected"] = np.nan
        return prior_names, audit

    if selection_mode != "train_ic_corr":
        raise ValueError(f"unsupported selection_mode: {selection_mode}")

    min_samples = int(config.get("min_ic_samples", 5))
    min_train_rank_ic = float(config.get("min_train_rank_ic", 0.01))
    max_pairwise_corr = float(config.get("max_pairwise_corr", 0.7))
    score_transform = str(config.get("score_transform", "cs_rank_centered"))

    direction_map = {
        str(row.feature_name): int(row.resolved_direction)
        for row in directions_frame.itertuples()
    }
    ic_map = {
        str(row.feature_name): (
            float(row.train_rank_ic) if pd.notna(row.train_rank_ic) else float("nan")
        )
        for row in directions_frame.itertuples()
    }

    rank_cols: list[pd.Series] = []
    for name in prior_names:
        signed = train[name].astype(float) * float(direction_map[name])
        temp = train[["trade_date"]].copy()
        temp["signed_signal"] = signed
        rank_cols.append(
            apply_score_transform(temp, "signed_signal", score_transform).rename(name)
        )
    ranked = pd.concat([train[["trade_date"]], *rank_cols], axis=1)

    ranked_ics = sorted(
        ((name, ic_map[name]) for name in prior_names),
        key=lambda item: (-(item[1] if np.isfinite(item[1]) else -np.inf), item[0]),
    )

    ic_pass = [
        name
        for name, ic in ranked_ics
        if np.isfinite(ic) and float(ic) >= min_train_rank_ic
    ]
    corr_matrix = (
        mean_daily_abs_corr_matrix(ranked, ic_pass, min_samples)
        if len(ic_pass) >= 2
        else pd.DataFrame(index=ic_pass, columns=ic_pass, dtype=float)
    )

    selected: list[str] = []
    reason_map: dict[str, str] = {}
    max_corr_map: dict[str, float] = {}
    for name, ic in ranked_ics:
        if not np.isfinite(ic) or float(ic) < min_train_rank_ic:
            reason_map[name] = "dropped_weak_train_rank_ic"
            max_corr_map[name] = np.nan
            continue
        if not selected:
            selected.append(name)
            reason_map[name] = "selected"
            max_corr_map[name] = 0.0
            continue
        pairwise = [
            float(corr_matrix.loc[name, other])
            for other in selected
            if name in corr_matrix.index
            and other in corr_matrix.columns
            and pd.notna(corr_matrix.loc[name, other])
        ]
        max_abs = float(max(pairwise)) if pairwise else 0.0
        max_corr_map[name] = max_abs
        if max_abs >= max_pairwise_corr:
            reason_map[name] = "dropped_high_corr"
            continue
        selected.append(name)
        reason_map[name] = "selected"

    if not selected:
        best_name, _best_ic = ranked_ics[0]
        selected = [best_name]
        reason_map[best_name] = "selected_fallback_best_train_ic"
        max_corr_map[best_name] = 0.0

    audit = directions_frame.copy()
    audit["selected"] = audit["feature_name"].map(lambda n: str(n) in set(selected))
    audit["selection_reason"] = audit["feature_name"].map(
        lambda n: reason_map.get(str(n), "not_considered")
    )
    audit["max_abs_corr_vs_selected"] = audit["feature_name"].map(
        lambda n: max_corr_map.get(str(n), np.nan)
    )
    audit["min_train_rank_ic"] = min_train_rank_ic
    audit["max_pairwise_corr"] = max_pairwise_corr
    return selected, audit


def build_equal_weight_prediction(
    frame: pd.DataFrame,
    feature_names: list[str],
    directions: dict[str, int],
    score_transform: str,
    missing_policy: str,
) -> tuple[pd.Series, pd.Series]:
    """
    Direction-align each factor, cs-rank within day, then equal-weight average.

    missing_policy:
      - mean_available: average non-missing ranks; all-missing -> NaN
      - require_all: any missing universe factor -> NaN for that row
    """
    if missing_policy not in {"mean_available", "require_all"}:
        raise ValueError(f"unsupported missing_policy: {missing_policy}")

    rank_columns: list[pd.Series] = []
    for name in feature_names:
        direction = int(directions[name])
        signed = frame[name].astype(float) * float(direction)
        temp = frame[["trade_date"]].copy()
        temp["signed_signal"] = signed
        rank_columns.append(
            apply_score_transform(temp, "signed_signal", score_transform).rename(name)
        )

    ranks = pd.concat(rank_columns, axis=1)
    n_used = ranks.notna().sum(axis=1).astype(float)
    if missing_policy == "require_all":
        complete = n_used.eq(float(len(feature_names)))
        prediction = ranks.mean(axis=1, skipna=False)
        prediction = prediction.where(complete)
        n_used = n_used.where(complete, 0.0)
    else:
        prediction = ranks.mean(axis=1, skipna=True)
        prediction = prediction.where(n_used.gt(0))
        n_used = n_used.where(n_used.gt(0), 0.0)

    prediction.name = "prediction"
    n_used.name = "n_factors_used"
    return prediction, n_used


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
    return_column = config["raw_return_column"]
    target_column = config.get("model_target_column", "target_cs_rank_5d")
    annualization = float(config["annualization"])
    if daily_ic is None:
        daily_ic = _daily_ic_frame(predictions, "prediction", return_column, min_samples)

    rank_std = float(daily_ic["rank_ic"].std(ddof=1)) if len(daily_ic) > 1 else np.nan
    rank_mean = float(daily_ic["rank_ic"].mean()) if not daily_ic.empty else np.nan
    result: dict[str, Any] = {
        "pearson_ic_mean": float(daily_ic["pearson_ic"].mean()) if not daily_ic.empty else np.nan,
        "rank_ic_mean": rank_mean,
        "rank_ic_hit_rate": float(daily_ic["rank_ic"].gt(0).mean()) if not daily_ic.empty else np.nan,
        "rank_icir": float(rank_mean / rank_std) if rank_std and rank_std > 0 else np.nan,
        "ic_days": int(len(daily_ic)),
        "prediction_rows": int(len(predictions)),
        "prediction_days": int(predictions["trade_date"].nunique()) if not predictions.empty else 0,
        "strategy_days": int(strategy["trade_date"].nunique()) if not strategy.empty else 0,
    }

    if not predictions.empty and "n_factors_used" in predictions.columns:
        used = predictions["n_factors_used"].replace(0, np.nan).dropna()
        result["n_factors_used_mean"] = float(used.mean()) if not used.empty else np.nan
        result["n_factors_used_p25"] = float(used.quantile(0.25)) if not used.empty else np.nan
        result["n_factors_used_p50"] = float(used.quantile(0.50)) if not used.empty else np.nan
        result["n_factors_used_p75"] = float(used.quantile(0.75)) if not used.empty else np.nan
        result["prediction_coverage"] = float(predictions["prediction"].notna().mean())

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
        net_stats = (
            return_statistics(strategy[column], annualization)
            if not strategy.empty and column in strategy.columns
            else return_statistics(pd.Series(dtype=float), annualization)
        )
        for key, value in net_stats.items():
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


def evaluate_fold(
    train: pd.DataFrame,
    test: pd.DataFrame,
    candidates: pd.DataFrame,
    feature_names: list[str],
    fold_id: str,
    test_year: int,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    min_samples = int(config.get("min_ic_samples", 5))
    target_column = config.get("model_target_column", "target_cs_rank_5d")
    directions_frame = resolve_directions(
        train, candidates, config["raw_return_column"], min_samples
    )
    directions_frame = directions_frame.copy()
    directions_frame["fold_id"] = fold_id
    directions_frame["test_year"] = int(test_year)

    selected_names, selection_audit = select_fold_universe(
        train, candidates, directions_frame, config
    )
    # Keep caller prior order for none-mode; selection order for train_ic_corr.
    if str(config.get("selection_mode", "none") or "none") in {"none", "", "all_equal"}:
        selected_names = [n for n in feature_names if n in set(selected_names)]
    selection_audit = selection_audit.copy()
    selection_audit["fold_id"] = fold_id
    selection_audit["test_year"] = int(test_year)
    selection_audit["prior_universe_size"] = int(len(feature_names))
    selection_audit["selected_universe_size"] = int(len(selected_names))

    direction_map = {
        str(row.feature_name): int(row.resolved_direction)
        for row in directions_frame.itertuples()
    }

    keep_cols = [
        "trade_date",
        "product",
        "sector",
        config["raw_return_column"],
        target_column,
        *selected_names,
    ]
    for column in config.get("optional_horizon_columns", []):
        if column in test.columns and column not in keep_cols:
            keep_cols.append(column)

    predicted = test[[c for c in keep_cols if c in test.columns]].copy()
    prediction, n_used = build_equal_weight_prediction(
        predicted,
        selected_names,
        direction_map,
        config["score_transform"],
        config.get("missing_policy", "mean_available"),
    )
    predicted["prediction"] = prediction.to_numpy()
    predicted["n_factors_used"] = n_used.to_numpy()
    predicted["fold_id"] = fold_id
    predicted["test_year"] = int(test_year)
    predicted = predicted.drop(columns=selected_names)

    strategy = build_strategy_returns(predicted, config)
    if not strategy.empty:
        strategy = strategy.copy()
        strategy["fold_id"] = fold_id
        strategy["test_year"] = int(test_year)

    daily_ic = _daily_ic_frame(
        predicted,
        "prediction",
        config["raw_return_column"],
        min_samples,
    )
    if not daily_ic.empty:
        daily_ic = daily_ic.copy()
        daily_ic["fold_id"] = fold_id
        daily_ic["test_year"] = int(test_year)

    fold = {
        "fold_id": fold_id,
        "test_year": int(test_year),
        "prior_universe_size": int(len(feature_names)),
        "universe_size": int(len(selected_names)),
        "selection_mode": str(config.get("selection_mode", "none") or "none"),
        "combination_scheme": config.get("combination_scheme", "all_equal"),
        "missing_policy": config.get("missing_policy", "mean_available"),
        "score_transform": config["score_transform"],
        "train_start": train["trade_date"].min().date().isoformat(),
        "train_end": train["trade_date"].max().date().isoformat(),
        "train_rows": int(len(train)),
        "test_rows": int(len(predicted)),
        "directions_from_catalog": int(
            (directions_frame["direction_source"] == "catalog").sum()
        ),
        "directions_from_train_ic": int(
            (directions_frame["direction_source"] == "train_rank_ic").sum()
        ),
        **summarize_prediction_quality(predicted, strategy, config, daily_ic),
        **optional_horizon_rank_ics(predicted, config),
    }
    return predicted, strategy, fold, daily_ic, directions_frame, selection_audit


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
    universe: pd.DataFrame,
    overall: dict[str, Any],
    fold_metrics: pd.DataFrame,
    fold_directions: pd.DataFrame,
    bundle_info: dict[str, Any] | None = None,
    fold_selected: pd.DataFrame | None = None,
    v1_summary: dict[str, Any] | None = None,
) -> None:
    n_universe = int(len(universe))
    raw_return = config["raw_return_column"]
    target_column = config.get("model_target_column", "target_cs_rank_5d")
    cost_text = "/".join(_cost_labels(config))
    base_bps = config.get("base_cost_bps_one_way", 5)
    holding = config.get("holding_days", 5)
    rebalance = config.get("rebalance_frequency_days", holding)
    selection_mode = str(config.get("selection_mode", "none") or "none")
    is_v2 = selection_mode == "train_ic_corr"
    model_version = str(config.get("model_version", "equal_weight_multi"))

    usage_counts = (
        universe.groupby(["usage_type"], as_index=False)
        .size()
        .rename(columns={"size": "字段数"})
    )
    direction_counts = (
        universe.groupby(["catalog_direction"], as_index=False)
        .size()
        .rename(columns={"size": "字段数"})
    )

    universe_display = universe[
        [c for c in ["feature_name", "source_factor", "usage_type", "catalog_direction"] if c in universe.columns]
    ].copy()
    universe_display["catalog_direction"] = universe_display["catalog_direction"].map(
        lambda v: _format(v, 0) if pd.notna(v) else "—"
    )

    oos_years = sorted(fold_metrics["test_year"].astype(int).unique().tolist()) if not fold_metrics.empty else []
    year_span = f"{oos_years[0]}—{oos_years[-1]}" if oos_years else "—"

    selected_sizes = (
        fold_metrics["universe_size"].astype(float)
        if "universe_size" in fold_metrics.columns and not fold_metrics.empty
        else pd.Series(dtype=float)
    )
    mean_selected = float(selected_sizes.mean()) if not selected_sizes.empty else float(n_universe)

    overall_pred = pd.DataFrame([
        {"指标": "模型版本", "结果": model_version},
        {"指标": "组合方案", "结果": config.get("combination_scheme", "all_equal")},
        {"指标": "筛选模式", "结果": selection_mode},
        {"指标": "样本外年份", "结果": year_span},
        {"指标": "先验宇宙大小 N", "结果": str(n_universe)},
        {"指标": "折均入选数", "结果": _format(mean_selected, 1)},
        {"指标": "缺失策略", "结果": config.get("missing_policy", "mean_available")},
        {"指标": "score_transform", "结果": config["score_transform"]},
        {"指标": "日均 Pearson IC", "结果": _format(overall.get("pearson_ic_mean"))},
        {"指标": "日均 RankIC", "结果": _format(overall.get("rank_ic_mean"))},
        {"指标": "RankIC 胜率", "结果": _format(overall.get("rank_ic_hit_rate"))},
        {"指标": "RankICIR", "结果": _format(overall.get("rank_icir"))},
        {"指标": f"相对 {target_column} 的 R²", "结果": _format(overall.get("target_r2"))},
        {"指标": f"相对 {target_column} 的 MAE", "结果": _format(overall.get("target_mae"))},
    ])

    strategy_rows = [
        {"口径": "毛收益", "年化收益": _format(overall.get("gross_annual_return")),
         "年化波动": _format(overall.get("gross_annual_volatility")),
         "Sharpe": _format(overall.get("gross_sharpe")),
         "最大回撤": _format(overall.get("gross_max_drawdown")),
         "总收益": _format(overall.get("gross_total_return")),
         "胜率": _format(overall.get("gross_hit_rate"))},
    ]
    for label in _cost_labels(config):
        strategy_rows.append({
            "口径": f"净 {label}bp",
            "年化收益": _format(overall.get(f"net_{label}bps_annual_return")),
            "年化波动": _format(overall.get(f"net_{label}bps_annual_volatility")),
            "Sharpe": _format(overall.get(f"net_{label}bps_sharpe")),
            "最大回撤": _format(overall.get(f"net_{label}bps_max_drawdown")),
            "总收益": _format(overall.get(f"net_{label}bps_total_return")),
            "胜率": _format(overall.get(f"net_{label}bps_hit_rate")),
        })
    strategy_table = pd.DataFrame(strategy_rows)

    yearly_cols = [
        "fold_id",
        "test_year",
        "prior_universe_size",
        "universe_size",
        "rank_ic_mean",
        "rank_ic_hit_rate",
        "gross_sharpe",
        "net_3bps_sharpe",
        "net_5bps_sharpe",
        "gross_annual_return",
        "strategy_days",
        "n_factors_used_mean",
        "prediction_coverage",
    ]
    yearly = fold_metrics[[c for c in yearly_cols if c in fold_metrics.columns]].copy()
    yearly_display = yearly.copy()
    for column in yearly_display.columns:
        if column in {"test_year", "prior_universe_size", "universe_size", "strategy_days"}:
            yearly_display[column] = yearly_display[column].astype(int).astype(str)
        elif pd.api.types.is_float_dtype(yearly_display[column]):
            yearly_display[column] = yearly_display[column].map(_format)

    coverage = pd.DataFrame([
        {"指标": "策略调仓日数", "结果": str(int(overall.get("strategy_days", 0)))},
        {"指标": "预测交易日数", "结果": str(int(overall.get("prediction_days", 0)))},
        {"指标": "策略覆盖率", "结果": _format(overall.get("strategy_coverage"))},
        {"指标": "预测行覆盖率", "结果": _format(overall.get("prediction_coverage"))},
        {"指标": "n_factors_used 均值", "结果": _format(overall.get("n_factors_used_mean"))},
        {"指标": "n_factors_used P25", "结果": _format(overall.get("n_factors_used_p25"))},
        {"指标": "n_factors_used P50", "结果": _format(overall.get("n_factors_used_p50"))},
        {"指标": "n_factors_used P75", "结果": _format(overall.get("n_factors_used_p75"))},
    ])

    dir_summary_rows: list[dict[str, Any]] = []
    if not fold_directions.empty:
        for feature_name, part in fold_directions.groupby("feature_name", sort=True):
            sources = sorted(part["direction_source"].unique())
            resolved = sorted({int(v) for v in part["resolved_direction"].tolist()})
            dir_summary_rows.append({
                "feature_name": feature_name,
                "catalog_direction": int(part["catalog_direction"].iloc[0]),
                "resolved_directions": ",".join(str(v) for v in resolved),
                "direction_source": ",".join(sources),
            })
    dir_summary = pd.DataFrame(dir_summary_rows)

    unknown = (
        fold_directions.loc[fold_directions["direction_source"].eq("train_rank_ic")]
        .copy()
        if not fold_directions.empty
        else pd.DataFrame()
    )
    if not unknown.empty:
        unknown_display = unknown[
            [c for c in ["fold_id", "test_year", "feature_name", "resolved_direction", "train_rank_ic"] if c in unknown.columns]
        ].copy()
        for column in unknown_display.columns:
            if pd.api.types.is_float_dtype(unknown_display[column]):
                unknown_display[column] = unknown_display[column].map(_format)
            elif column == "test_year":
                unknown_display[column] = unknown_display[column].astype(int).astype(str)
            elif column == "resolved_direction":
                unknown_display[column] = unknown_display[column].astype(int).astype(str)
    else:
        unknown_display = pd.DataFrame([{"说明": "全部宇宙因子均有目录方向，无需训练窗定号。"}])

    selection_block = ""
    if is_v2 and fold_selected is not None and not fold_selected.empty:
        per_fold = (
            fold_selected.loc[fold_selected["selected"].astype(bool)]
            .groupby(["fold_id", "test_year"], as_index=False)
            .agg(
                n_selected=("feature_name", "size"),
                selected_features=("feature_name", lambda s: ", ".join(sorted(map(str, s)))),
            )
            .sort_values("test_year")
        )
        per_fold_display = per_fold.copy()
        per_fold_display["test_year"] = per_fold_display["test_year"].astype(int).astype(str)
        per_fold_display["n_selected"] = per_fold_display["n_selected"].astype(int).astype(str)

        reason_counts = (
            fold_selected.groupby(["selection_reason"], as_index=False)
            .size()
            .rename(columns={"size": "记录数"})
            .sort_values("记录数", ascending=False)
        )
        selection_block = "\n".join([
            "## 折内训练窗筛选摘要",
            "",
            f"- `selection_mode = {selection_mode}`：仅用该折训练窗；**未**使用样本外 IC/Sharpe 挑因子。",
            f"- 剔弱阈值 `min_train_rank_ic = {config.get('min_train_rank_ic', 0.01)}`。",
            f"- 去重阈值 `max_pairwise_corr = {config.get('max_pairwise_corr', 0.7)}`（训练窗日均 |横截面相关|）。",
            "",
            "各折入选数量与名单：",
            "",
            _markdown(per_fold_display),
            "",
            "筛选原因计数（跨折汇总）：",
            "",
            _markdown(reason_counts),
            "",
        ])

    compare_block = ""
    if is_v2 and v1_summary:
        compare = pd.DataFrame([
            {
                "版本": "v1 全体92等权",
                "宇宙/入选": str(v1_summary.get("universe_size", 92)),
                "RankIC": _format(v1_summary.get("rank_ic_mean")),
                "RankIC胜率": _format(v1_summary.get("rank_ic_hit_rate")),
                "毛Sharpe": _format(v1_summary.get("gross_sharpe")),
                "净3bp Sharpe": _format(v1_summary.get("net_3bps_sharpe")),
                "净5bp Sharpe": _format(v1_summary.get("net_5bps_sharpe")),
            },
            {
                "版本": "v2 筛弱/去重等权",
                "宇宙/入选": f"{n_universe}先验 / 折均{_format(mean_selected, 1)}",
                "RankIC": _format(overall.get("rank_ic_mean")),
                "RankIC胜率": _format(overall.get("rank_ic_hit_rate")),
                "毛Sharpe": _format(overall.get("gross_sharpe")),
                "净3bp Sharpe": _format(overall.get("net_3bps_sharpe")),
                "净5bp Sharpe": _format(overall.get("net_5bps_sharpe")),
            },
        ])
        compare_block = "\n".join([
            "## 与 v1（全体等权）对照",
            "",
            "同协议、同 handoff；v2 仅改变先验宇宙与折内训练窗筛选。",
            "",
            _markdown(compare),
            "",
        ])

    handoff_block = ""
    if bundle_info:
        handoff_rows = pd.DataFrame([
            {"项目": "handoff目录", "值": bundle_info.get("handoff_directory", "")},
            {"项目": "特征文件", "值": bundle_info.get("feature_file", "")},
            {"项目": "注册表", "值": bundle_info.get("registry_file", "")},
            {"项目": "标签文件", "值": bundle_info.get("label_file", "")},
            {"项目": "折定义", "值": config.get("fold_file", "fold_definitions.csv")},
            {"项目": "注册特征数", "值": str(bundle_info.get("n_features", ""))},
            {"项目": "特征行数", "值": str(bundle_info.get("feature_rows", ""))},
            {"项目": "特征日期", "值": f"{bundle_info.get('feature_start', '')} — {bundle_info.get('feature_end', '')}"},
            {"项目": "品种数", "值": str(bundle_info.get("n_products", ""))},
        ])
        handoff_block = "\n".join([
            "## Handoff 数据接口",
            "",
            _markdown(handoff_rows),
            "",
        ])

    if is_v2:
        title = "# 等权多因子模型测试报告（筛弱/去重等权 v2）"
        blurb = [
            "> 本报告由 handoff 矩阵接口自动生成；模型为**先验 predictor + 折内训练窗剔弱/去相关后的等权合成**。",
            "> 筛选仅使用该折训练窗；**不**用样本外表现挑因子，不做子集枚举、不估计回归系数。",
            "> 2025 不参与方向选择或主评价。",
        ]
        model_bullets = [
            f"- 模型：`{model_version}` / `combination_scheme = all_equal` + `selection_mode = train_ic_corr`。",
            "- 先验宇宙：仅 `usage_type=predictor`；排除 quality_flag / risk / conditional。",
            f"- 折内剔弱：训练窗方向对齐后日均 RankIC ≥ `{config.get('min_train_rank_ic', 0.01)}`。",
            f"- 折内去重：按训练窗 RankIC 从高到低贪心纳入；与已入选因子日均 |相关| ≥ `{config.get('max_pairwise_corr', 0.7)}` 则跳过。",
            "- 合成：定向 → `cs_rank_centered` → 对入选且非缺失因子等权平均。",
        ]
        universe_line = (
            f"- usage_type 过滤：{config.get('candidate_usage_types')}\n"
            f"- 先验宇宙大小 N = **{n_universe}**（predictor-only）；折均入选 **{_format(mean_selected, 1)}**。\n"
            f"- 开发样本截止：{config.get('development_end', '2024-12-31')}"
        )
        boundary = [
            "1. 点时：因子在 T 日收盘后形成，信号仅可用于 T+1 开盘交易；切分、方向选择、筛选与评价均未使用测试段。",
            "2. 本模型是筛弱/去重后的等权多因子，**不是**单因子排行榜，也**不是** Ridge / OLS 加权结果。",
            "3. 入选名单允许折间变化；阈值写死在配置中，未对阈值做网格搜索。",
            f"4. 成本仅为单边 {cost_text} 基点敏感性，不得解读为已核验的真实手续费或冲击成本。",
            "5. 2025 未参与主评价；若另做补充样本外，须单独配置与目录。",
            "6. 未核验合约乘数、涨跌停、成交容量与真实交易成本，结果仅属研究回测。",
            "7. v1（92 全体等权）仍是无选择地板基准；本版用于检验去弱/去重是否提升稳健性。",
        ]
    else:
        title = "# 等权多因子模型测试报告（全体等权）"
        blurb = [
            "> 本报告由 handoff 矩阵接口自动生成；模型为**全体等权多因子合成**，不估计回归系数、不做 Top-K、不枚举子集。",
            "> 2025 不参与方向选择或主评价。",
        ]
        model_bullets = [
            "- 模型：方案 B / `combination_scheme = all_equal` —— 主宇宙内全部字段等权合成单一 `prediction`。",
            "- **未做**因子挑选、子集枚举、IC/Sharpe 加权或回归拟合。",
        ]
        universe_line = (
            f"- usage_type 过滤：{config.get('candidate_usage_types')}\n"
            f"- 宇宙大小 N = **{n_universe}**（「全体」= `ml_feature_registry.csv` 全部注册字段，与 Ridge / handoff 一致）。\n"
            f"- 开发样本截止：{config.get('development_end', '2024-12-31')}"
        )
        boundary = [
            "1. 点时：因子在 T 日收盘后形成，信号仅可用于 T+1 开盘交易；切分、方向选择与评价均未使用测试段。",
            "2. 本模型是全体等权多因子地板基准，**不是**单因子排行榜，也**不是** Ridge / OLS 加权结果。",
            "3. 弱因子与强因子同权；相关因子会重复计入同一信息——这是本版主动接受的代价。",
            f"4. 成本仅为单边 {cost_text} 基点敏感性，不得解读为已核验的真实手续费或冲击成本。",
            "5. 2025 未参与主评价；若另做补充样本外，须单独配置与目录，并标注为补充证据、非封存测试。",
            "6. 未核验合约乘数、涨跌停、成交容量与真实交易成本，结果仅属研究回测。",
            "7. 「全体」指 handoff 注册表全部字段（当前 92），与 Ridge 输入宇宙一致；弱信号与 quality_flag 同权是本地板基准主动接受的代价。",
        ]

    lines = [
        title,
        "",
        *blurb,
        "",
        handoff_block,
        "## 模型与标签选择",
        "",
        *model_bullets,
        f"- 分数变换：`{config['score_transform']}`"
        + ("（同日横截面中心化排名，映射到[-1,1]）" if config["score_transform"] == "cs_rank_centered" else "")
        + "。",
        f"- 缺失策略：`{config.get('missing_policy', 'mean_available')}`（对可用因子等权平均）。",
        f"- 原始收益标签：`{raw_return}`（同一真实合约开盘→开盘，持有 {holding} 个交易日）。",
        f"- 对照目标：`{target_column}`（仅作可选对照，不参与任何选择）。",
        f"- IC 与策略收益一律相对 `{raw_return}` 计算。",
        "- 外层验证：读取 `fold_definitions.csv` 的扩展窗口（2015 起点不变，2019–2024 逐年验证）；"
        f"训练净化规则为 `{config.get('label_end_column', 'label_end_date_5d')} < valid_start`。",
        "- 方向：读取因子目录 `expected_direction`；若为 0，仅在该折训练窗内用日均 RankIC 符号选向并冻结。",
        "",
        "## 策略定义",
        "",
        "- 每个调仓信号日对**合成分** `prediction` 排序，最高 20% 做多、最低 20% 做空。",
        "- 多头权重合计 0.5，空头权重合计 0.5；净敞口 0，总敞口 1；组内等权。",
        f"- T 收盘形成信号，T+1 开盘成交；持有 {holding} 日，每 {rebalance} 日调仓。",
        f"- 主成本口径单边 {base_bps} bp；敏感性报告单边 {cost_text} bp。",
        "- 调仓日有效品种数 < 10 则跳过组合；IC 要求当日有效样本 ≥ 5 且两侧有变异。",
        "",
        "## 候选宇宙",
        "",
        universe_line,
        "",
        _markdown(usage_counts),
        "",
        "目录期望方向分布（+1 / -1 / 0）：",
        "",
        _markdown(direction_counts),
        "",
        "完整先验宇宙名单：",
        "",
        _markdown(universe_display),
        "",
        selection_block,
        "## 方向规则与分折方向解析摘要",
        "",
        "各因子方向来源汇总：",
        "",
        _markdown(dir_summary if not dir_summary.empty else pd.DataFrame([{"说明": "无"}])),
        "",
        "训练窗定号明细（仅 `expected_direction=0` 的因子）：",
        "",
        _markdown(unknown_display),
        "",
        "## 总体样本外预测结果",
        "",
        _markdown(overall_pred),
        "",
        compare_block,
        f"## 策略总体结果（毛 / {cost_text} bp）",
        "",
        _markdown(strategy_table),
        "",
        "## 分年度扩展窗口结果",
        "",
        "分年度 RankIC、毛 Sharpe、净 3bp / 5bp Sharpe：",
        "",
        _markdown(yearly_display),
        "",
        "## 覆盖率与 n_factors_used 摘要",
        "",
        _markdown(coverage),
        "",
        "## 使用边界",
        "",
        *boundary,
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(
    config_path: str | Path,
    year_filter: list[int] | None = None,
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    root = config_path.parents[2]
    config = json.loads(config_path.read_text(encoding="utf-8"))

    if config.get("combination_scheme") != "all_equal":
        raise ValueError("equal_weight_multi requires combination_scheme=all_equal")

    selection_mode = str(config.get("selection_mode", "none") or "none")
    output_dir = root / config["output_directory"]
    report_path = root / config["report_file"]
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    bundle = load_handoff_bundle(root, config)
    candidates = select_candidates(bundle.registry, config)
    feature_names = candidates["feature_name"].tolist()
    missing = sorted(set(feature_names) - set(bundle.feature_names))
    if missing:
        raise ValueError(f"registered features missing from handoff matrix: {missing}")

    universe = candidates[
        [c for c in ["feature_name", "source_factor", "source_factor_id", "usage_type", "expected_direction", "category"] if c in candidates.columns]
    ].copy()
    universe = universe.rename(columns={"expected_direction": "catalog_direction"})
    universe.to_csv(output_dir / "universe_factors.csv", index=False)
    print(f"prior universe size N={len(feature_names)} selection_mode={selection_mode}")

    raw_return = config["raw_return_column"]
    target_column = config.get("model_target_column", "target_cs_rank_5d")
    label_end_column = config.get("label_end_column", "label_end_date_5d")

    # Join full handoff then keep universe features; preserve NaNs for missing_policy.
    merged = merge_features_and_labels(bundle, raw_return_column=raw_return)
    label_meta = [c for c in merged.columns if c.startswith("label_end_date_") or c == raw_return]
    data = merged[KEYS + ["sector"] + feature_names + label_meta].copy()

    development_end = pd.Timestamp(config.get("development_end", "2024-12-31"))
    data = data.loc[data["trade_date"].le(development_end)].copy()
    data[target_column] = cross_sectional_rank_target(data, raw_return)
    data = data.dropna(subset=[raw_return, target_column]).copy()

    folds = load_fold_definitions(bundle.directory, config)
    folds.to_csv(output_dir / "fold_definitions_used.csv", index=False)
    if year_filter is not None:
        allowed = {int(y) for y in year_filter}
        folds = folds.loc[folds["valid_start"].dt.year.isin(allowed)].copy()
        if folds.empty:
            raise ValueError(f"year_filter {year_filter} matched no fold definitions")

    bundle_info = handoff_summary(bundle)
    (output_dir / "handoff_summary.json").write_text(
        json.dumps(bundle_info, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    all_predictions: list[pd.DataFrame] = []
    all_strategy: list[pd.DataFrame] = []
    all_fold_metrics: list[dict[str, Any]] = []
    all_daily_ic: list[pd.DataFrame] = []
    all_directions: list[pd.DataFrame] = []
    all_selected: list[pd.DataFrame] = []

    for fold_row in folds.itertuples(index=False):
        fold_id = str(fold_row.fold_id)
        test_year = int(pd.Timestamp(fold_row.valid_start).year)
        outer_train, outer_test = split_fold(
            data,
            fold_row,
            target_column=target_column,
            label_end_column=label_end_column,
        )
        if outer_train.empty or outer_test.empty:
            continue

        predicted, strategy, fold, daily_ic, directions, selection_audit = evaluate_fold(
            outer_train,
            outer_test,
            candidates,
            feature_names,
            fold_id,
            test_year,
            config,
        )
        all_predictions.append(predicted)
        if not strategy.empty:
            all_strategy.append(strategy)
        all_fold_metrics.append(fold)
        if not daily_ic.empty:
            all_daily_ic.append(daily_ic)
        all_directions.append(directions)
        all_selected.append(selection_audit)
        print(
            f"{fold_id}: selected={fold.get('universe_size')}/{fold.get('prior_universe_size')} "
            f"rank_ic={fold.get('rank_ic_mean'):.4f} "
            f"gross_sharpe={fold.get('gross_sharpe'):.4f} "
            f"net_5bps_sharpe={fold.get('net_5bps_sharpe'):.4f} "
            f"strategy_days={fold.get('strategy_days')}"
        )

    if not all_predictions:
        raise RuntimeError("no OOS predictions were produced")

    predictions = pd.concat(all_predictions, ignore_index=True)
    strategy = pd.concat(all_strategy, ignore_index=True) if all_strategy else pd.DataFrame()
    fold_metrics = pd.DataFrame(all_fold_metrics).sort_values("test_year").reset_index(drop=True)
    daily_ic = pd.concat(all_daily_ic, ignore_index=True) if all_daily_ic else pd.DataFrame()
    fold_directions = (
        pd.concat(all_directions, ignore_index=True)
        if all_directions
        else pd.DataFrame()
    )
    fold_selected = (
        pd.concat(all_selected, ignore_index=True)
        if all_selected
        else pd.DataFrame()
    )

    overall = summarize_prediction_quality(predictions, strategy, config, daily_ic)
    overall.update(optional_horizon_rank_ics(predictions, config))

    pred_out_cols = [
        "trade_date",
        "product",
        "sector",
        "prediction",
        raw_return,
        target_column,
        "fold_id",
        "test_year",
        "n_factors_used",
    ]
    predictions[[c for c in pred_out_cols if c in predictions.columns]].to_csv(
        output_dir / "oos_predictions.csv.gz",
        index=False,
        compression="gzip",
    )
    strategy.to_csv(output_dir / "strategy_daily_returns.csv", index=False)
    fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    fold_metrics.to_csv(output_dir / "yearly_metrics.csv", index=False)
    fold_directions.to_csv(output_dir / "fold_directions.csv", index=False)
    fold_selected.to_csv(output_dir / "fold_selected_factors.csv", index=False)
    daily_ic.to_csv(output_dir / "daily_prediction_ic.csv", index=False)

    v1_summary: dict[str, Any] | None = None
    v1_summary_path = root / "modeling" / "equal_weight_multi_v1" / "outputs" / "run_summary.json"
    if selection_mode == "train_ic_corr" and v1_summary_path.exists():
        try:
            v1_summary = json.loads(v1_summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            v1_summary = None

    write_report(
        report_path,
        config,
        universe,
        overall,
        fold_metrics,
        fold_directions,
        bundle_info=bundle_info,
        fold_selected=fold_selected,
        v1_summary=v1_summary,
    )

    base_bps = float(config.get("base_cost_bps_one_way", 5))
    base_label = str(int(base_bps)) if float(base_bps).is_integer() else str(base_bps).replace(".", "p")
    mean_selected = (
        float(fold_metrics["universe_size"].mean())
        if "universe_size" in fold_metrics.columns and not fold_metrics.empty
        else float(len(feature_names))
    )
    if selection_mode == "train_ic_corr":
        note = (
            "equal-weight multi-factor after predictor prior + train-window "
            "RankIC filter and greedy correlation dedup; no OOS peeking, no regression weights"
        )
    else:
        note = (
            "all-equal multi-factor composite on configured prior universe; "
            "no factor selection, no subset search, no regression weights"
        )
    summary = {
        "model_version": config["model_version"],
        "combination_scheme": config.get("combination_scheme", "all_equal"),
        "selection_mode": selection_mode,
        "missing_policy": config.get("missing_policy", "mean_available"),
        "score_transform": config["score_transform"],
        "min_train_rank_ic": config.get("min_train_rank_ic"),
        "max_pairwise_corr": config.get("max_pairwise_corr"),
        "protocol": {
            "primary_target": raw_return,
            "split": "expanding_window_from_fold_definitions",
            "holding_days": config.get("holding_days", 5),
            "rebalance_frequency_days": config.get("rebalance_frequency_days", 5),
            "base_cost_bps_one_way": base_bps,
            "cost_scenarios_bps_one_way": config["one_way_cost_bps"],
        },
        "prior_universe_size": len(feature_names),
        "universe_size": len(feature_names),
        "mean_selected_universe_size": mean_selected,
        "features": feature_names,
        "folds": fold_metrics["fold_id"].tolist() if "fold_id" in fold_metrics.columns else [],
        "handoff": bundle_info,
        "oos_rows": int(len(predictions)),
        "prediction_days": int(overall.get("prediction_days", 0)),
        "strategy_days": int(overall.get("strategy_days", 0)),
        "pearson_ic_mean": overall.get("pearson_ic_mean"),
        "rank_ic_mean": overall.get("rank_ic_mean"),
        "rank_ic_hit_rate": overall.get("rank_ic_hit_rate"),
        "rank_icir": overall.get("rank_icir"),
        "gross_sharpe": overall.get("gross_sharpe"),
        f"net_{base_label}bps_sharpe": overall.get(f"net_{base_label}bps_sharpe"),
        "net_3bps_sharpe": overall.get("net_3bps_sharpe"),
        "n_factors_used_mean": overall.get("n_factors_used_mean"),
        "v1_comparison": {
            "rank_ic_mean": v1_summary.get("rank_ic_mean") if v1_summary else None,
            "gross_sharpe": v1_summary.get("gross_sharpe") if v1_summary else None,
            "net_5bps_sharpe": v1_summary.get("net_5bps_sharpe") if v1_summary else None,
        } if selection_mode == "train_ic_corr" else None,
        "report": str(report_path),
        "note": note,
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="等权多因子合成与扩展窗口样本外多空评价")
    parser.add_argument("--config", default="modeling/equal_weight_multi_v1/config.json")
    parser.add_argument(
        "--years",
        type=str,
        default=None,
        help="冒烟：逗号分隔测试年，例如 2019,2020",
    )
    args = parser.parse_args(argv)
    year_filter = None
    if args.years:
        year_filter = [int(part.strip()) for part in args.years.split(",") if part.strip()]
    print(json.dumps(run(args.config, year_filter=year_filter), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
