from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

from .definitions import FACTOR_DEFINITIONS


def _safe_correlation(left: pd.Series, right: pd.Series, rank: bool) -> float:
    valid = left.notna() & right.notna()
    if valid.sum() < 3:
        return np.nan
    x = left[valid]
    y = right[valid]
    if x.nunique() < 2 or y.nunique() < 2:
        return np.nan
    if rank:
        x = x.rank(method="average")
        y = y.rank(method="average")
    return float(x.corr(y))


def _hac_mean_test(values: pd.Series, max_lag: int = 3) -> tuple[float, float]:
    array = values.dropna().to_numpy(dtype=float)
    count = len(array)
    if count < 3:
        return np.nan, np.nan
    centered = array - array.mean()
    long_run_variance = float(centered @ centered / count)
    for lag in range(1, min(max_lag, count - 1) + 1):
        weight = 1.0 - lag / (max_lag + 1.0)
        autocovariance = float(centered[lag:] @ centered[:-lag] / count)
        long_run_variance += 2.0 * weight * autocovariance
    standard_error = math.sqrt(max(long_run_variance, 0.0) / count)
    if standard_error == 0:
        return np.nan, np.nan
    t_stat = float(array.mean() / standard_error)
    p_value = float(2.0 * stats.t.sf(abs(t_stat), df=count - 1))
    return t_stat, p_value


def _performance_metrics(returns: pd.Series, periods_per_year: float) -> dict[str, float]:
    values = returns.dropna().astype(float)
    if values.empty:
        return {
            "factor_return": np.nan,
            "annual_return": np.nan,
            "annual_volatility": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
            "calmar": np.nan,
        }
    equity = (1.0 + values).cumprod()
    total_return = float(equity.iloc[-1] - 1.0)
    years = len(values) / periods_per_year
    annual_return = (
        float(equity.iloc[-1] ** (1.0 / years) - 1.0)
        if years > 0 and equity.iloc[-1] > 0
        else np.nan
    )
    annual_volatility = float(values.std(ddof=1) * math.sqrt(periods_per_year))
    sharpe = (
        float(values.mean() / values.std(ddof=1) * math.sqrt(periods_per_year))
        if values.std(ddof=1) > 0
        else np.nan
    )
    drawdown = equity / equity.cummax() - 1.0
    maximum_drawdown = float(drawdown.min())
    calmar = (
        float(annual_return / abs(maximum_drawdown))
        if pd.notna(annual_return) and maximum_drawdown < 0
        else np.nan
    )
    return {
        "factor_return": total_return,
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "sharpe": sharpe,
        "max_drawdown": maximum_drawdown,
        "calmar": calmar,
    }


def _benjamini_hochberg(values: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=values.index, dtype=float)
    valid = values.dropna().clip(0.0, 1.0).sort_values()
    if valid.empty:
        return result
    count = len(valid)
    ranks = np.arange(1, count + 1, dtype=float)
    adjusted = valid.to_numpy(dtype=float) * count / ranks
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result.loc[valid.index] = np.clip(adjusted, 0.0, 1.0)
    return result


def _factor_portfolio(
    data: pd.DataFrame,
    factor_id: str,
    orientation: float,
    long_fraction: float,
    short_fraction: float,
    cost_scenarios: list[int],
) -> tuple[pd.DataFrame, dict[str, float], pd.DataFrame]:
    if data[factor_id].notna().sum() == 0:
        return pd.DataFrame(), {}, pd.DataFrame()
    rows: list[dict[str, object]] = []
    quantile_rows: list[dict[str, object]] = []
    previous_weights: dict[str, float] = {}
    rebalance_dates = sorted(data["trade_date"].unique())[::5]
    for trade_date in rebalance_dates:
        cross_section = data.loc[
            data["trade_date"].eq(trade_date),
            ["product", factor_id, "future_return_5d"],
        ].dropna()
        if len(cross_section) < 8 or cross_section[factor_id].nunique() < 2:
            continue
        cross_section = cross_section.assign(
            oriented_signal=orientation * cross_section[factor_id]
        ).sort_values(["oriented_signal", "product"])
        side_count = max(
            1,
            min(
                int(math.floor(len(cross_section) * long_fraction)),
                int(math.floor(len(cross_section) * short_fraction)),
            ),
        )
        short_side = cross_section.head(side_count)
        long_side = cross_section.tail(side_count)
        weights = {
            **{product: -0.5 / side_count for product in short_side["product"]},
            **{product: 0.5 / side_count for product in long_side["product"]},
        }
        gross_return = float(
            sum(
                weights[row.product] * row.future_return_5d
                for row in pd.concat([short_side, long_side]).itertuples()
            )
        )
        products = set(previous_weights) | set(weights)
        turnover = float(
            sum(abs(weights.get(item, 0.0) - previous_weights.get(item, 0.0)) for item in products)
        )
        result: dict[str, object] = {
            "trade_date": trade_date,
            "factor_id": factor_id,
            "gross_return": gross_return,
            "turnover": turnover,
            "long_count": side_count,
            "short_count": side_count,
        }
        for cost in cost_scenarios:
            result[f"net_return_{cost}bps"] = gross_return - turnover * cost / 10_000.0
        rows.append(result)
        previous_weights = weights

        ranked = cross_section.copy()
        try:
            ranked["quantile"] = pd.qcut(
                ranked["oriented_signal"].rank(method="first"),
                5,
                labels=False,
            ) + 1
        except ValueError:
            continue
        for quantile, subset in ranked.groupby("quantile"):
            quantile_rows.append(
                {
                    "trade_date": trade_date,
                    "factor_id": factor_id,
                    "quantile": int(quantile),
                    "return_5d": float(subset["future_return_5d"].mean()),
                    "count": int(len(subset)),
                }
            )

    portfolio = pd.DataFrame(rows)
    quantiles = pd.DataFrame(quantile_rows)
    if portfolio.empty:
        return portfolio, {}, quantiles
    base_column = "net_return_5bps"
    metrics = _performance_metrics(portfolio[base_column], 252.0 / 5.0)
    metrics["turnover"] = float(portfolio["turnover"].mean())
    t_stat, p_value = _hac_mean_test(portfolio[base_column])
    metrics["t_stat"] = t_stat
    metrics["p_value"] = p_value
    for cost in cost_scenarios:
        cost_metrics = _performance_metrics(
            portfolio[f"net_return_{cost}bps"],
            252.0 / 5.0,
        )
        metrics[f"annual_return_{cost}bps"] = cost_metrics["annual_return"]
        metrics[f"sharpe_{cost}bps"] = cost_metrics["sharpe"]

    if not quantiles.empty:
        means = quantiles.groupby("quantile")["return_5d"].mean()
        metrics["monotonicity"] = _safe_correlation(
            pd.Series(means.index, index=means.index),
            means,
            rank=True,
        )
        metrics["group_return_spread"] = float(means.iloc[-1] - means.iloc[0])
    else:
        metrics["monotonicity"] = np.nan
        metrics["group_return_spread"] = np.nan
    return portfolio, metrics, quantiles


def screen_factors(
    raw_factors: pd.DataFrame,
    labelled_panel: pd.DataFrame,
    evaluation_start: pd.Timestamp,
    evaluation_end: pd.Timestamp,
    minimum_cross_section: int,
    long_fraction: float,
    short_fraction: float,
    cost_scenarios: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    label_columns = [
        "trade_date",
        "product",
        "future_return_1d",
        "future_return_5d",
        "label_end_date_5d",
        "future_return_20d",
    ]
    data = raw_factors.merge(
        labelled_panel[label_columns],
        on=["trade_date", "product"],
        how="left",
        validate="one_to_one",
    )
    data = data.loc[
        data["trade_date"].between(evaluation_start, evaluation_end)
        & data["label_end_date_5d"].le(evaluation_end)
    ].copy()
    definition_map = {item.factor_id: item for item in FACTOR_DEFINITIONS}
    metric_rows: list[dict[str, object]] = []
    yearly_rows: list[dict[str, object]] = []
    portfolio_frames: list[pd.DataFrame] = []
    quantile_frames: list[pd.DataFrame] = []

    for factor_id, definition in definition_map.items():
        orientation = -1.0 if definition.expected_direction == "negative" else 1.0
        daily_rows: list[dict[str, object]] = []
        if data[factor_id].notna().any():
            for trade_date, cross_section in data.groupby("trade_date", sort=True):
                valid = cross_section[[factor_id, "future_return_5d"]].dropna()
                if len(valid) < minimum_cross_section:
                    continue
                oriented = orientation * valid[factor_id]
                daily_rows.append(
                    {
                        "trade_date": trade_date,
                        "pearson_ic": _safe_correlation(
                            oriented,
                            valid["future_return_5d"],
                            rank=False,
                        ),
                        "rank_ic": _safe_correlation(
                            oriented,
                            valid["future_return_5d"],
                            rank=True,
                        ),
                    }
                )
        daily = pd.DataFrame(daily_rows)
        if daily.empty:
            ic_mean = rank_ic = ic_std = ic_ir = np.nan
            rank_ic_std = rank_ic_ir = hit_rate = np.nan
            p_high = p_low = rank_ic_t_stat = rank_ic_p_value = np.nan
        else:
            ic_mean = float(daily["pearson_ic"].mean())
            rank_ic = float(daily["rank_ic"].mean())
            ic_std = float(daily["pearson_ic"].std(ddof=1))
            ic_ir = float(ic_mean / ic_std) if ic_std > 0 else np.nan
            rank_ic_std = float(daily["rank_ic"].std(ddof=1))
            rank_ic_ir = (
                float(rank_ic / rank_ic_std)
                if rank_ic_std > 0
                else np.nan
            )
            hit_rate = float(daily["rank_ic"].gt(0).mean())
            p_high = float(daily["pearson_ic"].gt(0.02).mean())
            p_low = float(daily["pearson_ic"].lt(-0.02).mean())
            rank_ic_t_stat, rank_ic_p_value = _hac_mean_test(
                daily["rank_ic"],
                max_lag=5,
            )
            daily["year"] = daily["trade_date"].dt.year
            for year, subset in daily.groupby("year"):
                yearly_rows.append(
                    {
                        "factor_id": factor_id,
                        "year": int(year),
                        "ic_mean": float(subset["pearson_ic"].mean()),
                        "rank_ic": float(subset["rank_ic"].mean()),
                        "observations": int(len(subset)),
                    }
                )

        portfolio, performance, quantiles = _factor_portfolio(
            data,
            factor_id,
            orientation,
            long_fraction,
            short_fraction,
            cost_scenarios,
        )
        if not portfolio.empty:
            portfolio_frames.append(portfolio)
        if not quantiles.empty:
            quantile_frames.append(quantiles)
        metric_rows.append(
            {
                "factor_id": factor_id,
                "factor_name": definition.factor_name,
                "category": definition.category,
                "protocol_role": definition.protocol_role,
                "usage_type": definition.usage_type,
                "economic_meaning": definition.economic_meaning,
                "expected_direction": definition.expected_direction,
                "orientation": orientation,
                "coverage": float(data[factor_id].notna().mean()),
                "ic_observations": int(len(daily)),
                "ic_mean": ic_mean,
                "rank_ic": rank_ic,
                "ic_std": ic_std,
                "ic_ir": ic_ir,
                "rank_ic_std": rank_ic_std,
                "rank_ic_ir": rank_ic_ir,
                "rank_ic_hit_rate": hit_rate,
                "p_ic_gt_002": p_high,
                "p_ic_lt_neg002": p_low,
                "rank_ic_t_stat": rank_ic_t_stat,
                "rank_ic_p_value": rank_ic_p_value,
                **performance,
            }
        )
    metrics = pd.DataFrame(metric_rows)
    metrics["rank_ic_fdr_q"] = _benjamini_hochberg(
        metrics["rank_ic_p_value"]
    )
    metrics["portfolio_fdr_q"] = _benjamini_hochberg(metrics["p_value"])
    return (
        metrics,
        pd.DataFrame(yearly_rows),
        pd.concat(portfolio_frames, ignore_index=True) if portfolio_frames else pd.DataFrame(),
        pd.concat(quantile_frames, ignore_index=True) if quantile_frames else pd.DataFrame(),
    )


def classify_factors(
    factor_registry: pd.DataFrame,
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    result = factor_registry.merge(metrics, on="factor_id", how="left", suffixes=("", "_metric"))

    def classify(row: pd.Series) -> str:
        if not bool(row["available"]):
            return "F 数据不足无法判断"
        if row["usage_type"] in {"risk"}:
            return "D 风险变量"
        if row["usage_type"] in {"conditional", "quality_flag"}:
            return "C 条件变量"
        evidence = (
            float(row.get("rank_ic", np.nan)) >= 0.02
            if pd.notna(row.get("rank_ic"))
            else False
        )
        economic = (
            float(row.get("annual_return", np.nan)) > 0
            if pd.notna(row.get("annual_return"))
            else False
        )
        significant = (
            float(row.get("rank_ic_fdr_q", np.nan)) <= 0.10
            if pd.notna(row.get("rank_ic_fdr_q"))
            else False
        )
        if row["protocol_role"] == "core" and evidence and significant and economic:
            return "A 核心预测因子"
        if economic and (evidence or significant):
            return "B 辅助预测因子"
        return "E 暂不使用因子"

    result["screening_class"] = result.apply(classify, axis=1)
    result["enters_ml"] = result["available"].astype(bool)
    result["ml_inclusion_note"] = (
        "机器学习接口采用预注册与数据质量硬门；统计结果只作研究报告，不用于回看式选特征"
    )
    return result


def robustness_diagnostics(
    raw_factors: pd.DataFrame,
    labelled_panel: pd.DataFrame,
    evaluation_start: pd.Timestamp,
    evaluation_end: pd.Timestamp,
    minimum_cross_section: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    label_columns = [
        "trade_date",
        "product",
        "sector",
        "future_return_1d",
        "label_end_date_1d",
        "future_return_5d",
        "label_end_date_5d",
        "future_return_20d",
        "label_end_date_20d",
    ]
    data = raw_factors.drop(columns=["sector"]).merge(
        labelled_panel[label_columns],
        on=["trade_date", "product"],
        how="left",
        validate="one_to_one",
    )
    data = data.loc[data["trade_date"].between(evaluation_start, evaluation_end)].copy()
    definitions = {item.factor_id: item for item in FACTOR_DEFINITIONS}
    horizon_rows: list[dict[str, object]] = []
    product_rows: list[dict[str, object]] = []
    sector_rows: list[dict[str, object]] = []

    def rowwise_correlation(
        left: pd.DataFrame,
        right: pd.DataFrame,
        rank: bool,
        minimum: int,
    ) -> pd.Series:
        common_columns = sorted(set(left.columns) & set(right.columns))
        x = left.reindex(columns=common_columns)
        y = right.reindex(columns=common_columns)
        if rank:
            x = x.rank(axis=1, method="average")
            y = y.rank(axis=1, method="average")
        valid = x.notna() & y.notna()
        count = valid.sum(axis=1)
        x_mean = x.where(valid).mean(axis=1)
        y_mean = y.where(valid).mean(axis=1)
        x_centered = x.sub(x_mean, axis=0).where(valid)
        y_centered = y.sub(y_mean, axis=0).where(valid)
        numerator = (x_centered * y_centered).sum(axis=1)
        denominator = np.sqrt(
            x_centered.pow(2).sum(axis=1) * y_centered.pow(2).sum(axis=1)
        )
        return numerator.div(denominator.where(denominator.gt(0))).where(
            count.ge(minimum)
        )

    label_pivots = {
        horizon: data.assign(
            bounded_target=data[f"future_return_{horizon}d"].where(
                data[f"label_end_date_{horizon}d"].le(evaluation_end)
            )
        ).pivot(
            index="trade_date",
            columns="product",
            values="bounded_target",
        )
        for horizon in [1, 5, 20]
    }

    for factor_id, definition in definitions.items():
        orientation = -1.0 if definition.expected_direction == "negative" else 1.0
        oriented = orientation * data[factor_id]
        if oriented.notna().sum() == 0:
            for horizon in [1, 5, 20, "5d_signal_delayed_1d"]:
                horizon_rows.append(
                    {
                        "factor_id": factor_id,
                        "horizon_days": horizon,
                        "ic_mean": np.nan,
                        "rank_ic": np.nan,
                        "date_count": 0,
                    }
                )
            for product in sorted(data["product"].unique()):
                product_rows.append(
                    {
                        "factor_id": factor_id,
                        "product": product,
                        "time_series_ic": np.nan,
                        "time_series_rank_ic": np.nan,
                        "observations": 0,
                    }
                )
            for sector in sorted(data["sector"].unique()):
                sector_rows.append(
                    {
                        "factor_id": factor_id,
                        "sector": sector,
                        "rank_ic": np.nan,
                        "date_count": 0,
                    }
                )
            continue
        factor_pivot = data.assign(oriented_factor=oriented).pivot(
            index="trade_date",
            columns="product",
            values="oriented_factor",
        )
        delayed = oriented.groupby(data["product"], sort=False).shift(1)
        for horizon in [1, 5, 20]:
            daily_ic = rowwise_correlation(
                factor_pivot,
                label_pivots[horizon],
                rank=False,
                minimum=minimum_cross_section,
            )
            daily_rank_ic = rowwise_correlation(
                factor_pivot,
                label_pivots[horizon],
                rank=True,
                minimum=minimum_cross_section,
            )
            horizon_rows.append(
                {
                    "factor_id": factor_id,
                    "horizon_days": horizon,
                    "ic_mean": float(daily_ic.mean()),
                    "rank_ic": float(daily_rank_ic.mean()),
                    "date_count": int(daily_rank_ic.notna().sum()),
                }
            )
        delayed_pivot = data.assign(delayed_factor=delayed).pivot(
            index="trade_date",
            columns="product",
            values="delayed_factor",
        )
        delayed_rank_ic = rowwise_correlation(
            delayed_pivot,
            label_pivots[5],
            rank=True,
            minimum=minimum_cross_section,
        )
        horizon_rows.append(
            {
                "factor_id": factor_id,
                "horizon_days": "5d_signal_delayed_1d",
                "ic_mean": np.nan,
                "rank_ic": float(delayed_rank_ic.mean()),
                "date_count": int(delayed_rank_ic.notna().sum()),
            }
        )

        for product in factor_pivot.columns:
            left = factor_pivot[product]
            right = label_pivots[5][product]
            product_rows.append(
                {
                    "factor_id": factor_id,
                    "product": product,
                    "time_series_ic": _safe_correlation(
                        left,
                        right,
                        False,
                    ),
                    "time_series_rank_ic": _safe_correlation(
                        left,
                        right,
                        True,
                    ),
                    "observations": int((left.notna() & right.notna()).sum()),
                }
            )
        product_sector = data.drop_duplicates("product").set_index("product")["sector"]
        for sector in sorted(data["sector"].unique()):
            sector_products = product_sector.loc[
                product_sector.eq(sector)
            ].index.tolist()
            values = rowwise_correlation(
                factor_pivot.reindex(columns=sector_products),
                label_pivots[5].reindex(columns=sector_products),
                rank=True,
                minimum=3,
            )
            sector_rows.append(
                {
                    "factor_id": factor_id,
                    "sector": sector,
                    "rank_ic": float(values.mean()),
                    "date_count": int(values.notna().sum()),
                }
            )
    return (
        pd.DataFrame(horizon_rows),
        pd.DataFrame(product_rows),
        pd.DataFrame(sector_rows),
    )


def parameter_sensitivity_diagnostics(
    raw_factors: pd.DataFrame,
    labelled_panel: pd.DataFrame,
    evaluation_start: pd.Timestamp,
    evaluation_end: pd.Timestamp,
    minimum_cross_section: int,
) -> pd.DataFrame:
    base = labelled_panel[
        [
            "trade_date",
            "product",
            "product_return_1d",
            "open_interest",
            "future_return_5d",
            "label_end_date_5d",
        ]
    ].merge(
        raw_factors[
            [
                "trade_date",
                "product",
                "carry_annualized",
                "basis_fresh",
                "term_structure_slope",
            ]
        ],
        on=["trade_date", "product"],
        how="left",
        validate="one_to_one",
    )
    base = base.sort_values(["product", "trade_date"]).reset_index(drop=True)
    safe = base["product_return_1d"].where(base["product_return_1d"].gt(-1.0))
    base["log_return"] = np.log1p(safe)
    base["log_oi"] = np.log(base["open_interest"].where(base["open_interest"].gt(0)))
    products = base["product"]
    dates = base["trade_date"]
    signals: list[tuple[str, int, pd.Series, float]] = []

    for window in [40, 60, 80]:
        rolling_sum = (
            base.groupby("product", sort=False)["log_return"]
            .rolling(window, min_periods=window)
            .sum()
            .reset_index(level=0, drop=True)
            .sort_index()
        )
        rolling_std = (
            base.groupby("product", sort=False)["log_return"]
            .rolling(window, min_periods=window)
            .std()
            .reset_index(level=0, drop=True)
            .sort_index()
        )
        signals.append(
            (
                "tsmom",
                window,
                rolling_sum.div(
                    (rolling_std * np.sqrt(window)).where(rolling_std.gt(0))
                ),
                1.0,
            )
        )
    for window in [10, 20, 40]:
        signals.extend(
            [
                (
                    "carry_change",
                    window,
                    base["carry_annualized"].groupby(products, sort=False).diff(window),
                    1.0,
                ),
                (
                    "basis_momentum",
                    window,
                    base["basis_fresh"].groupby(products, sort=False).diff(window),
                    1.0,
                ),
                (
                    "short_reversal",
                    window,
                    -base.groupby("product", sort=False)["log_return"]
                    .rolling(window, min_periods=window)
                    .sum()
                    .reset_index(level=0, drop=True)
                    .sort_index(),
                    1.0,
                ),
                (
                    "realized_volatility",
                    window,
                    base.groupby("product", sort=False)["log_return"]
                    .rolling(window, min_periods=window)
                    .std()
                    .reset_index(level=0, drop=True)
                    .sort_index()
                    * np.sqrt(252.0),
                    1.0,
                ),
                (
                    "oi_growth",
                    window,
                    base["log_oi"].groupby(products, sort=False).diff(window),
                    1.0,
                ),
                (
                    "term_structure_momentum",
                    window,
                    base["term_structure_slope"].groupby(products, sort=False).diff(
                        window
                    ),
                    1.0,
                ),
            ]
        )

    evaluation = (
        base["trade_date"].between(evaluation_start, evaluation_end)
        & base["label_end_date_5d"].le(evaluation_end)
    )
    rows: list[dict[str, object]] = []
    for family, window, signal, orientation in signals:
        frame = pd.DataFrame(
            {
                "trade_date": dates,
                "signal": orientation * signal,
                "target": base["future_return_5d"],
            }
        ).loc[evaluation]
        daily_values: list[float] = []
        for _, subset in frame.groupby("trade_date", sort=False):
            valid = subset[["signal", "target"]].dropna()
            if len(valid) < minimum_cross_section:
                continue
            value = _safe_correlation(valid["signal"], valid["target"], rank=True)
            if pd.notna(value):
                daily_values.append(value)
        series = pd.Series(daily_values, dtype=float)
        t_stat, p_value = _hac_mean_test(series, max_lag=5)
        rows.append(
            {
                "factor_family": family,
                "lookback_window": window,
                "rank_ic": float(series.mean()) if not series.empty else np.nan,
                "rank_ic_std": float(series.std(ddof=1)) if len(series) > 1 else np.nan,
                "rank_ic_t_stat": t_stat,
                "rank_ic_p_value": p_value,
                "date_count": int(len(series)),
                "coverage": float(signal.loc[evaluation].notna().mean()),
                "diagnostic_only": True,
            }
        )
    result = pd.DataFrame(rows)
    result["rank_ic_fdr_q"] = _benjamini_hochberg(result["rank_ic_p_value"])
    return result
