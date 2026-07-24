from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ProjectConfig
from .definitions import FACTOR_DEFINITIONS


def _rolling(
    frame: pd.DataFrame,
    column: str,
    window: int,
    operation: str,
    minimum: int | None = None,
) -> pd.Series:
    grouped = frame.groupby("product", sort=False)[column]
    roller = grouped.rolling(window, min_periods=minimum or window)
    result = getattr(roller, operation)()
    return result.reset_index(level=0, drop=True).sort_index()


def _prior_rolling_z(
    values: pd.Series,
    products: pd.Series,
    window: int,
    minimum: int,
) -> pd.Series:
    lagged = values.groupby(products, sort=False).shift(1)
    temporary = pd.DataFrame({"product": products, "lagged": lagged})
    mean = _rolling(temporary, "lagged", window, "mean", minimum)
    standard_deviation = _rolling(temporary, "lagged", window, "std", minimum)
    return (values - mean).div(standard_deviation.where(standard_deviation.gt(0)))


def _cross_sectional_z(values: pd.Series, dates: pd.Series) -> pd.Series:
    grouped = values.groupby(dates, sort=False)
    median = grouped.transform("median")
    absolute_deviation = (values - median).abs()
    mad = absolute_deviation.groupby(dates, sort=False).transform("median")
    scale = 1.4826 * mad
    scale = scale.where(scale.gt(0), grouped.transform("std"))
    return (values - median).div(scale.where(scale.gt(0)))


def _seasonal_prior_z(
    values: pd.Series,
    products: pd.Series,
    dates: pd.Series,
) -> pd.Series:
    iso_week = dates.dt.isocalendar().week.astype(int)
    temporary = pd.DataFrame(
        {"product": products, "iso_week": iso_week, "value": values}
    )

    def transform(group: pd.Series) -> pd.Series:
        prior = group.shift(1)
        mean = prior.expanding(min_periods=5).mean()
        standard_deviation = prior.expanding(min_periods=5).std()
        return (group - mean).div(standard_deviation.where(standard_deviation.gt(0)))

    return temporary.groupby(["product", "iso_week"], sort=False)["value"].transform(
        transform
    )


def _available_fundamental(
    frame: pd.DataFrame,
    value_column: str,
    observation_column: str,
    maximum_age: int,
    lag_trading_days: int,
    invalid_column: str | None = None,
) -> tuple[pd.Series, pd.Series]:
    grouped = frame.groupby("product", sort=False)
    values = grouped[value_column].shift(lag_trading_days)
    observations = grouped[observation_column].shift(lag_trading_days)
    age = (frame["trade_date"] - observations).dt.days
    valid = (
        values.notna()
        & observations.notna()
        & observations.lt(frame["trade_date"])
        & age.between(0, maximum_age)
    )
    if invalid_column is not None:
        invalid = grouped[invalid_column].shift(lag_trading_days).fillna(0).astype(bool)
        valid &= ~invalid
    return values.where(valid), observations.where(valid)


def _event_innovation_z(
    values: pd.Series,
    observations: pd.Series,
    products: pd.Series,
    window: int = 60,
    minimum: int = 20,
) -> pd.Series:
    changed = observations.ne(observations.groupby(products, sort=False).shift(1))
    innovation = _prior_rolling_z(values, products, window, minimum)
    return innovation.where(changed & observations.notna())


def _idiosyncratic_volatility(
    returns: pd.Series,
    market_returns: pd.Series,
    sector_returns: pd.Series,
    products: pd.Series,
    window: int = 60,
) -> pd.Series:
    output = pd.Series(np.nan, index=returns.index, dtype=float)
    for indices in products.groupby(products, sort=False).groups.values():
        ordered_indices = list(indices)
        y = returns.loc[ordered_indices].to_numpy(dtype=float)
        market = market_returns.loc[ordered_indices].to_numpy(dtype=float)
        sector = sector_returns.loc[ordered_indices].to_numpy(dtype=float)
        values = np.full(len(ordered_indices), np.nan, dtype=float)
        for end in range(window - 1, len(ordered_indices)):
            start = end - window + 1
            y_window = y[start : end + 1]
            x_window = np.column_stack(
                [
                    np.ones(window),
                    market[start : end + 1],
                    sector[start : end + 1],
                ]
            )
            valid = np.isfinite(y_window) & np.isfinite(x_window).all(axis=1)
            if valid.sum() < int(window * 0.8):
                continue
            coefficients, *_ = np.linalg.lstsq(
                x_window[valid],
                y_window[valid],
                rcond=None,
            )
            residual = y_window[valid] - x_window[valid] @ coefficients
            values[end] = residual.std(ddof=1) * np.sqrt(252.0)
        output.loc[ordered_indices] = values
    return output


def _rolling_product_market_correlation(
    returns: pd.Series,
    market_returns: pd.Series,
    products: pd.Series,
    window: int,
) -> pd.Series:
    output = pd.Series(np.nan, index=returns.index, dtype=float)
    for indices in products.groupby(products, sort=False).groups.values():
        ordered_indices = list(indices)
        left = returns.loc[ordered_indices]
        right = market_returns.loc[ordered_indices]
        output.loc[ordered_indices] = (
            left.rolling(window, min_periods=window)
            .corr(right)
            .to_numpy()
        )
    return output


def _aggregate_factor_momentum(
    factors: pd.DataFrame,
    daily_returns: pd.Series,
    products: pd.Series,
    dates: pd.Series,
) -> pd.Series:
    base = [
        "tsmom_60",
        "carry_annualized",
        "short_reversal_5",
        "sector_relative_value_60",
    ]
    factor_pnl: list[pd.Series] = []
    for name in base:
        lagged = factors[name].groupby(products, sort=False).shift(1)
        ranks = lagged.groupby(dates, sort=False).rank(pct=True)
        centered = ranks - 0.5
        normalizer = centered.abs().groupby(dates, sort=False).transform("sum")
        weights = centered.div(normalizer.where(normalizer.gt(0)))
        pnl = (weights * daily_returns).groupby(dates, sort=False).sum(min_count=8)
        factor_pnl.append(pnl)
    aggregate = pd.concat(factor_pnl, axis=1).mean(axis=1, skipna=False)
    momentum = aggregate.rolling(20, min_periods=10).sum()
    return dates.map(momentum)


def build_factors(
    panel: pd.DataFrame,
    config: ProjectConfig,
) -> pd.DataFrame:
    out = panel[["trade_date", "product", "sector"]].copy()
    work = panel.copy()
    safe_return = work["product_return_1d"].where(work["product_return_1d"].gt(-1.0))
    work["log_return"] = np.log1p(safe_return)
    products = work["product"]
    dates = work["trade_date"]
    point_in_time = config.raw["data"]["point_in_time"]
    lag = int(point_in_time["release_lag_trading_days"])

    cumulative_log_returns: dict[int, pd.Series] = {}
    return_volatility: dict[int, pd.Series] = {}
    for window in [5, 20, 60, 120, 252]:
        cumulative_log_returns[window] = _rolling(work, "log_return", window, "sum")
        return_volatility[window] = _rolling(work, "log_return", window, "std")

    out["tsmom_60"] = cumulative_log_returns[60].div(
        (return_volatility[60] * np.sqrt(60)).where(return_volatility[60].gt(0))
    )
    out["csmom_60"] = cumulative_log_returns[60].groupby(dates, sort=False).rank(
        pct=True
    )
    absolute_path = _rolling(
        work.assign(abs_log_return=work["log_return"].abs()),
        "abs_log_return",
        60,
        "sum",
    )
    out["trend_efficiency_60"] = cumulative_log_returns[60].abs().div(
        absolute_path.where(absolute_path.gt(0))
    )
    out["multi_horizon_trend"] = pd.concat(
        [
            np.sign(cumulative_log_returns[20]),
            np.sign(cumulative_log_returns[60]),
            np.sign(cumulative_log_returns[120]),
        ],
        axis=1,
    ).mean(axis=1, skipna=False)

    out["carry_annualized"] = work["carry_annualized"]
    out["carry_change_20"] = out["carry_annualized"] - out.groupby(
        "product", sort=False
    )["carry_annualized"].shift(20)

    fresh_spot, spot_observation = _available_fundamental(
        work,
        "spot_price",
        "spot_observation_date",
        int(point_in_time["spot_max_age_calendar_days"]),
        lag,
    )
    fresh_spot = fresh_spot.where(
        ~work["product"].isin(config.raw["data"]["spot_unit_excluded_products"])
    )
    spot_observation = spot_observation.where(fresh_spot.notna())
    out["basis_fresh"] = (
        (fresh_spot - work["close"]) / work["close"].where(work["close"].gt(0))
    )
    basis_change = out["basis_fresh"].groupby(products, sort=False).diff(1)
    out["basis_shock_20"] = _prior_rolling_z(
        basis_change,
        products,
        20,
        10,
    )
    out["basis_momentum_20"] = out["basis_fresh"].groupby(
        products, sort=False
    ).diff(20)
    out["curve_curvature"] = work["curve_curvature"]

    log_price = np.log(work["close"].where(work["close"].gt(0)))
    prior_log_price = log_price.groupby(products, sort=False).shift(1)
    price_temporary = pd.DataFrame({"product": products, "value": prior_log_price})
    value_mean = _rolling(price_temporary, "value", 252, "mean", 126)
    value_std = _rolling(price_temporary, "value", 252, "std", 126)
    out["value_252"] = -(log_price - value_mean).div(value_std.where(value_std.gt(0)))
    out["short_reversal_5"] = -cumulative_log_returns[5]

    fresh_stock, stock_observation = _available_fundamental(
        work,
        "stock_current",
        "stock_observation_date",
        int(point_in_time["stock_max_age_calendar_days"]),
        lag,
    )
    stock_seasonal_z = _seasonal_prior_z(fresh_stock, products, dates)
    out["inventory_scarcity"] = -stock_seasonal_z
    stock_delta_5 = fresh_stock.groupby(products, sort=False).diff(5)
    out["inventory_change_5"] = -stock_delta_5
    out["inventory_surprise"] = -_event_innovation_z(
        fresh_stock,
        stock_observation,
        products,
    )

    fresh_warehouse, warehouse_observation = _available_fundamental(
        work,
        "warehouse_receipt_current",
        "warehouse_receipt_observation_date",
        int(point_in_time["warehouse_max_age_calendar_days"]),
        lag,
        invalid_column="warehouse_receipt_conflict",
    )
    warehouse_seasonal_z = _seasonal_prior_z(fresh_warehouse, products, dates)
    out["warehouse_tightness"] = -warehouse_seasonal_z
    basis_z = _cross_sectional_z(out["basis_fresh"], dates)
    carry_z = _cross_sectional_z(out["carry_annualized"], dates)
    component_frame = pd.concat(
        [out["inventory_scarcity"], out["warehouse_tightness"], basis_z, carry_z],
        axis=1,
    )
    out["tightness_composite"] = component_frame.mean(axis=1, skipna=True).where(
        component_frame.notna().sum(axis=1).ge(3)
    )

    log_oi = np.log(work["open_interest"].where(work["open_interest"].gt(0)))
    out["oi_growth_20"] = log_oi.groupby(products, sort=False).diff(20)
    out["price_oi_interaction_20"] = (
        cumulative_log_returns[20] * out["oi_growth_20"]
    )
    log_volume = np.log(work["volume"].where(work["volume"].gt(0)))
    out["volume_shock_20"] = _prior_rolling_z(log_volume, products, 20, 10)
    work["amihud_raw"] = (
        safe_return.abs()
        .div(work["amount"].where(work["amount"].gt(0)))
        .mul(1_000_000.0)
    )
    out["amihud_20"] = _rolling(work, "amihud_raw", 20, "mean")
    out["turnover_oi"] = work["volume"].div(
        work["open_interest"].where(work["open_interest"].gt(0))
    )
    out["realized_volatility_20"] = return_volatility[20] * np.sqrt(252.0)
    out["low_volatility_20"] = -out["realized_volatility_20"].groupby(
        dates, sort=False
    ).rank(pct=True)
    out["realized_skewness_60"] = _rolling(work, "log_return", 60, "skew")

    trailing_return_5 = np.expm1(cumulative_log_returns[5])
    out["seasonality_3y"] = pd.concat(
        [
            trailing_return_5.groupby(products, sort=False).shift(252),
            trailing_return_5.groupby(products, sort=False).shift(504),
            trailing_return_5.groupby(products, sort=False).shift(756),
        ],
        axis=1,
    ).mean(axis=1, skipna=False)
    sector_return_60 = cumulative_log_returns[60].groupby(
        [dates, work["sector"]], sort=False
    ).transform("mean")
    out["sector_relative_value_60"] = cumulative_log_returns[60] - sector_return_60

    information_components = pd.concat(
        [out["inventory_scarcity"], out["warehouse_tightness"], basis_z, carry_z],
        axis=1,
    )
    information_mean = information_components.mean(axis=1, skipna=True)
    information_agreement = (
        np.sign(information_components).mean(axis=1, skipna=True).abs()
    )
    enough_information = information_components.notna().sum(axis=1).ge(3)
    out["information_consistency"] = (
        information_mean * information_agreement
    ).where(enough_information)
    out["information_divergence"] = (
        _cross_sectional_z(out["tightness_composite"], dates)
        - _cross_sectional_z(out["tsmom_60"], dates)
    )
    out["warehouse_surprise"] = -_event_innovation_z(
        fresh_warehouse,
        warehouse_observation,
        products,
    )
    out["momentum_quality_60"] = out["tsmom_60"] * out["trend_efficiency_60"]
    out["vol_managed_momentum"] = np.sign(cumulative_log_returns[60]).div(
        out["realized_volatility_20"].where(out["realized_volatility_20"].gt(0))
    )

    market_daily_return = work["log_return"].groupby(dates, sort=False).transform("mean")
    sector_daily_return = work["log_return"].groupby(
        [dates, work["sector"]], sort=False
    ).transform("mean")
    out["idiosyncratic_volatility_60"] = _idiosyncratic_volatility(
        work["log_return"],
        market_daily_return,
        sector_daily_return,
        products,
    )
    out["term_structure_slope"] = work["term_structure_slope"]
    out["term_structure_momentum"] = out["term_structure_slope"].groupby(
        products, sort=False
    ).diff(20)
    out["term_structure_acceleration"] = out[
        "term_structure_momentum"
    ].groupby(products, sort=False).diff(20)
    out["basis_surprise"] = _event_innovation_z(
        out["basis_fresh"],
        spot_observation,
        products,
    )
    out["inventory_acceleration"] = -(
        stock_delta_5 - stock_delta_5.groupby(products, sort=False).shift(5)
    )
    out["inventory_price_divergence"] = (
        _cross_sectional_z(-stock_delta_5, dates)
        - _cross_sectional_z(cumulative_log_returns[20], dates)
    )
    oi_delta_1 = log_oi.groupby(products, sort=False).diff(1)
    out["oi_surprise_60"] = _prior_rolling_z(oi_delta_1, products, 60, 20)
    volume_change_20 = log_volume.groupby(products, sort=False).diff(20)
    out["volume_price_divergence_20"] = (
        _cross_sectional_z(cumulative_log_returns[20], dates)
        - _cross_sectional_z(volume_change_20, dates)
    )
    out["momentum_carry_interaction"] = (
        _cross_sectional_z(out["tsmom_60"], dates)
        * _cross_sectional_z(out["carry_annualized"], dates)
    )
    tightness_z = _cross_sectional_z(out["tightness_composite"], dates)
    out["nonlinear_tightness"] = np.sign(tightness_z) * tightness_z.abs().pow(2)

    corr_20 = _rolling_product_market_correlation(
        work["log_return"],
        market_daily_return,
        products,
        20,
    )
    corr_120 = _rolling_product_market_correlation(
        work["log_return"],
        market_daily_return,
        products,
        120,
    )
    out["correlation_risk_20_120"] = corr_20 - corr_120

    event_news = pd.concat(
        [
            out["inventory_surprise"],
            out["warehouse_surprise"],
            out["basis_surprise"],
        ],
        axis=1,
    )
    out["fundamental_composite_news"] = event_news.mean(
        axis=1,
        skipna=True,
    ).where(event_news.notna().sum(axis=1).ge(2))
    out["fundamental_underreaction"] = (
        out["fundamental_composite_news"]
        - _cross_sectional_z(work["log_return"], dates)
    ).where(out["fundamental_composite_news"].notna())
    out["factor_momentum"] = _aggregate_factor_momentum(
        out,
        safe_return,
        products,
        dates,
    )

    for definition in FACTOR_DEFINITIONS:
        if definition.implementation_status == "unavailable":
            out[definition.factor_id] = np.nan

    expected = {definition.factor_id for definition in FACTOR_DEFINITIONS}
    missing = sorted(expected - set(out.columns))
    if missing:
        raise RuntimeError(f"factor implementation missing: {missing}")
    if len(expected) != 57:
        raise AssertionError("factor implementation registry must contain 57 factors")
    return out
