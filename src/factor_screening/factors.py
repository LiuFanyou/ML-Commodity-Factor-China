from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import ProjectConfig


@dataclass(frozen=True)
class FeatureDefinition:
    feature_name: str
    source_factor_id: int
    source_factor: str
    transformation: str


def _gtransform(panel: pd.DataFrame, series: pd.Series, method: str, window: int, min_periods: int | None = None) -> pd.Series:
    minimum = min_periods or max(5, window // 2)
    grouped = series.groupby(panel["product"])
    rolling = grouped.rolling(window, min_periods=minimum)
    if method == "sum":
        result = rolling.sum()
    elif method == "mean":
        result = rolling.mean()
    elif method == "std":
        result = rolling.std(ddof=0)
    elif method == "skew":
        result = rolling.skew()
    else:
        raise ValueError(method)
    return result.reset_index(level=0, drop=True).sort_index()


def _lag(panel: pd.DataFrame, series: pd.Series, periods: int) -> pd.Series:
    return series.groupby(panel["product"]).shift(periods)


def _historical_z(panel: pd.DataFrame, series: pd.Series, window: int) -> pd.Series:
    lagged = _lag(panel, series, 1)
    mean = _gtransform(panel, lagged, "mean", window)
    std = _gtransform(panel, lagged, "std", window).replace(0, np.nan)
    return (series - mean) / std


def _rolling_corr(panel: pd.DataFrame, left: pd.Series, right: pd.Series, window: int) -> pd.Series:
    frame = pd.DataFrame({"product": panel["product"], "left": left, "right": right})
    result = frame.groupby("product", group_keys=False).apply(
        lambda x: x["left"].rolling(window, min_periods=max(10, window // 2)).corr(x["right"]),
        include_groups=False,
    )
    return result.reset_index(level=0, drop=True).sort_index()


def build_curve_features(market: pd.DataFrame, config: ProjectConfig) -> pd.DataFrame:
    curve_cfg = config.raw["curve"]
    valid = market.loc[
        market["basic_liquid"].fillna(False)
        & market["is_standard_delivery_contract"].fillna(False)
        & market["close"].gt(0)
        & market["days_to_delisting"].between(
            curve_cfg["minimum_days_to_delisting"],
            curve_cfg["maximum_days_to_delisting"],
        )
    ].copy()
    valid["tau"] = valid["days_to_delisting"] / 365.0
    valid["log_price"] = np.log(valid["close"])
    valid["weight"] = np.sqrt(valid["open_interest"].clip(lower=1.0))
    valid = valid.sort_values(["trade_date", "product", "days_to_delisting", "delivery_yyyymm"])
    valid["curve_rank"] = valid.groupby(["trade_date", "product"]).cumcount()

    keys = ["trade_date", "product"]
    first_three = valid.loc[valid["curve_rank"].lt(3)].pivot(index=keys, columns="curve_rank", values=["log_price", "tau"])
    first_three.columns = [f"{a}_{int(b)}" for a, b in first_three.columns]
    first_three = first_three.reset_index()
    dtau = first_three.get("tau_1") - first_three.get("tau_0")
    first_three["carry"] = (first_three.get("log_price_0") - first_three.get("log_price_1")) / dtau.replace(0, np.nan)
    first_three["curve_gap"] = first_three.get("log_price_0") - first_three.get("log_price_1")
    first_three["curve_curvature"] = (
        2 * first_three.get("log_price_1") - first_three.get("log_price_0") - first_three.get("log_price_2")
    )

    valid["wx"] = valid["weight"] * valid["tau"]
    valid["wy"] = valid["weight"] * valid["log_price"]
    valid["wxy"] = valid["weight"] * valid["tau"] * valid["log_price"]
    valid["wx2"] = valid["weight"] * valid["tau"] ** 2
    sums = valid.groupby(keys).agg(
        n_contracts=("ts_code", "nunique"),
        sw=("weight", "sum"),
        swx=("wx", "sum"),
        swy=("wy", "sum"),
        swxy=("wxy", "sum"),
        swx2=("wx2", "sum"),
    ).reset_index()
    numerator = sums["swxy"] - sums["swx"] * sums["swy"] / sums["sw"]
    denominator = sums["swx2"] - sums["swx"] ** 2 / sums["sw"]
    sums["term_structure_slope"] = -(numerator / denominator.replace(0, np.nan))
    sums.loc[sums["n_contracts"].lt(curve_cfg["minimum_contracts_for_slope"]), "term_structure_slope"] = np.nan
    out = sums[keys + ["n_contracts", "term_structure_slope"]].merge(
        first_three[keys + ["carry", "curve_gap", "curve_curvature"]], on=keys, how="left"
    )
    return out.sort_values(keys).reset_index(drop=True)


def _seasonality_three_year_month(panel: pd.DataFrame) -> pd.Series:
    temp = panel[["product", "trade_date", "signal_log_return"]].copy()
    temp["year"] = temp["trade_date"].dt.year
    temp["month"] = temp["trade_date"].dt.month
    monthly = temp.groupby(["product", "year", "month"], as_index=False)["signal_log_return"].sum()
    records: list[pd.DataFrame] = []
    for offset in (1, 2, 3):
        shifted = monthly.copy()
        shifted["year"] = shifted["year"] + offset
        shifted = shifted.rename(columns={"signal_log_return": f"lag_{offset}"})
        records.append(shifted)
    seasonal = monthly[["product", "year", "month"]]
    for shifted in records:
        seasonal = seasonal.merge(shifted, on=["product", "year", "month"], how="left")
    seasonal["seasonality_3y_month"] = seasonal[["lag_1", "lag_2", "lag_3"]].mean(axis=1, skipna=False)
    temp = temp.merge(seasonal[["product", "year", "month", "seasonality_3y_month"]], on=["product", "year", "month"], how="left")
    return temp["seasonality_3y_month"].set_axis(panel.index)


def build_factor_matrix(panel: pd.DataFrame, curve: pd.DataFrame, config: ProjectConfig) -> tuple[pd.DataFrame, list[FeatureDefinition]]:
    data = panel.merge(curve, on=["trade_date", "product"], how="left", validate="one_to_one")
    data = data.sort_values(["product", "trade_date"]).reset_index(drop=True)
    definitions: list[FeatureDefinition] = []

    def add(name: str, values: pd.Series, factor_id: int, source: str, transformation: str) -> None:
        data[name] = values.replace([np.inf, -np.inf], np.nan)
        definitions.append(FeatureDefinition(name, factor_id, source, transformation))

    log_price = data["research_log_price"]
    log_ret = data["signal_log_return"]
    abs_ret = log_ret.abs()
    log_oi = np.log(data["selected_open_interest"].where(data["selected_open_interest"].gt(0)))
    log_volume = np.log(data["selected_volume"].where(data["selected_volume"].gt(0)))

    momentum: dict[int, pd.Series] = {}
    efficiency: dict[int, pd.Series] = {}
    realized_vol: dict[int, pd.Series] = {}
    for window in (20, 60, 120):
        momentum[window] = log_price - _lag(data, log_price, window)
        realized_vol[window] = np.sqrt(_gtransform(data, log_ret.pow(2), "sum", window))
        tsmom = momentum[window] / realized_vol[window].replace(0, np.nan)
        add(f"tsmom_{window}", tsmom, 1, "时间序列动量", f"{window}日累计收益/已实现波动率")
        rank = momentum[window].groupby(data["trade_date"]).rank(pct=True) - 0.5
        add(f"csmom_{window}", rank, 2, "横截面动量", f"{window}日收益横截面百分位")
        efficiency[window] = momentum[window].abs() / _gtransform(data, abs_ret, "sum", window).replace(0, np.nan)
        add(f"trend_efficiency_{window}", efficiency[window], 3, "趋势效率", f"{window}日净路径/总路径")
        add(f"realized_vol_{window}", realized_vol[window], 24, "已实现波动率", f"{window}日平方收益和开方")
        add(f"momentum_quality_{window}", tsmom * efficiency[window], 41, "动量质量", f"tsmom_{window}×trend_efficiency_{window}")
        annual_vol = _gtransform(data, log_ret, "std", window) * np.sqrt(252)
        vmm = np.sign(momentum[window]) * 0.10 / annual_vol.replace(0, np.nan)
        add(f"vol_managed_momentum_{window}", vmm, 42, "波动率管理动量", f"方向×10%目标波动/{window}日年化波动")

    multi = sum(np.sign(momentum[w]) for w in (20, 60, 120)) / 3.0
    add("multi_horizon_trend", multi, 4, "多周期趋势一致性", "20/60/120日趋势方向均值")

    add("carry", data["carry"], 5, "Carry", "最近两个合格流动合约年化对数斜率")
    add("curve_curvature", data["curve_curvature"], 10, "期限结构曲率", "最近三个合格流动合约对数曲率")
    add("term_structure_slope", data["term_structure_slope"], 44, "期限结构斜率", "全部合格流动合约按持仓开方加权回归")
    for window in (5, 20, 60):
        carry_change = data["carry"] - _lag(data, data["carry"], window)
        slope_change = data["term_structure_slope"] - _lag(data, data["term_structure_slope"], window)
        gap_change = data["curve_gap"] - _lag(data, data["curve_gap"], window)
        add(f"carry_change_{window}", carry_change, 6, "Carry变化", f"Carry的{window}日变化")
        add(f"basis_momentum_{window}", gap_change, 9, "基差动量", f"近远月对数价差的{window}日变化")
        add(f"term_structure_momentum_{window}", slope_change, 45, "期限结构动量", f"曲线斜率的{window}日变化")
        if window in (5, 20):
            previous_change = _lag(data, slope_change, window)
            add(f"term_structure_acceleration_{window}", slope_change - previous_change, 46, "期限结构加速度", f"曲线斜率{window}日二阶差分")

    for window in (60, 120, 252):
        mean = _gtransform(data, log_price, "mean", window)
        std = _gtransform(data, log_price, "std", window).replace(0, np.nan)
        add(f"value_{window}", -(log_price - mean) / std, 11, "价值因子", f"相对{window}日滚动价格锚的反向标准化偏离")
    for window in (5, 20):
        add(f"reversal_{window}", -momentum.get(window, log_price - _lag(data, log_price, window)), 12, "短期反转", f"{window}日收益反向值")
        oi_growth = log_oi - _lag(data, log_oi, window)
        add(f"oi_growth_{window}", oi_growth, 18, "持仓量增长", f"持仓量对数{window}日变化")
    add("price_oi_20", momentum[20] * (log_oi - _lag(data, log_oi, 20)), 19, "价格—持仓量交互", "20日收益×20日持仓增长")

    for window in (20, 60):
        add(f"volume_shock_{window}", _historical_z(data, log_volume, window), 21, "成交量冲击", f"成交量相对前{window}日历史分布的标准分")
        oi_change_1d = log_oi - _lag(data, log_oi, 1)
        add(f"oi_surprise_{window}", _historical_z(data, oi_change_1d, window), 52, "持仓意外", f"单日持仓变化相对前{window}日历史分布的标准分")
    illiq_raw = log_ret.abs() / data["turnover_amount_raw"].replace(0, np.nan)
    add("amihud_20", _gtransform(data, illiq_raw, "mean", 20), 22, "Amihud非流动性", "20日单位成交额绝对收益均值")
    add("volume_oi_turnover", data["selected_volume"] / data["selected_open_interest"].replace(0, np.nan), 23, "成交量—持仓量换手率", "当日成交量/持仓量")
    for window in (20, 60):
        low_vol = -realized_vol[window].groupby(data["trade_date"]).rank(pct=True)
        add(f"low_vol_{window}", low_vol, 25, "低波动", f"{window}日波动率的负横截面排名")
    for window in (60, 120):
        add(f"realized_skew_{window}", _gtransform(data, log_ret, "skew", window), 26, "已实现偏度", f"{window}日收益偏度")

    add("seasonality_3y_month", _seasonality_three_year_month(data), 27, "季节性", "前三年相同月份累计收益均值")

    market_return = data.groupby("trade_date")["signal_log_return"].transform("mean")
    sector_return = data.groupby(["trade_date", "sector"])["signal_log_return"].transform("mean")
    residual_return = log_ret - sector_return
    add("idiosyncratic_vol_60", _gtransform(data, residual_return, "std", 60), 43, "特质波动", "剔除同板块日收益后的60日残差波动")
    sector_momentum = _gtransform(data, sector_return, "sum", 60)
    corr = _rolling_corr(data, log_ret, sector_return, 60)
    std_i = _gtransform(data, log_ret, "std", 60)
    std_s = _gtransform(data, sector_return, "std", 60)
    beta = corr * std_i / std_s.replace(0, np.nan)
    add("relative_value_residual_60", -(momentum[60] - beta * sector_momentum), 29, "相对价值残差", "60日收益剔除滚动板块暴露后取反")
    corr_short = _rolling_corr(data, log_ret, market_return, 20)
    corr_long = _rolling_corr(data, log_ret, market_return, 120)
    add("correlation_risk_20_120", corr_short - corr_long, 57, "相关性风险", "20日市场相关性减120日市场相关性")

    volume_change_20 = log_volume - _lag(data, log_volume, 20)
    add("volume_price_divergence_20", _historical_z(data, momentum[20], 120) - _historical_z(data, volume_change_20, 120), 53, "量价背离", "20日价格变化与成交量变化的历史标准分差")

    for left, right in config.raw["features"]["pre_registered_interactions"]:
        if left in data and right in data:
            name = f"interaction__{left}__{right}"
            add(name, data[left] * data[right], 54, "因子交互", f"预注册交互：{left}×{right}")

    key_columns = ["trade_date", "product", "sector"]
    factor_columns = [definition.feature_name for definition in definitions]
    return data[key_columns + factor_columns], definitions
