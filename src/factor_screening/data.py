from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import ProjectConfig


def _read_until(
    path: Path,
    date_col: str,
    end_date: str,
    usecols: list[str] | None = None,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    end = pd.Timestamp(end_date)
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=150_000, low_memory=False):
        chunk[date_col] = pd.to_datetime(chunk[date_col], errors="coerce")
        chunk = chunk.loc[chunk[date_col].le(end)]
        if not chunk.empty:
            parts.append(chunk)
    if not parts:
        return pd.DataFrame(columns=usecols or [])
    return pd.concat(parts, ignore_index=True)


def load_universe(config: ProjectConfig) -> pd.DataFrame:
    universe = pd.read_csv(config.data_root / "core30_universe.csv")
    return universe[["product", "sector", "exchange", "observed_first_date", "observed_last_date"]]


def load_panel(config: ProjectConfig, end_date: str) -> pd.DataFrame:
    returns_cols = [
        "product",
        "exchange",
        "trade_date",
        "return_close_to_close",
        "primary_return",
        "return_eligible",
        "roll_execution_today",
    ]
    selection_cols = [
        "trade_date",
        "product",
        "selected_ts_code",
        "selected_contract_code",
        "selected_delivery_yyyymm",
        "selected_close",
        "selected_volume",
        "selected_open_interest",
        "selection_available_date",
    ]
    market_cols = ["trade_date", "ts_code", "turnover_amount_raw"]

    returns = _read_until(
        config.tables / "research_returns_daily.csv.gz",
        "trade_date",
        end_date,
        returns_cols,
    )
    selections = _read_until(
        config.tables / "main_selection_daily.csv.gz",
        "trade_date",
        end_date,
        selection_cols,
    )
    market = _read_until(
        config.tables / "market_contract_daily.csv.gz",
        "trade_date",
        end_date,
        market_cols,
    )
    selected_amount = selections[["trade_date", "product", "selected_ts_code"]].merge(
        market,
        left_on=["trade_date", "selected_ts_code"],
        right_on=["trade_date", "ts_code"],
        how="left",
        validate="one_to_one",
    )[["trade_date", "product", "turnover_amount_raw"]]

    panel = returns.merge(selections, on=["trade_date", "product"], how="left", validate="one_to_one")
    panel = panel.merge(selected_amount, on=["trade_date", "product"], how="left", validate="one_to_one")
    panel = panel.merge(load_universe(config)[["product", "sector"]], on="product", how="left", validate="many_to_one")
    panel = panel.sort_values(["product", "trade_date"]).reset_index(drop=True)

    panel["signal_return"] = pd.to_numeric(panel["return_close_to_close"], errors="coerce")
    panel["tradable_return"] = pd.to_numeric(panel["primary_return"], errors="coerce")
    panel["selected_close"] = pd.to_numeric(panel["selected_close"], errors="coerce")
    panel["selected_volume"] = pd.to_numeric(panel["selected_volume"], errors="coerce")
    panel["selected_open_interest"] = pd.to_numeric(panel["selected_open_interest"], errors="coerce")
    panel["turnover_amount_raw"] = pd.to_numeric(panel["turnover_amount_raw"], errors="coerce")
    panel["signal_log_return"] = np.log1p(panel["signal_return"].clip(lower=-0.999999))
    panel["research_log_price"] = panel["signal_log_return"].fillna(0.0).groupby(panel["product"]).cumsum()
    return panel


def load_curve_market(config: ProjectConfig, end_date: str) -> pd.DataFrame:
    columns = [
        "trade_date",
        "product",
        "ts_code",
        "delivery_yyyymm",
        "days_to_delisting",
        "close",
        "open_interest",
        "basic_liquid",
        "is_standard_delivery_contract",
    ]
    market = _read_until(
        config.tables / "market_contract_daily.csv.gz",
        "trade_date",
        end_date,
        columns,
    )
    for column in ["days_to_delisting", "close", "open_interest", "delivery_yyyymm"]:
        market[column] = pd.to_numeric(market[column], errors="coerce")
    return market


def data_availability_summary(config: ProjectConfig) -> pd.DataFrame:
    rows = [
        ("价格/OHLC", "market_contract_daily", True, "2015-2025", "严格点时可用"),
        ("成交量", "market_contract_daily", True, "2015-2025", "严格点时可用"),
        ("成交额", "market_contract_daily", True, "2015-2025", "原始单位保留，适合品种内标准化"),
        ("持仓量", "market_contract_daily", True, "2015-2025", "严格点时可用"),
        ("全合约期限结构", "market_contract_daily", True, "2015-2025", "动态流动性准入"),
        ("主力选择", "main_selection_daily", True, "2015-2025", "当日收盘选择，下一交易日可用"),
        ("可交易基础收益", "research_returns_daily", True, "2015-2025", "前日选择合约的次日开收收益"),
        ("真实现货价格", "无", False, "无", "旧现货文件实为会员成本"),
        ("新仓单", "warehouse_receipts_daily_new", False, "2015-2025", "真实公布时间未确认"),
        ("新库存/可用量", "warehouse_inventory_periodic_new", False, "2015-2025", "公布时间及字段语义未确认"),
        ("旧库存", "legacy_inventory_weekly", False, "2020-2025", "仅有假定可用日，单位和语义部分未确认"),
        ("旧仓单", "legacy_warehouse_receipts_weekly", False, "2020-2025", "覆盖8个品种且仅有假定可用日"),
        ("商业交易者分类持仓", "无", False, "无", "会员成本不能替代持仓份额"),
        ("产业链配比/成本", "无", False, "无", "缺少确认后的产业图、配比和成本"),
        ("手续费与合约乘数", "contract_rules/metadata", False, "2015-2025", "手续费单位及乘数未完成核验"),
    ]
    return pd.DataFrame(rows, columns=["data_type", "source", "strict_point_in_time_usable", "coverage", "note"])
