"""Shared backtest and evaluation helpers for factor models."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

KEYS = ["trade_date", "product"]
DEFAULT_RETURN_COLUMN = "future_return_1d"


def cross_sectional_rank_target(frame: pd.DataFrame, return_column: str) -> pd.Series:
    """Map each date's valid returns to centered ranks in [-1, 1]."""
    grouped = frame.groupby("trade_date")[return_column]
    rank = grouped.rank(method="average")
    count = grouped.transform("count")
    denominator = (count - 1).replace(0, np.nan)
    return 2.0 * (rank - (count + 1.0) / 2.0) / denominator


def purge_last_dates(frame: pd.DataFrame, days: int) -> pd.DataFrame:
    if days <= 0 or frame.empty:
        return frame.copy()
    dates = np.sort(frame["trade_date"].dropna().unique())
    if len(dates) <= days:
        return frame.iloc[0:0].copy()
    return frame.loc[~frame["trade_date"].isin(dates[-days:])].copy()


def resolve_return_column(frame: pd.DataFrame, return_column: str | None = None) -> str:
    """Pick the evaluation return column present on ``frame``."""
    if return_column and return_column in frame.columns:
        return return_column
    if DEFAULT_RETURN_COLUMN in frame.columns:
        return DEFAULT_RETURN_COLUMN
    candidates = [c for c in frame.columns if str(c).startswith("future_return_")]
    if len(candidates) == 1:
        return candidates[0]
    raise KeyError(
        f"return column {return_column!r} not found; available={list(frame.columns)}"
    )


def daily_correlations(
    frame: pd.DataFrame,
    return_column: str | None = None,
) -> pd.DataFrame:
    """Compute daily Pearson / Rank IC between prediction and forward return."""
    ret_col = resolve_return_column(frame, return_column)
    rows: list[dict[str, Any]] = []
    for date, part in frame.groupby("trade_date", sort=True):
        valid = part[["prediction", ret_col]].dropna()
        if len(valid) < 5 or valid["prediction"].nunique() < 2 or valid[ret_col].nunique() < 2:
            continue
        rows.append({
            "trade_date": date,
            "pearson_ic": valid["prediction"].corr(valid[ret_col], method="pearson"),
            "rank_ic": valid["prediction"].corr(valid[ret_col], method="spearman"),
        })
    return pd.DataFrame(rows)


# Backward-compatible private alias used by older callers.
_daily_correlations = daily_correlations


def mean_daily_rank_ic(frame: pd.DataFrame, return_column: str | None = None) -> float:
    daily = daily_correlations(frame, return_column=return_column)
    return float(daily["rank_ic"].mean()) if not daily.empty else float("nan")


def construct_daily_portfolio(
    part: pd.DataFrame,
    long_fraction: float,
    return_column: str | None = None,
) -> dict[str, Any] | None:
    ret_col = resolve_return_column(part, return_column)
    valid = part[["prediction", ret_col]].dropna().sort_values("prediction")
    if len(valid) < 10:
        return None
    bucket = max(1, int(np.floor(len(valid) * long_fraction)))
    if bucket * 2 > len(valid):
        return None
    short = valid.head(bucket)
    long = valid.tail(bucket)
    long_return = float(long[ret_col].mean())
    short_asset_return = float(short[ret_col].mean())
    gross_return = 0.5 * long_return - 0.5 * short_asset_return
    return {
        "instruments": len(valid),
        "long_count": bucket,
        "short_count": bucket,
        "long_return": long_return,
        "short_asset_return": short_asset_return,
        "gross_return": gross_return,
        "gross_exposure": 1.0,
        "net_exposure": 0.0,
        # Conservative full reshuffle each rebalance / signal date.
        "round_trip_turnover": 2.0,
    }


def select_rebalance_dates(dates: np.ndarray, rebalance_frequency_days: int) -> set[Any]:
    """Pick every N-th unique trading date as a rebalance / signal date."""
    ordered = np.sort(pd.to_datetime(dates))
    if len(ordered) == 0:
        return set()
    step = max(1, int(rebalance_frequency_days))
    return set(ordered[::step])


def build_strategy_returns(predictions: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Build long-short strategy returns under the shared execution protocol.

    Defaults preserve the historical daily open→close engine used by older
    baselines. Protocol-aligned configs should set:

    - ``strategy_return_column`` / ``raw_return_column`` = ``future_return_5d``
    - ``holding_days`` = 5
    - ``rebalance_frequency_days`` = 5
    - ``one_way_cost_bps`` = [0, 3, 5, 10]
    """
    rows: list[dict[str, Any]] = []
    fraction = float(config.get("long_short_fraction", config.get("long_fraction", 0.2)))
    costs = [float(value) for value in config["one_way_cost_bps"]]
    return_column = (
        config.get("strategy_return_column")
        or config.get("raw_return_column")
        or DEFAULT_RETURN_COLUMN
    )
    holding_days = int(config.get("holding_days", 1))
    rebalance_days = int(config.get("rebalance_frequency_days", holding_days))
    rebalance_dates = select_rebalance_dates(
        predictions["trade_date"].dropna().unique(),
        rebalance_days,
    )

    for date, part in predictions.groupby("trade_date", sort=True):
        if pd.Timestamp(date) not in rebalance_dates:
            continue
        portfolio = construct_daily_portfolio(part, fraction, return_column=return_column)
        if portfolio is None:
            continue
        row = {"trade_date": date, **portfolio}
        for bps in costs:
            label = str(int(bps)) if float(bps).is_integer() else str(bps).replace(".", "p")
            row[f"net_{label}bps_return"] = (
                row["gross_return"] - row["round_trip_turnover"] * bps / 10_000.0
            )
        rows.append(row)
    return pd.DataFrame(rows)


def return_statistics(returns: pd.Series, annualization: float) -> dict[str, float]:
    clean = pd.Series(returns, dtype=float).dropna()
    ann = float(annualization)
    if clean.empty:
        return {
            "annual_return": np.nan,
            "annual_volatility": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
            "total_return": np.nan,
            "hit_rate": np.nan,
        }
    volatility = float(clean.std(ddof=1))
    equity = (1.0 + clean).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    return {
        "annual_return": float(clean.mean() * ann),
        "annual_volatility": float(volatility * np.sqrt(ann)),
        "sharpe": float(clean.mean() / volatility * np.sqrt(ann)) if volatility > 0 else np.nan,
        "max_drawdown": float(drawdown.min()),
        "total_return": float(equity.iloc[-1] - 1.0),
        "hit_rate": float(clean.gt(0).mean()),
    }
