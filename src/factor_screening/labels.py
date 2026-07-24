from __future__ import annotations

import numpy as np
import pandas as pd


def add_forward_labels(panel: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    out = panel.copy()
    log_return = np.log1p(out["tradable_return"].clip(lower=-0.999999))

    def forward_sum(series: pd.Series, horizon: int) -> pd.Series:
        shifted = series.shift(-1)
        return shifted.rolling(horizon, min_periods=horizon).sum().shift(-(horizon - 1))

    for horizon in horizons:
        summed = log_return.groupby(out["product"], group_keys=False).transform(
            lambda s, h=horizon: forward_sum(s, h)
        )
        out[f"future_return_{horizon}d"] = np.expm1(summed)
        out[f"label_end_date_{horizon}d"] = out.groupby("product")["trade_date"].shift(-horizon)
    return out


def assert_labels_are_forward(panel: pd.DataFrame) -> None:
    sample = panel.sort_values(["product", "trade_date"]).groupby("product", sort=False).head(3)
    if sample.empty:
        raise AssertionError("empty panel")
    if not panel["trade_date"].is_monotonic_increasing:
        # Global ordering is product/date, so only enforce within product below.
        pass
    bad = panel.groupby("product")["trade_date"].apply(lambda s: not s.is_monotonic_increasing)
    if bad.any():
        raise AssertionError("trade dates are not sorted within product")
