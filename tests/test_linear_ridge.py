from __future__ import annotations

import numpy as np
import pandas as pd

from factor_modeling.common import (
    build_strategy_returns,
    construct_daily_portfolio,
    cross_sectional_rank_target,
    purge_last_dates,
    select_rebalance_dates,
)
from factor_modeling.handoff import apply_label_end_purge, split_fold



def test_cross_sectional_rank_target_is_centered_and_bounded() -> None:
    frame = pd.DataFrame({
        "trade_date": pd.to_datetime(["2024-01-02"] * 3 + ["2024-01-03"] * 3),
        "future_return_5d": [0.01, -0.02, 0.00, 0.03, 0.02, -0.01],
    })
    target = cross_sectional_rank_target(frame, "future_return_5d")
    assert np.allclose(target.groupby(frame["trade_date"]).mean(), 0.0)
    assert target.min() == -1.0
    assert target.max() == 1.0


def test_purge_removes_last_unique_training_date() -> None:
    frame = pd.DataFrame({
        "trade_date": pd.to_datetime(["2023-12-27", "2023-12-28", "2023-12-29"]),
        "product": ["A", "A", "A"],
    })
    purged = purge_last_dates(frame, 1)
    assert purged["trade_date"].max() == pd.Timestamp("2023-12-28")


def test_daily_portfolio_is_market_neutral_and_equal_sided() -> None:
    frame = pd.DataFrame({
        "prediction": np.arange(10, dtype=float),
        "future_return_5d": np.linspace(-0.01, 0.01, 10),
    })
    result = construct_daily_portfolio(frame, 0.2, return_column="future_return_5d")
    assert result is not None
    assert result["long_count"] == 2
    assert result["short_count"] == 2
    assert result["gross_exposure"] == 1.0
    assert result["net_exposure"] == 0.0
    assert result["round_trip_turnover"] == 2.0
    assert result["gross_return"] > 0


def test_label_end_purge_keeps_only_non_overlapping_labels() -> None:
    frame = pd.DataFrame({
        "trade_date": pd.to_datetime(["2018-12-20", "2018-12-24", "2018-12-28"]),
        "label_end_date_5d": pd.to_datetime(["2018-12-27", "2018-12-31", "2019-01-07"]),
        "product": ["A", "A", "A"],
    })
    purged = apply_label_end_purge(
        frame,
        pd.Timestamp("2019-01-01"),
        label_end_column="label_end_date_5d",
    )
    # 2018-12-28 ends on 2019-01-07 and overlaps validation; earlier rows stay.
    assert list(purged["trade_date"]) == [
        pd.Timestamp("2018-12-20"),
        pd.Timestamp("2018-12-24"),
    ]


def test_split_fold_matches_expanding_window_contract() -> None:
    data = pd.DataFrame({
        "trade_date": pd.to_datetime(
            ["2018-06-01", "2018-12-20", "2019-06-01", "2020-06-01"]
        ),
        "product": ["A"] * 4,
        "target_cs_rank_5d": [0.1, 0.2, 0.3, 0.4],
        "label_end_date_5d": pd.to_datetime(
            ["2018-06-08", "2018-12-27", "2019-06-08", "2020-06-08"]
        ),
    })
    fold = pd.Series({
        "fold_id": "fold_2019",
        "train_start": pd.Timestamp("2015-01-05"),
        "train_end": pd.Timestamp("2018-12-31"),
        "valid_start": pd.Timestamp("2019-01-01"),
        "valid_end": pd.Timestamp("2019-12-31"),
        "purge_days": 5,
    })
    train, valid = split_fold(
        data,
        fold,
        target_column="target_cs_rank_5d",
        label_end_column="label_end_date_5d",
    )
    assert list(train["trade_date"]) == [pd.Timestamp("2018-06-01"), pd.Timestamp("2018-12-20")]
    assert list(valid["trade_date"]) == [pd.Timestamp("2019-06-01")]


def test_strategy_uses_five_day_rebalance_and_cost_grid() -> None:
    dates = pd.to_datetime([f"2024-01-{day:02d}" for day in range(2, 12)])
    rows = []
    for date in dates:
        for i in range(10):
            rows.append({
                "trade_date": date,
                "product": f"P{i}",
                "prediction": float(i),
                "future_return_5d": 0.001 * (i - 4.5),
            })
    predictions = pd.DataFrame(rows)
    config = {
        "long_short_fraction": 0.2,
        "holding_days": 5,
        "rebalance_frequency_days": 5,
        "raw_return_column": "future_return_5d",
        "one_way_cost_bps": [0, 3, 5, 10],
    }
    strategy = build_strategy_returns(predictions, config)
    expected = select_rebalance_dates(dates.to_numpy(), 5)
    assert set(pd.to_datetime(strategy["trade_date"])) == expected
    assert "net_0bps_return" in strategy.columns
    assert "net_3bps_return" in strategy.columns
    assert "net_5bps_return" in strategy.columns
    assert "net_10bps_return" in strategy.columns
    assert np.allclose(
        strategy["net_5bps_return"],
        strategy["gross_return"] - 2.0 * 5 / 10_000.0,
    )
