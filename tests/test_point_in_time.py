import pandas as pd

from unified_factor_system.factors import _available_fundamental


def test_fundamental_is_delayed_one_trading_day() -> None:
    frame = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"]),
            "product": ["X", "X", "X"],
            "value": [100.0, 110.0, 110.0],
            "observation": pd.to_datetime(
                ["2020-01-02", "2020-01-03", "2020-01-03"]
            ),
        }
    )
    available, observations = _available_fundamental(
        frame,
        "value",
        "observation",
        maximum_age=7,
        lag_trading_days=1,
    )
    assert pd.isna(available.iloc[0])
    assert available.iloc[1] == 100.0
    assert observations.iloc[1] == pd.Timestamp("2020-01-02")
    assert available.iloc[2] == 110.0
