import pandas as pd

from unified_factor_system.labels import add_forward_labels, assert_forward_labels


def test_forward_label_uses_same_actual_contract_open_to_open() -> None:
    dates = pd.date_range("2020-01-01", periods=5)
    panel = pd.DataFrame(
        {
            "trade_date": dates,
            "product": ["X"] * 5,
            "calendar_index": range(5),
            "mapped_actual_ts_code": ["X01.EX"] * 5,
        }
    )
    contracts = pd.DataFrame(
        {
            "trade_date": dates,
            "ts_code": ["X01.EX"] * 5,
            "open": [10.0, 20.0, 30.0, 40.0, 50.0],
            "volume": [100.0] * 5,
            "suspected_price_limit_lock": [0] * 5,
            "suspected_price_limit_direction": [0.0] * 5,
        }
    )
    labelled = add_forward_labels(panel, contracts, [1, 2])
    assert abs(labelled.loc[0, "future_return_1d"] - (30.0 / 20.0 - 1.0)) < 1e-12
    assert abs(labelled.loc[0, "future_return_2d"] - (40.0 / 20.0 - 1.0)) < 1e-12
    assert_forward_labels(labelled, [1, 2])


def test_locked_entry_is_not_labelled() -> None:
    dates = pd.date_range("2020-01-01", periods=3)
    panel = pd.DataFrame(
        {
            "trade_date": dates,
            "product": ["X"] * 3,
            "calendar_index": range(3),
            "mapped_actual_ts_code": ["X01.EX"] * 3,
        }
    )
    contracts = pd.DataFrame(
        {
            "trade_date": dates,
            "ts_code": ["X01.EX"] * 3,
            "open": [10.0, 20.0, 30.0],
            "volume": [100.0] * 3,
            "suspected_price_limit_lock": [0, 1, 0],
            "suspected_price_limit_direction": [0.0, 1.0, 0.0],
        }
    )
    labelled = add_forward_labels(panel, contracts, [1], True)
    assert pd.isna(labelled.loc[0, "future_return_1d"])
