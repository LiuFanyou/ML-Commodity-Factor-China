import pandas as pd

from factor_screening.config import load_config
from factor_screening.point_in_time import run_point_in_time_audit


def test_development_rejects_2025():
    config = load_config()
    panel = pd.DataFrame(
        {
            "trade_date": [pd.Timestamp("2025-01-02")],
            "product": ["A"],
            "selection_available_date": [pd.Timestamp("2025-01-03")],
        }
    )
    try:
        run_point_in_time_audit(panel, config, "development")
    except AssertionError as error:
        assert "sealed" in str(error)
    else:
        raise AssertionError("2025 data should be rejected in development")


def test_selection_must_be_available_after_signal_date():
    config = load_config()
    panel = pd.DataFrame(
        {
            "trade_date": [pd.Timestamp("2024-01-02")],
            "product": ["A"],
            "selection_available_date": [pd.Timestamp("2024-01-02")],
        }
    )
    try:
        run_point_in_time_audit(panel, config, "development")
    except AssertionError as error:
        assert "point-in-time" in str(error)
    else:
        raise AssertionError("same-day selection availability should fail")
