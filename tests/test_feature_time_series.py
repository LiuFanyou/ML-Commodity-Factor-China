import numpy as np
import pandas as pd

from unified_factor_system.features import _prior_rolling_date_z


def test_rolling_date_z_does_not_use_future_values() -> None:
    dates = pd.Series(pd.date_range("2020-01-01", periods=6))
    values = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 1000.0])
    original = _prior_rolling_date_z(values, dates, 3, 2, 10.0)
    changed = values.copy()
    changed.iloc[-1] = -1000.0
    alternative = _prior_rolling_date_z(changed, dates, 3, 2, 10.0)
    assert np.allclose(
        original.iloc[:-1],
        alternative.iloc[:-1],
        equal_nan=True,
    )
