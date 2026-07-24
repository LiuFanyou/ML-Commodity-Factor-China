import numpy as np
import pandas as pd

from unified_factor_system.features import _cross_sectional_robust_z


def test_cross_sectional_z_uses_only_same_date() -> None:
    dates = pd.Series(
        [pd.Timestamp("2020-01-01")] * 3 + [pd.Timestamp("2020-01-02")] * 3
    )
    values = pd.Series([1.0, 2.0, 3.0, 100.0, 200.0, 300.0])
    result = _cross_sectional_robust_z(values, dates, 5.0)
    assert np.allclose(result.iloc[:3], result.iloc[3:], equal_nan=True)
