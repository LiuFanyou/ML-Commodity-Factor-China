import numpy as np
import pandas as pd

from factor_screening.labels import add_forward_labels


def test_forward_labels_start_next_row():
    panel = pd.DataFrame(
        {
            "product": ["A"] * 4,
            "trade_date": pd.date_range("2020-01-01", periods=4),
            "tradable_return": [0.10, 0.20, 0.30, 0.40],
        }
    )
    result = add_forward_labels(panel, [1, 2])
    assert np.isclose(result.loc[0, "future_return_1d"], 0.20)
    assert np.isclose(result.loc[0, "future_return_2d"], 1.20 * 1.30 - 1)
    assert result.loc[0, "label_end_date_2d"] == pd.Timestamp("2020-01-03")
    assert pd.isna(result.loc[3, "future_return_1d"])
