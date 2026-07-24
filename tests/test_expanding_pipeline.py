from __future__ import annotations

import numpy as np
import pandas as pd

from factor_expanding.pipeline import expanding_splits, mask_training_labels


def test_expanding_splits_keep_2015_start() -> None:
    assert expanding_splits(2015, 2019, 2022) == [
        (2015, 2018, 2019),
        (2015, 2019, 2020),
        (2015, 2020, 2021),
        (2015, 2021, 2022),
    ]


def test_training_labels_crossing_test_boundary_are_masked() -> None:
    frame = pd.DataFrame({
        "trade_date": pd.to_datetime(["2018-12-27", "2018-12-28", "2019-01-02"]),
        "future_return_1d": [0.01, 0.02, 0.03],
        "label_end_date_1d": pd.to_datetime(["2018-12-28", "2019-01-02", "2019-01-03"]),
        "future_return_5d": [0.05, 0.06, 0.07],
        "label_end_date_5d": pd.to_datetime(["2019-01-04", "2019-01-07", "2019-01-08"]),
    })
    result = mask_training_labels(frame, [1, 5], 2019)
    assert len(result) == 2
    assert result["future_return_1d"].notna().sum() == 1
    assert result["future_return_5d"].isna().all()


def test_invalid_expanding_years_raise() -> None:
    with np.testing.assert_raises(ValueError):
        expanding_splits(2015, 2015, 2020)
