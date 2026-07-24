import numpy as np

from hcgnn.data import _cpd_label_for_window


def test_dynp_labels_rising_regime_long() -> None:
    values = np.r_[
        np.full(20, 100.0),
        np.linspace(100.0, 120.0, 20),
        np.full(20, 120.0),
    ]
    label = _cpd_label_for_window(
        values, threshold=0.05, model="l2", n_bkps=2, min_size=5, jump=1
    )
    assert label == 1.0


def test_dynp_labels_falling_regime_short() -> None:
    values = np.r_[
        np.full(20, 100.0),
        np.linspace(100.0, 88.0, 20),
        np.full(20, 88.0),
    ]
    label = _cpd_label_for_window(
        values, threshold=0.05, model="l2", n_bkps=2, min_size=5, jump=1
    )
    assert label == 0.0


def test_dynp_interpolates_missing_values_inside_window() -> None:
    values = np.r_[
        np.full(20, 100.0),
        np.linspace(100.0, 120.0, 20),
        np.full(20, 120.0),
    ]
    values[[4, 21, 42]] = np.nan
    label = _cpd_label_for_window(
        values, threshold=0.05, model="l2", n_bkps=2, min_size=5, jump=1
    )
    assert label == 1.0
