import pandas as pd

from factor_screening.config import load_config
from factor_screening.factors import build_curve_features


def test_curve_uses_same_date_only_and_positive_backwardation():
    config = load_config()
    rows = []
    for date, prices in [("2024-01-02", [110.0, 105.0, 100.0]), ("2024-01-03", [80.0, 100.0, 120.0])]:
        for index, price in enumerate(prices):
            rows.append(
                {
                    "trade_date": pd.Timestamp(date),
                    "product": "X",
                    "ts_code": f"X{index}",
                    "delivery_yyyymm": 202402 + index,
                    "days_to_delisting": 30 + 30 * index,
                    "close": price,
                    "open_interest": 1000.0,
                    "basic_liquid": True,
                    "is_standard_delivery_contract": True,
                }
            )
    market = pd.DataFrame(rows)
    full = build_curve_features(market, config)
    first_only = build_curve_features(market.loc[market["trade_date"].eq(pd.Timestamp("2024-01-02"))], config)
    assert full.loc[0, "carry"] > 0
    assert full.loc[0, "term_structure_slope"] > 0
    pd.testing.assert_series_equal(
        full.loc[0, ["carry", "curve_curvature", "term_structure_slope"]],
        first_only.loc[0, ["carry", "curve_curvature", "term_structure_slope"]],
    )
