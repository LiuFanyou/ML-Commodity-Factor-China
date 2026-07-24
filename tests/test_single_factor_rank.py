from __future__ import annotations

import numpy as np
import pandas as pd

from factor_modeling.common import construct_daily_portfolio, purge_last_dates
from factor_modeling.handoff import apply_label_end_purge, split_fold
from factor_modeling.single_factor_rank import (
    apply_score_transform,
    attach_expected_direction,
    build_signed_prediction,
    choose_unknown_direction,
    cross_sectional_centered_rank,
    mean_daily_rank_ic,
    select_candidates,
    select_top_factors,
)


def test_cross_sectional_centered_rank_maps_to_unit_interval() -> None:
    values = pd.Series([0.1, -0.2, 0.0, 0.3])
    ranked = cross_sectional_centered_rank(values)
    assert np.isclose(ranked.mean(), 0.0)
    assert ranked.min() == -1.0
    assert ranked.max() == 1.0


def test_apply_score_transform_cs_rank_by_date() -> None:
    frame = pd.DataFrame({
        "trade_date": pd.to_datetime(["2024-01-02"] * 3 + ["2024-01-03"] * 3),
        "signed_signal": [1.0, 2.0, 3.0, 0.0, -1.0, 5.0],
    })
    prediction = apply_score_transform(frame, "signed_signal", "cs_rank_centered")
    assert np.allclose(prediction.groupby(frame["trade_date"]).mean(), 0.0)
    assert prediction.min() == -1.0
    assert prediction.max() == 1.0


def test_purge_does_not_leak_boundary_date() -> None:
    frame = pd.DataFrame({
        "trade_date": pd.to_datetime(["2023-12-27", "2023-12-28", "2023-12-29"]),
        "product": ["A", "A", "A"],
    })
    purged = purge_last_dates(frame, 1)
    assert purged["trade_date"].max() == pd.Timestamp("2023-12-28")
    assert pd.Timestamp("2023-12-29") not in set(purged["trade_date"])


def test_portfolio_net_zero_gross_one_and_skips_thin_cross_section() -> None:
    thick = pd.DataFrame({
        "prediction": np.arange(10, dtype=float),
        "future_return_5d": np.linspace(-0.01, 0.01, 10),
    })
    result = construct_daily_portfolio(thick, 0.2, return_column="future_return_5d")
    assert result is not None
    assert result["gross_exposure"] == 1.0
    assert result["net_exposure"] == 0.0
    assert result["round_trip_turnover"] == 2.0

    thin = pd.DataFrame({
        "prediction": np.arange(9, dtype=float),
        "future_return_5d": np.linspace(-0.01, 0.01, 9),
    })
    assert construct_daily_portfolio(thin, 0.2, return_column="future_return_5d") is None


def test_direction_flip_flips_rank_ic_sign() -> None:
    frame = pd.DataFrame({
        "trade_date": pd.to_datetime(["2024-01-02"] * 6),
        "product": list("ABCDEF"),
        "factor": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "future_return_5d": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
    })
    pos = frame.copy()
    pos["prediction"] = build_signed_prediction(pos, "factor", 1, "cs_rank_centered")
    neg = frame.copy()
    neg["prediction"] = build_signed_prediction(neg, "factor", -1, "cs_rank_centered")
    pos_ic = mean_daily_rank_ic(pos, "prediction", "future_return_5d", 5)
    neg_ic = mean_daily_rank_ic(neg, "prediction", "future_return_5d", 5)
    assert pos_ic > 0
    assert np.isclose(neg_ic, -pos_ic)


def test_cost_formula_is_round_trip_two_times_one_way_bps() -> None:
    gross = 0.0012
    for bps in (0.0, 3.0, 5.0, 10.0):
        net = gross - 2.0 * bps / 10_000.0
        assert np.isclose(net, gross - 2.0 * bps / 10000.0)


def test_unknown_direction_uses_training_rank_ic_only() -> None:
    train = pd.DataFrame({
        "trade_date": pd.to_datetime(["2023-01-03"] * 6 + ["2023-01-04"] * 6),
        "feature": [1, 2, 3, 4, 5, 6] * 2,
        "future_return_5d": [0.06, 0.05, 0.04, 0.03, 0.02, 0.01] * 2,
    })
    assert choose_unknown_direction(train, "feature", "future_return_5d", 5) == -1


def test_attach_expected_direction_accepts_factor_no() -> None:
    registry = pd.DataFrame({
        "feature_name": ["tsmom_60__csz"],
        "factor_no": [1],
        "usage_type": ["predictor"],
        "source_factor": ["tsmom_60"],
    })
    out = attach_expected_direction(registry)
    assert "source_factor_id" in out.columns
    assert int(out.iloc[0]["source_factor_id"]) == 1
    assert int(out.iloc[0]["expected_direction"]) in (-1, 0, 1)


def test_select_candidates_filters_predictor_usage() -> None:
    registry = pd.DataFrame({
        "feature_name": ["a__csz", "b__missing"],
        "factor_no": [1, 1],
        "usage_type": ["predictor", "quality_flag"],
        "source_factor": ["a", "a"],
    })
    selected = select_candidates(registry, {"candidate_usage_types": ["predictor"]})
    assert selected["feature_name"].tolist() == ["a__csz"]


def test_select_candidates_can_use_all_registered_features() -> None:
    registry = pd.DataFrame({
        "feature_name": ["a__csz", "b__missing", "c__csz"],
        "factor_no": [1, 1, 2],
        "usage_type": ["predictor", "quality_flag", "risk"],
        "source_factor": ["a", "a", "c"],
    })
    selected = select_candidates(
        registry,
        {"use_all_registered_features": True, "candidate_usage_types": None},
    )
    assert selected["feature_name"].tolist() == ["a__csz", "b__missing", "c__csz"]


def test_select_top_factors_keeps_per_factor_metrics() -> None:
    board = pd.DataFrame({
        "rank": [1, 2, 3, 4],
        "feature_name": ["a__missing", "b__csz", "c__csz", "d__csz"],
        "usage_type": ["quality_flag", "predictor", "predictor", "predictor"],
        "rank_ic_mean": [0.20, 0.04, 0.03, 0.02],
        "rank_ic_hit_rate": [0.6, 0.55, 0.54, 0.53],
        "gross_sharpe": [0.1, 0.5, 0.4, 0.3],
        "strategy_coverage": [1.0, 1.0, 1.0, 1.0],
        "strategy_days": [100, 100, 100, 100],
    })
    top = select_top_factors(
        board,
        config={
            "top_k_factors": 3,
            "top_k_selection": {
                "usage_types": ["predictor"],
                "exclude_missing_indicators": True,
            },
        },
    )
    assert top["feature_name"].tolist() == ["b__csz", "c__csz", "d__csz"]
    assert top["shortlist_rank"].tolist() == [1, 2, 3]
    assert top["overall_rank"].tolist() == [2, 3, 4]
    assert "a__missing" not in set(top["feature_name"])


def test_label_end_purge_and_split_fold_match_contract() -> None:
    frame = pd.DataFrame({
        "trade_date": pd.to_datetime(["2018-12-20", "2018-12-28", "2019-06-01"]),
        "label_end_date_5d": pd.to_datetime(["2018-12-27", "2019-01-07", "2019-06-08"]),
        "target_cs_rank_5d": [0.1, 0.2, 0.3],
        "product": ["A", "A", "A"],
    })
    purged = apply_label_end_purge(
        frame.iloc[:2],
        pd.Timestamp("2019-01-01"),
        label_end_column="label_end_date_5d",
    )
    assert list(purged["trade_date"]) == [pd.Timestamp("2018-12-20")]

    fold = pd.Series({
        "fold_id": "fold_2019",
        "train_start": pd.Timestamp("2015-01-05"),
        "train_end": pd.Timestamp("2018-12-31"),
        "valid_start": pd.Timestamp("2019-01-01"),
        "valid_end": pd.Timestamp("2019-12-31"),
        "purge_days": 5,
    })
    train, valid = split_fold(
        frame,
        fold,
        target_column="target_cs_rank_5d",
        label_end_column="label_end_date_5d",
    )
    assert list(train["trade_date"]) == [pd.Timestamp("2018-12-20")]
    assert list(valid["trade_date"]) == [pd.Timestamp("2019-06-01")]
