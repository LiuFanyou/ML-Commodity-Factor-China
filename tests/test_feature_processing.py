import pandas as pd

from factor_screening.config import load_config
from factor_screening.feature_processing import build_ml_features


def test_ml_processing_adds_missing_indicator_without_future_dates():
    config = load_config()
    data = pd.DataFrame(
        {
            "trade_date": [pd.Timestamp("2024-01-02")] * 3,
            "product": ["A", "B", "C"],
            "sector": ["s", "s", "s"],
            "x": [1.0, None, 3.0],
        }
    )
    classes = pd.DataFrame(
        {
            "feature_name": ["x"],
            "source_factor": ["测试"],
            "source_factor_id": [1],
            "category": ["测试"],
            "economic_meaning": ["测试"],
            "usage_type": ["predictor"],
            "classification": ["B"],
            "enter_ml": [True],
            "correlation_cluster": [1],
            "cluster_representative": [True],
            "delta_rank_ic": [0.0],
        }
    )
    features, registry = build_ml_features(data, classes, config)
    assert "x__cs_robust_z" in features
    assert "x__missing" in features
    assert features["x__cs_robust_z"].notna().all()
    assert set(registry["usage_type"]) == {"predictor", "quality_flag"}
