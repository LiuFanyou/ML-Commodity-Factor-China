from __future__ import annotations

import pandas as pd

from .config import ProjectConfig
from .factors import FeatureDefinition, build_curve_features, build_factor_matrix


def build_feature_matrix_and_definitions(
    panel: pd.DataFrame,
    curve_market: pd.DataFrame,
    config: ProjectConfig,
) -> tuple[pd.DataFrame, list[FeatureDefinition], pd.DataFrame]:
    curve = build_curve_features(curve_market, config)
    factors, definitions = build_factor_matrix(panel, curve, config)
    return factors, definitions, curve
