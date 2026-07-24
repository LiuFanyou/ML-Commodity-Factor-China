from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd

from .definitions import FACTOR_DEFINITIONS


KEY_COLUMNS = ["trade_date", "product", "sector"]
EVENT_FACTORS = {
    "inventory_surprise",
    "warehouse_surprise",
    "basis_surprise",
    "fundamental_composite_news",
    "fundamental_underreaction",
}
TIME_SERIES_ONLY_FACTORS = {"factor_momentum"}


def _cross_sectional_robust_z(
    values: pd.Series,
    dates: pd.Series,
    clip: float,
) -> pd.Series:
    grouped = values.groupby(dates, sort=False)
    median = grouped.transform("median")
    absolute_deviation = (values - median).abs()
    mad = absolute_deviation.groupby(dates, sort=False).transform("median")
    scale = 1.4826 * mad
    standard_deviation = grouped.transform("std")
    scale = scale.where(scale.gt(0), standard_deviation)
    z_score = (values - median).div(scale.where(scale.gt(0)))
    return z_score.clip(-clip, clip)


def _prior_rolling_date_z(
    values: pd.Series,
    dates: pd.Series,
    window: int,
    minimum: int,
    clip: float,
) -> pd.Series:
    by_date = pd.DataFrame({"trade_date": dates, "value": values}).drop_duplicates(
        "trade_date"
    )
    by_date = by_date.sort_values("trade_date")
    prior = by_date["value"].shift(1)
    mean = prior.rolling(window, min_periods=minimum).mean()
    standard_deviation = prior.rolling(window, min_periods=minimum).std()
    by_date["z"] = (
        (by_date["value"] - mean)
        .div(standard_deviation.where(standard_deviation.gt(0)))
        .clip(-clip, clip)
    )
    return dates.map(by_date.set_index("trade_date")["z"])


def _correlation_clusters(
    matrix: pd.DataFrame,
    threshold: float,
) -> tuple[pd.DataFrame, dict[str, int], dict[int, str]]:
    correlation = matrix.corr(min_periods=100)
    columns = list(correlation.columns)
    parent = {column: column for column in columns}

    def find(item: str) -> str:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left_index, left in enumerate(columns):
        for right in columns[left_index + 1 :]:
            value = correlation.at[left, right]
            if pd.notna(value) and abs(value) >= threshold:
                union(left, right)

    ordered_roots: dict[str, int] = {}
    cluster_map: dict[str, int] = {}
    representative_map: dict[int, str] = {}
    for column in columns:
        root = find(column)
        if root not in ordered_roots:
            ordered_roots[root] = len(ordered_roots) + 1
        cluster_id = ordered_roots[root]
        cluster_map[column] = cluster_id
        representative_map.setdefault(cluster_id, column)
    return correlation, cluster_map, representative_map


def build_ml_features(
    raw_factors: pd.DataFrame,
    development_end: pd.Timestamp,
    minimum_dense_coverage: float,
    minimum_event_observations: int,
    minimum_event_instruments: int,
    missing_indicator_threshold: float,
    mad_clip: float,
    rolling_z_window: int,
    rolling_z_minimum: int,
    correlation_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    definitions = {item.factor_id: item for item in FACTOR_DEFINITIONS}
    development_mask = raw_factors["trade_date"].le(development_end)
    development = raw_factors.loc[development_mask]
    output = raw_factors[KEY_COLUMNS].copy()
    factor_registry_rows: list[dict[str, object]] = []
    feature_registry_rows: list[dict[str, object]] = []
    continuous_features: list[str] = []

    for factor_id, definition in definitions.items():
        development_values = development[factor_id]
        coverage = float(development_values.notna().mean())
        unique_values = int(development_values.nunique(dropna=True))
        observation_count = int(development_values.notna().sum())
        instrument_count = int(
            development.loc[development_values.notna(), "product"].nunique()
        )
        is_event = factor_id in EVENT_FACTORS
        event_pass = (
            is_event
            and observation_count >= minimum_event_observations
            and instrument_count >= minimum_event_instruments
        )
        hard_pass = (
            definition.implementation_status == "computed"
            and unique_values > 1
            and (coverage >= minimum_dense_coverage or event_pass)
        )
        factor_registry_rows.append(
            {
                **asdict(definition),
                "available": hard_pass,
                "development_coverage": coverage,
                "development_missing_rate": 1.0 - coverage,
                "development_unique_values": unique_values,
                "development_observations": observation_count,
                "development_instruments": instrument_count,
                "event_feature": is_event,
                "raw_available_date": (
                    raw_factors.loc[raw_factors[factor_id].notna(), "trade_date"].min()
                    if raw_factors[factor_id].notna().any()
                    else pd.NaT
                ),
                "ml_inclusion_rule": (
                    "pre_registered_and_data_quality_pass"
                    if hard_pass
                    else (
                        "unavailable_required_data_missing"
                        if definition.implementation_status == "unavailable"
                        else "excluded_by_data_quality_hard_gate"
                    )
                ),
            }
        )
        if not hard_pass:
            continue

        if definition.binary:
            feature_name = factor_id
            output[feature_name] = raw_factors[factor_id].fillna(0).astype("int8")
            transformation = "binary_fill_missing_zero"
        elif factor_id in TIME_SERIES_ONLY_FACTORS:
            feature_name = f"{factor_id}__tsz"
            standardized = _prior_rolling_date_z(
                raw_factors[factor_id],
                raw_factors["trade_date"],
                rolling_z_window,
                rolling_z_minimum,
                mad_clip,
            )
            output[feature_name] = standardized.fillna(0.0).astype("float32")
            transformation = (
                f"prior_only_rolling_date_zscore_window_{rolling_z_window}_"
                f"minimum_{rolling_z_minimum}_clip_{mad_clip:g}_missing_to_zero"
            )
            continuous_features.append(feature_name)
        else:
            feature_name = f"{factor_id}__csz"
            standardized = _cross_sectional_robust_z(
                raw_factors[factor_id],
                raw_factors["trade_date"],
                mad_clip,
            )
            output[feature_name] = standardized.fillna(0.0).astype("float32")
            transformation = (
                f"same_date_cross_sectional_median_MAD_zscore_clip_{mad_clip:g}_"
                "missing_to_cross_sectional_median"
            )
            continuous_features.append(feature_name)

        feature_registry_rows.append(
            {
                "feature_name": feature_name,
                "factor_no": definition.factor_no,
                "source_factor": factor_id,
                "transformation": transformation,
                "category": definition.category,
                "protocol_role": definition.protocol_role,
                "economic_meaning": definition.economic_meaning,
                "available_date": factor_registry_rows[-1]["raw_available_date"],
                "missing_rate": 1.0 - coverage,
                "usage_type": definition.usage_type,
                "is_interaction": definition.category == "factor_interaction",
                "is_missing_indicator": False,
                "recommended_default": True,
                "point_in_time_safe": True,
                "frequency": "daily_product_panel",
            }
        )

        if not definition.binary and 1.0 - coverage >= missing_indicator_threshold:
            missing_feature = f"{factor_id}__missing"
            output[missing_feature] = raw_factors[factor_id].isna().astype("int8")
            feature_registry_rows.append(
                {
                    "feature_name": missing_feature,
                    "factor_no": definition.factor_no,
                    "source_factor": factor_id,
                    "transformation": "raw_missing_indicator",
                    "category": "data_quality",
                    "protocol_role": "extension",
                    "economic_meaning": f"{definition.factor_name}原始值不可用标记",
                    "available_date": raw_factors["trade_date"].min(),
                    "missing_rate": 0.0,
                    "usage_type": "quality_flag",
                    "is_interaction": False,
                    "is_missing_indicator": True,
                    "recommended_default": True,
                    "point_in_time_safe": True,
                    "frequency": "daily_product_panel",
                }
            )

    feature_registry = pd.DataFrame(feature_registry_rows)
    feature_columns = feature_registry["feature_name"].tolist()
    development_features = output.loc[development_mask, feature_columns]
    correlation, clusters, representatives = _correlation_clusters(
        development_features,
        correlation_threshold,
    )
    feature_registry["correlation_cluster"] = feature_registry["feature_name"].map(clusters)
    feature_registry["cluster_representative"] = feature_registry[
        "correlation_cluster"
    ].map(representatives)
    feature_registry["retained_despite_correlation"] = True
    feature_registry["selection_uses_target"] = False
    feature_registry["orthogonalization_policy"] = (
        "no_global_target_orthogonalization; use registered residual factors or "
        "fold-local regularization"
    )

    if output[feature_columns].isna().any().any():
        raise AssertionError("machine-learning feature matrix contains missing values")
    if output.duplicated(KEY_COLUMNS).any():
        raise AssertionError("machine-learning feature matrix has duplicate keys")
    return (
        output,
        pd.DataFrame(factor_registry_rows),
        feature_registry,
        correlation,
    )
