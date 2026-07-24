from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


def cross_sectional_factor_correlation(data: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    values = data[["trade_date", *features]].copy()
    means = values.groupby("trade_date")[features].transform("mean")
    stds = values.groupby("trade_date")[features].transform("std").replace(0, np.nan)
    standardized = (values[features] - means) / stds
    return standardized.corr(min_periods=100)


def cluster_factors(correlation: pd.DataFrame, threshold: float) -> pd.DataFrame:
    if correlation.empty:
        return pd.DataFrame(columns=["feature_name", "correlation_cluster"])
    corr = correlation.fillna(0.0).clip(-1, 1)
    np.fill_diagonal(corr.values, 1.0)
    distance = (1.0 - corr.abs()).clip(0, 1)
    condensed = squareform(distance.values, checks=False)
    tree = linkage(condensed, method="average")
    labels = fcluster(tree, t=1.0 - threshold, criterion="distance")
    return pd.DataFrame({"feature_name": correlation.index, "correlation_cluster": labels})


def choose_cluster_representatives(
    clusters: pd.DataFrame,
    primary_metrics: pd.DataFrame,
) -> pd.DataFrame:
    scored = clusters.merge(
        primary_metrics[["feature_name", "rank_ic_q_bh", "rank_ic_mean", "coverage"]],
        on="feature_name",
        how="left",
    )
    scored["q_score"] = scored["rank_ic_q_bh"].fillna(1.0)
    scored["abs_ic"] = scored["rank_ic_mean"].abs().fillna(0.0)
    scored = scored.sort_values(
        ["correlation_cluster", "q_score", "abs_ic", "coverage", "feature_name"],
        ascending=[True, True, False, False, True],
    )
    scored["cluster_representative"] = ~scored.duplicated("correlation_cluster")
    return scored.drop(columns=["q_score", "abs_ic"])
