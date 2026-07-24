from __future__ import annotations

import numpy as np
import pandas as pd


def _ridge_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    valid_x: np.ndarray,
    alpha: float,
) -> np.ndarray:
    x_mean = train_x.mean(axis=0)
    y_mean = train_y.mean()
    centered_x = train_x - x_mean
    centered_y = train_y - y_mean
    gram = centered_x.T @ centered_x
    penalty = np.eye(gram.shape[0]) * alpha
    coefficients = np.linalg.solve(gram + penalty, centered_x.T @ centered_y)
    return (valid_x - x_mean) @ coefficients + y_mean


def _daily_rank_ic(
    dates: pd.Series,
    actual: np.ndarray,
    prediction: np.ndarray,
) -> float:
    frame = pd.DataFrame(
        {"trade_date": dates.to_numpy(), "actual": actual, "prediction": prediction}
    )
    values: list[float] = []
    for _, subset in frame.groupby("trade_date", sort=False):
        if len(subset) < 8 or subset["actual"].nunique() < 2 or subset["prediction"].nunique() < 2:
            continue
        values.append(
            float(
                subset["actual"].rank().corr(subset["prediction"].rank())
            )
        )
    return float(np.mean(values)) if values else np.nan


def incremental_ridge_tests(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    feature_registry: pd.DataFrame,
    folds: list[dict[str, object]],
    target: str = "future_return_5d",
    alpha: float = 10.0,
) -> pd.DataFrame:
    label_columns = [
        "trade_date",
        "product",
        target,
        "label_end_date_5d",
    ]
    data = features.merge(
        labels[label_columns],
        on=["trade_date", "product"],
        how="left",
        validate="one_to_one",
    )
    baseline_source_factors = {
        "tsmom_60",
        "carry_annualized",
        "amihud_20",
        "turnover_oi",
    }
    fixed_baseline_features = feature_registry.loc[
        feature_registry["source_factor"].isin(baseline_source_factors)
        & ~feature_registry["is_missing_indicator"].astype(bool)
        & ~feature_registry["is_interaction"].astype(bool),
        "feature_name",
    ].tolist()
    primary_features = feature_registry.loc[
        ~feature_registry["is_missing_indicator"].astype(bool)
        & ~feature_registry["is_interaction"].astype(bool)
        & feature_registry["usage_type"].eq("predictor"),
        ["feature_name", "source_factor"],
    ]
    rows: list[dict[str, object]] = []

    for candidate in primary_features.itertuples(index=False):
        candidate_name = candidate.feature_name
        baseline_features = [
            name for name in fixed_baseline_features if name != candidate_name
        ]
        fold_rows: list[dict[str, float]] = []
        for fold in folds:
            train_start = pd.Timestamp(str(fold["train_start"]))
            train_end = pd.Timestamp(str(fold["train_end"]))
            valid_start = pd.Timestamp(str(fold["valid_start"]))
            valid_end = pd.Timestamp(str(fold["valid_end"]))
            train_mask = (
                data["trade_date"].between(train_start, train_end)
                & data["label_end_date_5d"].lt(valid_start)
                & data[target].notna()
            )
            valid_mask = (
                data["trade_date"].between(valid_start, valid_end)
                & data["label_end_date_5d"].le(valid_end)
                & data[target].notna()
            )
            train = data.loc[train_mask]
            valid = data.loc[valid_mask]
            if len(train) < 500 or len(valid) < 100:
                continue
            train_y = train[target].to_numpy(dtype=float)
            valid_y = valid[target].to_numpy(dtype=float)
            train_baseline = train[baseline_features].to_numpy(dtype=float)
            valid_baseline = valid[baseline_features].to_numpy(dtype=float)
            baseline_prediction = _ridge_predict(
                train_baseline,
                train_y,
                valid_baseline,
                alpha,
            )
            full_features = baseline_features + [candidate_name]
            full_prediction = _ridge_predict(
                train[full_features].to_numpy(dtype=float),
                train_y,
                valid[full_features].to_numpy(dtype=float),
                alpha,
            )
            fold_rows.append(
                {
                    "delta_mse": float(
                        np.mean((valid_y - baseline_prediction) ** 2)
                        - np.mean((valid_y - full_prediction) ** 2)
                    ),
                    "delta_rank_ic": float(
                        _daily_rank_ic(
                            valid["trade_date"],
                            valid_y,
                            full_prediction,
                        )
                        - _daily_rank_ic(
                            valid["trade_date"],
                            valid_y,
                            baseline_prediction,
                        )
                    ),
                }
            )
        fold_frame = pd.DataFrame(fold_rows)
        rows.append(
            {
                "factor_id": candidate.source_factor,
                "feature_name": candidate_name,
                "ridge_alpha": alpha,
                "folds_evaluated": int(len(fold_frame)),
                "mean_oos_delta_mse": (
                    float(fold_frame["delta_mse"].mean()) if not fold_frame.empty else np.nan
                ),
                "mean_oos_delta_rank_ic": (
                    float(fold_frame["delta_rank_ic"].mean()) if not fold_frame.empty else np.nan
                ),
                "positive_delta_mse_fold_rate": (
                    float(fold_frame["delta_mse"].gt(0).mean()) if not fold_frame.empty else np.nan
                ),
                "positive_delta_rank_ic_fold_rate": (
                    float(fold_frame["delta_rank_ic"].gt(0).mean()) if not fold_frame.empty else np.nan
                ),
                "interpretation": (
                    "固定Ridge、相对预注册基线"
                    "(TSMOM、Carry、Amihud、Turnover)的逐项样本外增量诊断；"
                    "不用于决定交接包特征是否保留"
                ),
            }
        )
    return pd.DataFrame(rows)
