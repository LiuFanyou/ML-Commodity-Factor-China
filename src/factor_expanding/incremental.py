from __future__ import annotations

import numpy as np
import pandas as pd

from factor_screening.incremental import BASELINE_FEATURES, _daily_robust_z, _model
from factor_screening.metrics import cross_sectional_corr


def expanding_incremental_tests(
    data: pd.DataFrame,
    candidate_features: list[str],
    primary_horizon: int,
    train_start_year: int,
    first_inner_oos_year: int,
    last_inner_oos_year: int,
    minimum_assets: int,
) -> pd.DataFrame:
    """Estimate candidate incremental value using only nested training-period folds."""
    target = f"future_return_{primary_horizon}d"
    end_col = f"label_end_date_{primary_horizon}d"
    required = sorted(set(BASELINE_FEATURES + candidate_features))
    frame = data[["trade_date", "product", target, end_col, *required]].copy()
    frame[required] = _daily_robust_z(frame, required)
    frame["row_id"] = np.arange(len(frame))

    baseline_predictions: list[pd.DataFrame] = []
    fold_cache: list[tuple[int, pd.DataFrame, pd.DataFrame, np.ndarray]] = []
    for oos_year in range(first_inner_oos_year, last_inner_oos_year + 1):
        start = pd.Timestamp(f"{train_start_year}-01-01")
        cutoff = pd.Timestamp(f"{oos_year}-01-01")
        end = pd.Timestamp(f"{oos_year}-12-31")
        train = frame.loc[
            frame["trade_date"].between(start, cutoff - pd.Timedelta(days=1))
            & frame[end_col].lt(cutoff)
            & frame[target].notna()
        ]
        test = frame.loc[frame["trade_date"].between(cutoff, end) & frame[target].notna()]
        if train.empty or test.empty:
            continue
        baseline_model = _model().fit(train[BASELINE_FEATURES], train[target])
        baseline_pred = baseline_model.predict(test[BASELINE_FEATURES])
        base = test[["row_id", "trade_date", "product", target]].copy()
        base["baseline_prediction"] = baseline_pred
        base["oos_year"] = oos_year
        baseline_predictions.append(base)
        fold_cache.append((oos_year, train, test, baseline_pred))

    if not fold_cache:
        return pd.DataFrame(columns=["feature_name", "delta_rank_ic", "mse_reduction", "oos_rows"])

    baseline_all = pd.concat(baseline_predictions, ignore_index=True)
    baseline_ic = cross_sectional_corr(
        baseline_all, "baseline_prediction", target, minimum_assets, rank=True
    ).mean()
    baseline_mse = float(np.mean((baseline_all[target] - baseline_all["baseline_prediction"]) ** 2))
    rows: list[dict[str, object]] = []
    for feature in candidate_features:
        if feature in BASELINE_FEATURES:
            rows.append({
                "feature_name": feature,
                "baseline_rank_ic": baseline_ic,
                "candidate_rank_ic": baseline_ic,
                "delta_rank_ic": 0.0,
                "baseline_mse": baseline_mse,
                "candidate_mse": baseline_mse,
                "mse_reduction": 0.0,
                "oos_rows": len(baseline_all),
                "inner_oos_first_year": int(fold_cache[0][0]),
                "inner_oos_last_year": int(fold_cache[-1][0]),
            })
            continue
        predictions: list[pd.DataFrame] = []
        model_features = [*BASELINE_FEATURES, feature]
        for oos_year, train, test, baseline_pred in fold_cache:
            if train[feature].notna().sum() == 0:
                pred = baseline_pred
            else:
                pred = _model().fit(train[model_features], train[target]).predict(test[model_features])
            part = test[["row_id", "trade_date", "product", target]].copy()
            part["candidate_prediction"] = pred
            part["oos_year"] = oos_year
            predictions.append(part)
        candidate_all = pd.concat(predictions, ignore_index=True)
        candidate_ic = cross_sectional_corr(
            candidate_all, "candidate_prediction", target, minimum_assets, rank=True
        ).mean()
        candidate_mse = float(np.mean((candidate_all[target] - candidate_all["candidate_prediction"]) ** 2))
        rows.append({
            "feature_name": feature,
            "baseline_rank_ic": baseline_ic,
            "candidate_rank_ic": candidate_ic,
            "delta_rank_ic": float(candidate_ic - baseline_ic),
            "baseline_mse": baseline_mse,
            "candidate_mse": candidate_mse,
            "mse_reduction": float((baseline_mse - candidate_mse) / baseline_mse) if baseline_mse > 0 else np.nan,
            "oos_rows": len(candidate_all),
            "inner_oos_first_year": int(fold_cache[0][0]),
            "inner_oos_last_year": int(fold_cache[-1][0]),
        })
    return pd.DataFrame(rows)
