from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import ProjectConfig
from .metrics import cross_sectional_corr


BASELINE_FEATURES = ["tsmom_60", "carry", "realized_vol_20", "amihud_20"]


def _daily_robust_z(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    out = frame[["trade_date", *features]].copy()
    med = out.groupby("trade_date")[features].transform("median")
    mad = (out[features] - med).abs().groupby(out["trade_date"]).transform("median")
    scale = (1.4826 * mad).replace(0, np.nan)
    out[features] = ((out[features] - med) / scale).clip(-5, 5)
    return out[features]


def _model() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=False)),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=10.0)),
        ]
    )


def walk_forward_incremental_tests(
    data: pd.DataFrame,
    candidate_features: list[str],
    config: ProjectConfig,
) -> pd.DataFrame:
    target = f"future_return_{config.primary_horizon}d"
    end_col = f"label_end_date_{config.primary_horizon}d"
    required = sorted(set(BASELINE_FEATURES + candidate_features))
    frame = data[["trade_date", "product", target, end_col, *required]].copy()
    frame[required] = _daily_robust_z(frame, required)
    frame["row_id"] = np.arange(len(frame))
    train_years = int(config.raw["walk_forward"]["train_years"])
    first_year = int(config.raw["walk_forward"]["first_oos_year"])
    last_year = int(config.raw["walk_forward"]["last_development_oos_year"])
    minimum_assets = int(config.raw["screening"]["minimum_cross_section_assets"])

    baseline_predictions: list[pd.DataFrame] = []
    fold_cache: list[tuple[int, pd.DataFrame, pd.DataFrame, np.ndarray]] = []
    for oos_year in range(first_year, last_year + 1):
        start = pd.Timestamp(f"{oos_year - train_years}-01-01")
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
        baseline_all,
        "baseline_prediction",
        target,
        minimum_assets,
        rank=True,
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
            })
            continue
        predictions: list[pd.DataFrame] = []
        model_features = BASELINE_FEATURES + [feature]
        for oos_year, train, test, baseline_pred in fold_cache:
            if train[feature].notna().sum() == 0:
                part = test[["row_id", "trade_date", "product", target]].copy()
                part["candidate_prediction"] = baseline_pred
                part["oos_year"] = oos_year
                predictions.append(part)
                continue
            candidate_model = _model().fit(train[model_features], train[target])
            pred = candidate_model.predict(test[model_features])
            part = test[["row_id", "trade_date", "product", target]].copy()
            part["candidate_prediction"] = pred
            part["oos_year"] = oos_year
            predictions.append(part)
        candidate_all = pd.concat(predictions, ignore_index=True)
        candidate_ic = cross_sectional_corr(
            candidate_all,
            "candidate_prediction",
            target,
            minimum_assets,
            rank=True,
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
        })
    return pd.DataFrame(rows)
