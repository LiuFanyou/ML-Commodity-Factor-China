"""Strict rolling OOS comparison for regularized LightGBM, XGBoost and RF."""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import statsmodels.api as sm
import xgboost as xgb
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from statsmodels.stats.sandwich_covariance import cov_hac


ROOT = Path(__file__).resolve().parents[1]
HANDOFF = ROOT / "training_handoff" / "training_handoff"
OUT = ROOT / "delivery" / "all_features"
RANDOM_STATE = 20260724
TARGET = "future_return_5d"
BASE_COSTS = (0, 3, 5, 10)

MODEL_CONFIGS = {
    "lightgbm": {"params": {"n_estimators": 160, "learning_rate": 0.025, "num_leaves": 15,
        "max_depth": 4, "min_child_samples": 100, "subsample": 0.7, "subsample_freq": 1,
        "colsample_bytree": 0.7, "reg_alpha": 0.1, "reg_lambda": 10.0, "min_split_gain": 0.01,
        "random_state": RANDOM_STATE, "n_jobs": 4, "verbosity": -1}},
    "xgboost": {"params": {"n_estimators": 160, "learning_rate": 0.025, "max_depth": 3,
        "min_child_weight": 50, "gamma": 0.05, "subsample": 0.7, "colsample_bytree": 0.7,
        "reg_alpha": 0.1, "reg_lambda": 10.0, "objective": "reg:squarederror",
        "random_state": RANDOM_STATE, "n_jobs": 4, "verbosity": 0, "tree_method": "hist"}},
    "random_forest": {"params": {"n_estimators": 60, "max_depth": 6, "min_samples_split": 150,
        "min_samples_leaf": 75, "max_features": 0.7, "bootstrap": True, "max_samples": 0.7,
        "ccp_alpha": 1e-6, "random_state": RANDOM_STATE, "n_jobs": 4}},
}


def make_model(name: str):
    params = MODEL_CONFIGS[name]["params"]
    if name == "lightgbm":
        return lgb.LGBMRegressor(**params)
    if name == "xgboost":
        return xgb.XGBRegressor(**params)
    return RandomForestRegressor(**params)


def safe_corr(group: pd.DataFrame, method: str, target: str = TARGET) -> float:
    group = group[["prediction", target]].dropna()
    if len(group) < 5 or group["prediction"].nunique() < 2 or group[target].nunique() < 2:
        return np.nan
    return group["prediction"].corr(group[target], method=method)


def hac_test(returns: pd.Series) -> tuple[float, float]:
    values = returns.dropna().to_numpy()
    if len(values) < 10 or np.isclose(values.std(), 0):
        return np.nan, np.nan
    result = sm.OLS(values, np.ones((len(values), 1))).fit()
    stderr = float(np.sqrt(cov_hac(result, nlags=5)[0, 0]))
    t_stat = float(result.params[0] / stderr) if stderr else np.nan
    return t_stat, float(2 * stats.t.sf(abs(t_stat), df=len(values) - 1))


def annual_metrics(daily: pd.DataFrame) -> dict[str, float]:
    ret = daily["net_return"].fillna(0.0)
    nav = (1 + ret).cumprod()
    years = max(len(ret) / 252, 1 / 252)
    annual_return = nav.iloc[-1] ** (1 / years) - 1
    annual_vol = ret.std(ddof=1) * np.sqrt(252)
    sharpe = ret.mean() / ret.std(ddof=1) * np.sqrt(252) if ret.std(ddof=1) else np.nan
    drawdown = nav / nav.cummax() - 1
    t_stat, p_value = hac_test(ret)
    return {"factor_return": nav.iloc[-1] - 1, "annual_return": annual_return,
        "annual_volatility": annual_vol, "sharpe": sharpe, "max_drawdown": drawdown.min(),
        "calmar": annual_return / abs(drawdown.min()) if drawdown.min() else np.nan,
        "win_rate": (ret > 0).mean(), "daily_turnover": daily["turnover"].mean(),
        "t_stat": t_stat, "p_value": p_value}


def create_rebalance_positions(pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pred = pred.sort_values(["trade_date", "product"]).copy()
    eligible = pred.loc[pred["entry_execution_available"].eq(1) & pred["exit_execution_available_5d"].eq(1)
        & ~pred["entry_suspected_limit_lock"].eq(1) & ~pred["exit_suspected_limit_lock_5d"].eq(1)].copy()
    dates = pd.Index(sorted(eligible["trade_date"].unique()))[::5]
    rows = []
    for _, group in eligible.loc[eligible["trade_date"].isin(dates)].groupby("trade_date", sort=True):
        group = group.sort_values("prediction")
        n_side = max(1, int(np.floor(len(group) * 0.2)))
        short, long = group.head(n_side).copy(), group.tail(n_side).copy()
        short["side"], long["side"] = -1, 1
        selected = pd.concat([short, long], ignore_index=True)
        selected["weight"] = selected["side"] / (2 * n_side)
        rows.append(selected)
    positions = pd.concat(rows, ignore_index=True).sort_values(["entry_date", "mapped_actual_ts_code"])
    positions["entry_date"] = pd.to_datetime(positions["entry_date"])
    positions["label_end_date_5d"] = pd.to_datetime(positions["label_end_date_5d"])
    turnover_rows, previous = [], pd.Series(dtype=float)
    for date, group in positions.groupby("entry_date", sort=True):
        current = group.groupby("mapped_actual_ts_code")["weight"].sum()
        turnover_rows.append({"entry_date": date, "turnover": current.sub(previous, fill_value=0).abs().sum()})
        previous = current
    turnover_rows.append({"entry_date": positions["label_end_date_5d"].max(), "turnover": previous.abs().sum()})
    turnover = pd.DataFrame(turnover_rows).groupby("entry_date", as_index=False)["turnover"].sum()
    return positions, turnover


def daily_portfolio(positions: pd.DataFrame, turnover: pd.DataFrame, raw_open: pd.DataFrame) -> pd.DataFrame:
    holdings = positions[["mapped_actual_ts_code", "entry_date", "label_end_date_5d", "weight"]]
    joined = raw_open.merge(holdings, left_on="ts_code", right_on="mapped_actual_ts_code", how="inner")
    joined = joined.loc[joined["trade_date"].ge(joined["entry_date"]) & joined["trade_date"].lt(joined["label_end_date_5d"])]
    gross = joined.assign(weighted_return=joined["weight"] * joined["contract_return"]).groupby("trade_date", as_index=False)["weighted_return"].sum()
    gross = gross.rename(columns={"weighted_return": "gross_return"})
    dates = pd.DataFrame({"trade_date": sorted(set(raw_open["trade_date"]).intersection(set(gross["trade_date"]) | set(turnover["entry_date"])))})
    return dates.merge(gross, on="trade_date", how="left").merge(turnover, left_on="trade_date", right_on="entry_date", how="left").drop(columns="entry_date").fillna({"gross_return": 0.0, "turnover": 0.0})


def prediction_metrics(pred: pd.DataFrame) -> dict[str, float]:
    by_date = pred.groupby("trade_date")
    ic = by_date.apply(safe_corr, method="pearson", include_groups=False).dropna()
    rank5 = by_date.apply(safe_corr, method="spearman", include_groups=False)
    rank20 = by_date.apply(safe_corr, method="spearman", target="future_return_20d", include_groups=False)
    return {"ic_mean": ic.mean(), "rank_ic": rank5.mean(), "rank_ic_20d": rank20.mean(),
        "ic_std": ic.std(ddof=1), "ic_ir": ic.mean() / ic.std(ddof=1),
        "p_ic_gt_002": (ic > 0.02).mean(), "p_ic_lt_neg002": (ic < -0.02).mean()}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    checkpoints = OUT / "_checkpoints"; checkpoints.mkdir(exist_ok=True)
    data = pd.read_csv(HANDOFF / "ml_dataset_development.csv.gz", parse_dates=["trade_date", "entry_date", "label_end_date_5d"])
    registry = pd.read_csv(HANDOFF / "ml_feature_registry.csv")
    features = registry["feature_name"].tolist()
    folds = pd.read_csv(HANDOFF / "fold_definitions.csv", parse_dates=["train_start", "train_end", "valid_start", "valid_end"])
    data[features] = data[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    predictions, importances = {name: [] for name in MODEL_CONFIGS}, []
    cols = ["trade_date", "product", "sector", "mapped_actual_ts_code", "entry_date", "label_end_date_5d", TARGET,
        "future_return_1d", "future_return_20d", "entry_execution_available", "entry_suspected_limit_lock",
        "exit_suspected_limit_lock_5d", "exit_execution_available_5d"]
    for fold in folds.itertuples(index=False):
        train = data.loc[data["trade_date"].between(fold.train_start, fold.train_end) & data["label_end_date_5d"].lt(fold.valid_start) & data[TARGET].notna()]
        valid = data.loc[data["trade_date"].between(fold.valid_start, fold.valid_end) & data[TARGET].notna()]
        for name in MODEL_CONFIGS:
            pred_file, imp_file = checkpoints / f"{name}_{fold.fold_id}_predictions.csv", checkpoints / f"{name}_{fold.fold_id}_importance.csv"
            if pred_file.exists() and imp_file.exists():
                predictions[name].append(pd.read_csv(pred_file, parse_dates=["trade_date", "entry_date", "label_end_date_5d"]))
                importances.extend(pd.read_csv(imp_file).to_dict("records")); continue
            model = make_model(name); model.fit(train[features].to_numpy(dtype=np.float32), train[TARGET].to_numpy())
            part = valid[cols].copy(); part["prediction"] = model.predict(valid[features].to_numpy(dtype=np.float32)); part["fold_id"] = fold.fold_id; part["model"] = name
            fold_importance = [{"model": name, "fold_id": fold.fold_id, "feature_name": f, "importance": float(v)} for f, v in zip(features, model.feature_importances_)]
            predictions[name].append(part); importances.extend(fold_importance)
            part.to_csv(pred_file, index=False); pd.DataFrame(fold_importance).to_csv(imp_file, index=False)
        print(f"completed {fold.fold_id}: train={len(train)}, valid={len(valid)}", flush=True)
    raw = pd.read_csv(ROOT / "futures_contract_daily_30varieties_2015_2025.zip", compression="zip", usecols=["trade_date", "ts_code", "open"])
    raw["trade_date"] = pd.to_datetime(raw["trade_date"]); raw["open"] = pd.to_numeric(raw["open"], errors="coerce")
    raw = raw.dropna(subset=["open"]).sort_values(["ts_code", "trade_date"]); raw["next_open"] = raw.groupby("ts_code")["open"].shift(-1); raw["contract_return"] = raw["next_open"] / raw["open"] - 1
    summaries, costs, daily_all, positions_all, predictions_all = [], [], [], [], []
    for name, parts in predictions.items():
        pred = pd.concat(parts, ignore_index=True).sort_values(["trade_date", "product"]); predictions_all.append(pred)
        positions, turnover = create_rebalance_positions(pred); positions["model"] = name; positions_all.append(positions)
        gross = daily_portfolio(positions, turnover, raw)
        for cost in BASE_COSTS:
            daily = gross.copy(); daily["cost_bps_one_way"] = cost; daily["cost"] = daily["turnover"] * cost / 10000; daily["net_return"] = daily["gross_return"] - daily["cost"]; daily["nav"] = (1 + daily["net_return"]).cumprod(); daily["model"] = name
            daily_all.append(daily); values = annual_metrics(daily); costs.append({"model": name, "cost_bps_one_way": cost, **values})
            if cost == 5:
                yearly = pred.assign(year=pred["trade_date"].dt.year).groupby("year").apply(lambda x: safe_corr(x, "spearman"), include_groups=False)
                stable = bool((yearly > 0).sum() >= 4 and values["p_value"] < 0.05 and values["sharpe"] > 0)
                summaries.append({"model": name, "oos_folds": len(folds), "positive_rank_ic_years": int((yearly > 0).sum()), "stable_oos_criteria_met": stable, **prediction_metrics(pred), **values})
    pd.DataFrame(summaries).sort_values("sharpe", ascending=False).to_csv(OUT / "metrics_summary.csv", index=False)
    pd.DataFrame(costs).sort_values(["model", "cost_bps_one_way"]).to_csv(OUT / "cost_sensitivity.csv", index=False)
    pd.concat(daily_all, ignore_index=True).to_csv(OUT / "daily_returns.csv", index=False)
    pd.concat(positions_all, ignore_index=True).to_csv(OUT / "positions.csv", index=False)
    pd.concat(predictions_all, ignore_index=True).to_csv(OUT / "predictions.csv", index=False)
    pd.DataFrame(importances).groupby(["model", "feature_name"], as_index=False)["importance"].mean().to_csv(OUT / "feature_importance.csv", index=False)
    registry.to_csv(OUT / "feature_list.csv", index=False)
    (OUT / "experiment_config.json").write_text(json.dumps({"target": TARGET, "feature_count": len(features), "models": MODEL_CONFIGS, "cost_bps": BASE_COSTS, "random_state": RANDOM_STATE}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
