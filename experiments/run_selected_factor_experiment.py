"""Fold-local de-duplication and Top-K factor selection extension."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_oos_ml_experiment import (BASE_COSTS, HANDOFF, MODEL_CONFIGS, ROOT, TARGET,
    annual_metrics, create_rebalance_positions, daily_portfolio, make_model, prediction_metrics, safe_corr)


OUT = ROOT / "delivery" / "selected_features"
TOP_K = 30


def training_rank_ic_scores(train: pd.DataFrame, candidates: list[str]) -> pd.DataFrame:
    """Training-only mean absolute daily cross-sectional Spearman IC."""
    date = train["trade_date"]
    feature_ranks = train.groupby("trade_date")[candidates].rank(method="average")
    target_rank = train.groupby("trade_date")[TARGET].rank(method="average")
    x = feature_ranks - feature_ranks.groupby(date).transform("mean")
    y = target_rank - target_rank.groupby(date).transform("mean")
    numerator = x.mul(y, axis=0).groupby(date).sum()
    denominator = np.sqrt((x * x).groupby(date).sum().mul((y * y).groupby(date).sum(), axis=0))
    scores = (numerator / denominator).abs().mean().rename("mean_abs_train_rank_ic").reset_index()
    return scores.rename(columns={"index": "feature_name"}).sort_values(["mean_abs_train_rank_ic", "feature_name"], ascending=[False, True], na_position="last")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True); checkpoints = OUT / "_checkpoints"; checkpoints.mkdir(exist_ok=True)
    data = pd.read_csv(HANDOFF / "ml_dataset_development.csv.gz", parse_dates=["trade_date", "entry_date", "label_end_date_5d"])
    registry = pd.read_csv(HANDOFF / "ml_feature_registry.csv")
    all_features = registry["feature_name"].tolist()
    representatives = registry.loc[registry["feature_name"].eq(registry["cluster_representative"]), "feature_name"].drop_duplicates().tolist()
    folds = pd.read_csv(HANDOFF / "fold_definitions.csv", parse_dates=["train_start", "train_end", "valid_start", "valid_end"])
    data[all_features] = data[all_features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    predictions, importances, selections = {name: [] for name in MODEL_CONFIGS}, [], []
    cols = ["trade_date", "product", "sector", "mapped_actual_ts_code", "entry_date", "label_end_date_5d", TARGET,
        "future_return_1d", "future_return_20d", "entry_execution_available", "entry_suspected_limit_lock",
        "exit_suspected_limit_lock_5d", "exit_execution_available_5d"]
    for fold in folds.itertuples(index=False):
        train = data.loc[data["trade_date"].between(fold.train_start, fold.train_end) & data["label_end_date_5d"].lt(fold.valid_start) & data[TARGET].notna()]
        valid = data.loc[data["trade_date"].between(fold.valid_start, fold.valid_end) & data[TARGET].notna()]
        scores = training_rank_ic_scores(train, representatives); selected = scores.head(TOP_K)["feature_name"].tolist()
        scores["fold_id"] = fold.fold_id; scores["selected"] = scores["feature_name"].isin(selected); selections.append(scores)
        for name in MODEL_CONFIGS:
            pred_file, imp_file = checkpoints / f"{name}_{fold.fold_id}_predictions.csv", checkpoints / f"{name}_{fold.fold_id}_importance.csv"
            if pred_file.exists() and imp_file.exists():
                predictions[name].append(pd.read_csv(pred_file, parse_dates=["trade_date", "entry_date", "label_end_date_5d"])); importances.extend(pd.read_csv(imp_file).to_dict("records")); continue
            model = make_model(name); model.fit(train[selected].to_numpy(dtype=np.float32), train[TARGET].to_numpy())
            part = valid[cols].copy(); part["prediction"] = model.predict(valid[selected].to_numpy(dtype=np.float32)); part["fold_id"] = fold.fold_id; part["model"] = name
            fold_importance = [{"model": name, "fold_id": fold.fold_id, "feature_name": f, "importance": float(v)} for f, v in zip(selected, model.feature_importances_)]
            predictions[name].append(part); importances.extend(fold_importance); part.to_csv(pred_file, index=False); pd.DataFrame(fold_importance).to_csv(imp_file, index=False)
        print(f"completed {fold.fold_id}: candidates={len(representatives)}, selected={len(selected)}", flush=True)
    raw = pd.read_csv(ROOT / "futures_contract_daily_30varieties_2015_2025.zip", compression="zip", usecols=["trade_date", "ts_code", "open"])
    raw["trade_date"] = pd.to_datetime(raw["trade_date"]); raw["open"] = pd.to_numeric(raw["open"], errors="coerce")
    raw = raw.dropna(subset=["open"]).sort_values(["ts_code", "trade_date"]); raw["next_open"] = raw.groupby("ts_code")["open"].shift(-1); raw["contract_return"] = raw["next_open"] / raw["open"] - 1
    summaries, costs, daily_all, positions_all, predictions_all = [], [], [], [], []
    for name, parts in predictions.items():
        pred = pd.concat(parts, ignore_index=True).sort_values(["trade_date", "product"]); predictions_all.append(pred)
        positions, turnover = create_rebalance_positions(pred); positions["model"] = name; positions_all.append(positions); gross = daily_portfolio(positions, turnover, raw)
        for cost in BASE_COSTS:
            daily = gross.copy(); daily["cost_bps_one_way"] = cost; daily["cost"] = daily["turnover"] * cost / 10000; daily["net_return"] = daily["gross_return"] - daily["cost"]; daily["nav"] = (1 + daily["net_return"]).cumprod(); daily["model"] = name
            daily_all.append(daily); values = annual_metrics(daily); costs.append({"model": name, "cost_bps_one_way": cost, **values})
            if cost == 5:
                yearly = pred.assign(year=pred["trade_date"].dt.year).groupby("year").apply(lambda x: safe_corr(x, "spearman"), include_groups=False)
                stable = bool((yearly > 0).sum() >= 4 and values["p_value"] < 0.05 and values["sharpe"] > 0)
                summaries.append({"model": name, "oos_folds": len(folds), "positive_rank_ic_years": int((yearly > 0).sum()), "stable_oos_criteria_met": stable, "selected_features_per_fold": TOP_K, "candidate_representatives": len(representatives), **prediction_metrics(pred), **values})
    pd.concat(selections, ignore_index=True).to_csv(OUT / "fold_feature_selection.csv", index=False)
    pd.DataFrame(summaries).sort_values("sharpe", ascending=False).to_csv(OUT / "metrics_summary.csv", index=False)
    pd.DataFrame(costs).sort_values(["model", "cost_bps_one_way"]).to_csv(OUT / "cost_sensitivity.csv", index=False)
    pd.concat(daily_all, ignore_index=True).to_csv(OUT / "daily_returns.csv", index=False)
    pd.concat(positions_all, ignore_index=True).to_csv(OUT / "positions.csv", index=False)
    pd.concat(predictions_all, ignore_index=True).to_csv(OUT / "predictions.csv", index=False)
    pd.DataFrame(importances).groupby(["model", "feature_name"], as_index=False)["importance"].mean().to_csv(OUT / "feature_importance.csv", index=False)
    registry.loc[registry["feature_name"].isin(representatives)].to_csv(OUT / "candidate_representative_features.csv", index=False)
    (OUT / "experiment_config.json").write_text(json.dumps({"selection_method": "registered cluster representative then training-only Top-30 mean absolute daily Rank IC", "candidate_count": len(representatives), "selected_per_fold": TOP_K, "target": TARGET, "models": MODEL_CONFIGS}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
