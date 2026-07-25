#!/usr/bin/env python3
"""Minimal 2025 backtest against the unified execution protocol."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def load_panel() -> pd.DataFrame:
    features = pd.read_csv(ROOT / "ml_features_2025.csv.gz", parse_dates=["trade_date"])
    labels = pd.read_csv(
        ROOT / "labels_2025_evaluator_only.csv.gz",
        parse_dates=["trade_date", "entry_date", "label_end_date_5d"],
    )
    registry = pd.read_csv(ROOT / "ml_feature_registry.csv")
    feature_names = registry["feature_name"].tolist()
    missing = [c for c in feature_names if c not in features.columns]
    if missing:
        raise ValueError(f"missing features: {missing[:5]}")
    panel = features.merge(labels, on=["trade_date", "product"], how="inner", validate="one_to_one")
    usable = panel["future_return_5d"].notna()
    if "entry_execution_available" in panel.columns:
        usable &= panel["entry_execution_available"].fillna(0).astype(int).eq(1)
    if "exit_execution_available_5d" in panel.columns:
        usable &= panel["exit_execution_available_5d"].fillna(0).astype(int).eq(1)
    return panel.loc[usable].copy(), feature_names


def load_scores(panel: pd.DataFrame) -> pd.Series:
    pred_path = ROOT / "predictions_2025.csv"
    if pred_path.exists():
        pred = pd.read_csv(pred_path, parse_dates=["trade_date"])
        merged = panel.merge(pred[["trade_date", "product", "score"]], on=["trade_date", "product"], how="left")
        if merged["score"].isna().any():
            raise ValueError("predictions_2025.csv does not cover all evaluable rows")
        return merged["score"]
    # Interface smoke test only.
    rng = np.random.default_rng(42)
    return pd.Series(rng.normal(size=len(panel)), index=panel.index, name="score")


def backtest(panel: pd.DataFrame, score: pd.Series, cost_bps: float = 5.0) -> pd.DataFrame:
    data = panel.copy()
    data["score"] = score.to_numpy()
    dates = sorted(data["trade_date"].unique())
    rebalance_dates = dates[::5]
    rows = []
    prev = {}
    for dt in rebalance_dates:
        cross = data.loc[data["trade_date"].eq(dt), ["product", "score", "future_return_5d"]].dropna()
        if len(cross) < 8:
            continue
        cross = cross.sort_values(["score", "product"])
        n = max(1, int(np.floor(len(cross) * 0.2)))
        short = cross.head(n)
        long = cross.tail(n)
        weights = {
            **{p: -0.5 / n for p in short["product"]},
            **{p: 0.5 / n for p in long["product"]},
        }
        gross = float(sum(weights[r.product] * r.future_return_5d for r in pd.concat([short, long]).itertuples()))
        turnover = float(sum(abs(weights.get(p, 0.0) - prev.get(p, 0.0)) for p in set(weights) | set(prev)))
        rows.append(
            {
                "trade_date": dt,
                "gross_return": gross,
                "turnover": turnover,
                "net_return": gross - turnover * cost_bps / 10_000.0,
                "long_count": n,
                "short_count": n,
            }
        )
        prev = weights
    return pd.DataFrame(rows)


def summarize(returns: pd.Series, periods_per_year: float = 252.0 / 5.0) -> dict:
    mu = float(returns.mean())
    sigma = float(returns.std(ddof=1))
    ann_ret = (1.0 + mu) ** periods_per_year - 1.0
    ann_vol = sigma * np.sqrt(periods_per_year)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    wealth = (1.0 + returns).cumprod()
    max_dd = float((wealth / wealth.cummax() - 1.0).min())
    return {
        "n_rebalances": int(len(returns)),
        "annual_return": ann_ret,
        "annual_volatility": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "mean_turnover": float("nan"),
    }


def main() -> None:
    panel, feature_names = load_panel()
    scores = load_scores(panel)
    bt = backtest(panel, scores, cost_bps=5.0)
    stats = summarize(bt["net_return"])
    stats["mean_turnover"] = float(bt["turnover"].mean()) if len(bt) else np.nan
    print(f"features={len(feature_names)} rows={len(panel)} rebalances={len(bt)}")
    for key, value in stats.items():
        print(f"{key}: {value}")
    bt.to_csv(ROOT / "example_backtest_2025_returns.csv", index=False)
    print("wrote example_backtest_2025_returns.csv")


if __name__ == "__main__":
    main()
