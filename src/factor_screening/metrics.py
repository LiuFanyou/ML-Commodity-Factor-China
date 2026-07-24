from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .catalog import CATALOG_BY_ID, FACTOR_CATALOG
from .config import ProjectConfig
from .factors import FeatureDefinition


def cross_sectional_corr(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    minimum_assets: int,
    rank: bool = False,
) -> pd.Series:
    frame = data[["trade_date", x_col, y_col]].dropna().copy()
    if rank:
        frame["x"] = frame.groupby("trade_date")[x_col].rank(method="average")
        frame["y"] = frame.groupby("trade_date")[y_col].rank(method="average")
    else:
        frame["x"] = frame[x_col]
        frame["y"] = frame[y_col]
    count = frame.groupby("trade_date")["x"].transform("count")
    frame = frame.loc[count.ge(minimum_assets)]
    if frame.empty:
        return pd.Series(dtype=float, name="correlation")
    frame["xd"] = frame["x"] - frame.groupby("trade_date")["x"].transform("mean")
    frame["yd"] = frame["y"] - frame.groupby("trade_date")["y"].transform("mean")
    frame["xy"] = frame["xd"] * frame["yd"]
    frame["x2"] = frame["xd"] ** 2
    frame["y2"] = frame["yd"] ** 2
    grouped = frame.groupby("trade_date")[["xy", "x2", "y2"]].sum()
    result = grouped["xy"] / np.sqrt(grouped["x2"] * grouped["y2"])
    result.name = "correlation"
    return result.replace([np.inf, -np.inf], np.nan).dropna()


def newey_west_mean_t(series: pd.Series, maxlags: int) -> tuple[float, float]:
    clean = series.dropna().astype(float)
    if len(clean) < max(20, maxlags + 5):
        return np.nan, np.nan
    model = sm.OLS(clean.to_numpy(), np.ones((len(clean), 1))).fit(
        cov_type="HAC", cov_kwds={"maxlags": maxlags}
    )
    return float(model.tvalues[0]), float(model.pvalues[0])


def two_way_cluster_slope(
    data: pd.DataFrame,
    feature: str,
    label: str,
) -> tuple[float, float, float, int]:
    frame = data[["trade_date", "product", feature, label]].dropna().copy()
    if len(frame) < 200:
        return np.nan, np.nan, np.nan, len(frame)
    mean = frame.groupby("trade_date")[feature].transform("mean")
    std = frame.groupby("trade_date")[feature].transform("std").replace(0, np.nan)
    frame["x"] = (frame[feature] - mean) / std
    frame = frame.dropna(subset=["x"])
    if len(frame) < 200:
        return np.nan, np.nan, np.nan, len(frame)
    x = sm.add_constant(frame[["x"]], has_constant="add")
    groups = np.column_stack(
        [
            pd.factorize(frame["trade_date"])[0],
            pd.factorize(frame["product"])[0],
        ]
    )
    try:
        result = sm.OLS(frame[label].to_numpy(), x.to_numpy()).fit(
            cov_type="cluster", cov_kwds={"groups": groups}
        )
        return float(result.params[1]), float(result.tvalues[1]), float(result.pvalues[1]), len(frame)
    except (ValueError, np.linalg.LinAlgError):
        return np.nan, np.nan, np.nan, len(frame)


def benjamini_hochberg(pvalues: pd.Series) -> pd.Series:
    values = pvalues.astype(float)
    valid = values.dropna().sort_values()
    output = pd.Series(np.nan, index=values.index, dtype=float)
    if valid.empty:
        return output
    m = len(valid)
    ranked = valid.to_numpy() * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1].clip(0, 1)
    output.loc[valid.index] = adjusted
    return output


def _group_return_statistics(
    data: pd.DataFrame,
    feature: str,
    label: str,
    groups: int,
) -> dict[str, float]:
    frame = data[["trade_date", feature, label]].dropna().copy()
    if frame.empty:
        return {"group_spread": np.nan, "group_monotonicity": np.nan, "group_days": 0}
    frame["pct_rank"] = frame.groupby("trade_date")[feature].rank(pct=True, method="first")
    frame["group"] = np.minimum((frame["pct_rank"] * groups).apply(np.ceil).astype(int), groups)
    means = frame.groupby(["trade_date", "group"])[label].mean().unstack()
    if 1 not in means or groups not in means:
        return {"group_spread": np.nan, "group_monotonicity": np.nan, "group_days": len(means)}
    spread = means[groups] - means[1]
    group_average = means.mean(axis=0)
    monotonicity = pd.Series(group_average.index, index=group_average.index).corr(group_average, method="spearman")
    return {
        "group_spread": float(spread.mean()),
        "group_monotonicity": float(monotonicity) if pd.notna(monotonicity) else np.nan,
        "group_days": int(spread.notna().sum()),
    }


def _portfolio_statistics(
    data: pd.DataFrame,
    feature: str,
    groups: int,
    costs_bps: list[float],
    annualization: int,
) -> dict[str, float]:
    frame = data[["trade_date", "product", feature, "future_return_1d"]].dropna().copy()
    frame["pct_rank"] = frame.groupby("trade_date")[feature].rank(pct=True, method="first")
    frame["side"] = 0.0
    frame.loc[frame["pct_rank"].le(1.0 / groups), "side"] = -1.0
    frame.loc[frame["pct_rank"].gt(1.0 - 1.0 / groups), "side"] = 1.0
    counts = frame.loc[frame["side"].ne(0)].groupby(["trade_date", "side"])["product"].transform("count")
    frame.loc[frame["side"].ne(0), "weight"] = 0.5 * frame.loc[frame["side"].ne(0), "side"] / counts
    frame["weight"] = frame["weight"].fillna(0.0)
    gross = (frame["weight"] * frame["future_return_1d"]).groupby(frame["trade_date"]).sum()
    weights = frame.pivot(index="trade_date", columns="product", values="weight").fillna(0.0)
    turnover = 0.5 * weights.diff().abs().sum(axis=1)
    result: dict[str, float] = {
        "gross_annual_return": float(gross.mean() * annualization),
        "gross_sharpe": float(gross.mean() / gross.std(ddof=0) * np.sqrt(annualization)) if gross.std(ddof=0) > 0 else np.nan,
        "average_one_way_turnover": float(turnover.mean()),
    }
    wealth = (1 + gross.fillna(0)).cumprod()
    drawdown = wealth / wealth.cummax() - 1
    result["gross_max_drawdown"] = float(drawdown.min())
    for bps in costs_bps:
        net = gross - turnover.reindex(gross.index).fillna(0) * float(bps) / 10_000.0
        suffix = str(int(bps))
        result[f"net_{suffix}bps_annual_return"] = float(net.mean() * annualization)
        result[f"net_{suffix}bps_sharpe"] = float(net.mean() / net.std(ddof=0) * np.sqrt(annualization)) if net.std(ddof=0) > 0 else np.nan
    return result


def compute_screening_metrics(
    data: pd.DataFrame,
    definitions: list[FeatureDefinition],
    config: ProjectConfig,
    start_date: str,
    end_date: str,
) -> dict[str, pd.DataFrame]:
    mask = data["trade_date"].between(pd.Timestamp(start_date), pd.Timestamp(end_date))
    sample = data.loc[mask].copy()
    minimum_assets = int(config.raw["screening"]["minimum_cross_section_assets"])
    groups = int(config.raw["screening"]["quantile_groups"])
    costs = [float(x) for x in config.raw["screening"]["cost_bps_one_way"]]
    annualization = int(config.raw["screening"]["annualization_days"])
    rows: list[dict[str, object]] = []
    daily_rows: list[pd.DataFrame] = []
    yearly_rows: list[dict[str, object]] = []
    product_rows: list[dict[str, object]] = []
    sector_rows: list[dict[str, object]] = []

    for definition in definitions:
        feature = definition.feature_name
        coverage = float(sample[feature].notna().mean())
        unchanged = sample.groupby("product")[feature].diff().eq(0) & sample[feature].notna()
        stale_rate = float(unchanged.mean())
        for horizon in config.horizons:
            label = f"future_return_{horizon}d"
            pearson = cross_sectional_corr(sample, feature, label, minimum_assets, rank=False)
            rank_ic = cross_sectional_corr(sample, feature, label, minimum_assets, rank=True)
            lag = max(1, horizon - 1)
            rank_t, rank_p = newey_west_mean_t(rank_ic, lag)
            pearson_t, pearson_p = newey_west_mean_t(pearson, lag)
            nonoverlap = rank_ic.iloc[::horizon] if horizon > 1 else rank_ic
            nonoverlap_t, nonoverlap_p = newey_west_mean_t(nonoverlap, 1)
            pooled_beta, pooled_t, pooled_p, pooled_n = two_way_cluster_slope(sample, feature, label)
            group_stats = _group_return_statistics(sample, feature, label, groups)
            row: dict[str, object] = {
                "feature_name": feature,
                "source_factor_id": definition.source_factor_id,
                "source_factor": definition.source_factor,
                "horizon": horizon,
                "coverage": coverage,
                "missing_rate": 1 - coverage,
                "stale_rate": stale_rate,
                "ic_mean": float(pearson.mean()) if len(pearson) else np.nan,
                "ic_std": float(pearson.std(ddof=0)) if len(pearson) else np.nan,
                "icir": float(pearson.mean() / pearson.std(ddof=0)) if len(pearson) and pearson.std(ddof=0) > 0 else np.nan,
                "ic_hit_rate": float((pearson > 0).mean()) if len(pearson) else np.nan,
                "ic_t_nw": pearson_t,
                "ic_p_nw": pearson_p,
                "rank_ic_mean": float(rank_ic.mean()) if len(rank_ic) else np.nan,
                "rank_ic_std": float(rank_ic.std(ddof=0)) if len(rank_ic) else np.nan,
                "rank_icir": float(rank_ic.mean() / rank_ic.std(ddof=0)) if len(rank_ic) and rank_ic.std(ddof=0) > 0 else np.nan,
                "rank_ic_hit_rate": float((rank_ic > 0).mean()) if len(rank_ic) else np.nan,
                "rank_ic_t_nw": rank_t,
                "rank_ic_p_nw": rank_p,
                "ic_days": int(len(pearson)),
                "rank_ic_days": int(len(rank_ic)),
                "nonoverlap_rank_ic_mean": float(nonoverlap.mean()) if len(nonoverlap) else np.nan,
                "nonoverlap_rank_ic_t": nonoverlap_t,
                "nonoverlap_rank_ic_p": nonoverlap_p,
                "pooled_beta": pooled_beta,
                "pooled_t_two_way_cluster": pooled_t,
                "pooled_p_two_way_cluster": pooled_p,
                "pooled_observations": pooled_n,
                **group_stats,
            }
            if horizon == 1:
                row.update(_portfolio_statistics(sample, feature, groups, costs, annualization))
            rows.append(row)
            if len(rank_ic):
                daily_rows.append(pd.DataFrame({
                    "trade_date": rank_ic.index,
                    "feature_name": feature,
                    "horizon": horizon,
                    "rank_ic": rank_ic.values,
                }))

            year_series = rank_ic.groupby(rank_ic.index.year).agg(["mean", "count"])
            for year, values in year_series.iterrows():
                yearly_rows.append({
                    "feature_name": feature,
                    "horizon": horizon,
                    "year": int(year),
                    "rank_ic_mean": float(values["mean"]),
                    "rank_ic_days": int(values["count"]),
                })

        primary_label = f"future_return_{config.primary_horizon}d"
        for product, part in sample[["product", feature, primary_label]].dropna().groupby("product"):
            product_rows.append({
                "feature_name": feature,
                "product": product,
                "correlation": part[feature].corr(part[primary_label], method="spearman"),
                "observations": len(part),
            })
        for sector, part in sample[["sector", feature, primary_label]].dropna().groupby("sector"):
            sector_rows.append({
                "feature_name": feature,
                "sector": sector,
                "correlation": part[feature].corr(part[primary_label], method="spearman"),
                "observations": len(part),
            })

    summary = pd.DataFrame(rows)
    primary = summary["horizon"].eq(config.primary_horizon)
    summary.loc[primary, "rank_ic_q_bh"] = benjamini_hochberg(summary.loc[primary, "rank_ic_p_nw"])
    daily = pd.concat(daily_rows, ignore_index=True) if daily_rows else pd.DataFrame()
    return {
        "summary": summary,
        "daily_ic": daily,
        "yearly": pd.DataFrame(yearly_rows),
        "product": pd.DataFrame(product_rows),
        "sector": pd.DataFrame(sector_rows),
    }


def conceptual_catalog_frame() -> pd.DataFrame:
    return pd.DataFrame([asdict(spec) for spec in FACTOR_CATALOG])
