from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import ruptures as rpt
from joblib import Parallel, delayed


PAPER_UNIVERSE = [
    "AG", "AL", "AP", "AU", "A", "BC", "BU", "B", "CF", "CS", "CU",
    "CY", "C", "EG", "FG", "FU", "HC", "IC", "IF", "IH", "I", "JD",
    "JM", "J", "L", "MA", "M", "NI", "NR", "OI", "PB", "PG", "PK",
    "PP", "P", "RB", "RM", "RU", "SA", "SC", "SN", "SP", "SR", "TA",
    "UR", "V", "Y", "ZC", "ZN",
]

FEATURE_COLUMNS = [
    "settle_index", "settle_return", "close_return", "intraday_return",
    "high_low_range", "log_volume", "log_open_interest", "spot_return",
    "basis", "spot_age", "log_warehouse_receipt", "warehouse_change",
    "warehouse_age", "warehouse_conflict", "log_stock", "stock_age",
    "stock_missing", "log_available_stock", "available_stock_age",
    "available_stock_missing",
]

INTEGRATED_NUMERIC_COLUMNS = [
    "pre_close", "pre_settle", "open", "high", "low", "close", "settle",
    "volume", "open_interest", "spot_price", "spot_price_age_days",
    "warehouse_receipt_current", "warehouse_receipt_change",
    "warehouse_receipt_age_days", "warehouse_receipt_conflict",
    "stock_current", "stock_age_days", "available_stock_current",
    "available_stock_age_days",
]


@dataclass(frozen=True)
class Fold:
    fold: int
    train_start: str
    train_end: str
    val_start: str
    val_end: str
    test_start: str
    test_end: str


def _exchange_alias(exchange: str) -> str:
    return {"SHFE": "SHF", "CZCE": "ZCE"}.get(exchange, exchange)


def paper_main_codes(data_dir: Path) -> list[str]:
    basic = pd.read_csv(data_dir / "fut_basic_all.csv", usecols=["exchange", "product"])
    basic["product"] = basic["product"].astype(str).str.upper()
    mapping = {
        row.product: f"{row.product}.{_exchange_alias(row.exchange)}"
        for row in basic.drop_duplicates(["product", "exchange"]).itertuples()
        if row.product in PAPER_UNIVERSE
    }
    mapping.update({"IC": "IC.CFX", "IF": "IF.CFX", "IH": "IH.CFX"})
    missing = sorted(set(PAPER_UNIVERSE) - set(mapping))
    if missing:
        raise ValueError(f"Cannot resolve paper-universe products: {missing}")
    return [mapping[p] for p in PAPER_UNIVERSE]


def load_integrated_daily(path: Path, last_year: int) -> pd.DataFrame:
    """Build observable next-day main-contract rows from the integrated CSV.

    The contract traded on date t is selected using open interest and volume from
    date t-1.  Consequently neither contract choice nor the target return uses
    information first observed after the trading decision.
    """
    cols = [
        "trade_date", "ts_code", "product_code", "delist_date", "quote_unit",
        "spot_price_unit", *INTEGRATED_NUMERIC_COLUMNS,
    ]
    data = pd.read_csv(path, usecols=cols, low_memory=False)
    data["trade_date"] = pd.to_datetime(data["trade_date"])
    data["delist_date"] = pd.to_datetime(data["delist_date"])
    data = data[data["trade_date"].dt.year <= last_year].copy()
    data = data.drop_duplicates(["trade_date", "ts_code"], keep="last")

    ranked = data.sort_values(
        ["trade_date", "product_code", "open_interest", "volume", "delist_date", "ts_code"],
        ascending=[True, True, False, False, False, True],
        na_position="last",
    )
    leaders = ranked.drop_duplicates(["trade_date", "product_code"], keep="first")[
        ["trade_date", "product_code", "ts_code"]
    ].rename(columns={"trade_date": "selection_date"})
    calendar = pd.DatetimeIndex(sorted(data["trade_date"].unique()))
    next_date = pd.Series(calendar[1:], index=calendar[:-1])
    leaders["trade_date"] = leaders["selection_date"].map(next_date)
    leaders = leaders.dropna(subset=["trade_date"])

    selected = leaders.merge(
        data,
        on=["trade_date", "product_code", "ts_code"],
        how="inner",
        validate="one_to_one",
    ).sort_values(["product_code", "trade_date"])
    selected["close_return"] = np.log(selected["close"] / selected["pre_close"])
    selected["settle_return"] = np.log(selected["settle"] / selected["pre_settle"])
    for col in ("close_return", "settle_return"):
        selected.loc[~np.isfinite(selected[col]), col] = np.nan
    selected["close_index"] = selected.groupby("product_code")["close_return"].transform(
        lambda s: 100.0 * np.exp(s.fillna(0.0).cumsum())
    )
    selected["settle_index"] = selected.groupby("product_code")["settle_return"].transform(
        lambda s: 100.0 * np.exp(s.fillna(0.0).cumsum())
    )
    return selected


def make_folds(dates: pd.DatetimeIndex, first_test_year: int, last_test_year: int) -> list[Fold]:
    folds: list[Fold] = []
    for test_year in range(first_test_year, last_test_year + 1):
        train_dates = dates[dates.year <= test_year - 2]
        val_dates = dates[dates.year == test_year - 1]
        test_dates = dates[dates.year == test_year]
        if not len(train_dates) or not len(val_dates) or not len(test_dates):
            continue
        folds.append(Fold(
            fold=len(folds),
            train_start=str(train_dates.min().date()), train_end=str(train_dates.max().date()),
            val_start=str(val_dates.min().date()), val_end=str(val_dates.max().date()),
            test_start=str(test_dates.min().date()), test_end=str(test_dates.max().date()),
        ))
    return folds


def build_integrated_panels(data: pd.DataFrame) -> tuple[pd.DatetimeIndex, dict[str, pd.DataFrame]]:
    dates = pd.DatetimeIndex(sorted(data["trade_date"].unique()))
    codes = sorted(data["product_code"].unique())
    raw: dict[str, pd.DataFrame] = {}
    panel_columns = [*INTEGRATED_NUMERIC_COLUMNS, "close_return", "settle_return",
                     "close_index", "settle_index"]
    for col in panel_columns:
        raw[col] = data.pivot(index="trade_date", columns="product_code", values=col).reindex(
            index=dates, columns=codes
        )
    return dates, raw


def select_nodes(raw: dict[str, pd.DataFrame], fold: Fold, min_coverage: float) -> list[str]:
    close = raw["close"]
    periods = [
        (fold.train_start, fold.train_end),
        (fold.val_start, fold.val_end),
        (fold.test_start, fold.test_end),
    ]
    keep = pd.Series(True, index=close.columns)
    for start, end in periods:
        keep &= close.loc[start:end].notna().mean() >= min_coverage
    return close.columns[keep].tolist()


def _signed_log1p(frame: pd.DataFrame) -> pd.DataFrame:
    return np.sign(frame) * np.log1p(np.abs(frame))


def engineer_features(raw: dict[str, pd.DataFrame], nodes: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = {k: v[nodes].copy() for k, v in raw.items()}
    fresh_spot = p["spot_price"].where(p["spot_price_age_days"] <= 30)
    spot_return = np.log(fresh_spot).diff()
    basis = np.log(fresh_spot / p["settle"])
    # FU and SC spot benchmarks are USD-denominated while futures are CNY-denominated.
    for incompatible in ("FU", "SC"):
        if incompatible in basis.columns:
            basis[incompatible] = np.nan
    stock_missing = p["stock_current"].isna().astype(float)
    available_missing = p["available_stock_current"].isna().astype(float)
    features = [
        p["settle_index"], p["settle_return"], p["close_return"],
        np.log(p["close"] / p["open"]), np.log(p["high"] / p["low"]),
        np.log1p(p["volume"].clip(lower=0)),
        np.log1p(p["open_interest"].clip(lower=0)), spot_return, basis,
        np.log1p(p["spot_price_age_days"].clip(lower=0)),
        np.log1p(p["warehouse_receipt_current"].clip(lower=0)),
        _signed_log1p(p["warehouse_receipt_change"]),
        np.log1p(p["warehouse_receipt_age_days"].clip(lower=0)),
        p["warehouse_receipt_conflict"],
        np.log1p(p["stock_current"].clip(lower=0)),
        np.log1p(p["stock_age_days"].clip(lower=0)), stock_missing,
        np.log1p(p["available_stock_current"].clip(lower=0)),
        np.log1p(p["available_stock_age_days"].clip(lower=0)), available_missing,
    ]
    x = np.stack([f.to_numpy(dtype=np.float32) for f in features], axis=-1)
    factor_price = p["close_index"].ffill(limit=3).to_numpy(dtype=np.float32)
    daily_return = p["close_return"].to_numpy(dtype=np.float32)
    return x, factor_price, daily_return


def _future_window(arr: np.ndarray, start: int, length: int) -> np.ndarray:
    return arr[start : start + length]


_CPD_LABEL_CACHE: dict[tuple[Any, ...], np.ndarray] = {}


def _interpolate_finite(values: np.ndarray) -> np.ndarray:
    """Fill rare gaps without allowing observations outside the CPD window."""
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros_like(values)
    indexes = np.arange(len(values))
    return np.interp(indexes, indexes[finite], values[finite])


def _cpd_label_for_window(
    values: np.ndarray,
    threshold: float,
    model: str,
    n_bkps: int,
    min_size: int,
    jump: int,
) -> np.float32:
    """Paper Eq. (4)-(6): Dynp segmentation then direction of the first change point.

    ``ruptures`` returns segment end indexes.  The first returned endpoint is the
    first future change point, and the next endpoint closes its following regime.
    Equation (6) labels that change point long when the regime's maximum rise from
    the change-point price exceeds ``threshold``; otherwise it is labelled short.
    """
    signal = _interpolate_finite(values)
    max_bkps = len(signal) // min_size - 1
    effective_bkps = min(n_bkps, max_bkps)
    if effective_bkps < 1:
        raise ValueError(
            f"CPD window length {len(signal)} is too short for min_size={min_size}"
        )
    breakpoints = rpt.Dynp(
        model=model, min_size=min_size, jump=jump
    ).fit(signal).predict(n_bkps=effective_bkps)
    change_index = int(breakpoints[0])
    next_index = int(breakpoints[1]) if len(breakpoints) > 1 else len(signal)
    change_price = signal[change_index]
    following_regime = signal[change_index:next_index]
    rise = float(np.max(following_regime) - change_price)
    return np.float32(rise > threshold * abs(change_price))


def _cpd_labels_for_node(
    series: np.ndarray,
    future_starts: np.ndarray,
    window: int,
    threshold: float,
    model: str,
    n_bkps: int,
    min_size: int,
    jump: int,
) -> np.ndarray:
    return np.asarray([
        _cpd_label_for_window(
            series[start : start + window], threshold, model, n_bkps, min_size, jump
        )
        for start in future_starts
    ], dtype=np.float32)


def _compute_cpd_labels(
    close: np.ndarray,
    future_starts: np.ndarray,
    window: int,
    threshold: float,
    model: str,
    n_bkps: int,
    min_size: int,
    jump: int,
    n_jobs: int,
) -> np.ndarray:
    # Four rolling folds currently share the same 37-node panel.  Content hashing
    # lets the expensive exact Dynp labels be safely reused during one preparation run.
    digest = hashlib.sha256(np.ascontiguousarray(close).view(np.uint8)).digest()
    key = (
        digest, tuple(future_starts.tolist()), window, threshold, model,
        n_bkps, min_size, jump,
    )
    if key not in _CPD_LABEL_CACHE:
        by_node = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(_cpd_labels_for_node)(
                close[:, node], future_starts, window, threshold,
                model, n_bkps, min_size, jump,
            )
            for node in range(close.shape[1])
        )
        _CPD_LABEL_CACHE[key] = np.stack(by_node, axis=1)
    return _CPD_LABEL_CACHE[key]


def create_samples(
    dates: pd.DatetimeIndex,
    x: np.ndarray,
    close: np.ndarray,
    daily_return: np.ndarray,
    input_window: int,
    gap_window: int,
    ma_window: int,
    cpd_window: int,
    cpd_threshold: float,
    cpd_model: str,
    cpd_n_bkps: int,
    cpd_min_size: int,
    cpd_jump: int,
    cpd_n_jobs: int,
) -> dict[str, np.ndarray]:
    longest = max(gap_window, ma_window, cpd_window)
    anchor_indexes = np.arange(input_window - 1, len(dates) - longest)
    cpd_labels = _compute_cpd_labels(
        close, anchor_indexes + 1, cpd_window, cpd_threshold, cpd_model,
        cpd_n_bkps, cpd_min_size, cpd_jump, cpd_n_jobs,
    )
    samples: dict[str, list[np.ndarray] | list[int]] = {
        "x": [], "price": [], "price_return": [], "current_close": [],
        "gap": [], "ma": [], "cpd": [],
        "anchor_idx": [], "factor_end_idx": [],
    }
    for row, t in enumerate(anchor_indexes):
        xw = x[t - input_window + 1 : t + 1]
        current = close[t]
        next_close = close[t + 1]
        gap_future = _future_window(close, t + 1, gap_window)
        ma_future = _future_window(close, t + 1, ma_window)
        price = next_close
        price_return = daily_return[t + 1]
        gap = (np.nanmax(gap_future, axis=0) - np.nanmin(gap_future, axis=0)) / gap_window
        ma = np.nanmean(ma_future, axis=0)
        cpd = cpd_labels[row]
        # Sparse stock fields are legitimate and have explicit missing indicators.
        # Reject only when core market features are insufficient or the next-day
        # return target is not observable for every retained graph node.
        if np.isfinite(xw[..., :7]).mean() < 0.90:
            continue
        if not (np.isfinite(next_close).all() and np.isfinite(price_return).all()):
            continue
        samples["x"].append(np.transpose(xw, (1, 0, 2)))
        samples["price"].append(price.astype(np.float32))
        samples["price_return"].append(price_return.astype(np.float32))
        samples["current_close"].append(current.astype(np.float32))
        samples["gap"].append(gap.astype(np.float32))
        samples["ma"].append(ma.astype(np.float32))
        samples["cpd"].append(cpd)
        samples["anchor_idx"].append(t)
        samples["factor_end_idx"].append(t + longest)
    return {k: np.asarray(v) for k, v in samples.items()}


def _fit_feature_scaler(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Z-score each futures series using training data only, as in the paper.
    mean = np.nanmean(x, axis=(0, 2), keepdims=True)
    std = np.nanstd(x, axis=(0, 2), keepdims=True)
    mean = np.nan_to_num(mean, nan=0.0)
    std = np.nan_to_num(std, nan=1.0)
    return mean.astype(np.float32), np.maximum(std, 1e-6).astype(np.float32)


def _fit_target_scaler(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(y, axis=0, keepdims=True)
    std = np.nanstd(y, axis=0, keepdims=True)
    return mean.astype(np.float32), np.maximum(std, 1e-6).astype(np.float32)


def prepare_fold(
    fold: Fold,
    dates: pd.DatetimeIndex,
    raw: dict[str, pd.DataFrame],
    cfg: dict[str, Any],
    out_dir: Path,
) -> dict[str, Any]:
    nodes = select_nodes(raw, fold, cfg["min_coverage"])
    if len(nodes) < cfg["min_nodes"]:
        raise ValueError(f"Fold {fold.fold} retained only {len(nodes)} nodes")
    x, close, daily_return = engineer_features(raw, nodes)
    arrays = create_samples(
        dates, x, close, daily_return, cfg["input_window"], cfg["gap_window"],
        cfg["ma_window"], cfg["cpd_window"], cfg["cpd_threshold"],
        cfg["cpd_model"], cfg["cpd_n_bkps"], cfg["cpd_min_size"],
        cfg["cpd_jump"], cfg["cpd_n_jobs"],
    )
    anchor_dates = dates[arrays["anchor_idx"]]
    factor_end_dates = dates[arrays["factor_end_idx"]]
    # Purging rule: all auxiliary labels must be fully observable inside their split.
    masks = {
        "train": (anchor_dates >= fold.train_start) & (factor_end_dates <= fold.train_end),
        "val": (anchor_dates >= fold.val_start) & (factor_end_dates <= fold.val_end),
        "test": (anchor_dates >= fold.test_start) & (factor_end_dates <= fold.test_end),
    }
    train_x = arrays["x"][masks["train"]]
    x_mean, x_std = _fit_feature_scaler(train_x)
    target_scalers = {
        task: _fit_target_scaler(arrays[task][masks["train"]])
        for task in ("price", "gap", "ma")
    }
    payload: dict[str, np.ndarray] = {
        "x_mean": x_mean, "x_std": x_std,
    }
    for task, (mean, std) in target_scalers.items():
        payload[f"{task}_mean"] = mean
        payload[f"{task}_std"] = std
    split_rows = []
    for split, mask in masks.items():
        split_x = np.nan_to_num((arrays["x"][mask] - x_mean) / x_std).astype(np.float32)
        payload[f"{split}_x"] = split_x
        payload[f"{split}_dates"] = anchor_dates[mask].strftime("%Y-%m-%d").to_numpy(dtype="U10")
        payload[f"{split}_anchor_idx"] = arrays["anchor_idx"][mask]
        payload[f"{split}_current_close"] = arrays["current_close"][mask].astype(np.float32)
        payload[f"{split}_price_return_raw"] = arrays["price_return"][mask].astype(np.float32)
        for task in ("price", "gap", "ma"):
            mean, std = target_scalers[task]
            payload[f"{split}_{task}"] = np.nan_to_num(
                (arrays[task][mask] - mean) / std
            ).astype(np.float32)
            payload[f"{split}_{task}_raw"] = arrays[task][mask].astype(np.float32)
        payload[f"{split}_cpd"] = arrays["cpd"][mask].astype(np.float32)
        split_rows.append({"split": split, "samples": int(mask.sum()),
                           "cpd_positive_rate": float(arrays["cpd"][mask].mean()),
                           "first_anchor": str(anchor_dates[mask].min().date()),
                           "last_anchor": str(anchor_dates[mask].max().date())})
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / f"fold_{fold.fold}.npz", **payload)
    meta = {
        **fold.__dict__, "nodes": nodes, "n_nodes": len(nodes),
        "features": FEATURE_COLUMNS, "splits": split_rows,
        "purge_days": max(cfg["gap_window"], cfg["ma_window"], cfg["cpd_window"]),
        "cpd": {
            "method": "ruptures.Dynp",
            "model": cfg["cpd_model"],
            "window": cfg["cpd_window"],
            "n_bkps": cfg["cpd_n_bkps"],
            "min_size": cfg["cpd_min_size"],
            "jump": cfg["cpd_jump"],
            "threshold": cfg["cpd_threshold"],
            "target": "direction_after_first_future_change_point",
        },
    }
    (out_dir / f"fold_{fold.fold}.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return meta
