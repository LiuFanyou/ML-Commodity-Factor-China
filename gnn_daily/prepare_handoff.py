from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from hcgnn.data import _compute_cpd_labels


def adjusted_close_panel(csv_path: Path, dates: pd.DatetimeIndex, products: list[str]) -> np.ndarray:
    cols = [
        "trade_date", "product_code", "close", "main_contract_changed",
        "old_contract_open_on_roll", "new_contract_open_on_roll",
    ]
    frame = pd.read_csv(csv_path, usecols=cols, parse_dates=["trade_date"])
    frame = frame[frame["trade_date"].isin(dates) & frame["product_code"].isin(products)].copy()
    valid_roll = (
        frame["main_contract_changed"].eq(1)
        & frame["old_contract_open_on_roll"].gt(0)
        & frame["new_contract_open_on_roll"].gt(0)
    )
    frame["roll_multiplier"] = 1.0
    frame.loc[valid_roll, "roll_multiplier"] = (
        frame.loc[valid_roll, "old_contract_open_on_roll"]
        / frame.loc[valid_roll, "new_contract_open_on_roll"]
    )
    frame = frame.sort_values(["product_code", "trade_date"])
    frame["cumulative_adjustment"] = frame.groupby("product_code")["roll_multiplier"].cumprod()
    frame["adjusted_close"] = frame["close"] * frame["cumulative_adjustment"]
    panel = frame.pivot(index="trade_date", columns="product_code", values="adjusted_close")
    return panel.reindex(index=dates, columns=products).to_numpy(dtype=np.float32)


def future_regression_labels(price: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    labels = np.full(price.shape, np.nan, dtype=np.float32)
    masks = np.zeros(price.shape, dtype=bool)
    for t in range(len(price) - window):
        future = price[t + 1 : t + 1 + window]
        valid = np.isfinite(future).all(axis=0)
        if window == 20:
            value = (np.nanmax(future, axis=0) - np.nanmin(future, axis=0)) / window
        else:
            value = np.nanmean(future, axis=0)
        labels[t, valid] = value[valid]
        masks[t, valid] = True
    return labels, masks


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare factor-handoff panel for protocol GNN")
    parser.add_argument("--config", default="config_handoff.yaml")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    base = config_path.parent
    handoff = (base / cfg["handoff_dir"]).resolve()
    out = (base / cfg["output_dir"] / "prepared").resolve()
    out.mkdir(parents=True, exist_ok=True)

    registry = pd.read_csv(handoff / "ml_feature_registry.csv")
    feature_names = registry["feature_name"].tolist()
    features = pd.read_csv(handoff / "ml_features_development.csv.gz", parse_dates=["trade_date"])
    labels = pd.read_csv(
        handoff / "labels_development.csv.gz",
        parse_dates=["trade_date", "label_end_date_1d", "label_end_date_5d", "label_end_date_20d"],
        low_memory=False,
    )
    cutoff = pd.Timestamp(cfg["development_end"])
    features = features[features["trade_date"] <= cutoff]
    labels = labels[labels["trade_date"] <= cutoff]
    data = features.merge(labels, on=["trade_date", "product"], how="inner", validate="one_to_one")
    dates = pd.DatetimeIndex(sorted(data["trade_date"].unique()))
    products = sorted(data["product"].unique())
    sectors = features.drop_duplicates("product").set_index("product")["sector"].reindex(products)

    factor_panel = np.stack([
        data.pivot(index="trade_date", columns="product", values=name)
        .reindex(index=dates, columns=products).to_numpy(dtype=np.float32)
        for name in feature_names
    ], axis=-1)
    payload: dict[str, np.ndarray] = {
        "dates": dates.strftime("%Y-%m-%d").to_numpy(dtype="U10"),
        "products": np.asarray(products, dtype="U8"),
        "sectors": sectors.to_numpy(dtype="U16"),
        "features": factor_panel,
        "feature_names": np.asarray(feature_names, dtype="U80"),
    }
    for target in ("future_return_1d", "future_return_5d", "future_return_20d"):
        panel = data.pivot(index="trade_date", columns="product", values=target).reindex(
            index=dates, columns=products
        ).to_numpy(dtype=np.float32)
        payload[target] = panel
        payload[f"{target}_mask"] = np.isfinite(panel)
    label_end = data.pivot(index="trade_date", columns="product", values="label_end_date_5d").reindex(
        index=dates, columns=products
    )
    payload["label_end_date_5d"] = label_end.apply(
        lambda col: col.dt.strftime("%Y-%m-%d")
    ).fillna("").to_numpy(dtype="U10")

    price = adjusted_close_panel((base / cfg["main_csv"]).resolve(), dates, products)
    payload["adjusted_close"] = price
    dc = cfg["data"]
    auxiliary_path = cfg.get("reuse_auxiliary_panel")
    if auxiliary_path:
        reused = np.load((base / auxiliary_path).resolve(), allow_pickle=False)
        if not np.array_equal(reused["dates"], payload["dates"]):
            raise ValueError("Cannot reuse auxiliary targets: date indexes differ")
        if not np.array_equal(reused["products"], payload["products"]):
            raise ValueError("Cannot reuse auxiliary targets: product indexes differ")
        if not np.allclose(reused["adjusted_close"], price, equal_nan=True):
            raise ValueError("Cannot reuse auxiliary targets: adjusted-close panels differ")
        for name in ("gap", "gap_mask", "ma", "ma_mask", "cpd", "cpd_mask"):
            payload[name] = reused[name]
    else:
        gap, gap_mask = future_regression_labels(price, 20)
        ma, ma_mask = future_regression_labels(price, 40)
        payload.update({"gap": gap, "gap_mask": gap_mask, "ma": ma, "ma_mask": ma_mask})
        cpd = np.full(price.shape, np.nan, dtype=np.float32)
        cpd_mask = np.zeros(price.shape, dtype=bool)
        anchors = np.arange(0, len(dates) - dc["cpd_window"])
        computed = _compute_cpd_labels(
            price, anchors + 1, dc["cpd_window"], dc["cpd_threshold"], dc["cpd_model"],
            dc["cpd_n_bkps"], dc["cpd_min_size"], dc["cpd_jump"], dc["cpd_n_jobs"],
        )
        future_valid = np.stack([
            np.isfinite(price[t + 1 : t + 1 + dc["cpd_window"]]).all(axis=0)
            for t in anchors
        ])
        cpd[anchors] = computed
        cpd_mask[anchors] = future_valid
        payload.update({"cpd": cpd, "cpd_mask": cpd_mask})

    np.savez_compressed(out / "protocol_panel.npz", **payload)
    kept_folds = pd.read_csv(handoff / "fold_definitions.csv")
    kept_folds = kept_folds[pd.to_datetime(kept_folds["valid_end"]) <= cutoff]
    kept_folds.to_csv(out / "fold_definitions.csv", index=False)
    meta = {
        "source_handoff": str(handoff),
        "development_end": str(cutoff.date()),
        "n_dates": len(dates), "n_products": len(products), "n_features": len(feature_names),
        "products": products, "features": feature_names,
        "cpd": {k: dc[k] for k in dc if k.startswith("cpd_")},
        "reused_auxiliary_panel": str((base / auxiliary_path).resolve()) if auxiliary_path else None,
        "folds": kept_folds.to_dict(orient="records"),
    }
    (out / "protocol_panel.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: meta[k] for k in ("development_end", "n_dates", "n_products", "n_features")}, indent=2))


if __name__ == "__main__":
    main()
