from __future__ import annotations

import argparse
import copy
import json
import random
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from hcgnn.model import HCGNNDaily
from hcgnn.trainer import ContinualTrainer, choose_device, predict


def _masked_node_stats(values: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    safe = np.where(mask, values, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean = np.nanmean(safe, axis=0, keepdims=True)
        std = np.nanstd(safe, axis=0, keepdims=True)
    return np.nan_to_num(mean, nan=0.0).astype(np.float32), np.maximum(
        np.nan_to_num(std, nan=1.0), 1e-6
    ).astype(np.float32)


def _end_mask(dates: pd.DatetimeIndex, anchors: np.ndarray, horizon: int, boundary: pd.Timestamp) -> np.ndarray:
    result = np.zeros(len(anchors), dtype=bool)
    possible = anchors + horizon < len(dates)
    result[possible] = dates[anchors[possible] + horizon] < boundary
    return result


def build_fold_data(panel: np.lib.npyio.NpzFile, fold: pd.Series, input_window: int) -> dict[str, np.ndarray]:
    dates = pd.DatetimeIndex(pd.to_datetime(panel["dates"]))
    anchors = np.arange(input_window - 1, len(dates))
    # NPZ members are decompressed on every ``panel[key]`` access.  Load the
    # feature cube once; accessing it inside this loop retains thousands of
    # separately decompressed backing arrays and exhausts memory at 92 features.
    feature_cube = panel["features"]
    sequences = np.stack([feature_cube[i - input_window + 1 : i + 1] for i in anchors])
    sequences = np.transpose(sequences, (0, 2, 1, 3)).astype(np.float32)
    anchor_dates = dates[anchors]
    train_start, train_end = pd.Timestamp(fold.train_start), pd.Timestamp(fold.train_end)
    valid_start, valid_end = pd.Timestamp(fold.valid_start), pd.Timestamp(fold.valid_end)
    train_rows = (
        (anchor_dates >= train_start) & (anchor_dates <= train_end)
        & (dates[anchors - input_window + 1] >= train_start)
    )
    valid_rows = (anchor_dates >= valid_start) & (anchor_dates <= valid_end)

    train_seq = sequences[train_rows]
    mean = np.nanmean(train_seq, axis=(0, 1, 2), keepdims=True)
    std = np.nanstd(train_seq, axis=(0, 1, 2), keepdims=True)
    mean, std = np.nan_to_num(mean, nan=0.0), np.maximum(np.nan_to_num(std, nan=1.0), 1e-6)
    sequences = np.nan_to_num((sequences - mean) / std).astype(np.float32)

    price_raw = panel["future_return_5d"][anchors].astype(np.float32)
    price_mask = panel["future_return_5d_mask"][anchors].astype(bool)
    label_end = panel["label_end_date_5d"][anchors]
    label_end_days = np.where(label_end == "", "NaT", label_end).astype("datetime64[D]")
    train_price_boundary = label_end_days < np.datetime64(valid_start.date())
    price_mean = np.asarray([[np.nanmean(price_raw[train_rows][price_mask[train_rows]])]], dtype=np.float32)
    price_std = np.asarray([[max(np.nanstd(price_raw[train_rows][price_mask[train_rows]]), 1e-6)]], dtype=np.float32)

    data: dict[str, np.ndarray] = {"price_mean": price_mean, "price_std": price_std}
    task_source = {"cpd": panel["cpd"][anchors], "gap": panel["gap"][anchors],
                   "ma": panel["ma"][anchors]}
    task_masks = {"cpd": panel["cpd_mask"][anchors], "gap": panel["gap_mask"][anchors],
                  "ma": panel["ma_mask"][anchors]}
    horizons = {"cpd": 60, "gap": 20, "ma": 40}
    row_sets = {"train": train_rows, "val": valid_rows, "test": valid_rows}
    for split, rows in row_sets.items():
        data[f"{split}_x"] = sequences[rows]
        data[f"{split}_dates"] = anchor_dates[rows].strftime("%Y-%m-%d").to_numpy(dtype="U10")
        data[f"{split}_price_raw"] = price_raw[rows]
        data[f"{split}_price"] = np.nan_to_num((price_raw[rows] - price_mean) / price_std).astype(np.float32)
        mask = price_mask[rows].copy()
        if split == "train":
            mask &= train_price_boundary[rows]
        data[f"{split}_price_mask"] = mask

    for task, values in task_source.items():
        train_mask_all = task_masks[task] & _end_mask(dates, anchors, horizons[task], valid_start)[:, None]
        seen_in_training = train_mask_all[train_rows].any(axis=0)
        val_end_ok = np.zeros(len(anchors), dtype=bool)
        possible = anchors + horizons[task] < len(dates)
        val_end_ok[possible] = dates[anchors[possible] + horizons[task]] <= valid_end
        val_mask_all = task_masks[task] & val_end_ok[:, None] & seen_in_training[None, :]
        if task == "cpd":
            task_mean, task_std = np.zeros((1, values.shape[1]), dtype=np.float32), np.ones(
                (1, values.shape[1]), dtype=np.float32
            )
            scaled = np.nan_to_num(values).astype(np.float32)
        else:
            task_mean, task_std = _masked_node_stats(values[train_rows], train_mask_all[train_rows])
            scaled = np.nan_to_num((values - task_mean) / task_std).astype(np.float32)
        data[f"{task}_mean"], data[f"{task}_std"] = task_mean, task_std
        for split, rows in row_sets.items():
            data[f"{split}_{task}"] = scaled[rows]
            data[f"{split}_{task}_mask"] = (train_mask_all if split == "train" else val_mask_all)[rows]
    return data


def prediction_metrics(actual: np.ndarray, predicted: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    pearson, rank_ic = [], []
    for y, p, valid in zip(actual, predicted, mask):
        if valid.sum() < 3:
            continue
        pearson.append(pd.Series(y[valid]).corr(pd.Series(p[valid]), method="pearson"))
        rank_ic.append(pd.Series(y[valid]).corr(pd.Series(p[valid]), method="spearman"))
    p_arr, r_arr = np.asarray(pearson), np.asarray(rank_ic)
    return {
        "ic_mean": float(np.nanmean(p_arr)), "rank_ic": float(np.nanmean(r_arr)),
        "ic_std": float(np.nanstd(p_arr)),
        "ic_ir": float(np.nanmean(p_arr) / (np.nanstd(p_arr) + 1e-12)),
        "p_ic_gt_002": float(np.nanmean(p_arr > 0.02)),
        "p_ic_lt_neg002": float(np.nanmean(p_arr < -0.02)),
        "rmse": float(np.sqrt(np.mean((predicted[mask] - actual[mask]) ** 2))),
        "direction_accuracy": float(np.mean(np.sign(predicted[mask]) == np.sign(actual[mask]))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train protocol GNN on factor-handoff data")
    parser.add_argument("--config", default="config_handoff.yaml")
    parser.add_argument("--fold", required=True)
    parser.add_argument("--variant", choices=["protocol_gnn", "hcgnn_continual"], required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if args.smoke:
        cfg["training"]["epochs_per_task"] = 1
        cfg["training"]["patience"] = 1
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    base = config_path.parent
    prepared = base / cfg["output_dir"] / "prepared"
    panel = np.load(prepared / "protocol_panel.npz", allow_pickle=False)
    folds = pd.read_csv(prepared / "fold_definitions.csv")
    selected = folds[folds["fold_id"] == args.fold]
    if len(selected) != 1:
        raise ValueError(f"Unknown fold: {args.fold}")
    fold = next(selected.itertuples(index=False))
    data = build_fold_data(panel, fold, cfg["data"]["input_window"])
    if args.smoke:
        # Smoke validates the complete train/predict/save interface, not model quality.
        # Limit each split to one batch so a CPU-only workstation does not run a
        # full rolling-fold epoch merely to check shapes and serialization.
        for split in ("train", "val", "test"):
            split_size = len(data[f"{split}_x"])
            limit = min(split_size, cfg["training"]["batch_size"])
            for key, value in list(data.items()):
                if (
                    key.startswith(f"{split}_")
                    and isinstance(value, np.ndarray)
                    and value.ndim > 0
                    and len(value) == split_size
                ):
                    data[key] = value[:limit]
    device = choose_device(cfg["training"]["device"])
    model = HCGNNDaily(n_features=data["train_x"].shape[-1], **cfg["model"])
    trainer = ContinualTrainer(model, cfg["training"], device)
    tasks = cfg["variants"][args.variant]
    history = trainer.fit(data, tasks)
    _, predicted = predict(model, data, "test", cfg["training"]["batch_size"], device)
    actual, mask = data["test_price_raw"], data["test_price_mask"]
    metrics = prediction_metrics(actual, predicted, mask)

    run_dir = base / cfg["output_dir"] / args.variant / args.fold
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": cfg, "products": panel["products"].tolist(),
                "features": panel["feature_names"].tolist(), "tasks": tasks}, run_dir / "model.pt")
    (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    result: dict[str, Any] = {"fold_id": args.fold, "variant": args.variant, "device": str(device), **metrics}
    (run_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    rows = []
    products, sectors = panel["products"], panel["sectors"]
    for date, y, pred, valid in zip(data["test_dates"], actual, predicted, mask):
        for j in np.flatnonzero(valid):
            rows.append({"trade_date": date, "product": products[j], "sector": sectors[j],
                         "future_return_5d": float(y[j]), "prediction": float(pred[j]),
                         "fold_id": args.fold, "variant": args.variant})
    pd.DataFrame(rows).to_csv(run_dir / "predictions.csv", index=False)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
