"""Shared loader for ``ml_2025_backtest_pack`` under the unified protocol."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from factor_modeling.common import KEYS
from factor_modeling.handoff import (
    apply_label_end_purge,
    load_handoff_bundle,
    merge_features_and_labels,
)


def apply_execution_filters(panel: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    raw_return = config["raw_return_column"]
    usable = panel[raw_return].notna()
    if not bool(config.get("apply_execution_filters", True)):
        return panel.loc[usable].copy()
    if "entry_execution_available" in panel.columns:
        usable &= panel["entry_execution_available"].fillna(0).astype(int).eq(1)
    if "exit_execution_available_5d" in panel.columns:
        usable &= panel["exit_execution_available_5d"].fillna(0).astype(int).eq(1)
    return panel.loc[usable].copy()


def load_development_train(
    root: Path,
    config: dict[str, Any],
    feature_names: list[str],
) -> pd.DataFrame:
    """Load 2015–2024 training panel from ``training_handoff`` and purge into 2025."""
    bundle = load_handoff_bundle(root, config)
    raw_return = config["raw_return_column"]
    data = merge_features_and_labels(bundle, raw_return_column=raw_return)
    keep = [c for c in KEYS + ["sector"] + feature_names if c in data.columns]
    label_meta = [
        c
        for c in data.columns
        if c == raw_return or c == config.get("label_end_column", "label_end_date_5d")
    ]
    data = data[keep + [c for c in label_meta if c not in keep]].copy()
    data = data.loc[
        data["trade_date"].dt.year.between(
            int(config.get("train_start_year", 2015)),
            int(config.get("train_end_year", 2024)),
        )
    ].copy()
    valid_start = pd.Timestamp(f"{int(config.get('test_year', 2025))}-01-01")
    return apply_label_end_purge(
        data,
        valid_start,
        label_end_column=config.get("label_end_column", "label_end_date_5d"),
        purge_trading_days=int(config.get("purge_trading_days", 5)),
    )


def load_2025_pack(
    root: Path,
    config: dict[str, Any],
    feature_names: list[str],
) -> pd.DataFrame:
    """Load evaluable 2025 panel from ``ml_2025_backtest_pack``."""
    pack_dir = root / config.get("pack_2025_directory", "ml_2025_backtest_pack")
    feature_path = pack_dir / config.get("sealed_feature_file", "ml_features_2025.csv.gz")
    label_path = pack_dir / config.get("sealed_label_file", "labels_2025_evaluator_only.csv.gz")
    registry_path = pack_dir / config.get("pack_registry_file", "ml_feature_registry.csv")
    for path in (feature_path, label_path, registry_path):
        if not path.is_file():
            raise FileNotFoundError(f"2025 backtest pack file missing: {path}")

    pack_registry = pd.read_csv(registry_path)
    pack_names = pack_registry["feature_name"].tolist()
    if pack_names != feature_names:
        missing = sorted(set(feature_names) - set(pack_names))
        extra = sorted(set(pack_names) - set(feature_names))
        raise ValueError(
            "2025 pack registry does not match development feature list; "
            f"missing={missing[:10]} extra={extra[:10]}"
        )

    features = pd.read_csv(feature_path, parse_dates=["trade_date"])
    labels = pd.read_csv(label_path, parse_dates=["trade_date"])
    for column in ["entry_date", config.get("label_end_column", "label_end_date_5d")]:
        if column in labels.columns:
            labels[column] = pd.to_datetime(labels[column], errors="coerce")

    missing_features = sorted(set(feature_names) - set(features.columns))
    if missing_features:
        raise ValueError(f"2025 feature matrix missing columns: {missing_features[:20]}")

    raw_return = config["raw_return_column"]
    if raw_return not in labels.columns:
        raise ValueError(f"2025 label matrix missing {raw_return}")

    keep_feat = [c for c in KEYS + ["sector"] + feature_names if c in features.columns]
    keep_lab = [
        c
        for c in KEYS
        + [
            raw_return,
            config.get("label_end_column", "label_end_date_5d"),
            "entry_execution_available",
            "exit_execution_available_5d",
            "entry_suspected_limit_lock",
            "exit_suspected_limit_lock_5d",
        ]
        if c in labels.columns
    ]
    panel = features[keep_feat].merge(labels[keep_lab], on=KEYS, how="inner", validate="one_to_one")
    test_year = int(config.get("test_year", 2025))
    if set(panel["trade_date"].dt.year.unique()) != {test_year}:
        years = sorted(panel["trade_date"].dt.year.unique().tolist())
        raise ValueError(f"2025 pack contains unexpected years: {years}")
    return apply_execution_filters(panel, config)


def ensure_empty_output(output_dir: Path, sentinel_name: str = "metrics_2025.csv") -> Path:
    sentinel = output_dir / sentinel_name
    if sentinel.exists():
        raise FileExistsError(
            f"2025 supplemental already completed; overwrite is forbidden ({sentinel})"
        )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    return sentinel
