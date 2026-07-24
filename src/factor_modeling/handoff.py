"""Training-handoff matrix interface shared by Ridge and related models.

This module only loads and validates the frozen handoff package.
It does not compute factors or labels from raw market data.

Expected package layout (under ``data_directory``):

- ``ml_feature_registry.csv``      : feature_name (+ optional metadata)
- ``ml_features_*.csv.gz``         : trade_date, product, sector, feature columns
- ``labels_*.csv.gz``              : trade_date, product, future_return_*d, label_end_date_*d
- ``fold_definitions.csv``         : expanding-window train/valid bounds
- ``training_contract.json``       : optional contract overrides

Join key: ``trade_date + product``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

KEYS = ["trade_date", "product"]
IDENTIFIER_COLUMNS = ["trade_date", "product", "sector"]


@dataclass(frozen=True)
class HandoffBundle:
    """Validated handoff matrices ready for model training."""

    directory: Path
    contract: dict[str, Any]
    registry: pd.DataFrame
    features: pd.DataFrame
    labels: pd.DataFrame
    feature_names: list[str]

    @property
    def n_features(self) -> int:
        return len(self.feature_names)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_handoff_directory(root: Path, config: dict[str, Any]) -> Path:
    """Resolve handoff directory from model config.

    Priority:
    1. ``handoff_directory``
    2. ``data_directory``
    """
    relative = config.get("handoff_directory") or config.get("data_directory")
    if not relative:
        raise ValueError("config must set handoff_directory or data_directory")
    path = root / relative
    if not path.exists():
        raise FileNotFoundError(f"handoff directory not found: {path}")
    return path


def load_contract(handoff_dir: Path, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load training_contract.json if present, then overlay model-config file names."""
    config = config or {}
    contract_name = config.get("contract_file", "training_contract.json")
    contract_path = handoff_dir / contract_name
    contract: dict[str, Any] = {}
    if contract_path.exists():
        contract = _read_json(contract_path)

    # Model config wins for explicit file pointers so a new handoff drop-in is easy.
    for key in ("feature_file", "registry_file", "label_file"):
        if key in config and config[key]:
            contract[key] = config[key]
        elif key not in contract:
            defaults = {
                "feature_file": "ml_features_development.csv.gz",
                "registry_file": "ml_feature_registry.csv",
                "label_file": "labels_development.csv.gz",
            }
            contract[key] = defaults[key]

    contract.setdefault("join_keys", list(KEYS))
    contract.setdefault("identifier_columns", list(IDENTIFIER_COLUMNS))
    return contract


def _require_columns(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def _assert_unique_keys(frame: pd.DataFrame, name: str) -> None:
    duplicated = int(frame.duplicated(KEYS).sum())
    if duplicated:
        raise ValueError(f"{name} has {duplicated} duplicate trade_date+product keys")


def load_registry(path: Path) -> tuple[pd.DataFrame, list[str]]:
    registry = pd.read_csv(path)
    if "feature_name" not in registry.columns:
        raise ValueError(f"registry missing feature_name column: {path}")
    feature_names = [str(name) for name in registry["feature_name"].tolist()]
    if not feature_names:
        raise ValueError(f"registry is empty: {path}")
    if len(feature_names) != len(set(feature_names)):
        raise ValueError(f"registry feature_name contains duplicates: {path}")
    return registry, feature_names


def load_handoff_bundle(root: Path, config: dict[str, Any]) -> HandoffBundle:
    """Load and validate the handoff feature/label package.

    Does not engineer factors. Expects matrices already prepared to the
    training-handoff contract. Swap files under the handoff directory when a
    new ``data_final``-based factor package is ready.
    """
    handoff_dir = resolve_handoff_directory(root, config)
    contract = load_contract(handoff_dir, config)

    registry_path = handoff_dir / contract["registry_file"]
    feature_path = handoff_dir / contract["feature_file"]
    label_path = handoff_dir / contract["label_file"]
    for path in (registry_path, feature_path, label_path):
        if not path.exists():
            raise FileNotFoundError(f"handoff file not found: {path}")

    registry, feature_names = load_registry(registry_path)
    features = pd.read_csv(feature_path, parse_dates=["trade_date"])
    labels = pd.read_csv(label_path, parse_dates=["trade_date"])

    _require_columns(features, list(IDENTIFIER_COLUMNS), "feature matrix")
    _require_columns(labels, list(KEYS), "label matrix")
    _assert_unique_keys(features, "feature matrix")
    _assert_unique_keys(labels, "label matrix")

    missing_features = sorted(set(feature_names) - set(features.columns))
    if missing_features:
        raise ValueError(
            "registered features missing from feature matrix: "
            + ", ".join(missing_features[:20])
            + (" ..." if len(missing_features) > 20 else "")
        )

    # Keep only interface columns + registered features; ignore extras.
    feature_keep = list(IDENTIFIER_COLUMNS) + feature_names
    features = features.loc[:, feature_keep].copy()
    for name in feature_names:
        features[name] = pd.to_numeric(features[name], errors="coerce")

    return HandoffBundle(
        directory=handoff_dir,
        contract=contract,
        registry=registry,
        features=features,
        labels=labels,
        feature_names=feature_names,
    )


def load_fold_definitions(
    handoff_dir: Path,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Load expanding-window folds from the handoff package.

    Expected columns:
    ``fold_id, train_start, train_end, valid_start, valid_end, purge_days``
    """
    config = config or {}
    fold_name = config.get("fold_file") or config.get("split_file") or "fold_definitions.csv"
    path = handoff_dir / fold_name
    if not path.exists():
        raise FileNotFoundError(f"fold definitions not found: {path}")
    folds = pd.read_csv(
        path,
        parse_dates=["train_start", "train_end", "valid_start", "valid_end"],
    )
    required = ["fold_id", "train_start", "train_end", "valid_start", "valid_end"]
    _require_columns(folds, required, "fold definitions")
    if "purge_days" not in folds.columns:
        folds["purge_days"] = int(config.get("purge_trading_days", 5))
    return folds.sort_values("valid_start").reset_index(drop=True)


def apply_label_end_purge(
    train: pd.DataFrame,
    valid_start: pd.Timestamp,
    *,
    label_end_column: str,
    purge_trading_days: int = 0,
) -> pd.DataFrame:
    """Drop training rows whose label horizon overlaps the validation start."""
    from factor_modeling.common import purge_last_dates

    out = train.copy()
    start = pd.Timestamp(valid_start)
    if label_end_column in out.columns:
        out = out.loc[out[label_end_column].notna() & out[label_end_column].lt(start)].copy()
    elif purge_trading_days > 0:
        out = purge_last_dates(out, purge_trading_days)
    return out


def split_fold(
    data: pd.DataFrame,
    fold: pd.Series | Any,
    *,
    target_column: str,
    label_end_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split one expanding-window fold with label-end purge."""
    train = data.loc[
        data["trade_date"].between(fold.train_start, fold.train_end)
        & data[target_column].notna()
    ].copy()
    train = apply_label_end_purge(
        train,
        fold.valid_start,
        label_end_column=label_end_column,
        purge_trading_days=int(getattr(fold, "purge_days", 5)),
    )
    valid = data.loc[
        data["trade_date"].between(fold.valid_start, fold.valid_end)
        & data[target_column].notna()
    ].copy()
    return train, valid


def merge_features_and_labels(
    bundle: HandoffBundle,
    *,
    raw_return_column: str,
    how: str = "inner",
) -> pd.DataFrame:
    """Join features and labels on trade_date+product."""
    if raw_return_column not in bundle.labels.columns:
        available = [c for c in bundle.labels.columns if c.startswith("future_return_")]
        raise ValueError(
            f"label column {raw_return_column!r} not found; available targets: {available}"
        )
    label_cols = list(KEYS) + [raw_return_column]
    # Preserve optional label metadata if present.
    for optional in (
        "label_end_date_1d",
        "label_end_date_5d",
        "label_end_date_20d",
        "mapped_actual_ts_code",
        "entry_date",
    ):
        if optional in bundle.labels.columns:
            label_cols.append(optional)

    labels = bundle.labels[label_cols].copy()
    for column in labels.columns:
        if column.startswith("label_end_date_") or column in {"entry_date"}:
            labels[column] = pd.to_datetime(labels[column], errors="coerce")

    merged = bundle.features.merge(
        labels,
        on=KEYS,
        how=how,
        validate="one_to_one",
    )
    return merged


def handoff_summary(bundle: HandoffBundle) -> dict[str, Any]:
    return {
        "handoff_directory": str(bundle.directory),
        "n_features": bundle.n_features,
        "feature_rows": len(bundle.features),
        "label_rows": len(bundle.labels),
        "feature_start": str(bundle.features["trade_date"].min().date()),
        "feature_end": str(bundle.features["trade_date"].max().date()),
        "n_products": int(bundle.features["product"].nunique()),
        "registry_file": bundle.contract.get("registry_file"),
        "feature_file": bundle.contract.get("feature_file"),
        "label_file": bundle.contract.get("label_file"),
    }
