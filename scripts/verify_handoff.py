from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    handoff = root / "training_handoff"
    sealed = root / "sealed_evaluation"
    registry = pd.read_csv(handoff / "ml_feature_registry.csv")
    factor_registry = pd.read_csv(handoff / "factor_registry.csv")
    feature_names = registry["feature_name"].tolist()
    features = pd.read_csv(
        handoff / "ml_features_development.csv.gz",
        parse_dates=["trade_date"],
        low_memory=False,
    )
    labels = pd.read_csv(
        handoff / "labels_development.csv.gz",
        parse_dates=[
            "trade_date",
            "label_end_date_1d",
            "label_end_date_5d",
            "label_end_date_20d",
        ],
        low_memory=False,
    )
    dataset = pd.read_csv(
        handoff / "ml_dataset_development.csv.gz",
        parse_dates=[
            "trade_date",
            "label_end_date_1d",
            "label_end_date_5d",
            "label_end_date_20d",
        ],
        low_memory=False,
    )
    folds = pd.read_csv(
        handoff / "fold_definitions.csv",
        parse_dates=["train_start", "train_end", "valid_start", "valid_end"],
    )
    sealed_features = pd.read_csv(
        sealed / "ml_features_2025.csv.gz",
        parse_dates=["trade_date"],
        low_memory=False,
    )

    assert registry["feature_name"].is_unique
    assert len(factor_registry) == 57
    assert factor_registry["factor_no"].tolist() == list(range(1, 58))
    assert features.columns.tolist() == ["trade_date", "product", "sector", *feature_names]
    assert not features.duplicated(["trade_date", "product"]).any()
    assert not labels.duplicated(["trade_date", "product"]).any()
    assert not dataset.duplicated(["trade_date", "product"]).any()
    assert not features[feature_names].isna().any().any()
    assert not dataset[feature_names].isna().any().any()
    assert features["trade_date"].max() <= pd.Timestamp("2024-12-31")
    assert sealed_features["trade_date"].min() >= pd.Timestamp("2025-01-01")
    assert not dataset["label_end_date_5d"].gt(pd.Timestamp("2024-12-31")).any()
    assert len(features) == len(labels) == len(dataset)
    for fold in folds.itertuples(index=False):
        assert fold.train_end < fold.valid_start
        assert fold.valid_start <= fold.valid_end
        train = dataset.loc[
            dataset["trade_date"].between(fold.train_start, fold.train_end)
            & dataset["label_end_date_5d"].lt(fold.valid_start)
            & dataset["future_return_5d"].notna()
        ]
        assert not train.empty
        assert train["label_end_date_5d"].max() < fold.valid_start

    expected_hashes = {}
    for line in (handoff / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, filename = line.split("  ", maxsplit=1)
        expected_hashes[filename] = digest
    for filename, expected in expected_hashes.items():
        assert sha256(handoff / filename) == expected, filename
    run_manifest = json.loads(
        (root / "outputs" / "RUN_MANIFEST.json").read_text(encoding="utf-8")
    )
    config = json.loads((root / "config" / "project.json").read_text(encoding="utf-8"))
    data_file = (root / config["data"]["contract_file"]).resolve()
    assert sha256(data_file) == run_manifest["data_sha256"]
    factor_library = root / "中国商品期货Alpha因子库V2.1_分类版.md"
    assert sha256(factor_library) == run_manifest["factor_library_sha256"]
    for relative_path, expected in run_manifest["source_sha256"].items():
        assert sha256(root / relative_path) == expected, relative_path
    print(
        "handoff verified:",
        f"rows={len(dataset)}",
        f"features={len(feature_names)}",
        f"folds={len(folds)}",
    )


if __name__ == "__main__":
    main()
