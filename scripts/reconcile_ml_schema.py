from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from factor_screening.config import load_config
from factor_screening.freeze import verify_freeze_manifest
from factor_screening.pipeline import _build_research_frame


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    config = load_config("config/screening.json")
    verify_freeze_manifest(config)
    registry_path = config.root / "features" / "ml_feature_registry.csv"
    matrix_path = config.root / "features" / "ml_features.csv.gz"
    audit_path = config.root / "artifacts" / "sealed" / "feature_schema_reconciliation.json"

    registry = pd.read_csv(registry_path)
    matrix = pd.read_csv(matrix_path, parse_dates=["trade_date"])
    required = registry["feature_name"].tolist()
    missing_columns = [name for name in required if name not in matrix.columns]
    invalid = [name for name in missing_columns if not name.endswith("__missing")]
    if invalid:
        raise RuntimeError(f"non-mask registry fields are absent from final matrix: {invalid}")

    before_hash = _sha256(matrix_path)
    added: list[str] = []
    if missing_columns:
        research, _, _, _, _ = _build_research_frame(config, config.raw["sealed_end"], "sealed")
        keys = ["trade_date", "product", "sector"]
        source_names = [name.removesuffix("__missing") for name in missing_columns]
        masks = research[keys].copy()
        for registry_name, source_name in zip(missing_columns, source_names, strict=True):
            if source_name not in research.columns:
                raise RuntimeError(f"cannot reconstruct missingness for {registry_name}: {source_name} absent")
            masks[registry_name] = research[source_name].isna().astype("int8")
            added.append(registry_name)
        matrix = matrix.merge(masks, on=keys, how="left", validate="one_to_one")

    absent_after = [name for name in required if name not in matrix.columns]
    if absent_after:
        raise RuntimeError(f"schema reconciliation failed: {absent_after}")
    ordered = ["trade_date", "product", "sector", *required]
    matrix = matrix[ordered]
    temporary = matrix_path.with_name("ml_features.reconciled.csv.gz")
    matrix.to_csv(temporary, index=False, compression="gzip")
    temporary.replace(matrix_path)

    payload = {
        "purpose": "align the final ML matrix to the frozen development feature registry",
        "sealed_metrics_recomputed": False,
        "factor_values_changed": False,
        "added_quality_mask_columns": added,
        "registry_feature_count": len(required),
        "matrix_row_count": len(matrix),
        "matrix_hash_before": before_hash,
        "matrix_hash_after": _sha256(matrix_path),
    }
    audit_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
