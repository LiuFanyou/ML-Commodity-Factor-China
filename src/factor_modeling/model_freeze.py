from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from factor_screening.config import load_config
from factor_screening.freeze import verify_freeze_manifest


MODEL_FREEZE_FILES = [
    "src/factor_modeling/__init__.py",
    "src/factor_modeling/linear_ridge.py",
    "src/factor_modeling/model_freeze.py",
    "src/factor_modeling/supplemental_2025.py",
    "modeling/linear_ridge_v1/config.json",
    "modeling/linear_ridge_v1/supplemental_2025_config.json",
    "modeling/linear_ridge_v1/README.md",
    "training_handoff/ml_feature_registry.csv",
    "features/ml_features.csv.gz",
    "training_handoff/ml_features_development.csv.gz",
    "training_handoff/labels_development.csv.gz",
    "modeling/linear_ridge_v1/outputs/fold_metrics.csv",
    "modeling/linear_ridge_v1/outputs/coefficient_summary.csv",
    "modeling/linear_ridge_v1/outputs/run_summary.json",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def create_model_freeze_manifest(root: Path) -> Path:
    root = root.resolve()
    path = root / "modeling" / "linear_ridge_v1" / "MODEL_FREEZE_MANIFEST.json"
    if path.exists():
        raise FileExistsError(f"model freeze manifest already exists: {path}")

    factor_config = load_config(root / "config" / "screening.json")
    verify_freeze_manifest(factor_config)
    missing = [name for name in MODEL_FREEZE_FILES if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"model freeze files missing: {missing}")

    factor_manifest = root / "artifacts" / "FREEZE_MANIFEST.json"
    payload: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_version": "linear_ridge_v1",
        "status": "frozen_before_supplemental_2025",
        "supplemental_train_period": "2022-01-01/2024-12-31",
        "supplemental_test_period": "2025-01-01/2025-12-31",
        "factor_freeze_manifest_sha256": sha256(factor_manifest),
        "frozen_hashes": {name: sha256(root / name) for name in MODEL_FREEZE_FILES},
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def verify_model_freeze_manifest(root: Path) -> dict[str, Any]:
    root = root.resolve()
    factor_config = load_config(root / "config" / "screening.json")
    verify_freeze_manifest(factor_config)

    path = root / "modeling" / "linear_ridge_v1" / "MODEL_FREEZE_MANIFEST.json"
    if not path.exists():
        raise FileNotFoundError("2025 supplemental test requires MODEL_FREEZE_MANIFEST.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    factor_manifest = root / "artifacts" / "FREEZE_MANIFEST.json"
    if sha256(factor_manifest) != payload["factor_freeze_manifest_sha256"]:
        raise RuntimeError("factor freeze manifest changed after model freeze")
    for name, expected in payload["frozen_hashes"].items():
        file_path = root / name
        if not file_path.is_file():
            raise FileNotFoundError(f"frozen model file missing: {name}")
        if sha256(file_path) != expected:
            raise RuntimeError(f"frozen model file changed: {name}")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="冻结Ridge V1模型及2025补充测试接口")
    parser.add_argument("action", choices=["create", "verify"])
    parser.add_argument("--root", default=".")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    if args.action == "create":
        result: Any = str(create_model_freeze_manifest(root))
    else:
        result = verify_model_freeze_manifest(root)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
