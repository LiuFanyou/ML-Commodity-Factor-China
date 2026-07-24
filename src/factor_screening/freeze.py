from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from .config import ProjectConfig


STATIC_FREEZE_FILES = [
    "config/screening.json",
    "中国商品期货Alpha因子库V2.1_分类版.md",
]


def freeze_files(root: Path) -> list[str]:
    source_files = [str(path.relative_to(root)) for path in sorted((root / "src" / "factor_screening").glob("*.py"))]
    return [*STATIC_FREEZE_FILES, *source_files]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def create_freeze_manifest(config: ProjectConfig) -> Path:
    path = config.root / "artifacts" / "FREEZE_MANIFEST.json"
    if path.exists():
        raise FileExistsError("freeze manifest already exists")
    files = {name: sha256(config.root / name) for name in freeze_files(config.root)}
    development_outputs = [
        "artifacts/development/feature_classification.csv",
        "artifacts/development/conceptual_factor_classification.csv",
        "features/ml_feature_registry.csv",
    ]
    outputs = {name: sha256(config.root / name) for name in development_outputs}
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_version": config.raw["dataset_version"],
        "development_end": config.raw["development_end"],
        "sealed_start": config.raw["sealed_start"],
        "sealed_end": config.raw["sealed_end"],
        "source_hashes": files,
        "development_output_hashes": outputs,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def verify_freeze_manifest(config: ProjectConfig) -> dict[str, object]:
    path = config.root / "artifacts" / "FREEZE_MANIFEST.json"
    if not path.exists():
        raise FileNotFoundError("sealed evaluation requires artifacts/FREEZE_MANIFEST.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    for name, expected in payload["source_hashes"].items():
        actual = sha256(config.root / name)
        if actual != expected:
            raise RuntimeError(f"frozen source changed: {name}")
    for name, expected in payload["development_output_hashes"].items():
        actual = sha256(config.root / name)
        if actual != expected:
            raise RuntimeError(f"frozen development output changed: {name}")
    return payload
