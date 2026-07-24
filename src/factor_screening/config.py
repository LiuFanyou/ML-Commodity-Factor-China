from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ProjectConfig:
    root: Path
    raw: dict[str, Any]

    @property
    def data_root(self) -> Path:
        return self.root / self.raw["data_root"]

    @property
    def tables(self) -> Path:
        return self.data_root / "tables"

    @property
    def primary_horizon(self) -> int:
        return int(self.raw["labels"]["primary_horizon"])

    @property
    def horizons(self) -> list[int]:
        return [int(x) for x in self.raw["labels"]["horizons"]]


def load_config(path: str | Path = "config/screening.json") -> ProjectConfig:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return ProjectConfig(root=config_path.parent.parent, raw=raw)
