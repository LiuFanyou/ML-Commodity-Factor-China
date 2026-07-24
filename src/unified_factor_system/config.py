from __future__ import annotations

import json
import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ProjectConfig:
    project_root: Path
    repo_root: Path
    raw: dict[str, Any]

    def resolve_repo_path(self, value: str) -> Path:
        return (self.project_root / value).resolve()

    @property
    def data_file(self) -> Path:
        return self.resolve_repo_path(self.raw["data"]["contract_file"])

    @property
    def data_dictionary(self) -> Path:
        return self.resolve_repo_path(self.raw["data"]["data_dictionary"])

    @property
    def continuous_daily_files(self) -> list[Path]:
        pattern = self.raw["data"]["continuous_daily_glob"]
        return [
            Path(value).resolve()
            for value in sorted(glob.glob(str(self.project_root / pattern)))
        ]

    @property
    def output_root(self) -> Path:
        return self.project_root / "outputs"

    @property
    def handoff_root(self) -> Path:
        return self.project_root / "training_handoff"

    @property
    def sealed_root(self) -> Path:
        return self.project_root / "sealed_evaluation"

    @property
    def feature_root(self) -> Path:
        return self.project_root / "features"

    @property
    def report_root(self) -> Path:
        return self.project_root / "reports"


def load_config(path: str | Path) -> ProjectConfig:
    config_path = Path(path).resolve()
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    project_root = config_path.parent.parent
    return ProjectConfig(
        project_root=project_root,
        repo_root=project_root,
        raw=raw,
    )
