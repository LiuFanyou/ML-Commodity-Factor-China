from __future__ import annotations

from pathlib import Path

from factor_modeling.model_freeze import sha256


def test_sha256_changes_when_file_changes(tmp_path: Path) -> None:
    sample = tmp_path / "sample.txt"
    sample.write_text("before", encoding="utf-8")
    before = sha256(sample)
    sample.write_text("after", encoding="utf-8")
    assert sha256(sample) != before


def test_model_freeze_file_list_has_no_result_sentinel() -> None:
    from factor_modeling.model_freeze import MODEL_FREEZE_FILES

    assert "modeling/linear_ridge_v1/supplemental_2025/metrics_2025.csv" not in MODEL_FREEZE_FILES
