from pathlib import Path

from unified_factor_system.config import load_config
from unified_factor_system.definitions import definitions_frame


def test_user_amendment_disables_t_minus_1_reselection() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "config" / "project.json")
    assert config.raw["main_contract"]["t_minus_1_selection_required"] is False
    assert config.raw["main_contract"]["rule"].startswith("direct_tushare")


def test_v21_registry_contains_all_57_factors() -> None:
    registry = definitions_frame()
    assert len(registry) == 57
    assert registry["factor_no"].tolist() == list(range(1, 58))


def test_expanding_folds_and_sealed_2025() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "config" / "project.json")
    folds = config.raw["folds"]
    assert [fold["fold_id"] for fold in folds] == [
        "fold_2019",
        "fold_2020",
        "fold_2021",
        "fold_2022",
        "fold_2023",
        "fold_2024",
    ]
    assert {fold["train_start"] for fold in folds} == {"2015-01-05"}
    assert config.raw["sealed_fold"]["fold_id"] == "sealed_2025"
