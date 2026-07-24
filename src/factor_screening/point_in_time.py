from __future__ import annotations

import pandas as pd

from .config import ProjectConfig


def run_point_in_time_audit(panel: pd.DataFrame, config: ProjectConfig, phase: str) -> dict[str, object]:
    selection_dates = pd.to_datetime(panel["selection_available_date"], errors="coerce")
    same_or_before = selection_dates.le(panel["trade_date"]).fillna(False)
    checks: dict[str, object] = {
        "selection_available_strictly_after_signal_date": not bool(same_or_before.any()),
        "selection_bad_row_count": int(same_or_before.sum()),
        "panel_key_unique": not bool(panel.duplicated(["trade_date", "product"]).any()),
        "fundamental_tables_loaded_into_strict_panel": False,
        "phase": phase,
        "panel_date_min": str(panel["trade_date"].min().date()),
        "panel_date_max": str(panel["trade_date"].max().date()),
    }
    if phase == "development":
        development_end = pd.Timestamp(config.raw["development_end"])
        checks["development_excludes_sealed_period"] = bool(panel["trade_date"].max() <= development_end)
    if not all(
        bool(checks[key])
        for key in [
            "selection_available_strictly_after_signal_date",
            "panel_key_unique",
        ]
    ):
        raise AssertionError(f"point-in-time audit failed: {checks}")
    if phase == "development" and not checks["development_excludes_sealed_period"]:
        raise AssertionError("development phase includes sealed 2025 data")
    return checks
