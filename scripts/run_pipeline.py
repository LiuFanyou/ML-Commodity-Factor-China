from __future__ import annotations

from pathlib import Path

from unified_factor_system.pipeline import run_pipeline


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    result = run_pipeline(root / "config" / "project.json")
    print(
        "completed",
        result["project_version"],
        "development_rows=",
        result["handoff"]["development_rows"],
        "feature_count=",
        result["handoff"]["feature_count"],
    )
