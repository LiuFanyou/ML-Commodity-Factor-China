"""Regenerate top-k shortlist / report from existing single-factor outputs."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from factor_modeling.single_factor_rank import (
    factor_metric_record,
    select_top_factors,
    write_report,
)


def main() -> int:
    config_path = ROOT / "modeling" / "single_factor_rank_v1" / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_dir = ROOT / config["output_directory"]
    report_path = ROOT / config["report_file"]

    leaderboard = pd.read_csv(output_dir / "leaderboard.csv")
    fold_metrics = pd.read_csv(output_dir / "fold_metrics.csv")
    factor_metrics = pd.read_csv(output_dir / "factor_metrics.csv")
    bundle_info = json.loads((output_dir / "handoff_summary.json").read_text(encoding="utf-8"))
    summary = json.loads((output_dir / "run_summary.json").read_text(encoding="utf-8"))

    top_k = int(config.get("top_k_factors", 3))
    top_factors = select_top_factors(leaderboard, config=config)
    top_factor_records = [factor_metric_record(row, config) for _, row in top_factors.iterrows()]
    per_factor_records = [factor_metric_record(row, config) for _, row in leaderboard.iterrows()]

    selection = config.get("top_k_selection") or {}
    selection_rule = [
        "eligible_usage_types=" + str(selection.get("usage_types", ["predictor"])),
        "exclude_missing_indicators=" + str(selection.get("exclude_missing_indicators", True)),
        "rank_ic_mean desc",
        "rank_ic_hit_rate desc",
        "gross_sharpe desc",
    ]

    top_factors.to_csv(output_dir / "top3_factors.csv", index=False)
    (output_dir / "top3_factors.json").write_text(
        json.dumps(
            {
                "selection_rule": selection_rule,
                "top_k": top_k,
                "note": (
                    "Full 92-field evaluation kept; shortlist restricted to predictor "
                    "universe excluding missing indicators."
                ),
                "factors": top_factor_records,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    summary["selection_rule"] = selection_rule
    summary["top_k_factors"] = top_k
    summary["top3_factors"] = top_factor_records
    summary["per_factor_metrics"] = per_factor_records
    summary["note"] = (
        "Primary audit output is per-factor metrics for all registered fields; "
        "top-k shortlist is selected only within predictor non-missing fields."
    )
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    candidates = leaderboard[["feature_name", "source_factor", "usage_type"]].copy()
    if "expected_direction" not in candidates.columns and "catalog_direction" in leaderboard.columns:
        candidates["expected_direction"] = leaderboard["catalog_direction"]
    write_report(
        report_path,
        config,
        candidates,
        factor_metrics,
        fold_metrics,
        leaderboard,
        top_factors,
        bundle_info,
    )
    print(json.dumps({"top3": top_factor_records}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
