"""Run or regenerate Ridge v1 2025 supplemental report.

Examples:

    # Regenerate markdown from already-saved CSVs (no feature matrix needed)
    py modeling/linear_ridge_v1/run_supplemental_2025.py --from-artifacts

    # Full re-run (requires features/ml_features.csv.gz + data_research + empty output)
    py modeling/linear_ridge_v1/run_supplemental_2025.py
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    from factor_modeling.supplemental_2025 import main as supplemental_main

    config = Path(__file__).resolve().parent / "supplemental_2025_config.json"
    return supplemental_main(["--config", str(config), *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())
