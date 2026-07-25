"""Run protocol-aligned Ridge v1 2025 supplemental backtest."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    from factor_modeling.protocol_supplemental_2025 import main as protocol_main

    config = Path(__file__).resolve().parent / "supplemental_2025_protocol_config.json"
    return protocol_main(["--config", str(config), *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())
