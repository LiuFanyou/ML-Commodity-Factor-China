"""Run single_factor_rank_v1 from this directory."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    from factor_modeling.single_factor_rank import main as rank_main

    config = Path(__file__).resolve().parent / "config.json"
    return rank_main(["--config", str(config), *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())
