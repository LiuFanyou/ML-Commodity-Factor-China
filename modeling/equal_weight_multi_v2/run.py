"""Run equal_weight_multi_v2 from this directory."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    from factor_modeling.equal_weight_multi import main as equal_weight_main

    config = Path(__file__).resolve().parent / "config.json"
    return equal_weight_main(["--config", str(config), *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())
