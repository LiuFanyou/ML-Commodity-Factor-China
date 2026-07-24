"""Run Ridge linear regression via the handoff matrix interface."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from factor_modeling.linear_ridge import main


if __name__ == "__main__":
    raise SystemExit(main(["--config", str(Path(__file__).with_name("config.json"))]))
