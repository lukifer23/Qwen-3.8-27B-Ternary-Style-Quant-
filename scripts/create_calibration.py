#!/usr/bin/env python3
"""Build the pilot (default) calibration set and a separate holdout."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from q38ternary.calibration import build_split  # noqa: E402
from q38ternary.config import load_config  # noqa: E402
from q38ternary.utils.logging import setup_logging  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", default=None, help="pilot | standard | deep")
    parser.add_argument("--holdout", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config()
    log = setup_logging("create_calibration", cfg.root)
    tier = args.tier or cfg.calibration_tier
    if args.holdout:
        manifest = build_split(cfg, tier=tier, holdout=True)
    else:
        manifest = build_split(cfg, tier=tier, holdout=False)
    log.info("packed %s sequences of length %s", manifest["sequences"], manifest["length"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
