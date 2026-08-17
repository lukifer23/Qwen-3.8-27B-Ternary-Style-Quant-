#!/usr/bin/env python3
"""Scales-only reconstruction of one cached layer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from q38ternary.config import load_config  # noqa: E402
from q38ternary.reconstruction.pilot import run_one_layer  # noqa: E402
from q38ternary.utils.logging import setup_logging  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    cfg = load_config()
    log = setup_logging("reconstruct_block", cfg.root)
    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    report, _packed = run_one_layer(cfg, args.layer, device=device, steps=args.steps, lr=args.lr)
    out = cfg.resolve("artifacts", "reports", f"layer_{args.layer:03d}.json")
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log.info("wrote %s gate=%s", out, report["gate_pass"])
    return 0 if report["gate_pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
