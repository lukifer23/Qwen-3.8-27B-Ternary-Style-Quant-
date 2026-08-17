#!/usr/bin/env python3
"""Run scales-only reconstruction on every configured pilot layer and write the gate report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from q38ternary.config import load_config  # noqa: E402
from q38ternary.reconstruction.pilot import run_one_layer, write_pilot_report  # noqa: E402
from q38ternary.utils.logging import setup_logging  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default=None)
    parser.add_argument("--layers", default=None, help="comma-separated, default from config")
    args = parser.parse_args(argv)

    cfg = load_config()
    log = setup_logging("layer_pilot", cfg.root)
    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    layers = [int(x) for x in args.layers.split(",")] if args.layers else list(cfg.pilot_layers)
    reports = []
    for layer_idx in layers:
        log.info("=== layer %s ===", layer_idx)
        report, _packed = run_one_layer(cfg, layer_idx, device=device, steps=args.steps, lr=args.lr)
        reports.append(report)
    path = write_pilot_report(cfg, reports)
    log.info("wrote %s", path)
    if not any(item["gate_pass"] for item in reports):
        log.error("G4 FAIL: reconstruction did not improve any pilot layer")
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
