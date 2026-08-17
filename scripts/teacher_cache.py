#!/usr/bin/env python3
"""Cache teacher in/out activations for the six pilot layers only."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from q38ternary.calibration import load_token_array  # noqa: E402
from q38ternary.config import load_config  # noqa: E402
from q38ternary.streaming_model import compare_to_reference  # noqa: E402
from q38ternary.teacher import cache_pilot_layers  # noqa: E402
from q38ternary.utils.logging import setup_logging  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    cfg = load_config()
    log = setup_logging("teacher_cache", cfg.root)
    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.validate:
        tokens = load_token_array(cfg)[:1, :16]
        metrics = compare_to_reference(tokens, cfg.model_local_dir)
        log.info("G2 %s", metrics)
        return 0

    tokens = load_token_array(cfg)
    log.info("caching %s sequences on %s for layers %s", tokens.shape, device, cfg.pilot_layers)
    out = cache_pilot_layers(cfg, tokens, device=device, chunk_size=args.chunk_size)
    log.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
