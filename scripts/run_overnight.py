#!/usr/bin/env python3
"""Overnight unattended job: reconstruct all layers, write a runnable GGUF, smoke-test.

Prints live [layer/total] elapsed/ETA/gpu/ram lines. Safe to re-run; completed
layers are skipped.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from q38ternary.config import load_config  # noqa: E402
from q38ternary.overnight import run  # noqa: E402
from q38ternary.utils.logging import setup_logging  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--start-layer", type=int, default=0)
    parser.add_argument("--end-layer", type=int, default=63)
    parser.add_argument("--force", action="store_true", help="run even if the GPU is already busy")
    parser.add_argument("--skip-gguf", action="store_true")
    parser.add_argument(
        "--compile",
        action="store_true",
        dest="compile_if_missing",
        help="cmake-build Prism llama.cpp if llama-cli is not on PATH",
    )
    args = parser.parse_args(argv)

    import os

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    cfg = load_config()
    setup_logging("overnight", cfg.root)
    run(
        cfg,
        device=args.device,
        steps=args.steps,
        lr=args.lr,
        chunk_size=args.chunk_size,
        start_layer=args.start_layer,
        end_layer=args.end_layer,
        force=args.force,
        skip_gguf=args.skip_gguf,
        compile_if_missing=args.compile_if_missing,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
