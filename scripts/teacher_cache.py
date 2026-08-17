#!/usr/bin/env python3
"""Run the streaming teacher on the calibration set, or validate Gate G2."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from q38ternary.calibration import load_token_array  # noqa: E402
from q38ternary.config import load_config  # noqa: E402
from q38ternary.streaming_model import compare_to_reference, forward_hidden  # noqa: E402
from q38ternary.utils.logging import setup_logging  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate", action="store_true", help="Gate G2 short-sequence check")
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--max-sequences", type=int, default=1)
    args = parser.parse_args(argv)

    cfg = load_config()
    log = setup_logging("teacher_cache", cfg.root)
    model_dir = cfg.model_local_dir

    if args.validate:
        tokens = load_token_array(cfg)[:1, :32]
        metrics = compare_to_reference(tokens, model_dir)
        log.info("G2 pass %s", metrics)
        cfg.resolve("artifacts").mkdir(parents=True, exist_ok=True)
        (cfg.resolve("artifacts") / "gate_g2.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )
        return 0

    tokens = load_token_array(cfg)[: args.max_sequences]
    result = forward_hidden(tokens, model_dir, through_layer=args.layer, want_logits=False)
    log.info("cached %s layers", len(result["hidden"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
