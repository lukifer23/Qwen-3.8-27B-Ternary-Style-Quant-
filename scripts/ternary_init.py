#!/usr/bin/env python3
"""Initialize a weight tensor (or a .npy file) with one of the four ternary initializers."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from q38ternary.config import load_config  # noqa: E402
from q38ternary.quant.ternary import (  # noqa: E402
    dequantize,
    quantize_absolute,
    quantize_activation_weighted,
    quantize_hessian_diag,
    quantize_search,
)
from q38ternary.utils.logging import setup_logging  # noqa: E402
from q38ternary.utils.manifest import write_manifest  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="NumPy .npy weight matrix")
    parser.add_argument("--activations", default=None, help="Optional .npy activation matrix X")
    parser.add_argument(
        "--initializer",
        choices=("absolute", "search", "activation", "hessian"),
        default="search",
    )
    parser.add_argument("--tau", type=float, default=0.0)
    parser.add_argument("--group-size", type=int, default=None)
    parser.add_argument("--out", required=True, help="Directory for codes.npy / scales.json")
    args = parser.parse_args(argv)

    cfg = load_config()
    log = setup_logging("ternary_init", cfg.root)
    group_size = args.group_size or cfg.group_size
    weights = np.load(args.input)

    if args.initializer == "absolute":
        packed = quantize_absolute(weights, group_size=group_size, tau=args.tau)
    elif args.initializer == "search":
        packed = quantize_search(weights, group_size=group_size)
    else:
        if not args.activations:
            raise SystemExit("--activations is required for activation/hessian initializers")
        act = np.load(args.activations)
        if args.initializer == "activation":
            packed = quantize_activation_weighted(weights, act, group_size=group_size)
        else:
            packed = quantize_hessian_diag(weights, act, group_size=group_size)

    recon = dequantize(packed)
    mse = float(np.mean((weights - recon) ** 2))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "codes.npy", packed.codes)
    np.save(out / "scales.npy", packed.scales)
    meta = {
        "scheme": packed.scheme,
        "group_size": packed.group_size,
        "shape": list(packed.original_shape),
        "weight_mse": mse,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    write_manifest(cfg, out, kind="ternary_init", extra=meta)
    log.info("wrote %s mse=%.6e scheme=%s", out, mse, packed.scheme)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
