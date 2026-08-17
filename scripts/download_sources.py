#!/usr/bin/env python3
"""Clone third-party repos, fetch the teacher config, write the architecture map."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from q38ternary.architecture import write_architecture_reports  # noqa: E402
from q38ternary.config import load_config  # noqa: E402
from q38ternary.hf import download_config, verify_teacher_identity, read_config_json  # noqa: E402
from q38ternary.inventory import write_inventory  # noqa: E402
from q38ternary.runtime import inspect_runtimes  # noqa: E402
from q38ternary.sources import sync_third_party  # noqa: E402
from q38ternary.size import write_size_report  # noqa: E402
from q38ternary.utils.logging import setup_logging  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-clone", action="store_true")
    parser.add_argument("--weights", action="store_true", help="also download the full BF16 snapshot")
    args = parser.parse_args(argv)

    cfg = load_config()
    log = setup_logging("download_sources", cfg.root)

    if not args.skip_clone:
        versions = sync_third_party(cfg)
        log.info("pinned %s", {k: v["commit"] for k, v in versions.items()})

    dest = download_config(cfg)
    config = read_config_json(dest)
    verify_teacher_identity(cfg, config)
    arch = write_architecture_reports(cfg, dest)
    inventory = write_inventory(cfg, dest)
    sizes = write_size_report(cfg, arch, inventory=inventory)
    runtimes = inspect_runtimes(cfg)
    log.info("runtimes %s", runtimes)
    log.info(
        "inventory language_params=%s index_gb=%s",
        inventory["language_parameters"],
        inventory["index_total_size_gb"],
    )
    log.info("predicted BF16 %.2f GB, hybrid v0.1 %.2f GB, ternary deployed %.2f GB",
             sizes["predicted_gguf_gb"]["BF16"],
             sizes["predicted_gguf_gb"]["hybrid_v01"],
             sizes["predicted_gguf_gb"]["all_ternary_g128_deployed"])

    if args.weights:
        from q38ternary.hf import download_weights

        download_weights(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
