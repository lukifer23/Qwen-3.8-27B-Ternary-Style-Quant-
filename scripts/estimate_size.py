#!/usr/bin/env python3
"""Print predicted footprints from the discovered architecture."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from q38ternary.architecture import write_architecture_reports  # noqa: E402
from q38ternary.config import load_config  # noqa: E402
from q38ternary.hf import read_config_json  # noqa: E402
from q38ternary.size import write_size_report  # noqa: E402


def main() -> int:
    cfg = load_config()
    model_dir = cfg.model_local_dir
    if (model_dir / "config.json").is_file():
        arch = write_architecture_reports(cfg, model_dir)
    else:
        print("config.json not downloaded yet; using the checked-in architecture discovery path.", file=sys.stderr)
        # Still require a config — refuse to invent one.
        raise SystemExit("Run scripts/download_sources.py first (config-only is enough).")
    payload = write_size_report(cfg, arch)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
