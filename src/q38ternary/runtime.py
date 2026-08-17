"""Inspect the cloned llama.cpp trees and decide which binary owns ternary g128."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from q38ternary.config import AppConfig

_QK2_RE = re.compile(r"#define\s+QK2_0\s+(\d+)")


def q2_group_size(tree: Path) -> int | None:
    header = tree / "ggml" / "src" / "ggml-common.h"
    if not header.is_file():
        return None
    text = header.read_text(encoding="utf-8", errors="replace")
    match = _QK2_RE.search(text)
    return int(match.group(1)) if match else None


def inspect_runtimes(cfg: AppConfig) -> dict[str, Any]:
    upstream = cfg.resolve("third_party", "llama.cpp")
    prism = cfg.resolve("third_party", "prism-llama.cpp")
    up_g = q2_group_size(upstream)
    pr_g = q2_group_size(prism)
    ternary_tree = None
    if pr_g == 128:
        ternary_tree = str(prism)
    elif up_g == 128:
        ternary_tree = str(upstream)
    return {
        "upstream": {"path": str(upstream), "q2_group": up_g, "exists": upstream.is_dir()},
        "prism": {"path": str(prism), "q2_group": pr_g, "exists": prism.is_dir()},
        "ternary_runtime": "prism" if pr_g == 128 else ("upstream" if up_g == 128 else None),
        "baseline_runtime": "upstream" if upstream.is_dir() else None,
        "ternary_tree": ternary_tree,
        "note": (
            "Prism Q2_0 is group-128; current upstream Q2_0 is group-64. "
            "Use the Prism CUDA build for our ternary GGUF. "
            "Use upstream for conventional Unsloth/Qwen GGUF baselines."
        ),
    }
