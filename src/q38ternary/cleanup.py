"""Delete activation caches, leftover student HF trees, and partial GGUFs."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from q38ternary.config import AppConfig

log = logging.getLogger("q38ternary.cleanup")


def _rm(path: Path) -> None:
    if path.is_file():
        path.unlink()
        log.info("removed %s", path)
    elif path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
        log.info("removed dir %s", path)


def cleanup_after_layer(cache_root: Path, completed_layer: int) -> None:
    """Keep at most the current layer's outputs. Drop anything older than L-1."""
    stale = completed_layer - 1
    if stale < 0:
        return
    for path in cache_root.glob(f"layer_{stale:03d}_*"):
        _rm(path)


def cleanup_run(cfg: AppConfig, *, keep_checkpoints: bool = True) -> None:
    """End-of-run cleanup. Never deletes reconstructed codes/scales unless asked."""
    _rm(cfg.resolve("models", "output", "hf-student"))
    _rm(cfg.resolve("cache", "activations", "full"))
    for partial in cfg.resolve("models", "output").glob("*.gguf.partial"):
        _rm(partial)
    if not keep_checkpoints:
        _rm(cfg.resolve("checkpoints", "reconstructed"))
