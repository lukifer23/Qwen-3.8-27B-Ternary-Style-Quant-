"""Timestamped stage logs and the experiment registry."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from q38ternary.config import AppConfig, repo_root


def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M")


def log_path(stage: str, root: Path | None = None) -> Path:
    base = root or repo_root()
    directory = base / "artifacts" / "logs"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{_stamp()}-{stage}.log"


def setup_logging(stage: str, root: Path | None = None, level: int = logging.INFO) -> logging.Logger:
    """Configure root logging to a timestamped file and stderr."""
    path = log_path(stage, root)
    logger = logging.getLogger("q38ternary")
    logger.setLevel(level)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False
    logger.info("logging to %s", path)
    return logger


def registry_path(root: Path | None = None) -> Path:
    base = root or repo_root()
    directory = base / "artifacts"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / "experiment_registry.jsonl"


def record_experiment(
    cfg: AppConfig,
    *,
    stage: str,
    metrics: Mapping[str, Any] | None = None,
    output_path: str | Path | None = None,
    duration_seconds: float | None = None,
    extra: Mapping[str, Any] | None = None,
) -> None:
    """Append one experiment record. Never raises away a finished run's metrics."""
    record: dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "config_hash": cfg.hash(),
        "model_repo": cfg.model_repo,
        "seed": cfg.seed,
        "duration_seconds": duration_seconds,
        "metrics": dict(metrics or {}),
        "output_path": str(output_path) if output_path else None,
    }
    if extra:
        record.update(dict(extra))
    path = registry_path(cfg.root)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=str) + "\n")
