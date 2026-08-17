"""Resumable stage checkpoints. If reconstruction dies at layer 37, the next run continues there."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from q38ternary.config import AppConfig


@dataclass
class StageState:
    stage: str
    layer: int = 0
    layers_total: int = 64
    status: str = "pending"
    config_hash: str = ""
    rng_state: list[int] | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StageState":
        known = {k: data[k] for k in ("stage", "layer", "layers_total", "status", "config_hash", "rng_state", "metrics", "extra") if k in data}
        return cls(**known)


def checkpoint_path(cfg: AppConfig, stage: str) -> Path:
    directory = cfg.resolve("checkpoints", stage)
    directory.mkdir(parents=True, exist_ok=True)
    return directory / "state.json"


def save_stage(cfg: AppConfig, state: StageState) -> Path:
    path = checkpoint_path(cfg, state.stage)
    payload = json.dumps(state.to_dict(), indent=2, default=str)
    fd, tmp = tempfile.mkstemp(prefix="ckpt-", suffix=".json", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    return path


def load_stage(cfg: AppConfig, stage: str) -> StageState | None:
    path = checkpoint_path(cfg, stage)
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return StageState.from_dict(data)


def resume_layer(cfg: AppConfig, stage: str, layers_total: int = 64) -> int:
    """Return the first unfinished layer index."""
    state = load_stage(cfg, stage)
    if state is None:
        return 0
    if state.config_hash and state.config_hash != cfg.hash():
        return 0
    if state.status == "complete":
        return layers_total
    return int(state.layer)
