"""Every pipeline output carries a sidecar manifest."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from q38ternary.config import AppConfig


def manifest_path_for(output: Path) -> Path:
    if output.suffix:
        return output.with_suffix(output.suffix + ".manifest.json")
    return output / "manifest.json"


def write_manifest(
    cfg: AppConfig,
    output: Path,
    *,
    kind: str,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    record: dict[str, Any] = {
        "kind": kind,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output": str(output),
        "config_hash": cfg.hash(),
        "model_repo": cfg.model_repo,
        "seed": cfg.seed,
        "group_size": cfg.group_size,
    }
    if extra:
        record.update(dict(extra))
    path = manifest_path_for(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    return path


def read_manifest(output: Path) -> dict[str, Any]:
    path = manifest_path_for(output)
    if not path.is_file():
        raise FileNotFoundError(f"No manifest for {output} (looked at {path})")
    return json.loads(path.read_text(encoding="utf-8"))
