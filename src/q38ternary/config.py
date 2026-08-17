"""Load, merge, and hash the YAML configuration set."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

CONFIG_FILES = (
    "project.yaml",
    "hardware.yaml",
    "calibration.yaml",
    "quantization.yaml",
    "reconstruction.yaml",
    "evaluation.yaml",
)


def repo_root(start: Path | None = None) -> Path:
    """Walk upward from *start* (or this file) until config/project.yaml exists."""
    here = (start or Path(__file__)).resolve()
    if here.is_file():
        here = here.parent
    for candidate in (here, *here.parents):
        if (candidate / "config" / "project.yaml").is_file():
            return candidate
    raise FileNotFoundError(
        "Could not locate repo root (missing config/project.yaml)."
    )


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping, got {type(data)}")
    return data


def _deep_get(mapping: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


@dataclass(frozen=True)
class AppConfig:
    """Resolved view of every file under config/."""

    root: Path
    raw: dict[str, Any]

    @property
    def seed(self) -> int:
        value = _deep_get(self.raw, "project", "project", "seed")
        if value is None:
            value = _deep_get(self.raw, "calibration", "seed", default=42)
        return int(value)

    @property
    def name(self) -> str:
        return str(_deep_get(self.raw, "project", "project", "name", default="qwen38-ternary"))

    @property
    def model_repo(self) -> str:
        return str(_deep_get(self.raw, "project", "model", "repo", default="Qwen/Qwen3.8-27B"))

    @property
    def model_local_dir(self) -> Path:
        rel = _deep_get(
            self.raw, "project", "model", "local_dir", default="models/source/Qwen3.8-27B"
        )
        return (self.root / str(rel)).resolve()

    @property
    def text_only(self) -> bool:
        return bool(_deep_get(self.raw, "project", "model", "text_only", default=True))

    @property
    def include_vision(self) -> bool:
        return bool(_deep_get(self.raw, "project", "model", "include_vision", default=False))

    @property
    def include_mtp(self) -> bool:
        return bool(_deep_get(self.raw, "project", "model", "include_mtp", default=False))

    @property
    def group_size(self) -> int:
        return int(_deep_get(self.raw, "quantization", "group_size", default=128))

    @property
    def alphabet(self) -> tuple[int, ...]:
        values = _deep_get(self.raw, "quantization", "alphabet", default=[-1, 0, 1])
        return tuple(int(v) for v in values)

    @property
    def gpu_memory_limit_gb(self) -> float:
        return float(
            _deep_get(self.raw, "hardware", "limits", "gpu_memory_limit_gb")
            or _deep_get(self.raw, "project", "hardware", "gpu_memory_limit_gb", default=14.0)
        )

    @property
    def system_memory_limit_gb(self) -> float:
        return float(
            _deep_get(self.raw, "hardware", "limits", "system_memory_limit_gb")
            or _deep_get(self.raw, "project", "hardware", "system_memory_limit_gb", default=52.0)
        )

    @property
    def minimum_free_disk_gb(self) -> float:
        return float(
            _deep_get(self.raw, "hardware", "limits", "minimum_free_disk_gb")
            or _deep_get(self.raw, "project", "hardware", "minimum_free_disk_gb", default=180.0)
        )

    @property
    def preferred_free_disk_gb(self) -> float:
        return float(
            _deep_get(self.raw, "hardware", "limits", "preferred_free_disk_gb")
            or _deep_get(self.raw, "project", "hardware", "preferred_free_disk_gb", default=250.0)
        )

    @property
    def gpu_temp_pause_c(self) -> float:
        return float(_deep_get(self.raw, "hardware", "limits", "gpu_temp_pause_c", default=87.0))

    @property
    def pilot_layers(self) -> tuple[int, ...]:
        layers = _deep_get(
            self.raw, "reconstruction", "pilot_layers", default=[0, 3, 15, 31, 47, 63]
        )
        return tuple(int(i) for i in layers)

    @property
    def calibration_tier(self) -> str:
        return str(_deep_get(self.raw, "calibration", "default_tier", default="pilot"))

    def section(self, name: str) -> dict[str, Any]:
        data = self.raw.get(name, {})
        return data if isinstance(data, dict) else {}

    def hash(self) -> str:
        payload = json.dumps(self.raw, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:16]

    def resolve(self, *parts: str | Path) -> Path:
        return (self.root.joinpath(*[str(p) for p in parts])).resolve()


def load_config(root: Path | None = None) -> AppConfig:
    base = repo_root(root)
    raw: dict[str, Any] = {}
    missing: list[str] = []
    for name in CONFIG_FILES:
        path = base / "config" / name
        if not path.is_file():
            missing.append(str(path))
            continue
        raw[name.removesuffix(".yaml")] = _read_yaml(path)
    if missing:
        raise FileNotFoundError("Missing config files:\n  " + "\n  ".join(missing))
    return AppConfig(root=base, raw=raw)
