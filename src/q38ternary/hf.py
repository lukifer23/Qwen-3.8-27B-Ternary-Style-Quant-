"""Hugging Face snapshot helpers. Config/tokenizer first; weights only after inventory."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable

from q38ternary.config import AppConfig
from q38ternary.hardware import HardwareError, require_disk_for_download
from q38ternary.utils.manifest import write_manifest

log = logging.getLogger("q38ternary.hf")

CONFIG_PATTERNS = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "preprocessor_config.json",
    "processor_config.json",
    "video_preprocessor_config.json",
    "chat_template.jinja",
    "chat_template.json",
    "*.jinja",
    "LICENSE*",
    "README.md",
    "model.safetensors.index.json",
)


def _hub():
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required for downloads") from exc
    return hf_hub_download, snapshot_download


def download_config(cfg: AppConfig, dest: Path | None = None) -> Path:
    """Fetch config/tokenizer/index only. Does not pull weight shards."""
    dest = dest or (cfg.model_local_dir)
    dest.mkdir(parents=True, exist_ok=True)
    _, snapshot_download = _hub()
    log.info("downloading config/tokenizer for %s → %s", cfg.model_repo, dest)
    snapshot_download(
        repo_id=cfg.model_repo,
        local_dir=str(dest),
        allow_patterns=list(CONFIG_PATTERNS),
        resume_download=True,
    )
    config_path = dest / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"config.json missing after download into {dest}")
    write_manifest(cfg, dest, kind="hf_config", extra={"repo": cfg.model_repo})
    return dest


def download_weights(cfg: AppConfig, dest: Path | None = None) -> Path:
    """Full snapshot. Refuses to start if free disk is below the configured floor."""
    require_disk_for_download(cfg)
    dest = dest or cfg.model_local_dir
    dest.mkdir(parents=True, exist_ok=True)
    _, snapshot_download = _hub()
    log.info("downloading full snapshot %s → %s", cfg.model_repo, dest)
    snapshot_download(
        repo_id=cfg.model_repo,
        local_dir=str(dest),
        resume_download=True,
    )
    write_manifest(cfg, dest, kind="hf_weights", extra={"repo": cfg.model_repo})
    return dest


def read_config_json(model_dir: Path) -> dict:
    path = model_dir / "config.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def repo_revision(model_dir: Path) -> str | None:
    """Best-effort commit from the huggingface cache refs if present."""
    refs = list(model_dir.rglob("refs/main"))
    if refs:
        return refs[0].read_text(encoding="utf-8").strip()
    snap = model_dir / ".cache" / "huggingface" / "download"
    if snap.is_dir():
        return None
    return None


def verify_teacher_identity(cfg: AppConfig, config: dict) -> None:
    expected_repo = cfg.model_repo
    architectures = config.get("architectures") or []
    if "Qwen3_5ForConditionalGeneration" not in architectures and "Qwen3_8" not in str(architectures):
        # Discover, don't hard-fail on a renamed class — but refuse a totally different family.
        model_type = str(config.get("model_type") or "")
        if "qwen3" not in model_type.lower():
            raise HardwareError(
                f"{expected_repo} config model_type={model_type!r} architectures={architectures!r} "
                "does not look like the official Qwen3.8-27B teacher."
            )
    text = config.get("text_config") or {}
    hidden = int(text.get("hidden_size") or config.get("hidden_size") or 0)
    layers = int(text.get("num_hidden_layers") or config.get("num_hidden_layers") or 0)
    if hidden and hidden != 5120:
        log.warning("hidden_size=%s (expected 5120) — continuing with discovered value", hidden)
    if layers and layers != 64:
        log.warning("num_hidden_layers=%s (expected 64) — continuing with discovered value", layers)
