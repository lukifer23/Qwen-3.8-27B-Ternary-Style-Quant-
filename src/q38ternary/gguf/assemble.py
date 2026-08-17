"""Local GGUF assembly. Hugging Face export is a later, separate step."""

from __future__ import annotations

from pathlib import Path

from q38ternary.config import AppConfig
from q38ternary.gguf.writer import write_local_gguf


def assemble_local_gguf(cfg: AppConfig) -> Path:
    outfile = cfg.resolve("models", "output", "qwen38-ternary-v01-q2_0.gguf")
    ckpt = cfg.resolve("checkpoints", "reconstructed")
    return write_local_gguf(cfg, ckpt, outfile)
