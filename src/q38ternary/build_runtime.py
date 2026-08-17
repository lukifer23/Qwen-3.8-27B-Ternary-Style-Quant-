"""Find or compile a llama.cpp binary that can load Prism Q2_0 g128."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path

from q38ternary.config import AppConfig

log = logging.getLogger("q38ternary.build")


def find_existing_cli() -> Path | None:
    names = ("llama-cli.exe", "llama-cli", "llama.exe")
    extra = [
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs",
        Path(r"C:\Program Files"),
    ]
    which = shutil.which("llama-cli") or shutil.which("llama-cli.exe")
    if which:
        return Path(which)
    for root in extra:
        if not root.exists():
            continue
        for name in names:
            hits = list(root.rglob(name))
            if hits:
                return hits[0]
    return None


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    log.info("exec %s", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def compile_prism(cfg: AppConfig) -> Path:
    tree = cfg.resolve("third_party", "prism-llama.cpp")
    if shutil.which("cmake") is None:
        raise RuntimeError(
            "cmake is not on PATH. Reconstruction and GGUF are done; "
            "install Visual Studio Build Tools + CUDA toolkit, then re-run "
            "with --from-gguf to compile and smoke-test."
        )
    build = tree / "build"
    _run(["cmake", "-B", str(build), "-DGGML_CUDA=ON", "-DCMAKE_BUILD_TYPE=Release"], cwd=tree)
    _run(["cmake", "--build", str(build), "--config", "Release", "-j"], cwd=tree)
    for cand in build.rglob("llama-cli.exe"):
        return cand
    for cand in build.rglob("llama-cli"):
        return cand
    raise FileNotFoundError("cmake succeeded but llama-cli was not found")


def resolve_cli(cfg: AppConfig, *, compile_if_missing: bool) -> Path:
    existing = find_existing_cli()
    if existing:
        log.info("using existing binary %s", existing)
        return existing
    if compile_if_missing:
        return compile_prism(cfg)
    raise RuntimeError("no llama-cli on PATH and compile was not requested")
