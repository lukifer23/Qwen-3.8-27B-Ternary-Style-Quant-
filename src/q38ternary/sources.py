"""Clone pinned third-party repositories and record exact commits."""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Any

from q38ternary.config import AppConfig
from q38ternary.utils.manifest import write_manifest

log = logging.getLogger("q38ternary.sources")

DEFAULT_REPOS = (
    ("llama.cpp", "https://github.com/ggml-org/llama.cpp", "third_party/llama.cpp"),
    ("prism-llama.cpp", "https://github.com/PrismML-Eng/llama.cpp", "third_party/prism-llama.cpp"),
    ("bonsai-demo", "https://github.com/PrismML-Eng/Bonsai-demo", "third_party/bonsai-demo"),
    ("auto-round", "https://github.com/intel/auto-round", "third_party/auto-round"),
)


def _run_git(args: list[str], cwd: Path | None = None) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=str(cwd) if cwd else None,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed in {cwd}: {completed.stderr.strip() or completed.stdout.strip()}"
        )
    return completed.stdout.strip()


def _head_commit(path: Path) -> str:
    return _run_git(["rev-parse", "HEAD"], cwd=path)


def clone_or_update(url: str, dest: Path) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if (dest / ".git").is_dir():
        log.info("reusing existing checkout %s", dest)
        return _head_commit(dest)
    log.info("cloning %s → %s", url, dest)
    _run_git(["clone", "--depth", "1", url, str(dest)])
    return _head_commit(dest)


def sync_third_party(cfg: AppConfig) -> dict[str, Any]:
    versions_path = cfg.resolve("third_party", "versions.json")
    if versions_path.is_file():
        declared = json.loads(versions_path.read_text(encoding="utf-8"))
    else:
        declared = {}

    record: dict[str, Any] = {}
    for name, url, rel in DEFAULT_REPOS:
        dest = cfg.resolve(rel)
        existing = declared.get(name) or {}
        commit = clone_or_update(existing.get("url") or url, dest)
        record[name] = {
            "url": existing.get("url") or url,
            "path": rel.replace("\\", "/"),
            "commit": commit,
        }
        log.info("%s @ %s", name, commit)

    versions_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    write_manifest(cfg, versions_path, kind="third_party_versions", extra=record)
    return record
