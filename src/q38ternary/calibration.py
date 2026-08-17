"""Deterministic mixed calibration / holdout corpora. Never share rows with eval."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from q38ternary.config import AppConfig
from q38ternary.utils.manifest import write_manifest

log = logging.getLogger("q38ternary.calibration")


def _tokenizer(model_dir: Path):
    try:
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise RuntimeError("tokenizers is required to build the calibration set") from exc
    path = model_dir / "tokenizer.json"
    if not path.is_file():
        raise FileNotFoundError(f"tokenizer.json missing at {path}")
    return Tokenizer.from_file(str(path))


def _as_text(row: dict[str, Any], field: str, aux_fields: list[str] | None = None) -> str:
    value = row.get(field)
    if isinstance(value, list):
        # chat-style messages
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict):
                role = item.get("role", "")
                content = item.get("content", "")
                parts.append(f"{role}: {content}")
            else:
                parts.append(str(item))
        text = "\n".join(parts)
    else:
        text = "" if value is None else str(value)
    for extra in aux_fields or []:
        extra_val = row.get(extra)
        if extra_val:
            text += "\n" + str(extra_val)
    return text.strip()


def _iter_source(
    source: dict[str, Any],
    seed: int,
) -> Iterator[tuple[str, str, int]]:
    """Yield (source_name, text, row_index)."""
    from datasets import load_dataset

    name = str(source["name"])
    kwargs: dict[str, Any] = {"path": source["repo"], "split": source.get("split") or "train"}
    if source.get("config") and source["config"] != "default":
        kwargs["name"] = source["config"]
    try:
        ds = load_dataset(**kwargs)
    except Exception as exc:
        if source.get("optional"):
            log.warning("optional source %s failed: %s", name, exc)
            return
        raise
    # Deterministic shuffle of *indices*, not of the cached dataset object.
    n = len(ds)
    order = np.random.default_rng(seed).permutation(n)
    field = str(source.get("text_field") or "text")
    aux = list(source.get("aux_fields") or [])
    for raw_idx in order:
        idx = int(raw_idx)
        text = _as_text(ds[idx], field, aux)
        if len(text) < 32:
            continue
        yield name, text, idx


def _encode_ids(tokenizer, text: str) -> list[int]:
    return list(tokenizer.encode(text).ids)


def build_split(
    cfg: AppConfig,
    *,
    tier: str,
    holdout: bool = False,
) -> dict[str, Any]:
    section = cfg.section("calibration")
    seed = cfg.seed + (1 if holdout else 0)
    if holdout:
        sources = list(section.get("holdout") or [])
        n_seq = int((section.get("pilot") or {}).get("sequences") or 128)
        length = int((section.get("pilot") or {}).get("length") or 1024)
        out_dir = cfg.resolve("data", "evaluation")
        name = "holdout"
    else:
        sources = list(section.get("sources") or [])
        spec = section.get(tier) or section.get("pilot") or {}
        n_seq = int(spec.get("sequences") or 512)
        length = int(spec.get("length") or 1024)
        out_dir = cfg.resolve("data", "calibration")
        name = f"calibration_{tier}"

    tokenizer = _tokenizer(cfg.model_local_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sequences = np.zeros((n_seq, length), dtype=np.int32)
    records: list[dict[str, Any]] = []
    filled = 0
    carry: list[int] = []
    carry_meta: list[dict[str, Any]] = []

    for source in sources:
        if filled >= n_seq:
            break
        share = float(source.get("target_share") or (1.0 / max(len(sources), 1)))
        target = max(1, int(round(n_seq * share)))
        got = 0
        try:
            stream = _iter_source(source, seed)
        except Exception as exc:
            if source.get("optional"):
                log.warning("skipping optional %s: %s", source.get("name"), exc)
                continue
            raise
        for src_name, text, row_id in stream:
            ids = _encode_ids(tokenizer, text)
            if not ids:
                continue
            carry.extend(ids)
            carry_meta.append({"source": src_name, "row": row_id, "n_tokens": len(ids)})
            while len(carry) >= length and filled < n_seq and got < target:
                sequences[filled] = np.asarray(carry[:length], dtype=np.int32)
                records.append(
                    {
                        "index": filled,
                        "pieces": list(carry_meta),
                    }
                )
                carry = carry[length:]
                carry_meta = []
                filled += 1
                got += 1
            if filled >= n_seq:
                break

    if filled < n_seq:
        raise RuntimeError(
            f"{name}: only packed {filled}/{n_seq} sequences. "
            "Add another public source or lower the sequence count."
        )

    tokens_path = out_dir / f"{name}.npy"
    np.save(tokens_path, sequences)
    manifest = {
        "name": name,
        "tier": tier,
        "holdout": holdout,
        "seed": seed,
        "sequences": n_seq,
        "length": length,
        "tokenizer": str(cfg.model_local_dir / "tokenizer.json"),
        "model_repo": cfg.model_repo,
        "tokens_path": str(tokens_path),
        "sources": [s.get("name") for s in sources],
        "rows": records,
    }
    man_path = out_dir / "manifest.json" if not holdout else out_dir / "holdout_manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_manifest(cfg, tokens_path, kind=name, extra={"sequences": n_seq, "length": length})
    log.info("wrote %s (%s x %s)", tokens_path, n_seq, length)
    return manifest


def load_token_array(cfg: AppConfig, *, holdout: bool = False, tier: str | None = None) -> np.ndarray:
    tier = tier or cfg.calibration_tier
    if holdout:
        path = cfg.resolve("data", "evaluation", "holdout.npy")
        if not path.is_file():
            path = cfg.resolve("data", "evaluation", "holdout_manifest.json")
            raise FileNotFoundError(path)
        # tokens live next to the holdout manifest
        man = json.loads(cfg.resolve("data", "evaluation", "holdout_manifest.json").read_text(encoding="utf-8"))
        return np.load(man["tokens_path"])
    path = cfg.resolve("data", "calibration", f"calibration_{tier}.npy")
    if not path.is_file():
        raise FileNotFoundError(path)
    return np.load(path)
