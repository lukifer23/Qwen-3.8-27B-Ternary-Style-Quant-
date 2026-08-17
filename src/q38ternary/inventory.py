"""Tensor inventory from the official index + Hub safetensors metadata (no weight download)."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

from q38ternary.architecture import classify_tensor, layer_types_from_config
from q38ternary.config import AppConfig
from q38ternary.hf import read_config_json
from q38ternary.utils.manifest import write_manifest

log = logging.getLogger("q38ternary.inventory")

DTYPE_BYTES = {
    "BF16": 2,
    "F16": 2,
    "F32": 4,
    "F64": 8,
    "I8": 1,
    "I16": 2,
    "I32": 4,
    "I64": 8,
    "U8": 1,
    "BOOL": 1,
}


def _fetch_hub_metadata(repo_id: str) -> dict[str, dict[str, Any]]:
    """Shape/dtype for every tensor, without downloading shards."""
    try:
        from huggingface_hub import get_safetensors_metadata
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required for inventory metadata") from exc
    meta = get_safetensors_metadata(repo_id)
    tensors: dict[str, dict[str, Any]] = {}
    files = getattr(meta, "files_metadata", None) or {}
    for _fname, file_meta in files.items():
        file_tensors = getattr(file_meta, "tensors", None) or {}
        for name, info in file_tensors.items():
            shape = list(getattr(info, "shape", None) or [])
            dtype = getattr(info, "dtype", None)
            n_params = getattr(info, "parameter_count", None)
            tensors[name] = {
                "shape": shape,
                "dtype": str(dtype),
                "parameters": int(n_params) if n_params is not None else None,
            }
    return tensors


def _n_params(shape: list[int]) -> int:
    n = 1
    for dim in shape:
        n *= int(dim)
    return int(n) if shape else 0


def build_inventory(cfg: AppConfig, model_dir: Path | None = None) -> dict[str, Any]:
    model_dir = model_dir or cfg.model_local_dir
    config = read_config_json(model_dir)
    text = config.get("text_config") if isinstance(config.get("text_config"), dict) else config
    layer_types = layer_types_from_config(text)
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(index_path)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map: dict[str, str] = dict(index.get("weight_map") or {})
    total_size = float((index.get("metadata") or {}).get("total_size") or 0)

    try:
        hub_tensors = _fetch_hub_metadata(cfg.model_repo)
    except Exception as exc:
        log.warning("Hub metadata unavailable (%s); inventory will lack shapes", exc)
        hub_tensors = {}

    categories: dict[str, dict[str, Any]] = {
        name: {"tensors": 0, "parameters": 0, "bytes_bf16": 0, "names": []}
        for name in (
            "embedding",
            "lm_head",
            "full_attention",
            "gated_deltanet",
            "mlp",
            "norm",
            "mtp",
            "vision",
            "other",
        )
    }
    tensors_out: list[dict[str, Any]] = []
    for name, shard in weight_map.items():
        # Infer layer type from the name when possible.
        layer_type = None
        marker = ".layers."
        if marker in name:
            try:
                idx = int(name.split(marker, 1)[1].split(".", 1)[0])
                if 0 <= idx < len(layer_types):
                    layer_type = layer_types[idx]
            except ValueError:
                layer_type = None
        category = classify_tensor(name, layer_type)
        info = hub_tensors.get(name) or {}
        shape = list(info.get("shape") or [])
        dtype = info.get("dtype")
        n_params = _n_params(shape)
        width = DTYPE_BYTES.get(str(dtype).upper(), 2)
        nbytes = n_params * width
        entry = {
            "name": name,
            "shard": shard,
            "category": category,
            "shape": shape,
            "dtype": dtype,
            "parameters": n_params,
            "bytes": nbytes,
        }
        tensors_out.append(entry)
        bucket = categories[category]
        bucket["tensors"] += 1
        bucket["parameters"] += n_params
        bucket["bytes_bf16"] += nbytes
        bucket["names"].append(name)

    language_params = sum(
        categories[c]["parameters"]
        for c in ("embedding", "lm_head", "full_attention", "gated_deltanet", "mlp", "norm")
    )
    payload = {
        "repo": cfg.model_repo,
        "index_total_size_bytes": total_size,
        "index_total_size_gb": round(total_size / (1024**3), 3),
        "tensor_count": len(weight_map),
        "shard_count": len(set(weight_map.values())),
        "language_parameters": language_params,
        "categories": {
            name: {
                "tensors": bucket["tensors"],
                "parameters": bucket["parameters"],
                "bytes_bf16": bucket["bytes_bf16"],
                "gb_bf16": round(bucket["bytes_bf16"] / (1024**3), 3),
            }
            for name, bucket in categories.items()
        },
        "tensors": tensors_out,
    }
    return payload


def render_inventory_md(payload: dict[str, Any]) -> str:
    lines = [
        "# Model inventory",
        "",
        f"- repo: `{payload['repo']}`",
        f"- tensors: {payload['tensor_count']}",
        f"- shards: {payload['shard_count']}",
        f"- index total_size: {payload['index_total_size_gb']} GB",
        f"- language parameters (sum of known shapes): {payload['language_parameters']:,}",
        "",
        "| category | tensors | parameters | BF16 GB |",
        "|---|---:|---:|---:|",
    ]
    for name, bucket in payload["categories"].items():
        lines.append(
            f"| {name} | {bucket['tensors']} | {bucket['parameters']:,} | {bucket['gb_bf16']} |"
        )
    lines.append("")
    lines.append("Shapes come from Hugging Face safetensors metadata, not from loading shards.")
    lines.append("")
    return "\n".join(lines)


def write_inventory(cfg: AppConfig, model_dir: Path | None = None) -> dict[str, Any]:
    payload = build_inventory(cfg, model_dir)
    artifacts = cfg.resolve("artifacts")
    reports = artifacts / "reports"
    artifacts.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    json_path = artifacts / "model_inventory.json"
    md_path = artifacts / "model_inventory.md"
    report = reports / "01_model_inventory.md"
    # The full tensor list is large; keep it in JSON only.
    summary = {k: v for k, v in payload.items() if k != "tensors"}
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md = render_inventory_md(payload)
    md_path.write_text(md, encoding="utf-8")
    report.write_text(md, encoding="utf-8")
    write_manifest(cfg, json_path, kind="model_inventory", extra=summary)
    log.info("wrote %s", json_path)
    return payload
