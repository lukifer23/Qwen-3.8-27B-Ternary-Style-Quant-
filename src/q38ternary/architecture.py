"""Discover the Qwen3.8 hybrid layout from config + safetensors keys. Do not hard-code old Qwen names."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from q38ternary.config import AppConfig
from q38ternary.hf import read_config_json
from q38ternary.utils.manifest import write_manifest

log = logging.getLogger("q38ternary.architecture")

CATEGORIES = (
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


def _text_config(config: dict[str, Any]) -> dict[str, Any]:
    if isinstance(config.get("text_config"), dict):
        return config["text_config"]
    return config


def classify_tensor(name: str, layer_type: str | None = None) -> str:
    lower = name.lower()
    if "visual" in lower or "vision" in lower or lower.startswith("model.visual"):
        return "vision"
    if "mtp" in lower:
        return "mtp"
    if any(key in lower for key in ("embed_tokens", "tok_embeddings", "token_emb")):
        return "embedding"
    if "lm_head" in lower or lower.endswith("lm_head.weight"):
        return "lm_head"
    if any(key in lower for key in ("norm", "layernorm", "rmsnorm")):
        return "norm"
    if "mlp" in lower or "feed_forward" in lower:
        return "mlp"
    if layer_type == "full_attention" or "self_attn" in lower or "attn.q" in lower:
        if "linear_attn" in lower:
            return "gated_deltanet"
        return "full_attention"
    if layer_type == "linear_attention" or "linear_attn" in lower or "delta" in lower:
        return "gated_deltanet"
    return "other"


def layer_types_from_config(text: dict[str, Any]) -> list[str]:
    declared = text.get("layer_types")
    if isinstance(declared, list) and declared:
        return [str(item) for item in declared]
    n_layers = int(text.get("num_hidden_layers") or 0)
    interval = int(text.get("full_attention_interval") or 4)
    types: list[str] = []
    for index in range(n_layers):
        if interval > 0 and ((index + 1) % interval == 0):
            types.append("full_attention")
        else:
            types.append("linear_attention")
    return types


def build_layer_map(config: dict[str, Any], tensor_index: dict[str, Any] | None = None) -> dict[str, Any]:
    text = _text_config(config)
    types = layer_types_from_config(text)
    weight_map: dict[str, str] = {}
    if tensor_index:
        weight_map = dict(tensor_index.get("weight_map") or {})

    layers: list[dict[str, Any]] = []
    for index, layer_type in enumerate(types):
        prefix_candidates = (
            f"model.language_model.layers.{index}.",
            f"model.layers.{index}.",
            f"language_model.layers.{index}.",
        )
        tensors: list[dict[str, Any]] = []
        for name, shard in weight_map.items():
            if any(name.startswith(prefix) for prefix in prefix_candidates):
                tensors.append({"name": name, "shard": shard, "category": classify_tensor(name, layer_type)})
        layers.append(
            {
                "index": index,
                "type": layer_type,
                "is_full_attention": layer_type == "full_attention",
                "is_gated_deltanet": layer_type == "linear_attention",
                "tensors": tensors,
                "parameter_count": None,
            }
        )

    return {
        "architectures": config.get("architectures"),
        "model_type": config.get("model_type"),
        "transformers_version": config.get("transformers_version"),
        "hidden_size": text.get("hidden_size"),
        "intermediate_size": text.get("intermediate_size"),
        "num_hidden_layers": text.get("num_hidden_layers"),
        "vocab_size": text.get("vocab_size"),
        "head_dim": text.get("head_dim"),
        "num_attention_heads": text.get("num_attention_heads"),
        "num_key_value_heads": text.get("num_key_value_heads"),
        "full_attention_interval": text.get("full_attention_interval"),
        "linear_conv_kernel_dim": text.get("linear_conv_kernel_dim"),
        "linear_num_key_heads": text.get("linear_num_key_heads"),
        "linear_num_value_heads": text.get("linear_num_value_heads"),
        "linear_key_head_dim": text.get("linear_key_head_dim"),
        "linear_value_head_dim": text.get("linear_value_head_dim"),
        "mtp_num_hidden_layers": text.get("mtp_num_hidden_layers"),
        "has_vision_config": isinstance(config.get("vision_config"), dict),
        "layer_types": types,
        "full_attention_indices": [i for i, t in enumerate(types) if t == "full_attention"],
        "deltanet_indices": [i for i, t in enumerate(types) if t == "linear_attention"],
        "layers": layers,
    }


def _load_index(model_dir: Path) -> dict[str, Any] | None:
    path = model_dir / "model.safetensors.index.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_architecture_reports(cfg: AppConfig, model_dir: Path | None = None) -> dict[str, Any]:
    model_dir = model_dir or cfg.model_local_dir
    config = read_config_json(model_dir)
    index = _load_index(model_dir)
    payload = build_layer_map(config, index)
    payload["model_dir"] = str(model_dir)
    payload["repo"] = cfg.model_repo
    payload["config_hash"] = cfg.hash()

    artifacts = cfg.resolve("artifacts")
    artifacts.mkdir(parents=True, exist_ok=True)
    json_path = artifacts / "architecture.json"
    md_path = artifacts / "architecture.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(render_architecture_md(payload), encoding="utf-8")
    write_manifest(cfg, json_path, kind="architecture")
    log.info("wrote %s and %s", json_path, md_path)
    return payload


def render_architecture_md(payload: dict[str, Any]) -> str:
    lines = [
        "# Architecture inventory",
        "",
        f"- repo: `{payload.get('repo')}`",
        f"- architectures: `{payload.get('architectures')}`",
        f"- model_type: `{payload.get('model_type')}`",
        f"- hidden_size: {payload.get('hidden_size')}",
        f"- intermediate_size: {payload.get('intermediate_size')}",
        f"- layers: {payload.get('num_hidden_layers')}",
        f"- vocab: {payload.get('vocab_size')}",
        f"- vision present: {payload.get('has_vision_config')}",
        f"- MTP layers: {payload.get('mtp_num_hidden_layers')}",
        f"- full-attention indices: {payload.get('full_attention_indices')}",
        f"- DeltaNet count: {len(payload.get('deltanet_indices') or [])}",
        "",
        "## Per-layer type",
        "",
        "| index | type |",
        "|------:|------|",
    ]
    for layer in payload.get("layers") or []:
        lines.append(f"| {layer['index']} | {layer['type']} |")
    lines.append("")
    return "\n".join(lines)
