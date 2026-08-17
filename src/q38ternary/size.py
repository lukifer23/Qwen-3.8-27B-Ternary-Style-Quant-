"""Predict deployed size from the discovered architecture. Measure, don't guess later."""

from __future__ import annotations

import json
from typing import Any

from q38ternary.config import AppConfig
from q38ternary.quant.mixed_precision import BITS_PER_WEIGHT

# Conventional GGUF-ish deployed bits/weight for the control group.
SCHEME_BPW = {
    "bf16": 16.0,
    "q8": 8.5,
    "q4": 4.5,
    "q3": 3.5,
    "q2": 2.8,
    "ternary_g128": 2.125,
    "ternary_ideal": 1.71,
    "binary": 1.125,
}


def _language_params_from_arch(arch: dict[str, Any]) -> dict[str, int]:
    """Closed-form count from config. Replaced by shard inventory once weights exist."""
    hidden = int(arch["hidden_size"])
    intermediate = int(arch["intermediate_size"])
    vocab = int(arch["vocab_size"])
    n_layers = int(arch["num_hidden_layers"])
    full_idx = list(arch.get("full_attention_indices") or [])
    delta_idx = list(arch.get("deltanet_indices") or [])

    embed = vocab * hidden
    lm_head = vocab * hidden  # tie_word_embeddings is false on Qwen3.8-27B

    # SwiGLU: gate, up, down
    mlp_per = hidden * intermediate + hidden * intermediate + intermediate * hidden

    # Gated attention: Q (n_q * head_dim * hidden), KV, O, plus output gate
    n_q = int(arch.get("num_attention_heads") or 24)
    n_kv = int(arch.get("num_key_value_heads") or 4)
    head_dim = int(arch.get("head_dim") or 256)
    attn_per = (
        n_q * head_dim * hidden
        + 2 * n_kv * head_dim * hidden
        + hidden * (n_q * head_dim)
        + hidden  # output gate, small
    )

    lin_k = int(arch.get("linear_num_key_heads") or 16)
    lin_v = int(arch.get("linear_num_value_heads") or 48)
    lin_kd = int(arch.get("linear_key_head_dim") or 128)
    lin_vd = int(arch.get("linear_value_head_dim") or 128)
    conv_k = int(arch.get("linear_conv_kernel_dim") or 4)
    # Q, K, V, out_proj, plus a small conv on the value/state path.
    delta_per = (
        lin_k * lin_kd * hidden
        + lin_k * lin_kd * hidden
        + lin_v * lin_vd * hidden
        + hidden * (lin_v * lin_vd)
        + lin_v * lin_vd * conv_k
    )

    # Two RMSNorms per block, plus a few scalar/state tensors. Tiny vs matrices.
    norm_per = 2 * hidden

    attn_params = attn_per * len(full_idx)
    delta_params = delta_per * len(delta_idx)
    mlp_params = mlp_per * n_layers
    norm_params = norm_per * n_layers + hidden  # final norm

    language = embed + lm_head + attn_params + delta_params + mlp_params + norm_params
    return {
        "embedding": embed,
        "lm_head": lm_head,
        "full_attention": attn_params,
        "gated_deltanet": delta_params,
        "mlp": mlp_params,
        "norm": norm_params,
        "language_total": language,
    }


def _gb(params: int, bpw: float) -> float:
    return round(params * bpw / 8.0 / (1024**3), 3)


def estimate_from_inventory(inventory: dict[str, Any]) -> dict[str, int]:
    cats = inventory["categories"]
    return {
        "embedding": int(cats["embedding"]["parameters"]),
        "lm_head": int(cats["lm_head"]["parameters"]),
        "full_attention": int(cats["full_attention"]["parameters"]),
        "gated_deltanet": int(cats["gated_deltanet"]["parameters"]),
        "mlp": int(cats["mlp"]["parameters"]),
        "norm": int(cats["norm"]["parameters"]),
        "language_total": int(inventory["language_parameters"]),
        "vision": int(cats["vision"]["parameters"]),
        "mtp": int(cats["mtp"]["parameters"]),
    }


def estimate_from_architecture(arch: dict[str, Any], inventory: dict[str, Any] | None = None) -> dict[str, Any]:
    counts = estimate_from_inventory(inventory) if inventory else _language_params_from_arch(arch)
    lang = counts["language_total"]
    major = counts["full_attention"] + counts["gated_deltanet"] + counts["mlp"]
    embed_lm = counts["embedding"] + counts["lm_head"]
    high_prec = counts["norm"]

    models = {
        "BF16": _gb(lang, SCHEME_BPW["bf16"]),
        "Q8": _gb(lang, SCHEME_BPW["q8"]),
        "Q4": _gb(lang, SCHEME_BPW["q4"]),
        "Q3": _gb(lang, SCHEME_BPW["q3"]),
        "Q2": _gb(lang, SCHEME_BPW["q2"]),
        "all_ternary_g128_deployed": _gb(lang, SCHEME_BPW["ternary_g128"]),
        "all_ternary_ideal": _gb(lang, SCHEME_BPW["ternary_ideal"]),
        "hybrid_v01": round(
            _gb(major, SCHEME_BPW["ternary_g128"])
            + _gb(embed_lm, SCHEME_BPW["q4"])
            + _gb(high_prec, 16.0),
            3,
        ),
        "adaptive_7_5": 7.5,
        "adaptive_8": 8.0,
        "adaptive_9": 9.0,
        "binary_stretch": _gb(lang, SCHEME_BPW["binary"]),
    }
    return {
        "parameter_counts": counts,
        "bits_per_weight": SCHEME_BPW,
        "predicted_gguf_gb": models,
        "note": (
            "These are predicted language-model footprints from the discovered config. "
            "They exclude vision, MTP, GGUF metadata, and runtime KV/scratch. "
            "Do not report the ideal 1.71 bpw number as the deployed file size."
        ),
    }


def write_size_report(
    cfg: AppConfig,
    arch: dict[str, Any],
    inventory: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = estimate_from_architecture(arch, inventory=inventory)
    path = cfg.resolve("artifacts", "size_estimate.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
