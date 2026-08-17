"""Streaming BF16 teacher: one language layer at a time. Never load the full 27B on GPU."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from q38ternary.safetensors_io import ShardIndex

log = logging.getLogger("q38ternary.streaming")


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required for the streaming teacher") from exc
    return torch


def _require_layer_classes():
    try:
        from transformers import AutoConfig
        from transformers.models.qwen3_5.modeling_qwen3_5 import (
            Qwen3_5DecoderLayer,
            Qwen3_5RMSNorm,
            Qwen3_5TextRotaryEmbedding,
        )
    except ImportError as exc:
        raise RuntimeError(
            "transformers>=5.8 with Qwen3_5 is required for the streaming teacher"
        ) from exc
    return AutoConfig, Qwen3_5DecoderLayer, Qwen3_5RMSNorm, Qwen3_5TextRotaryEmbedding


def load_text_config(model_dir: Path):
    AutoConfig, *_ = _require_layer_classes()
    cfg = AutoConfig.from_pretrained(str(model_dir))
    text = cfg.text_config if hasattr(cfg, "text_config") else cfg
    if getattr(text, "_attn_implementation", None) is None:
        text._attn_implementation = "eager"
    return text


def _as_torch(array: np.ndarray, torch_mod):
    return torch_mod.from_numpy(np.ascontiguousarray(array))


def load_decoder_layer(store: ShardIndex, text_cfg, layer_idx: int, torch_mod):
    _, Qwen3_5DecoderLayer, _, _ = _require_layer_classes()
    layer = Qwen3_5DecoderLayer(text_cfg, layer_idx)
    prefix = f"model.language_model.layers.{layer_idx}."
    tensors = store.load_layer(layer_idx)
    state = {
        name[len(prefix) :]: _as_torch(array, torch_mod)
        for name, array in tensors.items()
        if name.startswith(prefix)
    }
    missing, unexpected = layer.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"layer {layer_idx} unexpected keys: {unexpected}")
    if missing:
        raise RuntimeError(f"layer {layer_idx} missing keys: {missing}")
    layer.eval()
    return layer


def load_final_norm(store: ShardIndex, text_cfg, torch_mod):
    _, _, Qwen3_5RMSNorm, _ = _require_layer_classes()
    for name in ("model.language_model.norm.weight", "model.norm.weight"):
        try:
            weight = store.load_tensor(name)
            break
        except KeyError:
            weight = None
    if weight is None:
        raise KeyError("final language-model RMSNorm weight not found")
    norm = Qwen3_5RMSNorm(int(text_cfg.hidden_size), eps=float(text_cfg.rms_norm_eps))
    with torch_mod.no_grad():
        norm.weight.copy_(_as_torch(weight, torch_mod))
    norm.eval()
    return norm


def embed(token_ids: np.ndarray, store: ShardIndex, torch_mod) -> Any:
    table = _as_torch(store.load_tensor("model.language_model.embed_tokens.weight"), torch_mod)
    ids = torch_mod.as_tensor(np.ascontiguousarray(token_ids), dtype=torch_mod.long)
    hidden = torch_mod.nn.functional.embedding(ids, table)
    del table
    return hidden


def lm_logits(hidden: Any, store: ShardIndex, torch_mod) -> Any:
    weight = _as_torch(store.load_tensor("lm_head.weight"), torch_mod)
    logits = torch_mod.nn.functional.linear(hidden, weight)
    del weight
    return logits


def position_embeddings(hidden: Any, text_cfg, torch_mod):
    *_, Qwen3_5TextRotaryEmbedding = _require_layer_classes()
    rotary = Qwen3_5TextRotaryEmbedding(config=text_cfg)
    seq = hidden.shape[1]
    position_ids = torch_mod.arange(seq, device=hidden.device)
    position_ids = position_ids.view(1, 1, -1).expand(3, hidden.shape[0], -1)
    return rotary(hidden, position_ids)


def attention_masks(text_cfg, hidden: Any, torch_mod) -> dict[str, Any]:
    from transformers.masking_utils import create_causal_mask, create_recurrent_attention_mask

    kwargs = {
        "config": text_cfg,
        "inputs_embeds": hidden,
        "attention_mask": None,
        "past_key_values": None,
        "position_ids": None,
    }
    return {
        "full_attention": create_causal_mask(**kwargs),
        "linear_attention": create_recurrent_attention_mask(**kwargs),
    }


def forward_hidden(
    token_ids: np.ndarray,
    model_dir: Path,
    *,
    through_layer: int | None = None,
    want_logits: bool = False,
    device: str = "cpu",
) -> dict[str, Any]:
    """Walk language layers sequentially. `through_layer` inclusive; None = all 64."""
    torch_mod = _require_torch()
    text_cfg = load_text_config(model_dir)
    last = int(text_cfg.num_hidden_layers) - 1 if through_layer is None else through_layer
    activations: dict[int, np.ndarray] = {}
    with ShardIndex(model_dir) as store:
        hidden = embed(token_ids, store, torch_mod).to(device)
        pos = position_embeddings(hidden, text_cfg, torch_mod)
        masks = attention_masks(text_cfg, hidden, torch_mod)
        layer_types = list(text_cfg.layer_types)
        for idx in range(last + 1):
            layer = load_decoder_layer(store, text_cfg, idx, torch_mod).to(device)
            mask = masks[layer_types[idx]]
            with torch_mod.no_grad():
                hidden = layer(
                    hidden,
                    position_embeddings=pos,
                    attention_mask=mask,
                    position_ids=None,
                    past_key_values=None,
                )
            activations[idx] = hidden.detach().to("cpu").float().numpy()
            del layer
            store.release_layer(idx)
            if torch_mod.cuda.is_available():
                torch_mod.cuda.empty_cache()
            log.info("teacher layer %s done shape=%s", idx, tuple(hidden.shape))
        logits_np = None
        if want_logits:
            norm = load_final_norm(store, text_cfg, torch_mod).to(device)
            with torch_mod.no_grad():
                hidden = norm(hidden)
                logits = lm_logits(hidden, store, torch_mod)
            logits_np = logits.detach().to("cpu").float().numpy()
            del norm, logits
    return {"hidden": activations, "logits": logits_np}


def compare_to_reference(
    token_ids: np.ndarray,
    model_dir: Path,
    *,
    atol: float = 5e-2,
    rtol: float = 5e-2,
) -> dict[str, float]:
    """Gate G2: streaming logits vs transformers on a short sequence.

    The reference path uses device_map=cpu and low_cpu_mem_usage. It is only
    for a tiny prompt. If it cannot fit, we skip the full-model reference and
    compare last-token self-consistency instead of silently passing.
    """
    torch_mod = _require_torch()
    from transformers import AutoModelForCausalLM

    streamed = forward_hidden(token_ids, model_dir, want_logits=True, device="cpu")
    if streamed["logits"] is None:
        raise RuntimeError("streaming teacher did not return logits")
    try:
        ref = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            dtype=torch_mod.bfloat16,
            low_cpu_mem_usage=True,
            device_map="cpu",
        )
        ref.eval()
        ids = torch_mod.as_tensor(np.ascontiguousarray(token_ids), dtype=torch_mod.long)
        with torch_mod.no_grad():
            out = ref(input_ids=ids, use_cache=False)
        ref_logits = out.logits.float().cpu().numpy()
        del ref, out
    except (torch_mod.cuda.OutOfMemoryError, MemoryError, OSError) as exc:
        raise RuntimeError(
            "Could not materialize the official Transformers reference on this machine. "
            f"{exc}"
        ) from exc
    a = streamed["logits"]
    b = ref_logits
    mse = float(np.mean((a - b) ** 2))
    max_abs = float(np.max(np.abs(a - b)))
    if not np.allclose(a, b, atol=atol, rtol=rtol):
        raise RuntimeError(
            f"G2 FAIL: streaming vs reference logits mse={mse:.4e} max_abs={max_abs:.4e}"
        )
    return {"mse": mse, "max_abs": max_abs}
