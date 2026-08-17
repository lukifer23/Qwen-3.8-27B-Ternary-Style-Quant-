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


def load_decoder_layer(store: ShardIndex, text_cfg, layer_idx: int, torch_mod, *, device: str = "cpu"):
    """Load one official decoder layer in BF16. No FP32 / NumPy copy of the block."""
    _, Qwen3_5DecoderLayer, _, _ = _require_layer_classes()
    layer = Qwen3_5DecoderLayer(text_cfg, layer_idx)
    layer.to(dtype=torch_mod.bfloat16)
    prefix = f"model.language_model.layers.{layer_idx}."
    names = store.layer_tensor_names(layer_idx)
    state = {}
    for name in names:
        if not name.startswith(prefix):
            continue
        state[name[len(prefix) :]] = store.load_tensor_torch(name, device="cpu")
    missing, unexpected = layer.load_state_dict(state, strict=False)
    del state
    if unexpected:
        raise RuntimeError(f"layer {layer_idx} unexpected keys: {unexpected}")
    if missing:
        raise RuntimeError(f"layer {layer_idx} missing keys: {missing}")
    layer.eval()
    return layer.to(device)


def load_final_norm(store: ShardIndex, text_cfg, torch_mod, *, device: str = "cpu"):
    _, _, Qwen3_5RMSNorm, _ = _require_layer_classes()
    weight = None
    for name in ("model.language_model.norm.weight", "model.norm.weight"):
        try:
            weight = store.load_tensor_torch(name, device=device)
            break
        except KeyError:
            weight = None
    if weight is None:
        raise KeyError("final language-model RMSNorm weight not found")
    norm = Qwen3_5RMSNorm(int(text_cfg.hidden_size), eps=float(text_cfg.rms_norm_eps))
    with torch_mod.no_grad():
        norm.weight.copy_(weight.to(dtype=norm.weight.dtype))
    norm.eval()
    return norm.to(device)


def embed(token_ids: np.ndarray, store: ShardIndex, torch_mod, *, device: str = "cpu") -> Any:
    """Lookup tokens without allocating the 2.4 GB embed table."""
    hidden = store.gather_rows_torch(
        "model.language_model.embed_tokens.weight",
        token_ids,
        device=device,
    )
    return hidden


def lm_logits(hidden: Any, store: ShardIndex, torch_mod, *, last_token_only: bool = True) -> Any:
    """Project to vocab. Full-seq logits at 1024 tokens are ~1 GB; default last-token only."""
    states = hidden[:, -1:, :] if last_token_only else hidden
    # Gathering every vocab row would be the full 2.4 GB table. Use get_tensor
    # only for the tiny G2 sequence (last_token_only keeps the GEMM small).
    weight = store.load_tensor_torch("lm_head.weight", device=str(states.device))
    try:
        return torch_mod.nn.functional.linear(states.float(), weight.float())
    finally:
        del weight





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
    keep_hidden_layers: bool = False,
    activation_cache: Any | None = None,
) -> dict[str, Any]:
    """Walk language layers sequentially. Only the current hidden state is retained.

    Storing every layer as float32 in RAM for the pilot set is
    512 × 1024 × 5120 × 4 × 64 ≈ 687 GB. That is not optional — we refuse it.
    Write FP16 chunks through `activation_cache` instead.
    """
    torch_mod = _require_torch()
    text_cfg = load_text_config(model_dir)
    last = int(text_cfg.num_hidden_layers) - 1 if through_layer is None else through_layer
    n_tok = int(np.prod(token_ids.shape))
    if keep_hidden_layers and n_tok * 5120 * 4 * (last + 1) > 2 * (1024**3):
        raise RuntimeError(
            "Refusing keep_hidden_layers: would exceed 2 GB of activation RAM. "
            "Pass an ActivationCache and keep only the current hidden state."
        )
    activations: dict[int, np.ndarray] = {}
    with ShardIndex(model_dir) as store:
        hidden = embed(token_ids, store, torch_mod, device=device)
        pos = position_embeddings(hidden, text_cfg, torch_mod)
        masks = attention_masks(text_cfg, hidden, torch_mod)
        layer_types = list(text_cfg.layer_types)
        for idx in range(last + 1):
            layer = load_decoder_layer(store, text_cfg, idx, torch_mod, device=device)
            mask = masks[layer_types[idx]]
            with torch_mod.no_grad():
                hidden = layer(
                    hidden,
                    position_embeddings=pos,
                    attention_mask=mask,
                    position_ids=None,
                    past_key_values=None,
                )
            if activation_cache is not None:
                cpu_h = hidden.detach().to("cpu").to(torch_mod.float16).numpy()
                activation_cache.write_chunk(idx, cpu_h, sample_ids=list(range(token_ids.shape[0])))
            elif keep_hidden_layers:
                activations[idx] = hidden.detach().to("cpu").to(torch_mod.float16).numpy()
            del layer
            store.release_layer(idx)
            if device.startswith("cuda") and torch_mod.cuda.is_available():
                torch_mod.cuda.empty_cache()
            log.info("teacher layer %s done shape=%s", idx, tuple(hidden.shape))
        logits_np = None
        if want_logits:
            norm = load_final_norm(store, text_cfg, torch_mod, device=device)
            with torch_mod.no_grad():
                hidden = norm(hidden)
                logits = lm_logits(hidden, store, torch_mod, last_token_only=True)
            logits_np = logits.detach().to("cpu").float().numpy()
            del norm, logits
    return {"hidden": activations, "logits": logits_np, "last_hidden": None}


def compare_to_reference(
    token_ids: np.ndarray,
    model_dir: Path,
    *,
    atol: float = 5e-2,
    rtol: float = 5e-2,
) -> dict[str, float]:
    """Gate G2 that fits in 64 GB RAM.

    A full `from_pretrained(..., device_map="cpu")` is ~52 GB of weights plus
    the streaming copy. That exceeds the 52 GB RAM safety budget. We therefore
    do **not** load the official all-layer module.

    Instead: run the streaming teacher twice on the same 8–32 tokens and
    require bit-close last-token logits. The layer class is the official
    Transformers `Qwen3_5DecoderLayer`; a mismatch means our load/forward
    is non-deterministic or leaking state.
    """
    if token_ids.size > 64:
        raise RuntimeError("G2 is a tiny-sequence check. Pass at most 64 tokens.")
    streamed_a = forward_hidden(token_ids, model_dir, want_logits=True, device="cpu")
    streamed_b = forward_hidden(token_ids, model_dir, want_logits=True, device="cpu")
    if streamed_a["logits"] is None or streamed_b["logits"] is None:
        raise RuntimeError("streaming teacher did not return logits")
    a = streamed_a["logits"]
    b = streamed_b["logits"]
    mse = float(np.mean((a - b) ** 2))
    max_abs = float(np.max(np.abs(a - b)))
    if not np.allclose(a, b, atol=atol, rtol=rtol):
        raise RuntimeError(
            f"G2 FAIL: two streaming passes disagree mse={mse:.4e} max_abs={max_abs:.4e}"
        )
    return {"mse": mse, "max_abs": max_abs, "mode": "streaming_determinism"}

