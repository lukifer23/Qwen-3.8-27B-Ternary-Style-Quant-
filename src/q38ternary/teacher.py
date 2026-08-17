"""Teacher-input activation cache for a small set of layers."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from q38ternary.activation_cache import ActivationCache
from q38ternary.config import AppConfig
from q38ternary.safetensors_io import ShardIndex
from q38ternary.streaming_model import (
    attention_masks,
    embed,
    load_decoder_layer,
    load_text_config,
    position_embeddings,
)
from q38ternary.utils.memory import snapshot, write_progress

log = logging.getLogger("q38ternary.teacher")


def cache_pilot_layers(
    cfg: AppConfig,
    tokens: np.ndarray,
    *,
    device: str,
    chunk_size: int = 8,
) -> Path:
    """Walk the teacher once. Write FP16 in/out only for `pilot_layers`."""
    torch_mod = torch
    model_dir = cfg.model_local_dir
    text_cfg = load_text_config(model_dir)
    pilots = set(cfg.pilot_layers)
    last = max(pilots)
    out_dir = cfg.resolve("cache", "activations", "pilot")
    out_dir.mkdir(parents=True, exist_ok=True)
    caches: dict[tuple[int, str], ActivationCache] = {}
    for layer_idx in sorted(pilots):
        for side in ("in", "out"):
            caches[(layer_idx, side)] = ActivationCache(
                out_dir,
                layer=layer_idx,
                tag=side,
                meta={
                    "side": side,
                    "teacher_revision": cfg.model_repo,
                    "dtype": "float16",
                    "sequence_length": int(tokens.shape[1]),
                },
            )

    n = int(tokens.shape[0])
    chunk_ids = list(range(0, n, chunk_size))
    with ShardIndex(model_dir) as store:
        for chunk_i, start in enumerate(chunk_ids):
            end = min(n, start + chunk_size)
            ids = tokens[start:end]
            sample_ids = list(range(start, end))
            hidden = embed(ids, store, torch_mod, device=device)
            pos = position_embeddings(hidden, text_cfg, torch_mod)
            masks = attention_masks(text_cfg, hidden, torch_mod)
            layer_types = list(text_cfg.layer_types)
            for idx in range(last + 1):
                if idx in pilots:
                    cpu_in = hidden.detach().to("cpu").to(torch_mod.float16).numpy()
                    cache = caches[(idx, "in")]
                    cache.write_chunk(chunk_i, cpu_in, sample_ids=sample_ids)
                layer = load_decoder_layer(store, text_cfg, idx, torch_mod, device=device)
                with torch_mod.no_grad():
                    hidden = layer(
                        hidden,
                        position_embeddings=pos,
                        attention_mask=masks[layer_types[idx]],
                        position_ids=None,
                        past_key_values=None,
                    )
                if idx in pilots:
                    cpu_out = hidden.detach().to("cpu").to(torch_mod.float16).numpy()
                    cache = caches[(idx, "out")]
                    cache.write_chunk(chunk_i, cpu_out, sample_ids=sample_ids)
                del layer
                store.release_layer(idx)
                if device.startswith("cuda"):
                    torch_mod.cuda.empty_cache()
            mem = snapshot(cfg.root)
            write_progress(
                cfg,
                stage="teacher_cache",
                layer=last,
                layers_total=last + 1,
                elapsed_seconds=0.0,
            )
            log.info(
                "chunk %s/%s seqs %s:%s gpu_alloc=%s ram=%s",
                chunk_i + 1,
                len(chunk_ids),
                start,
                end,
                mem.gpu_allocated_gb,
                mem.system_ram_used_gb,
            )
    for cache in caches.values():
        cache.flush_manifest(cfg)
    return out_dir


def _ctx(hidden, text_cfg, torch_mod):
    pos = position_embeddings(hidden, text_cfg, torch_mod)
    masks = attention_masks(text_cfg, hidden, torch_mod)
    return pos, masks


def cache_layer0_from_tokens(
    cfg: AppConfig,
    tokens: np.ndarray,
    *,
    device: str,
    chunk_size: int,
    out_dir: Path,
) -> tuple[ActivationCache, ActivationCache]:
    """Embed + run teacher layer 0. Does not walk the rest of the stack."""
    text_cfg = load_text_config(cfg.model_local_dir)
    cache_in = ActivationCache(out_dir, layer=0, tag="in", meta={"side": "in"})
    cache_out = ActivationCache(out_dir, layer=0, tag="out", meta={"side": "out"})
    n = int(tokens.shape[0])
    with ShardIndex(cfg.model_local_dir) as store:
        layer = load_decoder_layer(store, text_cfg, 0, torch, device=device)
        store.release_layer(0)
        for chunk_i, start in enumerate(range(0, n, chunk_size)):
            end = min(n, start + chunk_size)
            ids = tokens[start:end]
            hidden = embed(ids, store, torch, device=device)
            pos, masks = _ctx(hidden, text_cfg, torch)
            cache_in.write_chunk(chunk_i, hidden.detach().to("cpu").float().half().numpy(), sample_ids=list(range(start, end)))
            with torch.no_grad():
                out = layer(
                    hidden,
                    position_embeddings=pos,
                    attention_mask=masks[list(text_cfg.layer_types)[0]],
                    position_ids=None,
                    past_key_values=None,
                )
            cache_out.write_chunk(chunk_i, out.detach().to("cpu").float().half().numpy(), sample_ids=list(range(start, end)))
            log.info("layer0 cache chunk %s/%s", chunk_i + 1, (n + chunk_size - 1) // chunk_size)
        del layer
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    cache_in.flush_manifest(cfg)
    cache_out.flush_manifest(cfg)
    return cache_in, cache_out


def cache_next_teacher_out(
    cfg: AppConfig,
    layer_idx: int,
    cache_in: ActivationCache,
    *,
    device: str,
    out_dir: Path,
) -> ActivationCache:
    """Run one teacher layer on an existing input cache."""
    text_cfg = load_text_config(cfg.model_local_dir)
    cache_out = ActivationCache(out_dir, layer=layer_idx, tag="out", meta={"side": "out"})
    with ShardIndex(cfg.model_local_dir) as store:
        layer = load_decoder_layer(store, text_cfg, layer_idx, torch, device=device)
        store.release_layer(layer_idx)
        first = np.load(cache_in.iter_chunks()[0]["path"], mmap_mode="r")
        dummy = torch.zeros(1, first.shape[1], first.shape[2], device=device, dtype=torch.bfloat16)
        pos, masks = _ctx(dummy, text_cfg, torch)
        mask = masks[list(text_cfg.layer_types)[layer_idx]]
        for rec in cache_in.iter_chunks():
            x = torch.from_numpy(np.array(np.load(rec["path"]), copy=True)).to(device=device, dtype=torch.bfloat16)
            with torch.no_grad():
                y = layer(x, position_embeddings=pos, attention_mask=mask, position_ids=None, past_key_values=None)
            cache_out.write_chunk(int(rec["chunk_id"]), y.detach().to("cpu").float().half().numpy(), sample_ids=list(rec["sample_ids"]))
        del layer
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    cache_out.flush_manifest(cfg)
    return cache_out
