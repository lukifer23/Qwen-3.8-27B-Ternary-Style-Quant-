"""Run the 6-layer pilot: naive ternary vs scales-only reconstruction."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch

from q38ternary.activation_cache import ActivationCache
from q38ternary.config import AppConfig
from q38ternary.reconstruction.block_runner import (
    eval_layer,
    pack_linear_weights,
    reconstruct_scales,
)
from q38ternary.safetensors_io import ShardIndex
from q38ternary.streaming_model import attention_masks, load_decoder_layer, load_text_config, position_embeddings
from q38ternary.utils.manifest import write_manifest

log = logging.getLogger("q38ternary.pilot")


def _open_cache(root: Path, layer: int, side: str) -> ActivationCache:
    cache = ActivationCache(root, layer=layer, tag=side, meta={"side": side})
    if not cache._index_path.is_file():
        raise FileNotFoundError(cache._index_path)
    cache.iter_chunks()
    return cache


def _batches(
    cache_in: ActivationCache,
    cache_out: ActivationCache,
    *,
    device: str,
    batch_size: int,
    max_sequences: int | None,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    seen = 0
    for rec in cache_in.iter_chunks():
        x = np.load(rec["path"], mmap_mode="r")
        y = cache_out.load_chunk(int(rec["chunk_id"]))
        for start in range(0, x.shape[0], batch_size):
            if max_sequences is not None and seen >= max_sequences:
                return
            xb = np.array(x[start : start + batch_size], copy=True)
            yb = np.array(y[start : start + batch_size], copy=True)
            yield (
                torch.from_numpy(xb).to(device=device, dtype=torch.bfloat16),
                torch.from_numpy(yb).to(device=device, dtype=torch.bfloat16),
            )
            seen += xb.shape[0]


def run_one_layer(
    cfg: AppConfig,
    layer_idx: int,
    *,
    device: str,
    steps: int,
    lr: float,
    batch_size: int = 2,
    eval_sequences: int = 16,
    train_sequences: int = 64,
    cache_root: Path | None = None,
) -> dict[str, Any]:
    cache_root = Path(cache_root) if cache_root is not None else cfg.resolve("cache", "activations", "pilot")
    cache_in = _open_cache(cache_root, layer_idx, "in")
    cache_out = _open_cache(cache_root, layer_idx, "out")
    text_cfg = load_text_config(cfg.model_local_dir)
    with ShardIndex(cfg.model_local_dir) as store:
        layer = load_decoder_layer(store, text_cfg, layer_idx, torch, device=device)
        store.release_layer(layer_idx)

    first_x, _first_y = next(_batches(cache_in, cache_out, device=device, batch_size=1, max_sequences=1))
    seq = int(first_x.shape[1])
    hidden = int(first_x.shape[-1])
    dummy = torch.zeros(batch_size, seq, hidden, device=device, dtype=torch.bfloat16)
    pos = position_embeddings(dummy, text_cfg, torch)
    masks = attention_masks(text_cfg, dummy, torch)
    mask = masks[list(text_cfg.layer_types)[layer_idx]]
    del first_x, _first_y, dummy

    packed = pack_linear_weights(layer, group_size=cfg.group_size)
    # Bake naive ternary into the live weights for the baseline measurement.
    from q38ternary.reconstruction.block_runner import install_dequant_forwards, restore_forwards

    originals = install_dequant_forwards(layer, packed)
    eval_in, eval_tgt = [], []
    for i, pair in enumerate(
        _batches(cache_in, cache_out, device=device, batch_size=batch_size, max_sequences=eval_sequences)
    ):
        eval_in.append(pair[0])
        eval_tgt.append(pair[1])
    naive_metrics = _mean_eval(layer, eval_in, eval_tgt, pos, mask)
    restore_forwards(originals)

    train_batches = list(
        _batches(cache_in, cache_out, device=device, batch_size=batch_size, max_sequences=train_sequences)
    )
    train_last = reconstruct_scales(
        layer,
        packed,
        train_batches,
        position_embeddings=pos,
        attention_mask=mask,
        steps=steps,
        lr=lr,
    )
    recovered = _mean_eval(layer, eval_in, eval_tgt, pos, mask)

    improved_rel = recovered["relative_mse"] < naive_metrics["relative_mse"]
    improved_cos = recovered["cosine"] > naive_metrics["cosine"]
    report = {
        "layer": layer_idx,
        "layer_type": list(text_cfg.layer_types)[layer_idx],
        "packed_tensors": [
            {"name": p.name, "naive_weight_mse": p.naive_mse, "groups": int(p.scales.numel())}
            for p in packed
        ],
        "naive": naive_metrics,
        "reconstructed": recovered,
        "train_last": train_last,
        "improved_relative_mse": improved_rel,
        "improved_cosine": improved_cos,
        "gate_pass": bool(improved_rel or improved_cos),
        "steps": steps,
        "lr": lr,
        "device": device,
    }
    log.info(
        "layer %s naive rel=%.4e cos=%.4f -> recon rel=%.4e cos=%.4f gate=%s",
        layer_idx,
        naive_metrics["relative_mse"],
        naive_metrics["cosine"],
        recovered["relative_mse"],
        recovered["cosine"],
        report["gate_pass"],
    )
    del layer
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return report, packed


def _mean_eval(layer, inputs, targets, pos, mask) -> dict[str, float]:
    acc: dict[str, list[float]] = {}
    for inp, tgt in zip(inputs, targets, strict=True):
        metrics = eval_layer(layer, inp, tgt, position_embeddings=pos, attention_mask=mask)
        for key, value in metrics.items():
            acc.setdefault(key, []).append(value)
    return {key: float(np.mean(vals)) for key, vals in acc.items()}


def write_pilot_report(cfg: AppConfig, layers: list[dict[str, Any]]) -> Path:
    payload = {
        "layers": layers,
        "any_gate_pass": any(item["gate_pass"] for item in layers),
        "all_gate_pass": all(item["gate_pass"] for item in layers),
    }
    reports = cfg.resolve("artifacts", "reports")
    reports.mkdir(parents=True, exist_ok=True)
    json_path = reports / "layer_pilot.json"
    md_path = reports / "layer_pilot.md"
    numbered = reports / "03_layer_pilot.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# 03 Layer pilot",
        "",
        f"- layers: {len(layers)}",
        f"- all gates pass: {payload['all_gate_pass']}",
        "",
        "| layer | type | naive rel MSE | recon rel MSE | naive cos | recon cos | gate |",
        "|------:|------|--------------:|--------------:|----------:|----------:|------|",
    ]
    for item in layers:
        lines.append(
            f"| {item['layer']} | {item['layer_type']} | "
            f"{item['naive']['relative_mse']:.4e} | {item['reconstructed']['relative_mse']:.4e} | "
            f"{item['naive']['cosine']:.4f} | {item['reconstructed']['cosine']:.4f} | "
            f"{'PASS' if item['gate_pass'] else 'FAIL'} |"
        )
    lines.append("")
    md = "\n".join(lines)
    md_path.write_text(md, encoding="utf-8")
    numbered.write_text(md, encoding="utf-8")
    write_manifest(cfg, json_path, kind="layer_pilot", extra={"all_gate_pass": payload["all_gate_pass"]})
    return json_path
