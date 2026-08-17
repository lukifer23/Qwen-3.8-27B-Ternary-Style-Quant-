"""Scales-only reconstruction of one Qwen3.8 decoder block."""

from __future__ import annotations

import logging
import json
from dataclasses import dataclass
from pathlib import Path
from types import MethodType
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from q38ternary.quant.ternary import dequantize, quantize_search
from q38ternary.quant.torch_dequant import dequant_grouped
from q38ternary.reconstruction.losses import combined_loss, hidden_losses

log = logging.getLogger("q38ternary.reconstruct")

SKIP_SUBSTRINGS = (
    "norm.weight",
    "A_log",
    "dt_bias",
    "bias",
)


@dataclass
class PackedWeight:
    name: str
    codes: torch.Tensor
    scales: nn.Parameter
    group_size: int
    last_dim: int
    naive_mse: float


def eligible(name: str, tensor: torch.Tensor) -> bool:
    if tensor.ndim < 2:
        return False
    if tensor.numel() < 4096:
        return False
    lower = name.lower()
    return not any(skip in lower for skip in SKIP_SUBSTRINGS)


def _child(root: nn.Module, param_name: str) -> nn.Module:
    module = root
    for part in param_name.split(".")[:-1]:
        module = getattr(module, part)
    return module


def pack_linear_weights(layer: nn.Module, group_size: int = 128) -> list[PackedWeight]:
    packed: list[PackedWeight] = []
    for name, param in layer.named_parameters():
        if not eligible(name, param):
            continue
        cpu = param.detach().float().cpu().numpy()
        ternary = quantize_search(cpu, group_size=group_size)
        recon = dequantize(ternary)
        naive_mse = float(np.mean((cpu - recon) ** 2))
        codes = torch.from_numpy(np.ascontiguousarray(ternary.codes)).to(device=param.device)
        scales = nn.Parameter(
            torch.from_numpy(np.ascontiguousarray(ternary.scales.reshape(-1).astype(np.float32))).to(
                device=param.device
            )
        )
        packed.append(
            PackedWeight(
                name=name,
                codes=codes,
                scales=scales,
                group_size=group_size,
                last_dim=int(ternary.original_last_dim),
                naive_mse=naive_mse,
            )
        )
        del cpu, recon
        log.info("packed %s shape=%s weight_mse=%.4e", name, tuple(param.shape), naive_mse)
    return packed


def _safe_name(name: str) -> str:
    return name.replace(".", "__")


def save_packed(directory: Path, packed: list[PackedWeight], *, name_prefix: str = "") -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    index = []
    for item in packed:
        full = f"{name_prefix}{item.name}"
        stem = _safe_name(full)
        codes_path = directory / f"{stem}.codes.npy"
        scales_path = directory / f"{stem}.scales.npy"
        np.save(codes_path, item.codes.detach().cpu().numpy())
        np.save(scales_path, item.scales.detach().cpu().numpy())
        index.append(
            {
                "name": full,
                "codes": codes_path.name,
                "scales": scales_path.name,
                "group_size": item.group_size,
                "last_dim": item.last_dim,
                "naive_mse": item.naive_mse,
            }
        )
    (directory / "tensors.json").write_text(json.dumps(index, indent=2), encoding="utf-8")


def _weight(item: PackedWeight, dtype: torch.dtype) -> torch.Tensor:
    return dequant_grouped(item.codes, item.scales, item.group_size, item.last_dim).to(dtype=dtype)


def install_dequant_forwards(layer: nn.Module, packed: list[PackedWeight]) -> list[tuple[nn.Module, Any]]:
    """Patch Linear/Conv1d so ŵ = s q stays in the autograd graph."""
    originals: list[tuple[nn.Module, Any]] = []
    for item in packed:
        module = _child(layer, item.name)
        originals.append((module, module.forward))
        if isinstance(module, nn.Linear):

            def linear_fwd(mod: nn.Module, x: torch.Tensor, *, _item: PackedWeight = item) -> torch.Tensor:
                return F.linear(x, _weight(_item, x.dtype), mod.bias)

            module.forward = MethodType(linear_fwd, module)
        elif isinstance(module, nn.Conv1d):

            def conv_fwd(mod: nn.Conv1d, x: torch.Tensor, *, _item: PackedWeight = item) -> torch.Tensor:
                return F.conv1d(
                    x,
                    _weight(_item, x.dtype),
                    mod.bias,
                    stride=mod.stride,
                    padding=mod.padding,
                    dilation=mod.dilation,
                    groups=mod.groups,
                )

            module.forward = MethodType(conv_fwd, module)
        else:
            raise TypeError(f"unsupported module for {item.name}: {type(module)}")
    return originals


def restore_forwards(originals: list[tuple[nn.Module, Any]]) -> None:
    for module, fwd in originals:
        module.forward = fwd


@torch.no_grad()
def bake_weights(layer: nn.Module, packed: list[PackedWeight]) -> None:
    lookup = dict(layer.named_parameters())
    for item in packed:
        lookup[item.name].data.copy_(_weight(item, lookup[item.name].dtype).detach())


@torch.no_grad()
def eval_layer(
    layer: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    position_embeddings: Any,
    attention_mask: Any,
) -> dict[str, float]:
    layer.eval()
    pred = layer(
        inputs,
        position_embeddings=position_embeddings,
        attention_mask=attention_mask,
        position_ids=None,
        past_key_values=None,
    )
    parts = hidden_losses(pred, targets)
    return {k: float(v.detach().cpu()) for k, v in parts.items()}


def reconstruct_scales(
    layer: nn.Module,
    packed: list[PackedWeight],
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    *,
    position_embeddings: Any,
    attention_mask: Any,
    steps: int,
    lr: float,
) -> dict[str, float]:
    originals = install_dequant_forwards(layer, packed)
    opt = torch.optim.AdamW([item.scales for item in packed], lr=lr, weight_decay=0.0)
    layer.train()
    last: dict[str, float] = {}
    try:
        for step in range(steps):
            inp, tgt = batches[step % len(batches)]
            opt.zero_grad(set_to_none=True)
            pred = layer(
                inp,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                position_ids=None,
                past_key_values=None,
            )
            parts = hidden_losses(pred, tgt)
            loss = combined_loss(parts)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([item.scales for item in packed], 1.0)
            opt.step()
            last = {k: float(v.detach().cpu()) for k, v in parts.items()}
            last["loss"] = float(loss.detach().cpu())
            if step == 0 or (step + 1) % 25 == 0 or step + 1 == steps:
                log.info(
                    "step %s/%s loss=%.4e rel_mse=%.4e cos=%.4f",
                    step + 1,
                    steps,
                    last["loss"],
                    last["relative_mse"],
                    last["cosine"],
                )
    finally:
        restore_forwards(originals)
        bake_weights(layer, packed)
        layer.eval()
    return last
