"""Differentiable fake-ternary: forward uses codes/scales, export never ships latents."""

from __future__ import annotations

from typing import Any

import numpy as np

from q38ternary.quant.ste import ternary_ste
from q38ternary.quant.ternary import TernaryTensor, dequantize, quantize_search


def fake_ternary_numpy(weights: np.ndarray, group_size: int = 128) -> tuple[np.ndarray, TernaryTensor]:
    packed = quantize_search(weights, group_size=group_size)
    return dequantize(packed), packed


def fake_ternary_torch(latent: Any, group_size: int = 128) -> Any:
    """Quantize a torch latent on the forward pass; STE on the backward pass."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required for fake_ternary_torch") from exc

    np_w = latent.detach().to(dtype=torch.float32, device="cpu").numpy()
    packed = quantize_search(np_w, group_size=group_size)
    quant_np = dequantize(packed)
    quant = torch.from_numpy(np.ascontiguousarray(quant_np)).to(device=latent.device, dtype=latent.dtype)
    return ternary_ste(latent, quant)
