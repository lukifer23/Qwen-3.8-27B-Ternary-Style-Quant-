"""Straight-through estimator for ternary fake-quant."""

from __future__ import annotations

from typing import Any


def ternary_ste(latent: Any, quantized: Any) -> Any:
    """Forward = quantized, backward = identity through *latent*.

    Works with torch tensors. Raises if torch is not installed — reconstruction
    will not silently invent a gradient.
    """
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required for STE fake-quant") from exc

    class _TernarySTE(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, latent_w: Any, quant_w: Any) -> Any:
            return quant_w

        @staticmethod
        def backward(ctx: Any, grad_output: Any) -> tuple[Any, None]:
            return grad_output, None

    return _TernarySTE.apply(latent, quantized)
