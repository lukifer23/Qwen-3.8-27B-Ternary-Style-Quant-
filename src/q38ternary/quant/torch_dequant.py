"""GPU-resident dequant: ŵ = s q. Differentiable in the scale."""

from __future__ import annotations

from typing import Any

import torch


def dequant_grouped(codes: torch.Tensor, scales: torch.Tensor, group_size: int, last_dim: int) -> torch.Tensor:
    """codes: (..., last_dim) int8 in {-1,0,+1}; scales: flat or (..., n_groups[, 1])."""
    pad = (group_size - last_dim % group_size) % group_size
    if pad:
        codes = torch.nn.functional.pad(codes, (0, pad))
    groups = codes.shape[-1] // group_size
    grouped = codes.reshape(*codes.shape[:-1], groups, group_size).to(dtype=scales.dtype)
    scale = scales.reshape(*grouped.shape[:-1], 1)
    recon = grouped * scale
    return recon.reshape(*codes.shape[:-1], groups * group_size)[..., :last_dim]
