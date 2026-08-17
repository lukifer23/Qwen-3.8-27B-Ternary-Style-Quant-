"""Teacher/student hidden-state losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def hidden_losses(pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
    pred_f = pred.float()
    target_f = target.float()
    mse = F.mse_loss(pred_f, target_f)
    pred_flat = pred_f.reshape(-1, pred_f.shape[-1])
    tgt_flat = target_f.reshape(-1, target_f.shape[-1])
    cos = F.cosine_similarity(pred_flat, tgt_flat, dim=-1).mean()
    rel = mse / (target_f.pow(2).mean() + 1e-8)
    mae = (pred_f - target_f).abs().mean()
    max_err = (pred_f - target_f).abs().max()
    return {
        "mse": mse,
        "relative_mse": rel,
        "cosine": cos,
        "cosine_loss": 1.0 - cos,
        "mae": mae,
        "max_err": max_err,
    }


def combined_loss(parts: dict[str, torch.Tensor], *, hidden_w: float = 1.0, cos_w: float = 0.1) -> torch.Tensor:
    return hidden_w * parts["mse"] + cos_w * parts["cosine_loss"]
