"""Ternary initializers A–D and dequantization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from q38ternary.quant.grouping import group_axis, ungroup_axis
from q38ternary.quant.scaling import least_squares_scale
from q38ternary.quant.thresholds import assign_absolute, candidate_thresholds


@dataclass
class TernaryTensor:
    codes: np.ndarray  # int8, same logical shape as the source (no pad)
    scales: np.ndarray  # float32, shape (..., n_groups, 1)
    group_size: int
    original_shape: tuple[int, ...]
    original_last_dim: int
    scheme: str

    def dequantize(self) -> np.ndarray:
        return dequantize(self)


def dequantize(tensor: TernaryTensor) -> np.ndarray:
    grouped, _ = group_axis(tensor.codes.astype(np.float32), tensor.group_size)
    reconstructed = grouped * tensor.scales.astype(np.float32)
    return ungroup_axis(reconstructed, tensor.original_last_dim).reshape(tensor.original_shape)


def _pack(codes_grouped: np.ndarray, scales: np.ndarray, original: np.ndarray, group_size: int, scheme: str) -> TernaryTensor:
    original_last = original.shape[-1]
    codes = ungroup_axis(codes_grouped, original_last).astype(np.int8, copy=False)
    return TernaryTensor(
        codes=codes.reshape(original.shape),
        scales=scales.astype(np.float32, copy=False),
        group_size=group_size,
        original_shape=tuple(original.shape),
        original_last_dim=original_last,
        scheme=scheme,
    )


def quantize_absolute(weights: np.ndarray, group_size: int = 128, tau: float = 0.0) -> TernaryTensor:
    """Initializer A: fixed absolute threshold, then LS scale."""
    grouped, _ = group_axis(np.asarray(weights, dtype=np.float32), group_size)
    codes = assign_absolute(grouped, tau)
    scales = least_squares_scale(grouped, codes)
    return _pack(codes, scales, np.asarray(weights), group_size, "absolute_threshold")


def _mse(grouped: np.ndarray, codes: np.ndarray, scales: np.ndarray) -> np.ndarray:
    recon = codes.astype(np.float64) * scales.astype(np.float64)
    return np.mean((grouped.astype(np.float64) - recon) ** 2, axis=-1)


def quantize_search(weights: np.ndarray, group_size: int = 128) -> TernaryTensor:
    """Initializer B: search τ per group, keep the assignment with lowest ||W-sQ||²."""
    grouped, _ = group_axis(np.asarray(weights, dtype=np.float32), group_size)
    # One candidate set from the whole tensor is cheap and covers typical magnitudes;
    # we still evaluate each candidate independently per group.
    flat_candidates = candidate_thresholds(grouped)
    best_mse = np.full(grouped.shape[:-1], np.inf, dtype=np.float64)
    best_codes = np.zeros_like(grouped, dtype=np.int8)
    best_scales = np.zeros(grouped.shape[:-1] + (1,), dtype=np.float32)

    for tau in flat_candidates:
        codes = assign_absolute(grouped, float(tau))
        scales = least_squares_scale(grouped, codes)
        mse = _mse(grouped, codes, scales)
        better = mse < best_mse
        best_mse = np.where(better, mse, best_mse)
        # Expand `better` over the group axis for codes.
        better_codes = better[..., None]
        best_codes = np.where(better_codes, codes, best_codes)
        best_scales = np.where(better[..., None], scales, best_scales)

    return _pack(best_codes, best_scales, np.asarray(weights), group_size, "search_threshold")


def _weighted_scale(weights: np.ndarray, codes: np.ndarray, diag_h: np.ndarray) -> np.ndarray:
    """s = (qᵀ H w) / (qᵀ H q) with H diagonal. Guard zero denominator."""
    numerator = np.sum(diag_h * codes * weights, axis=-1, keepdims=True)
    denominator = np.sum(diag_h * codes * codes, axis=-1, keepdims=True)
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float64),
        where=denominator != 0,
    ).astype(np.float32)


def _group_diag(diag: np.ndarray, group_size: int, like: np.ndarray) -> np.ndarray:
    """Broadcast a per-input-channel diagonal onto grouped weight columns."""
    diag = np.asarray(diag, dtype=np.float64)
    if diag.ndim == 1:
        grouped, _ = group_axis(diag.reshape(1, -1), group_size)
        return np.broadcast_to(grouped, like.shape)
    grouped, _ = group_axis(diag, group_size)
    if grouped.shape[-2:] != like.shape[-2:]:
        raise ValueError(
            f"grouped diagonal {grouped.shape} incompatible with grouped weights {like.shape}"
        )
    return grouped


def quantize_activation_weighted(
    weights: np.ndarray,
    activations: np.ndarray,
    group_size: int = 128,
) -> TernaryTensor:
    """Initializer C: minimize ||XW - XW_q||² via the diagonal of H ≈ XᵀX.

    `activations` is X with shape (tokens, in_features) matching weights' last dim.
    """
    weights = np.asarray(weights, dtype=np.float32)
    X = np.asarray(activations, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"activations must be 2-D (tokens, in_features), got {X.shape}")
    if X.shape[1] != weights.shape[-1]:
        raise ValueError(
            f"activations in_features {X.shape[1]} != weight last dim {weights.shape[-1]}"
        )
    # Diagonal of XᵀX is cheap and always fits. Full H is optional (initializer D).
    diag = np.einsum("ti,ti->i", X, X)
    grouped, _ = group_axis(weights, group_size)
    diag_g = _group_diag(diag, group_size, grouped)

    flat_candidates = candidate_thresholds(grouped)
    best_err = np.full(grouped.shape[:-1], np.inf, dtype=np.float64)
    best_codes = np.zeros_like(grouped, dtype=np.int8)
    best_scales = np.zeros(grouped.shape[:-1] + (1,), dtype=np.float32)

    for tau in flat_candidates:
        codes = assign_absolute(grouped, float(tau))
        scales = _weighted_scale(grouped, codes, diag_g)
        recon = codes.astype(np.float64) * scales.astype(np.float64)
        # Weighted SSE per group: (w-wq)ᵀ diag(H) (w-wq)
        err = np.sum(diag_g * (grouped.astype(np.float64) - recon) ** 2, axis=-1)
        better = err < best_err
        best_err = np.where(better, err, best_err)
        best_codes = np.where(better[..., None], codes, best_codes)
        best_scales = np.where(better[..., None], scales, best_scales)

    return _pack(best_codes, best_scales, weights, group_size, "activation_weighted")


def quantize_hessian_diag(
    weights: np.ndarray,
    activations: np.ndarray,
    group_size: int = 128,
) -> TernaryTensor:
    """Initializer D (diagonal): same H≈XᵀX diagonal used by C, kept as a named entry point.

    A full Hessian is intentionally not materialized — it would not fit this machine
    for a 5120-d feature map times a long calibration set.
    """
    result = quantize_activation_weighted(weights, activations, group_size=group_size)
    result.scheme = "hessian_diag"
    return result
