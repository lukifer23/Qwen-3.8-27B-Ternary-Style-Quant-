"""Closed-form least-squares scale for a fixed ternary assignment."""

from __future__ import annotations

import numpy as np


def least_squares_scale(weights: np.ndarray, codes: np.ndarray, axis: int = -1) -> np.ndarray:
    """s = (wᵀq) / (qᵀq) per group. Guard qᵀq = 0 → s = 0."""
    if weights.shape != codes.shape:
        raise ValueError(f"shape mismatch: weights {weights.shape} vs codes {codes.shape}")
    numerator = np.sum(weights * codes, axis=axis, keepdims=True)
    denominator = np.sum(codes * codes, axis=axis, keepdims=True)
    scale = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float64),
        where=denominator != 0,
    )
    return scale.astype(np.float32, copy=False)
