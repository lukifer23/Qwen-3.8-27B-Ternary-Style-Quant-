"""Group a weight tensor along its last axis into blocks of `group_size`."""

from __future__ import annotations

import numpy as np


def pad_to_group(weights: np.ndarray, group_size: int) -> tuple[np.ndarray, int]:
    """Pad the last axis with zeros so its length is a multiple of *group_size*.

    Returns (padded, original_last_dim).
    """
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}")
    last = weights.shape[-1]
    remainder = last % group_size
    if remainder == 0:
        return np.ascontiguousarray(weights), last
    pad = group_size - remainder
    padded = np.pad(weights, [(0, 0)] * (weights.ndim - 1) + [(0, pad)], mode="constant")
    return np.ascontiguousarray(padded), last


def group_axis(weights: np.ndarray, group_size: int) -> tuple[np.ndarray, int]:
    """Reshape to (..., n_groups, group_size). Pads if needed."""
    padded, original = pad_to_group(weights, group_size)
    groups = padded.shape[-1] // group_size
    grouped = padded.reshape(*padded.shape[:-1], groups, group_size)
    return grouped, original


def ungroup_axis(grouped: np.ndarray, original_last_dim: int) -> np.ndarray:
    """Inverse of group_axis. Drops padding on the last axis."""
    merged = grouped.reshape(*grouped.shape[:-2], grouped.shape[-2] * grouped.shape[-1])
    return merged[..., :original_last_dim]
