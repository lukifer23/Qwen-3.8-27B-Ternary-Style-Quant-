"""Threshold rules that map a group of weights onto {-1, 0, +1}."""

from __future__ import annotations

import numpy as np


def assign_absolute(weights: np.ndarray, tau: np.ndarray | float) -> np.ndarray:
    """q = -1 if w < -τ, 0 if |w| ≤ τ, +1 if w > τ.

    Compare in the weight dtype so float32 boundary values are not pushed
    over the threshold by a float64 cast.
    """
    tau_arr = np.asarray(tau, dtype=weights.dtype)
    codes = np.sign(weights).astype(np.int8, copy=False)
    return np.where(np.abs(weights) <= tau_arr, 0, codes).astype(np.int8, copy=False)


def candidate_thresholds(group: np.ndarray) -> np.ndarray:
    """Percentiles, mean(|w|)·k and std(w)·k, plus magnitude breakpoints."""
    flat = np.asarray(group, dtype=np.float64).reshape(-1)
    abs_w = np.abs(flat)
    candidates = [0.0]
    for q in (0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70):
        candidates.append(float(np.quantile(abs_w, q)))
    mean_abs = float(np.mean(abs_w))
    std = float(np.std(flat))
    for scalar in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0):
        candidates.append(mean_abs * scalar)
        candidates.append(std * scalar)
    unique = np.unique(np.asarray(candidates, dtype=np.float64))
    return unique[np.isfinite(unique)]
