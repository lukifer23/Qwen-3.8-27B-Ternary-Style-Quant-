"""Quantization correctness: known vectors, scale formula, CUDA/CPU fake-quant."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from q38ternary.quant.scaling import least_squares_scale
from q38ternary.quant.ternary import (
    dequantize,
    quantize_absolute,
    quantize_activation_weighted,
    quantize_search,
)
from q38ternary.quant.thresholds import assign_absolute


def test_known_vector_absolute_threshold() -> None:
    weights = np.array([[-3.0, -0.1, 0.0, 0.1, 2.0, 4.0]], dtype=np.float32)
    packed = quantize_absolute(weights, group_size=8, tau=0.5)
    # last dim 6 pads to 8; last two codes are 0
    assert packed.codes.tolist() == [[-1, 0, 0, 0, 1, 1]]
    q = np.array([-1.0, 0.0, 0.0, 0.0, 1.0, 1.0], dtype=np.float64)
    w = weights.reshape(-1).astype(np.float64)
    expected_s = float((w @ q) / (q @ q))
    np.testing.assert_allclose(packed.scales.reshape(-1)[0], expected_s, rtol=1e-5)
    recon = dequantize(packed)
    assert recon.shape == weights.shape
    # Nonzero positions reconstruct as ±s
    np.testing.assert_allclose(recon[0, 0], -expected_s, rtol=1e-5)
    np.testing.assert_allclose(recon[0, 4], expected_s, rtol=1e-5)
    np.testing.assert_allclose(recon[0, 1], 0.0, atol=1e-7)


def test_scale_formula_zero_guard() -> None:
    weights = np.zeros((1, 4), dtype=np.float32)
    codes = np.zeros((1, 4), dtype=np.int8)
    scale = least_squares_scale(weights, codes)
    assert scale.shape == (1, 1)
    assert float(scale.reshape(-1)[0]) == 0.0


def test_assign_absolute_boundaries() -> None:
    w = np.array([-0.5, -0.4, 0.0, 0.4, 0.5], dtype=np.float32)
    codes = assign_absolute(w, 0.4)
    assert codes.tolist() == [-1, 0, 0, 0, 1]


def test_search_beats_bad_fixed_threshold() -> None:
    rng = np.random.default_rng(42)
    # Mixture: a few large weights and a pile of near-zeros.
    large = rng.choice([-2.0, 2.0], size=(4, 32))
    small = rng.normal(0.0, 0.05, size=(4, 96))
    weights = np.concatenate([large, small], axis=1).astype(np.float32)
    naive = quantize_absolute(weights, group_size=16, tau=5.0)  # everything → 0
    searched = quantize_search(weights, group_size=16)
    naive_err = float(np.mean((weights - dequantize(naive)) ** 2))
    search_err = float(np.mean((weights - dequantize(searched)) ** 2))
    assert search_err < naive_err
    assert not np.allclose(dequantize(searched), 0.0)


def test_activation_weighted_uses_salient_channels() -> None:
    rng = np.random.default_rng(0)
    in_features = 32
    out_features = 8
    W = rng.normal(0.0, 1.0, size=(out_features, in_features)).astype(np.float32)
    # Make channel 0 hugely more important than the others.
    X = rng.normal(0.0, 0.01, size=(64, in_features))
    X[:, 0] = rng.normal(0.0, 5.0, size=64)
    packed = quantize_activation_weighted(W, X, group_size=16)
    recon = dequantize(packed)
    assert recon.shape == W.shape
    # Reconstructing the salient input column should be no worse than naive MSE search.
    mse_aw = float(np.mean((X @ W.T - X @ recon.T) ** 2))
    mse_search = float(np.mean((X @ W.T - X @ dequantize(quantize_search(W, 16)).T) ** 2))
    assert mse_aw <= mse_search * 1.15


def test_codes_are_ternary_alphabet() -> None:
    rng = np.random.default_rng(1)
    W = rng.normal(size=(7, 200)).astype(np.float32)
    packed = quantize_search(W, group_size=128)
    unique = set(int(v) for v in np.unique(packed.codes))
    assert unique <= {-1, 0, 1}


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="torch not installed",
)
def test_fake_quant_cpu_matches_numpy() -> None:
    import torch

    from q38ternary.quant.fake_quant import fake_ternary_numpy, fake_ternary_torch

    rng = np.random.default_rng(3)
    W = rng.normal(size=(4, 64)).astype(np.float32)
    np_q, _ = fake_ternary_numpy(W, group_size=32)
    torch_q = fake_ternary_torch(torch.from_numpy(W.copy()), group_size=32)
    np.testing.assert_allclose(torch_q.detach().cpu().numpy(), np_q, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="torch not installed",
)
def test_ste_passes_gradient() -> None:
    import torch

    from q38ternary.quant.fake_quant import fake_ternary_torch

    latent = torch.randn(2, 32, requires_grad=True)
    out = fake_ternary_torch(latent, group_size=16)
    out.sum().backward()
    assert latent.grad is not None
    assert latent.grad.shape == latent.shape
    # STE is identity: a non-zero output gradient must reach the latent.
    assert float(latent.grad.abs().sum()) > 0.0
