"""Grouping, padding, and non-divisible last-dimension edge cases."""

from __future__ import annotations

import numpy as np
import pytest

from q38ternary.quant.grouping import group_axis, pad_to_group, ungroup_axis
from q38ternary.quant.ternary import dequantize, quantize_absolute, quantize_search


def test_pad_when_not_divisible() -> None:
    w = np.arange(10, dtype=np.float32).reshape(2, 5)
    padded, original = pad_to_group(w, 4)
    assert original == 5
    assert padded.shape == (2, 8)
    np.testing.assert_array_equal(padded[:, :5], w)
    np.testing.assert_array_equal(padded[:, 5:], 0)


def test_no_pad_when_divisible() -> None:
    w = np.ones((3, 8), dtype=np.float32)
    padded, original = pad_to_group(w, 4)
    assert original == 8
    assert padded.shape == (3, 8)


def test_group_ungroup_roundtrip() -> None:
    w = np.arange(30, dtype=np.float32).reshape(2, 3, 5)
    grouped, original = group_axis(w, 4)
    assert grouped.shape == (2, 3, 2, 4)
    restored = ungroup_axis(grouped, original)
    np.testing.assert_array_equal(restored, w)


def test_quantize_non_divisible_shape() -> None:
    rng = np.random.default_rng(4)
    w = rng.normal(size=(11, 100)).astype(np.float32)
    packed = quantize_search(w, group_size=128)
    recon = dequantize(packed)
    assert recon.shape == (11, 100)
    assert packed.codes.shape == (11, 100)
    # 100 is not divisible by 128 → one padded group.
    assert packed.scales.shape[-2] == 1


def test_bad_group_size() -> None:
    with pytest.raises(ValueError):
        pad_to_group(np.ones((2, 4)), 0)


def test_absolute_preserves_leading_dims() -> None:
    w = np.ones((2, 3, 10), dtype=np.float32)
    packed = quantize_absolute(w, group_size=8, tau=0.0)
    assert packed.codes.shape == (2, 3, 10)
    assert dequantize(packed).shape == (2, 3, 10)
