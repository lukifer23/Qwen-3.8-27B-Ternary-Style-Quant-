"""Q2_0 pack ↔ unpack must be bit-exact for ternary codes and scales."""

from __future__ import annotations

import numpy as np
import pytest

from q38ternary.gguf.prism_format import (
    PRISM_Q2_GROUP,
    UPSTREAM_Q2_GROUP,
    pack_q2_0,
    unpack_q2_0,
)


@pytest.mark.parametrize("group_size", [PRISM_Q2_GROUP, UPSTREAM_Q2_GROUP])
def test_pack_unpack_exact(group_size: int) -> None:
    rng = np.random.default_rng(7)
    n = group_size * 3 + 11  # not divisible
    codes = rng.choice(np.array([-1, 0, 1], dtype=np.int8), size=n)
    scales = rng.random(4).astype(np.float32) * 2.0 + 0.01
    blob = pack_q2_0(codes, scales, group_size=group_size)
    out_codes, out_scales = unpack_q2_0(blob, n, group_size=group_size)
    np.testing.assert_array_equal(out_codes, codes)
    # fp16 round-trip on the scales we actually stored (including the pad group).
    stored = np.array(
        [float(np.float16(s)) for s in list(scales)],
        dtype=np.float32,
    )
    np.testing.assert_allclose(out_scales, stored, rtol=0, atol=0)


def test_refuses_non_ternary_code() -> None:
    codes = np.array([0, 1, 2, -1], dtype=np.int8)
    with pytest.raises(ValueError, match="non-ternary"):
        pack_q2_0(codes, np.array([1.0], dtype=np.float32), group_size=4)


def test_block_size_matches_ggml() -> None:
    # sizeof(block_q2_0) = 2 + QK2_0/4
    blob = pack_q2_0(np.zeros(PRISM_Q2_GROUP, dtype=np.int8), np.array([1.5], np.float32), PRISM_Q2_GROUP)
    assert len(blob) == 2 + PRISM_Q2_GROUP // 4
    blob64 = pack_q2_0(np.zeros(UPSTREAM_Q2_GROUP, dtype=np.int8), np.array([1.5], np.float32), UPSTREAM_Q2_GROUP)
    assert len(blob64) == 2 + UPSTREAM_Q2_GROUP // 4
