"""FP values → ternary codes/scales → packed Q2_0 → unpack → reconstructed tensor."""

from __future__ import annotations

import numpy as np

from q38ternary.gguf.prism_format import PRISM_Q2_GROUP, pack_q2_0, unpack_q2_0
from q38ternary.quant.ternary import dequantize, quantize_search


def test_search_then_pack_then_unpack_matches() -> None:
    rng = np.random.default_rng(11)
    weights = rng.normal(size=(5, 200)).astype(np.float32)
    packed = quantize_search(weights, group_size=PRISM_Q2_GROUP)
    blob = pack_q2_0(packed.codes, packed.scales.reshape(-1), group_size=PRISM_Q2_GROUP)
    codes, scales = unpack_q2_0(blob, packed.codes.shape, group_size=PRISM_Q2_GROUP)
    np.testing.assert_array_equal(codes, packed.codes)
    # Reconstruct from unpacked codes using the (fp16-rounded) packed scales.
    from q38ternary.quant.ternary import TernaryTensor

    restored = TernaryTensor(
        codes=codes.reshape(packed.codes.shape),
        scales=scales.reshape(packed.scales.shape),
        group_size=packed.group_size,
        original_shape=packed.original_shape,
        original_last_dim=packed.original_last_dim,
        scheme=packed.scheme,
    )
    # Codes are exact. Reconstruction can only differ by fp16 scale rounding.
    a = dequantize(packed)
    b = dequantize(restored)
    np.testing.assert_allclose(b, a, rtol=1e-3, atol=1e-3)
