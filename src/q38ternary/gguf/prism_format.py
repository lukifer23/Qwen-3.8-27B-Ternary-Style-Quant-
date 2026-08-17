"""Q2_0 block layout as implemented in the inspected llama.cpp trees.

Inspected 2026-08-17:

* Prism fork `third_party/prism-llama.cpp` @ pinned commit: `QK2_0 = 128`
* Upstream `third_party/llama.cpp` @ pinned commit: `QK2_0 = 64`

Block (both trees), from `ggml-common.h`:

    struct block_q2_0 { ggml_half d; uint8_t qs[QK2_0 / 4]; }

Encoding (both trees, `quantize_row_q2_0_ref`):

    d = max_j |x_j|
    q = clamp(round(x / d) + 1, 0, 3)
    2-bit packing, 4 values per byte, low bits first

    00 → -1, 01 → 0, 10 → +1, 11 → +2   (decode: (q - 1) * d)

Our ternary exporter only emits q ∈ {0,1,2}. The +2 code is never used.
When we have a reconstructed per-group scale s, we store s as `d` rather
than the naive amax used by llama-quantize.
"""

from __future__ import annotations

import struct

import numpy as np

from q38ternary.quant.grouping import group_axis, ungroup_axis

PRISM_Q2_GROUP = 128
UPSTREAM_Q2_GROUP = 64

# 00, 01, 10  ↔  -1, 0, +1
TERNARY_TO_U2 = {-1: 0, 0: 1, 1: 2}
U2_TO_TERNARY = {0: -1, 1: 0, 2: 1, 3: 2}


def pack_q2_0(codes: np.ndarray, scales: np.ndarray, group_size: int = PRISM_Q2_GROUP) -> bytes:
    """Pack int8 codes in {-1,0,+1} plus per-group float scales into Q2_0 blocks.

    Groups along the last axis — the same convention as the ternary quantizer —
    then walks groups in C order. `scales` is one value per group.
    """
    grouped, _ = group_axis(np.asarray(codes, dtype=np.int8), group_size)
    n_groups = int(np.prod(grouped.shape[:-1]))
    scale_arr = np.asarray(scales, dtype=np.float32).reshape(-1)
    if scale_arr.size == 1:
        scale_arr = np.repeat(scale_arr, n_groups)
    if scale_arr.size != n_groups:
        raise ValueError(f"expected {n_groups} scales, got {scale_arr.size}")

    blocks = bytearray()
    grouped = grouped.reshape(n_groups, group_size)
    for group, scale in zip(grouped, scale_arr, strict=True):
        blocks += struct.pack("<e", float(scale))
        qs = bytearray(group_size // 4)
        for j, code in enumerate(group):
            q = TERNARY_TO_U2.get(int(code))
            if q is None:
                raise ValueError(f"refusing to pack non-ternary code {int(code)}")
            qs[j // 4] |= (q & 0x03) << ((j % 4) * 2)
        blocks += qs
    return bytes(blocks)


def unpack_q2_0(
    blob: bytes,
    shape: tuple[int, ...] | int,
    group_size: int = PRISM_Q2_GROUP,
) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of pack_q2_0. `shape` is the logical code tensor shape (last axis unpadded)."""
    if isinstance(shape, int):
        shape = (shape,)
    block_bytes = 2 + group_size // 4
    if len(blob) % block_bytes != 0:
        raise ValueError(f"blob length {len(blob)} is not a multiple of block size {block_bytes}")
    n_groups = len(blob) // block_bytes
    grouped = np.empty((n_groups, group_size), dtype=np.int8)
    scales = np.empty(n_groups, dtype=np.float32)
    for i in range(n_groups):
        off = i * block_bytes
        (scale,) = struct.unpack_from("<e", blob, off)
        scales[i] = scale
        qs = blob[off + 2 : off + block_bytes]
        for j in range(group_size):
            q = (qs[j // 4] >> ((j % 4) * 2)) & 0x03
            grouped[i, j] = U2_TO_TERNARY[q]
    groups_per_row = (shape[-1] + group_size - 1) // group_size
    expected = int(np.prod(shape[:-1])) * groups_per_row if shape[:-1] else groups_per_row
    if expected != n_groups:
        raise ValueError(f"shape {shape} implies {expected} groups, blob has {n_groups}")
    grouped = grouped.reshape(*shape[:-1], groups_per_row, group_size)
    codes = ungroup_axis(grouped, shape[-1])
    return codes, scales
