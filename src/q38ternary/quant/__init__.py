"""Ternary / mixed-bit quantization primitives."""

from q38ternary.quant.grouping import group_axis, pad_to_group, ungroup_axis
from q38ternary.quant.scaling import least_squares_scale
from q38ternary.quant.ternary import (
    TernaryTensor,
    dequantize,
    quantize_activation_weighted,
    quantize_absolute,
    quantize_hessian_diag,
    quantize_search,
)

__all__ = [
    "TernaryTensor",
    "dequantize",
    "group_axis",
    "least_squares_scale",
    "pad_to_group",
    "quantize_absolute",
    "quantize_activation_weighted",
    "quantize_hessian_diag",
    "quantize_search",
    "ungroup_axis",
]
