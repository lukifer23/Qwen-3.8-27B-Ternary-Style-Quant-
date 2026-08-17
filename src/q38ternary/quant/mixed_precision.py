"""Precision assignment and byte-cost estimates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

Scheme = Literal["ternary", "3bit", "4bit", "fp16", "fp32"]

# Deployed storage, not information-theoretic size.
# ternary g128: 2-bit codes + fp16 scale / 128 → 2.125 bits/weight
BITS_PER_WEIGHT: dict[str, float] = {
    "ternary": 2.125,
    "3bit": 3.5,   # typical Q3_K-style deployed, not a claim of exact GGUF layout
    "4bit": 4.5,   # typical Q4_K-style deployed
    "fp16": 16.0,
    "fp32": 32.0,
}


@dataclass(frozen=True)
class TensorAssignment:
    name: str
    n_params: int
    scheme: Scheme
    group_size: int = 128

    @property
    def bytes(self) -> int:
        return int(round(self.n_params * BITS_PER_WEIGHT[self.scheme] / 8.0))


def estimate_bytes(n_params: int, scheme: Scheme) -> int:
    return int(round(n_params * BITS_PER_WEIGHT[scheme] / 8.0))


def total_gb(assignments: Iterable[TensorAssignment]) -> float:
    return sum(item.bytes for item in assignments) / (1024**3)
