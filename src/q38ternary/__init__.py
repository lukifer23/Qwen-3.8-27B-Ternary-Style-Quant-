"""Activation-aware ternary / mixed-bit pipeline for Qwen3.8-27B."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("qwen38-ternary")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = ["__version__"]
