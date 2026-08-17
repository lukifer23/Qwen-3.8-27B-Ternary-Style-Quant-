"""Shard-aware safetensors loader. Never materializes the full 27B BF16 model."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger("q38ternary.safetensors_io")


class ShardIndex:
    def __init__(self, model_dir: Path) -> None:
        self.model_dir = Path(model_dir)
        index_path = self.model_dir / "model.safetensors.index.json"
        single = self.model_dir / "model.safetensors"
        if index_path.is_file():
            payload = json.loads(index_path.read_text(encoding="utf-8"))
            self.weight_map: dict[str, str] = dict(payload.get("weight_map") or {})
            self.metadata: dict[str, Any] = dict(payload.get("metadata") or {})
        elif single.is_file():
            self.weight_map = {}
            self.metadata = {"single_file": str(single)}
            self._single = single
        else:
            raise FileNotFoundError(
                f"No model.safetensors.index.json or model.safetensors under {model_dir}"
            )
        self._single = single if single.is_file() else None
        self._open_shards: dict[str, Any] = {}
        self._layer_cache: dict[int, dict[str, np.ndarray]] = {}
        self._framework = self._select_framework()

    @staticmethod
    def _select_framework() -> str:
        try:
            import torch  # noqa: F401

            return "pt"
        except ImportError:
            pass
        try:
            import ml_dtypes  # noqa: F401

            return "np"
        except ImportError as exc:
            raise RuntimeError(
                "Loading BF16 safetensors needs torch or ml_dtypes. "
                "Run scripts/bootstrap.ps1."
            ) from exc

    def tensor_names(self) -> list[str]:
        if self.weight_map:
            return list(self.weight_map)
        # Single-file fallback: list keys via safetensors.
        from safetensors import safe_open

        assert self._single is not None
        with safe_open(str(self._single), framework=self._framework) as handle:
            return list(handle.keys())

    def shard_for(self, name: str) -> Path:
        if self.weight_map:
            rel = self.weight_map.get(name)
            if rel is None:
                raise KeyError(name)
            return self.model_dir / rel
        if self._single is None:
            raise KeyError(name)
        return self._single

    def _handle(self, shard: Path):
        key = str(shard)
        handle = self._open_shards.get(key)
        if handle is None:
            from safetensors import safe_open

            handle = safe_open(key, framework=self._framework)
            self._open_shards[key] = handle
        return handle

    def load_tensor(self, name: str) -> np.ndarray:
        shard = self.shard_for(name)
        tensor = self._handle(shard).get_tensor(name)
        if hasattr(tensor, "detach"):
            import torch

            cpu = tensor.detach().cpu()
            if cpu.dtype in (torch.bfloat16, torch.float16):
                cpu = cpu.to(dtype=torch.float32)
            array = cpu.numpy()
        else:
            array = np.asarray(tensor)
            if str(array.dtype) in {"bfloat16", "torch.bfloat16", "float16"}:
                array = array.astype(np.float32, copy=False)
        return np.ascontiguousarray(array)

    def layer_tensor_names(self, layer_index: int) -> list[str]:
        # Do not substring-match "layers.0." — that also hits mtp.layers.0.*.
        prefixes = (
            f"model.language_model.layers.{layer_index}.",
            f"language_model.layers.{layer_index}.",
            f"model.layers.{layer_index}.",
        )
        return [name for name in self.tensor_names() if name.startswith(prefixes)]

    def load_layer(self, layer_index: int) -> dict[str, np.ndarray]:
        if layer_index in self._layer_cache:
            return self._layer_cache[layer_index]
        names = self.layer_tensor_names(layer_index)
        if not names:
            raise KeyError(f"no tensors for layer {layer_index}")
        # Open only the shards this layer needs.
        needed_shards = {str(self.shard_for(name)) for name in names}
        log.info("layer %s: %s tensors across %s shard(s)", layer_index, len(names), len(needed_shards))
        loaded = {name: self.load_tensor(name) for name in names}
        self._layer_cache[layer_index] = loaded
        return loaded

    def release_layer(self, layer_index: int) -> None:
        cached = self._layer_cache.pop(layer_index, None)
        if cached is not None:
            cached.clear()
        # Close shard handles that no remaining cached layer needs.
        still_needed: set[str] = set()
        for tensors in self._layer_cache.values():
            for name in tensors:
                still_needed.add(str(self.shard_for(name)))
        for key, handle in list(self._open_shards.items()):
            if key not in still_needed:
                close = getattr(handle, "close", None)
                if callable(close):
                    close()
                self._open_shards.pop(key, None)
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def close(self) -> None:
        for index in list(self._layer_cache):
            self.release_layer(index)
        for handle in self._open_shards.values():
            close = getattr(handle, "close", None)
            if callable(close):
                close()
        self._open_shards.clear()

    def __enter__(self) -> "ShardIndex":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def load_tensor(model_dir: Path, name: str) -> np.ndarray:
    with ShardIndex(model_dir) as store:
        return store.load_tensor(name)


def load_layer(model_dir: Path, layer_index: int) -> dict[str, np.ndarray]:
    with ShardIndex(model_dir) as store:
        # Copy out before close() releases the cache.
        return {k: np.array(v, copy=True) for k, v in store.load_layer(layer_index).items()}


def release_layer(store: ShardIndex, layer_index: int) -> None:
    store.release_layer(layer_index)
