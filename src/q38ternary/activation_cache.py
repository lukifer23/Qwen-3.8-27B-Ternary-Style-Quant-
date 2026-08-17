"""Bounded, chunked activation cache. Never one gigantic fragile file."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from q38ternary.config import AppConfig
from q38ternary.utils.manifest import write_manifest


class ActivationCache:
    def __init__(self, directory: Path, *, layer: int, meta: dict[str, Any]) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.layer = int(layer)
        self.meta = dict(meta)
        self._index: list[dict[str, Any]] = []
        self._index_path = self.directory / f"layer_{self.layer:03d}.index.json"

    def write_chunk(
        self,
        chunk_id: int,
        activations: np.ndarray,
        *,
        sample_ids: list[int],
    ) -> Path:
        path = self.directory / f"layer_{self.layer:03d}_chunk_{chunk_id:05d}.npy"
        array = np.ascontiguousarray(activations.astype(np.float16, copy=False))
        np.save(path, array)
        record = {
            "chunk_id": chunk_id,
            "path": str(path),
            "sample_ids": list(sample_ids),
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            **self.meta,
        }
        self._index.append(record)
        self._index_path.write_text(json.dumps(self._index, indent=2), encoding="utf-8")
        return path

    def flush_manifest(self, cfg: AppConfig) -> Path:
        write_manifest(
            cfg,
            self.directory / f"layer_{self.layer:03d}",
            kind="activation_cache",
            extra={"layer": self.layer, "chunks": len(self._index), **self.meta},
        )
        return self._index_path

    def iter_chunks(self) -> list[dict[str, Any]]:
        if self._index_path.is_file() and not self._index:
            self._index = json.loads(self._index_path.read_text(encoding="utf-8"))
        return list(self._index)

    def load_chunk(self, chunk_id: int) -> np.ndarray:
        for record in self.iter_chunks():
            if record["chunk_id"] == chunk_id:
                return np.load(record["path"])
        raise KeyError(chunk_id)
