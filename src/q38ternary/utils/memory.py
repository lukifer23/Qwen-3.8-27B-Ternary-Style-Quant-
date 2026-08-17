"""RAM / VRAM / disk / temperature polling and recoverable-OOM helpers."""

from __future__ import annotations

import logging
import shutil
import subprocess
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypeVar

from q38ternary.config import AppConfig, repo_root

log = logging.getLogger("q38ternary.memory")

T = TypeVar("T")


@dataclass
class MemorySnapshot:
    gpu_allocated_gb: float | None
    gpu_reserved_gb: float | None
    gpu_free_gb: float | None
    gpu_total_gb: float | None
    system_ram_used_gb: float | None
    system_ram_total_gb: float | None
    pagefile_used_gb: float | None
    disk_free_gb: float
    gpu_temp_c: float | None

    def as_dict(self) -> dict[str, float | None]:
        return asdict(self)


def _bytes_to_gb(value: int | float | None) -> float | None:
    if value is None:
        return None
    return round(float(value) / (1024**3), 3)


def _nvidia_query(*fields: str) -> list[str] | None:
    try:
        completed = subprocess.run(
            ["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    line = completed.stdout.strip().splitlines()
    if not line:
        return None
    return [part.strip() for part in line[0].split(",")]


def snapshot(root: Path | None = None) -> MemorySnapshot:
    base = root or repo_root()
    disk_free = shutil.disk_usage(base).free

    gpu_allocated = gpu_reserved = gpu_free = gpu_total = None
    try:
        import torch

        if torch.cuda.is_available():
            gpu_allocated = _bytes_to_gb(torch.cuda.memory_allocated())
            gpu_reserved = _bytes_to_gb(torch.cuda.memory_reserved())
            props = torch.cuda.get_device_properties(0)
            gpu_total = _bytes_to_gb(props.total_memory)
            free_b, total_b = torch.cuda.mem_get_info()
            gpu_free = _bytes_to_gb(free_b)
            gpu_total = gpu_total or _bytes_to_gb(total_b)
    except ImportError:
        pass

    query = _nvidia_query("memory.total", "memory.free", "temperature.gpu")
    gpu_temp = None
    if query and len(query) >= 3:
        try:
            total_mib = float(query[0])
            free_mib = float(query[1])
            gpu_temp = float(query[2])
            if gpu_total is None:
                gpu_total = round(total_mib / 1024.0, 3)
            if gpu_free is None:
                gpu_free = round(free_mib / 1024.0, 3)
        except ValueError:
            pass

    ram_used = ram_total = pagefile = None
    try:
        import psutil

        vm = psutil.virtual_memory()
        ram_used = _bytes_to_gb(vm.used)
        ram_total = _bytes_to_gb(vm.total)
        swap = psutil.swap_memory()
        pagefile = _bytes_to_gb(swap.used)
    except ImportError:
        pass

    return MemorySnapshot(
        gpu_allocated_gb=gpu_allocated,
        gpu_reserved_gb=gpu_reserved,
        gpu_free_gb=gpu_free,
        gpu_total_gb=gpu_total,
        system_ram_used_gb=ram_used,
        system_ram_total_gb=ram_total,
        pagefile_used_gb=pagefile,
        disk_free_gb=round(disk_free / (1024**3), 3),
        gpu_temp_c=gpu_temp,
    )


def empty_cuda() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        return


def write_progress(
    cfg: AppConfig,
    *,
    stage: str,
    layer: int = 0,
    layers_total: int = 64,
    elapsed_seconds: float = 0.0,
    estimated_remaining_seconds: float = 0.0,
) -> Path:
    mem = snapshot(cfg.root)
    payload = {
        "stage": stage,
        "layer": layer,
        "layers_total": layers_total,
        "elapsed_seconds": elapsed_seconds,
        "estimated_remaining_seconds": estimated_remaining_seconds,
        "gpu_memory_gb": mem.gpu_allocated_gb or mem.gpu_total_gb or 0,
        "system_memory_gb": mem.system_ram_used_gb or 0,
        **mem.as_dict(),
    }
    path = cfg.resolve("artifacts", "progress.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(__import__("json").dumps(payload, indent=2), encoding="utf-8")
    return path


class RecoverableOOM(RuntimeError):
    """Raised after retries are exhausted."""


def run_with_oom_backoff(
    fn: Callable[..., T],
    *args: object,
    batch_size: int,
    min_batch: int = 1,
    cfg: AppConfig | None = None,
    **kwargs: object,
) -> T:
    """Catch CUDA OOM, empty cache, shrink batch, retry. Never crash a multi-day run on the first OOM."""
    current = int(batch_size)
    last_error: BaseException | None = None
    while current >= min_batch:
        try:
            return fn(*args, batch_size=current, **kwargs)
        except RuntimeError as exc:
            message = str(exc).lower()
            if "out of memory" not in message:
                raise
            last_error = exc
            log.warning("OOM at batch_size=%s; emptying cache and retrying", current)
            empty_cuda()
            if cfg is not None:
                write_progress(cfg, stage="oom_retry", layer=0)
            if current == min_batch:
                break
            current = max(min_batch, current // 2)
            time.sleep(1)
    raise RecoverableOOM(f"OOM persisted down to batch_size={min_batch}") from last_error


def thermal_should_pause(cfg: AppConfig) -> bool:
    mem = snapshot(cfg.root)
    if mem.gpu_temp_c is None:
        return False
    return mem.gpu_temp_c >= cfg.gpu_temp_pause_c
