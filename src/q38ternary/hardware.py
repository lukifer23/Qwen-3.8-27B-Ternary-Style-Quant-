"""Hardware / toolchain discovery. Writes artifacts/hardware.json and environment.txt."""

from __future__ import annotations

import json
import logging
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from q38ternary.config import AppConfig, load_config

log = logging.getLogger("q38ternary.hardware")


class HardwareError(RuntimeError):
    """Unrecoverable environment problem (missing GPU after install, no disk, ...)."""


def _run(command: list[str] | str, *, shell: bool = False, timeout: int = 30) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            shell=shell,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        return {
            "command": command if isinstance(command, str) else " ".join(command),
            "ok": False,
            "returncode": None,
            "stdout": "",
            "stderr": str(exc),
        }
    except subprocess.TimeoutExpired:
        return {
            "command": command if isinstance(command, str) else " ".join(command),
            "ok": False,
            "returncode": None,
            "stdout": "",
            "stderr": "timeout",
        }
    return {
        "command": command if isinstance(command, str) else " ".join(command),
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def _which(name: str) -> str | None:
    return shutil.which(name)


def _cim(class_name: str, properties: list[str]) -> list[dict[str, str]]:
    if os.name != "nt":
        return []
    joined = ", ".join(properties)
    script = f"Get-CimInstance {class_name} | Select-Object {joined} | ConvertTo-Json -Compress"
    result = _run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
        timeout=45,
    )
    if not result["ok"] or not result["stdout"]:
        return []
    try:
        parsed = json.loads(result["stdout"])
    except json.JSONDecodeError:
        return []
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        return [parsed]
    return []


def _nvidia_smi_full() -> dict[str, Any]:
    result = _run(["nvidia-smi"], timeout=20)
    query = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used,memory.free,driver_version,temperature.gpu,power.draw,power.limit,utilization.gpu",
            "--format=csv,noheader",
        ],
        timeout=20,
    )
    gpus: list[dict[str, str]] = []
    if query["ok"] and query["stdout"]:
        header = [
            "name",
            "memory_total",
            "memory_used",
            "memory_free",
            "driver_version",
            "temperature_gpu",
            "power_draw",
            "power_limit",
            "utilization_gpu",
        ]
        for line in query["stdout"].splitlines():
            parts = [p.strip() for p in line.split(",")]
            gpus.append(dict(zip(header, parts, strict=False)))
    return {"raw": result, "gpus": gpus}


def _parse_mib(text: str | None) -> float | None:
    if not text:
        return None
    token = text.replace("MiB", "").replace("GiB", "").strip().split()[0]
    try:
        value = float(token)
    except ValueError:
        return None
    if "GiB" in (text or ""):
        return value
    return round(value / 1024.0, 3)


def _torch_probe() -> dict[str, Any]:
    info: dict[str, Any] = {
        "importable": False,
        "version": None,
        "cuda_build": None,
        "cuda_available": False,
        "device_name": None,
        "device_total_memory": None,
    }
    try:
        import torch
    except ImportError as exc:
        info["error"] = str(exc)
        return info
    info["importable"] = True
    info["version"] = torch.__version__
    info["cuda_build"] = getattr(torch.version, "cuda", None)
    info["cuda_available"] = bool(torch.cuda.is_available())
    if info["cuda_available"]:
        info["device_name"] = torch.cuda.get_device_name(0)
        info["device_total_memory"] = int(torch.cuda.get_device_properties(0).total_memory)
    return info


def collect_hardware(cfg: AppConfig) -> dict[str, Any]:
    disk = shutil.disk_usage(cfg.root)
    processors = _cim("Win32_Processor", ["Name", "NumberOfCores", "NumberOfLogicalProcessors", "MaxClockSpeed"])
    operating_system = _cim("Win32_OperatingSystem", ["Caption", "Version", "BuildNumber"])
    memory_modules = _cim("Win32_PhysicalMemory", ["Capacity", "Speed", "Manufacturer", "PartNumber"])
    volumes = _cim("Win32_Volume", ["DriveLetter", "Label", "Capacity", "FreeSpace", "FileSystem"])
    # Get-Volume is friendlier on modern Windows; keep both.
    get_volume = _run(
        [
            "powershell",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            "Get-Volume | Where-Object { $_.DriveLetter } | "
            "Select-Object DriveLetter, FileSystemLabel, Size, SizeRemaining | ConvertTo-Json -Compress",
        ]
    ) if os.name == "nt" else {"ok": False, "stdout": "", "stderr": "not-windows"}

    ram_bytes = 0
    for module in memory_modules:
        try:
            ram_bytes += int(module.get("Capacity") or 0)
        except (TypeError, ValueError):
            pass

    nvidia = _nvidia_smi_full()
    torch_info = _torch_probe()
    primary = nvidia["gpus"][0] if nvidia["gpus"] else {}

    snapshot = {
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "caption": (operating_system[0].get("Caption") if operating_system else None),
            "build": (operating_system[0].get("BuildNumber") if operating_system else None),
        },
        "python": {
            "version": sys.version,
            "executable": sys.executable,
        },
        "cpu": processors,
        "memory_modules": memory_modules,
        "memory_total_gb": round(ram_bytes / (1024**3), 3) if ram_bytes else None,
        "volumes_cim": volumes,
        "volumes": get_volume,
        "disk_workspace": {
            "path": str(cfg.root),
            "total_gb": round(disk.total / (1024**3), 3),
            "used_gb": round(disk.used / (1024**3), 3),
            "free_gb": round(disk.free / (1024**3), 3),
        },
        "nvidia_smi": nvidia,
        "gpu": {
            "name": primary.get("name"),
            "memory_total_gb": _parse_mib(primary.get("memory_total")),
            "memory_used_gb": _parse_mib(primary.get("memory_used")),
            "memory_free_gb": _parse_mib(primary.get("memory_free")),
            "driver_version": primary.get("driver_version"),
            "temperature_c": primary.get("temperature_gpu"),
            "power_draw": primary.get("power_draw"),
            "power_limit": primary.get("power_limit"),
        },
        "torch": torch_info,
        "toolchain": {
            "python": _run([sys.executable, "--version"]),
            "git": _run(["git", "--version"]),
            "cmake": _run(["cmake", "--version"]),
            "nvcc": _run(["nvcc", "--version"]),
            "uv": _run(["uv", "--version"]),
        },
        "which": {
            "python": sys.executable,
            "git": _which("git"),
            "cmake": _which("cmake"),
            "nvcc": _which("nvcc"),
            "uv": _which("uv"),
            "nvidia-smi": _which("nvidia-smi"),
        },
        "wsl2": _detect_wsl(),
        "limits": {
            "gpu_memory_limit_gb": cfg.gpu_memory_limit_gb,
            "system_memory_limit_gb": cfg.system_memory_limit_gb,
            "minimum_free_disk_gb": cfg.minimum_free_disk_gb,
            "preferred_free_disk_gb": cfg.preferred_free_disk_gb,
        },
    }
    snapshot["checks"] = evaluate_checks(cfg, snapshot)
    return snapshot


def _detect_wsl() -> dict[str, Any]:
    if os.name != "nt":
        return {"present": False, "reason": "not-windows"}
    result = _run(
        ["wsl", "--status"],
        timeout=15,
    )
    return {
        "present": result["ok"],
        "status": result["stdout"] or result["stderr"],
        "hard_dependency": False,
    }


def evaluate_checks(cfg: AppConfig, snapshot: dict[str, Any]) -> dict[str, Any]:
    free_gb = float(snapshot["disk_workspace"]["free_gb"])
    gpu = snapshot.get("gpu") or {}
    torch_info = snapshot.get("torch") or {}
    used_gb = gpu.get("memory_used_gb")
    issues: list[str] = []
    warnings: list[str] = []

    if free_gb < cfg.minimum_free_disk_gb:
        issues.append(
            f"Free disk {free_gb:.1f} GB is below the {cfg.minimum_free_disk_gb:.0f} GB floor. "
            "Refuse large downloads."
        )
    elif free_gb < cfg.preferred_free_disk_gb:
        warnings.append(
            f"Free disk {free_gb:.1f} GB is below the preferred {cfg.preferred_free_disk_gb:.0f} GB."
        )

    if not snapshot["which"].get("nvidia-smi"):
        issues.append("nvidia-smi is not on PATH. A CUDA GPU is required.")
    if gpu.get("name") is None:
        issues.append("No NVIDIA GPU reported by nvidia-smi.")

    if used_gb is not None and gpu.get("memory_total_gb"):
        headroom = float(gpu["memory_total_gb"]) - float(used_gb)
        if headroom < 4.0:
            warnings.append(
                f"GPU already has {used_gb:.2f} GB in use "
                f"({gpu['memory_total_gb']:.2f} GB total). "
                "Reconstruction will need that VRAM free. "
                "This process will not kill other GPU programs."
            )

    if torch_info.get("importable") and not torch_info.get("cuda_available"):
        issues.append(
            "torch is importable but torch.cuda.is_available() is False. "
            "Install an official CUDA PyTorch build before GPU reconstruction. "
            "Do not silently fall back to CPU for multi-day workloads."
        )

    if not snapshot["which"].get("cmake"):
        warnings.append("cmake is not on PATH. llama.cpp CUDA builds will need a compiler toolchain.")
    if not snapshot["which"].get("nvcc"):
        warnings.append("nvcc is not on PATH. llama.cpp CUDA builds will need the CUDA toolkit.")

    return {
        "ok": not issues,
        "issues": issues,
        "warnings": warnings,
        "cuda_ready": bool(torch_info.get("cuda_available")),
        "disk_ok_for_download": free_gb >= cfg.minimum_free_disk_gb,
    }


def render_environment_txt(snapshot: dict[str, Any]) -> str:
    lines = [
        f"collected_at: {snapshot['collected_at']}",
        f"host: {snapshot['hostname']}",
        f"os: {snapshot['os']['system']} {snapshot['os']['release']}",
        f"python: {snapshot['python']['version'].splitlines()[0]}",
        f"python_executable: {snapshot['python']['executable']}",
        "",
        "=== toolchain ===",
    ]
    for name, payload in snapshot["toolchain"].items():
        body = payload.get("stdout") or payload.get("stderr") or "not found"
        lines.append(f"{name}: {body.splitlines()[0] if body else 'not found'}")
    lines.append("")
    lines.append("=== nvidia-smi ===")
    raw = snapshot["nvidia_smi"]["raw"]
    lines.append(raw.get("stdout") or raw.get("stderr") or "nvidia-smi failed")
    lines.append("")
    lines.append("=== torch ===")
    lines.append(json.dumps(snapshot["torch"], indent=2))
    lines.append("")
    lines.append("=== checks ===")
    lines.append(json.dumps(snapshot["checks"], indent=2))
    return "\n".join(lines) + "\n"


def render_hardware_report(snapshot: dict[str, Any]) -> str:
    gpu = snapshot.get("gpu") or {}
    disk = snapshot["disk_workspace"]
    checks = snapshot["checks"]
    cpu_name = ""
    if snapshot["cpu"]:
        cpu_name = str(snapshot["cpu"][0].get("Name") or "")
    lines = [
        "# 00 Hardware",
        "",
        f"Collected at `{snapshot['collected_at']}` on `{snapshot['hostname']}`.",
        "",
        "## Machine",
        "",
        f"- OS: {snapshot['os'].get('caption') or (snapshot['os']['system'] + ' ' + snapshot['os']['release'])}",
        f"- CPU: {cpu_name or 'unknown'}",
        f"- RAM: {snapshot.get('memory_total_gb') or 'unknown'} GB",
        f"- GPU: {gpu.get('name') or 'none reported'}",
        f"- GPU memory: {gpu.get('memory_total_gb')} GB total, {gpu.get('memory_used_gb')} GB used, {gpu.get('memory_free_gb')} GB free",
        f"- Driver: {gpu.get('driver_version')}",
        f"- Workspace disk: {disk['free_gb']} GB free of {disk['total_gb']} GB at `{disk['path']}`",
        "",
        "## Toolchain",
        "",
    ]
    for name, payload in snapshot["toolchain"].items():
        body = payload.get("stdout") or payload.get("stderr") or "not found"
        first = body.splitlines()[0] if body else "not found"
        lines.append(f"- {name}: `{first}`")
    torch_info = snapshot["torch"]
    lines.extend(
        [
            "",
            "## PyTorch",
            "",
            f"- importable: {torch_info.get('importable')}",
            f"- version: {torch_info.get('version')}",
            f"- cuda build: {torch_info.get('cuda_build')}",
            f"- cuda available: {torch_info.get('cuda_available')}",
            f"- device: {torch_info.get('device_name')}",
            "",
            "## Checks",
            "",
            f"- ok: {checks['ok']}",
            f"- cuda_ready: {checks['cuda_ready']}",
            f"- disk_ok_for_download: {checks['disk_ok_for_download']}",
        ]
    )
    if checks["issues"]:
        lines.append("")
        lines.append("### Issues")
        lines.append("")
        for item in checks["issues"]:
            lines.append(f"- {item}")
    if checks["warnings"]:
        lines.append("")
        lines.append("### Warnings")
        lines.append("")
        for item in checks["warnings"]:
            lines.append(f"- {item}")
    lines.append("")
    lines.append("Power limits and clocks were not modified.")
    lines.append("")
    return "\n".join(lines)


def write_hardware_reports(cfg: AppConfig | None = None) -> dict[str, Any]:
    cfg = cfg or load_config()
    snapshot = collect_hardware(cfg)
    artifacts = cfg.resolve("artifacts")
    reports = artifacts / "reports"
    artifacts.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    hardware_json = artifacts / "hardware.json"
    environment_txt = artifacts / "environment.txt"
    report_md = reports / "00_hardware.md"

    hardware_json.write_text(json.dumps(snapshot, indent=2, default=str), encoding="utf-8")
    environment_txt.write_text(render_environment_txt(snapshot), encoding="utf-8")
    report_md.write_text(render_hardware_report(snapshot), encoding="utf-8")

    log.info("wrote %s", hardware_json)
    log.info("wrote %s", environment_txt)
    log.info("wrote %s", report_md)
    return snapshot


def require_disk_for_download(cfg: AppConfig, snapshot: dict[str, Any] | None = None) -> None:
    snapshot = snapshot or collect_hardware(cfg)
    if not snapshot["checks"]["disk_ok_for_download"]:
        raise HardwareError("; ".join(snapshot["checks"]["issues"]))


def require_cuda(snapshot: dict[str, Any]) -> None:
    if not snapshot["checks"]["cuda_ready"]:
        raise HardwareError(
            "CUDA PyTorch is required before GPU reconstruction. "
            + " ".join(snapshot["checks"]["issues"])
        )


def main(argv: list[str] | None = None) -> int:
    del argv
    from q38ternary.utils.logging import setup_logging

    cfg = load_config()
    setup_logging("detect_hardware", cfg.root)
    snapshot = write_hardware_reports(cfg)
    checks = snapshot["checks"]
    if checks["warnings"]:
        for warning in checks["warnings"]:
            log.warning(warning)
    if not checks["ok"]:
        for issue in checks["issues"]:
            log.error(issue)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
