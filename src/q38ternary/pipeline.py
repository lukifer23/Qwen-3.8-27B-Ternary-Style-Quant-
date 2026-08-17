"""Single entrypoint. Stages run in the spec order and stop on a failed gate."""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections.abc import Callable
from typing import Any

from q38ternary.config import AppConfig, load_config
from q38ternary.hardware import (
    HardwareError,
    require_cuda,
    require_disk_for_download,
    write_hardware_reports,
)
from q38ternary.utils.logging import record_experiment, setup_logging
from q38ternary.utils.memory import write_progress

log = logging.getLogger("q38ternary.pipeline")

# Ordered exactly as Project_Plan.md `--mode auto`.
AUTO_STAGES = (
    "detect_environment",
    "bootstrap_dependencies",
    "verify_cuda",
    "check_disk",
    "clone_dependencies",
    "pin_revisions",
    "download_model",
    "inspect_architecture",
    "create_baselines",
    "pilot_calibration",
    "layer_pilot",
    "evaluate_pilot",
    "choose_reconstruction",
    "full_calibration",
    "sensitivity_scan",
    "hybrid_precision_map",
    "reconstruct_all",
    "pack_checkpoint",
    "load_gguf",
    "benchmark",
    "target_worst_layers",
    "repack",
    "benchmark_again",
    "select_best_model",
    "final_report",
)


class GateFailure(RuntimeError):
    """A hard stop. Do not continue to the next expensive stage."""


def _stage_detect_environment(cfg: AppConfig, **_: Any) -> dict[str, Any]:
    snapshot = write_hardware_reports(cfg)
    return {"checks": snapshot["checks"], "gpu": snapshot.get("gpu")}


def _stage_bootstrap_dependencies(cfg: AppConfig, **_: Any) -> dict[str, Any]:
    required = ("yaml", "numpy", "psutil")
    missing = []
    for name in required:
        try:
            __import__(name)
        except ImportError:
            missing.append(name)
    if missing:
        raise GateFailure(
            f"missing Python packages {missing}. Run scripts/bootstrap.ps1 (or bootstrap.sh)."
        )
    return {"packages": list(required)}


def _stage_verify_cuda(cfg: AppConfig, **_: Any) -> dict[str, Any]:
    snapshot = write_hardware_reports(cfg)
    # Reconstruction later requires CUDA. Detection itself is allowed without torch.
    if snapshot["torch"].get("importable") and not snapshot["torch"].get("cuda_available"):
        require_cuda(snapshot)
    return {"cuda_ready": snapshot["checks"]["cuda_ready"], "torch": snapshot["torch"]}


def _stage_check_disk(cfg: AppConfig, **_: Any) -> dict[str, Any]:
    snapshot = write_hardware_reports(cfg)
    require_disk_for_download(cfg, snapshot)
    return {"free_gb": snapshot["disk_workspace"]["free_gb"]}


def _stage_clone_dependencies(cfg: AppConfig, **_: Any) -> dict[str, Any]:
    from q38ternary.sources import sync_third_party

    return sync_third_party(cfg)


def _stage_pin_revisions(cfg: AppConfig, **_: Any) -> dict[str, Any]:
    path = cfg.resolve("third_party", "versions.json")
    if not path.is_file():
        raise GateFailure("third_party/versions.json missing; clone_dependencies must run first")
    data = json.loads(path.read_text(encoding="utf-8"))
    missing = [name for name, meta in data.items() if not meta.get("commit")]
    if missing:
        raise GateFailure(f"unpinned third-party checkouts: {missing}")
    return {"versions": data}


def _stage_inspect_architecture(cfg: AppConfig, **_: Any) -> dict[str, Any]:
    from q38ternary.architecture import write_architecture_reports
    from q38ternary.inventory import write_inventory
    from q38ternary.runtime import inspect_runtimes
    from q38ternary.size import write_size_report

    if not (cfg.model_local_dir / "config.json").is_file():
        raise GateFailure("teacher config.json is missing; download_model must run first")
    arch = write_architecture_reports(cfg)
    inventory = write_inventory(cfg)
    sizes = write_size_report(cfg, arch, inventory=inventory)
    runtimes = inspect_runtimes(cfg)
    cfg.resolve("artifacts").mkdir(parents=True, exist_ok=True)
    cfg.resolve("artifacts", "runtime.json").write_text(
        json.dumps(runtimes, indent=2), encoding="utf-8"
    )
    return {
        "layers": arch.get("num_hidden_layers"),
        "full_attention": len(arch.get("full_attention_indices") or []),
        "deltanet": len(arch.get("deltanet_indices") or []),
        "language_parameters": inventory["language_parameters"],
        "predicted_gb": sizes["predicted_gguf_gb"],
        "ternary_runtime": runtimes["ternary_runtime"],
    }


def _stage_pilot_calibration(cfg: AppConfig, **_: Any) -> dict[str, Any]:
    from q38ternary.calibration import build_split

    calib = build_split(cfg, tier="pilot", holdout=False)
    hold = build_split(cfg, tier="pilot", holdout=True)
    return {
        "calibration_sequences": calib["sequences"],
        "holdout_sequences": hold["sequences"],
        "length": calib["length"],
    }


def _stage_download_model(cfg: AppConfig, **kwargs: Any) -> dict[str, Any]:
    from q38ternary.hf import download_config, download_weights, read_config_json, verify_teacher_identity

    dest = download_config(cfg)
    config = read_config_json(dest)
    verify_teacher_identity(cfg, config)
    if kwargs.get("weights"):
        download_weights(cfg)
        return {"dir": str(cfg.model_local_dir), "weights": True}
    return {"dir": str(dest), "weights": False}


STAGE_HANDLERS: dict[str, Callable[..., dict[str, Any]]] = {
    "detect_environment": _stage_detect_environment,
    "bootstrap_dependencies": _stage_bootstrap_dependencies,
    "verify_cuda": _stage_verify_cuda,
    "check_disk": _stage_check_disk,
    "clone_dependencies": _stage_clone_dependencies,
    "pin_revisions": _stage_pin_revisions,
    "download_model": _stage_download_model,
    "inspect_architecture": _stage_inspect_architecture,
    "pilot_calibration": _stage_pilot_calibration,
}

# Registered by later slices as they land. Missing names in AUTO_STAGES
# are a hard stop with a clear next-slice message — never a silent skip.
_UNIMPLEMENTED_HINT = (
    "This stage is not wired yet. The previous stage completed; implement the "
    "next slice rather than inventing a skip."
)


def run_stage(cfg: AppConfig, name: str, **kwargs: Any) -> dict[str, Any]:
    handler = STAGE_HANDLERS.get(name)
    if handler is None:
        raise GateFailure(f"stage {name!r} is not implemented. {_UNIMPLEMENTED_HINT}")
    write_progress(cfg, stage=name)
    started = time.perf_counter()
    log.info("stage start: %s", name)
    result = handler(cfg, **kwargs)
    duration = time.perf_counter() - started
    record_experiment(cfg, stage=name, metrics=result, duration_seconds=duration)
    log.info("stage done: %s (%.1fs)", name, duration)
    return result


def run_auto(cfg: AppConfig, *, start_from: str | None = None) -> None:
    started = False if start_from else True
    for name in AUTO_STAGES:
        if not started:
            if name == start_from:
                started = True
            else:
                continue
        run_stage(cfg, name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Qwen3.8-27B activation-aware ternary pipeline"
    )
    parser.add_argument(
        "--mode",
        choices=("auto", "pilot", "reconstruct", "detect"),
        default="detect",
    )
    parser.add_argument("--target", choices=("hybrid", "ternary"), default="hybrid")
    parser.add_argument("--budget-gb", type=float, default=None)
    parser.add_argument("--start-from", default=None, help="Resume auto mode at this stage name")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config()
    setup_logging(f"pipeline-{args.mode}", cfg.root)
    log.info("config hash %s seed %s", cfg.hash(), cfg.seed)
    try:
        if args.mode == "detect":
            run_stage(cfg, "detect_environment")
            run_stage(cfg, "check_disk")
            return 0
        if args.mode == "auto":
            run_auto(cfg, start_from=args.start_from)
            return 0
        if args.mode == "pilot":
            for name in (
                "detect_environment",
                "check_disk",
                "pilot_calibration",
                "layer_pilot",
                "evaluate_pilot",
            ):
                run_stage(cfg, name)
            return 0
        if args.mode == "reconstruct":
            kwargs = {"target": args.target, "budget_gb": args.budget_gb}
            run_stage(cfg, "reconstruct_all", **kwargs)
            return 0
    except (GateFailure, HardwareError) as exc:
        log.error("GATE STOP: %s", exc)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
