"""Unattended 64-layer reconstruct → GGUF → optional llama.cpp smoke test."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import time
from pathlib import Path

from q38ternary.activation_cache import ActivationCache
from q38ternary.calibration import load_token_array
from q38ternary.cleanup import cleanup_after_layer, cleanup_run
from q38ternary.config import AppConfig
from q38ternary.reconstruction.block_runner import save_packed
from q38ternary.reconstruction.pilot import run_one_layer
from q38ternary.teacher import cache_layer0_from_tokens, cache_next_teacher_out
from q38ternary.utils.checkpoints import StageState, save_stage
from q38ternary.utils.memory import snapshot, write_progress

log = logging.getLogger("q38ternary.overnight")


def _fmt(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


def _layer_done(ckpt_root: Path, layer: int) -> bool:
    path = ckpt_root / f"layer_{layer:03d}" / "metrics.json"
    if not path.is_file():
        return False
    data = json.loads(path.read_text(encoding="utf-8"))
    return bool(data.get("complete") or data.get("gate_pass") is not None)


def _open(root: Path, layer: int, tag: str) -> ActivationCache:
    cache = ActivationCache(root, layer=layer, tag=tag, meta={"side": tag})
    if cache._index_path.is_file():
        cache.iter_chunks()
    return cache


def _status(cfg: AppConfig, layer: int, total: int, started: float, extra: str = "") -> None:
    elapsed = time.perf_counter() - started
    remain = (elapsed / max(layer, 1)) * max(total - layer, 0) if layer else 0
    mem = snapshot(cfg.root)
    line = (
        f"[{layer}/{total}] elapsed {_fmt(elapsed)}  eta {_fmt(remain)}  "
        f"gpu_alloc={mem.gpu_allocated_gb}G  ram={mem.system_ram_used_gb}G  {extra}"
    )
    print(line, flush=True)
    log.info("%s", line)
    write_progress(
        cfg,
        stage="overnight",
        layer=layer,
        layers_total=total,
        elapsed_seconds=elapsed,
        estimated_remaining_seconds=remain,
    )


def ensure_calibration(cfg: AppConfig) -> None:
    try:
        tokens = load_token_array(cfg)
        print(f"calibration ready {tokens.shape}", flush=True)
    except FileNotFoundError as exc:
        raise SystemExit(
            "Local calibration file is missing. Overnight will not download datasets. "
            "Build it first with: python scripts/create_calibration.py --tier pilot"
        ) from exc


def reconstruct_layers(
    cfg: AppConfig,
    *,
    device: str,
    steps: int,
    lr: float,
    chunk_size: int,
    start_layer: int,
    end_layer: int,
) -> None:
    tokens = load_token_array(cfg)
    cache_root = cfg.resolve("cache", "activations", "full")
    ckpt_root = cfg.resolve("checkpoints", "reconstructed")
    cache_root.mkdir(parents=True, exist_ok=True)
    ckpt_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    total = end_layer - start_layer + 1

    for layer in range(start_layer, end_layer + 1):
        done_n = layer - start_layer
        _status(cfg, done_n, total, started, extra=f"next=layer {layer}")
        layer_dir = ckpt_root / f"layer_{layer:03d}"
        if _layer_done(ckpt_root, layer):
            print(f"  skip layer {layer} (already complete)", flush=True)
            continue

        if layer == 0:
            cache_layer0_from_tokens(cfg, tokens, device=device, chunk_size=chunk_size, out_dir=cache_root)
        else:
            prev = _open(cache_root, layer - 1, "out")
            if not prev.iter_chunks():
                raise FileNotFoundError(f"need layer {layer-1} teacher outputs first")
            # Point this layer's input index at the previous outputs (no 5 GB copy).
            alias = ActivationCache(cache_root, layer=layer, tag="in", meta={"side": "in", "alias_of": f"layer_{layer-1:03d}_out"})
            alias._index = list(prev.iter_chunks())
            alias._index_path.write_text(json.dumps(alias._index, indent=2), encoding="utf-8")
            cache_next_teacher_out(cfg, layer, alias, device=device, out_dir=cache_root)

        report, packed = run_one_layer(
            cfg,
            layer,
            device=device,
            steps=steps,
            lr=lr,
            cache_root=cache_root,
            train_sequences=min(64, int(tokens.shape[0])),
            eval_sequences=min(16, int(tokens.shape[0])),
        )
        report["complete"] = True
        layer_dir.mkdir(parents=True, exist_ok=True)
        (layer_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        save_packed(layer_dir, packed, name_prefix=f"model.language_model.layers.{layer}.")
        del packed
        if device.startswith("cuda"):
            import torch

            torch.cuda.empty_cache()

        cleanup_after_layer(cache_root, layer)

        save_stage(
            cfg,
            StageState(stage="overnight", layer=layer + 1, layers_total=end_layer + 1, status="running", config_hash=cfg.hash()),
        )
        print(
            f"  layer {layer} gate={report.get('gate_pass')} "
            f"naive_cos={report['naive']['cosine']:.4f} recon_cos={report['reconstructed']['cosine']:.4f}",
            flush=True,
        )

    save_stage(
        cfg,
        StageState(stage="overnight", layer=end_layer + 1, layers_total=end_layer + 1, status="complete", config_hash=cfg.hash()),
    )
    _status(cfg, total, total, started, extra="reconstruct done")


def assemble_gguf(cfg: AppConfig) -> Path:
    from q38ternary.gguf.assemble import assemble_local_gguf

    print("writing local GGUF from checkpoints + official shards (no HF tree)…", flush=True)
    outfile = assemble_local_gguf(cfg)
    print(f"GGUF written: {outfile}  ({outfile.stat().st_size / 1024**3:.2f} GB)", flush=True)
    return outfile


def smoke_gguf(cfg: AppConfig, gguf_path: Path, *, compile_if_missing: bool) -> None:
    from q38ternary.build_runtime import resolve_cli

    try:
        cli = resolve_cli(cfg, compile_if_missing=compile_if_missing)
    except Exception as exc:
        print(f"GGUF is on disk but llama-cli is not available: {exc}", flush=True)
        print(f"Load it later with a Prism-capable llama.cpp: llama-cli -m {gguf_path} -ngl 99", flush=True)
        return
    cmd = [
        str(cli),
        "-m",
        str(gguf_path),
        "-p",
        "Say hello in one short sentence.",
        "-n",
        "32",
        "-ngl",
        "99",
        "--temp",
        "0",
    ]
    print("smoke test:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=False)


def run(
    cfg: AppConfig,
    *,
    device: str,
    steps: int,
    lr: float,
    chunk_size: int,
    start_layer: int,
    end_layer: int,
    force: bool,
    skip_gguf: bool,
    compile_if_missing: bool,
) -> None:
    import os

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    used = 0.0
    mem = snapshot(cfg.root)
    if mem.gpu_total_gb and mem.gpu_free_gb is not None:
        used = float(mem.gpu_total_gb) - float(mem.gpu_free_gb)
    if used > 6.0 and not force:
        raise SystemExit(
            f"GPU already has {used:.1f} GB in use. Stop the other process "
            "or pass --force if you really want to share the card."
        )
    ensure_calibration(cfg)
    reconstruct_layers(
        cfg,
        device=device,
        steps=steps,
        lr=lr,
        chunk_size=chunk_size,
        start_layer=start_layer,
        end_layer=end_layer,
    )
    if skip_gguf:
        print("skipping GGUF assembly (--skip-gguf)", flush=True)
        return
    gguf_path = assemble_gguf(cfg)
    cleanup_run(cfg, keep_checkpoints=True)
    smoke_gguf(cfg, gguf_path, compile_if_missing=compile_if_missing)
    print("overnight complete.", flush=True)
