# qwen38-ternary

Activation-aware ternary / mixed-bit compression of official
[`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) so the language
model can run entirely on a 16 GB NVIDIA RTX 2000-class GPU through
llama.cpp-compatible CUDA inference.

Target path:

`Qwen3.8-27B BF16 (~54 GB) → activation-aware ternary / mixed-bit → ~7–9 GB`

## What this is — and what it is not

PrismML has publicly released ternary models, GGUFs, runtime kernels, a
llama.cpp fork, packing formats, and benchmarks. This repository is a
**clean research implementation inspired by that public representation and
runtime**. It is **not** a reproduction of PrismML’s proprietary
quality-recovery / training recipe, and it does not claim to be.

Ideal information content of `{-1,0,+1}` plus a group-128 FP16 scale is about
**1.71 bits/weight** (~5.9 GB). Deployed GGUF packing uses 2-bit slots
(~2.125 bits/weight, ~7.2 GB). Those numbers are not interchangeable. Reports
in this repo always list **deployed** file size, resident weight memory, and
runtime VRAM separately.

## Hardware target

Designed for a Windows 11 workstation: 24-core CPU, 64 GB RAM, ~16 GB CUDA
GPU, NVMe. The pipeline measures the actual machine at startup and refuses
to start a large download if free disk is below 180 GB.

Do not use Docker. WSL2 is optional and never required.

There is no admin interface and no authentication. This is a local CLI
research pipeline.

## Quick start

```powershell
.\scripts\bootstrap.ps1
python scripts\run_pipeline.py --mode detect
```

Full gated run (stops on a failed quality gate):

```powershell
.\scripts\run-all.ps1
```

Other entry points, once later slices are in place:

```powershell
python scripts\run_pipeline.py --mode pilot
python scripts\run_pipeline.py --mode reconstruct --target hybrid --budget-gb 8.5
python scripts\run_pipeline.py --mode reconstruct --target ternary
```

## Reproducible path

`bootstrap → quantize → pack → benchmark`

`--mode auto` walks the 25 stages in `Project_Plan.md` and **stops** when a
gate fails (streaming teacher mismatch, reconstruction that does not beat
naive ternary, GGUF round-trip failure, gibberish generation). It will not
burn days on a broken stage.

## Layout

Logic lives once under `src/q38ternary/`. Files in `scripts/` are thin CLIs.
Configuration is the six YAML files in `config/`. Large weights, caches, and
GGUFs are gitignored.

## License

Apache-2.0 for this repository. Third-party checkouts keep their own licenses;
pin exact commits in `third_party/versions.json` and preserve attribution.
