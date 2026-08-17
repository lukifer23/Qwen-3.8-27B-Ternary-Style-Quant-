# Qwen 3.8 27B Ternary-Style Quantization

Research code for activation-aware ternary and mixed-bit compression of the
official [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) language
model.

The deployment target is a text-only model that can run on a 16 GB NVIDIA
GPU through a llama.cpp-compatible CUDA runtime. The working size target is
approximately 7-9 GB, measured as deployed model size rather than an
information-theoretic bit count.

## Project status

This is an active research prototype. The public repository currently focuses
on conversion infrastructure and reproducible experiments:

- Qwen3.8 architecture discovery and tensor inventory
- hardware and disk-safety checks
- shard-aware safetensors loading
- BF16 streaming of one language layer at a time
- ternary initializers, activation-weighted scaling, and STE primitives
- Prism-style Q2 packing and round-trip tests
- calibration and holdout dataset configuration

The full reconstruction optimizer, production GGUF writer, CUDA runtime build,
quality evaluation, and final model selection are planned next. A model file
is not a successful result until size, memory, runtime, and quality are
measured together.

## Scope and research boundary

This is a clean implementation inspired by public ternary representations,
packing formats, and runtime code. It is not a reproduction of PrismML's
proprietary quality-recovery or training recipe, and it does not claim access
to that recipe.

The initial target is text inference. Vision weights, multimodal processing,
and MTP components are inventoried but excluded from the first deployment
path.

## Public repository policy

Large or machine-local files stay out of Git. In particular, do not commit:

- BF16 or quantized model weights, GGUFs, or calibration arrays
- Hugging Face caches, checkpoints, generated logs, or progress files
- local virtual environments or machine-specific hardware snapshots
- third-party source checkouts; record their commits in `third_party/versions.json`

Download the teacher model into `models/source/` on the local machine when a
workflow requires it. The model files are intentionally covered by
`.gitignore`.

## Quick start

On the supported Windows workstation:

```powershell
.\scripts\bootstrap.ps1
python scripts/run_pipeline.py --mode detect
```

Configuration, tokenizer, and inventory preparation can be run with:

```powershell
python scripts/download_sources.py --skip-clone
```

The full gated workflow is the eventual path:

```powershell
.\scripts\run-all.ps1
```

It is intentionally staged and stops at a failed quality or safety gate. Do
not start a full weight download or reconstruction until the local disk,
system memory, CUDA, and active-GPU checks pass.

## Layout

| Path | Purpose |
|---|---|
| `src/q38ternary/` | Reusable conversion, quantization, packing, and runtime code |
| `scripts/` | Thin command-line entry points |
| `config/` | Project, hardware, calibration, quantization, reconstruction, and evaluation settings |
| `tests/` | CPU-safe correctness and packing tests |
| `artifacts/` | Small, shareable architecture and inventory summaries |
| `models/`, `cache/`, `checkpoints/`, `data/` | Local-only generated data and model assets |
| `third_party/` | Local checkouts of pinned external projects |
| `Project_Plan.md` | Detailed research plan and acceptance criteria |

## License

This repository is Apache-2.0. Third-party checkouts retain their own
licenses and attribution requirements.
