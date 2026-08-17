# PROJECT GOAL: Build a Prism-Style Activation-Aware Ternary Qwen3.8-27B for a 16 GB RTX 2000 Workstation

## Mission

Build an end-to-end, reproducible local research pipeline that takes the official:

https://huggingface.co/Qwen/Qwen3.8-27B

and compresses the **language model** into an aggressively low-bit ternary representation inspired by PrismML's Ternary-Bonsai-27B work.

Primary target:

**Qwen3.8-27B BF16 (~54 GB) -> activation-aware ternary/mixed-bit model -> ~7-9 GB -> runnable entirely on a 16 GB NVIDIA RTX 2000 GPU through llama.cpp-compatible CUDA inference.**

This is not intended to be a naive "round everything to 2 bits" quantization.

The objective is to implement:

1. Hardware/environment discovery
2. Reproducible baseline inference
3. Calibration corpus generation
4. Layer/tensor sensitivity analysis
5. Ternary initialization using {-1, 0, +1}
6. Groupwise scaling, preferably g128
7. Activation-aware blockwise reconstruction
8. Progressive quantization through all 64 transformer blocks
9. Mixed-bit escape hatches for demonstrably sensitive tensors
10. GGUF/Q2-style packing
11. CUDA inference
12. Automated quality evaluation
13. Automatic identification/recovery of the worst layers
14. Final benchmarking against BF16/conventional GGUF baselines
15. A completely reproducible repository with scripts, logs, manifests, checkpoints, and documentation

The system should require as little user intervention as technically possible.

Do not stop merely because one preferred library lacks support. Inspect the architecture and implement missing glue code where feasible.

Do not use Docker.

---

# IMPORTANT RESEARCH PRINCIPLE

PrismML has publicly released:

- ternary models
- GGUFs
- runtime kernels
- llama.cpp forks
- model documentation
- packing formats
- benchmarking information

but do NOT assume that their complete internal quality-recovery/training recipe is available.

This project is therefore a **clean research implementation inspired by the public Prism representation/runtime**, not a claim that we possess PrismML's proprietary training pipeline.

Any README produced by this project must state that clearly.

---

# HARDWARE TARGET

Expected host hardware:

- OS: Windows 11
- CPU: Intel Core Ultra 9 285K
- CPU cores: 24
- RAM: 64 GB
- GPU: NVIDIA RTX 2000-class workstation GPU
- VRAM target: approximately 16 GB
- Storage: NVMe SSD
- CUDA-capable
- No Docker

DO NOT blindly assume these values.

At startup automatically collect:

```powershell
nvidia-smi
Get-CimInstance Win32_Processor
Get-CimInstance Win32_PhysicalMemory
Get-Volume
```

Also record:

```bash
python --version
git --version
cmake --version
nvcc --version
```

Write the result to:

```text
artifacts/hardware.json
artifacts/environment.txt
```

If WSL2 is already installed and functional, it may be used where advantageous.

If WSL2 is not installed, DO NOT make it a hard dependency. Prefer native Windows Python/CUDA rather than requiring a reboot or user intervention.

---

# RESOURCE SAFETY

The machine has only 64 GB system RAM and approximately 16 GB VRAM.

Never attempt to materialize the entire 27B BF16 model plus multiple copies, gradients, and optimizer states simultaneously.

Design for:

- memory mapping
- safetensors shard-level loading
- one-block-at-a-time reconstruction
- CPU offload
- GPU block streaming
- explicit object deletion
- torch.cuda.empty_cache()
- bounded activation caches
- resumable stages

Set conservative defaults.

Suggested usable GPU budget:

```text
13.0-14.0 GB
```

rather than intentionally allocating all 16 GB.

Suggested RAM safety budget:

```text
48-54 GB
```

before spilling to the pagefile.

Check disk space before any large download.

Target free working space:

```text
>= 180 GB
```

Preferred:

```text
>= 250 GB
```

If available storage is insufficient, fail with a clear explanation before downloading huge files.

---

# SOURCE MODELS AND REPOSITORIES

## Official teacher model

Qwen3.8-27B:

https://huggingface.co/Qwen/Qwen3.8-27B

Use this as the canonical source model.

Do not substitute a community finetune.

Before downloading weights:

1. Fetch config.json.
2. Verify model identifier.
3. Verify architecture.
4. Record revision/commit hash.
5. Record file hashes where practical.

Expected broad architecture:

- approximately 27B language parameters
- hidden dimension 5120
- 64 language layers
- FFN dimension 17408
- hybrid architecture
- repeated 3 Gated DeltaNet blocks + 1 Gated Attention block
- 16 repetitions of that four-layer pattern
- 248,320 padded vocabulary
- native multimodal model
- MTP components

The project will initially target **text inference only**.

Do NOT load the vision tower for Phase 1.

Do NOT require MTP/speculative decoding for Phase 1.

---

# PRISM REFERENCE IMPLEMENTATION

Clone/read these repositories and use them as technical references:

## Demo / documentation

https://github.com/PrismML-Eng/Bonsai-demo

## Prism llama.cpp fork

https://github.com/PrismML-Eng/llama.cpp

## Existing reference ternary 27B

https://huggingface.co/prism-ml/Ternary-Bonsai-27B-gguf

## Existing reference binary 27B

https://huggingface.co/prism-ml/Bonsai-27B-gguf

Prism's relevant ternary representation is conceptually:

```text
weight alphabet: {-1, 0, +1}
group size: 128
one FP16 scale per group
ideal information content: ~1.71 bits/weight
deployed packing: 2-bit slots + group scale
effective deployed storage: ~2.125 bits/weight
27B deployed language model: ~7.2 GB
```

Use Prism's implementation to understand:

- tensor packing
- GGUF types
- Q2_0/Q2_0_g128 implementation
- CUDA kernels
- supported tensor types
- metadata requirements
- model architecture handling
- conversion scripts
- quantization source code
- llama.cpp loading path

Do not blindly copy code in a license-incompatible way.

Preserve required attribution/licenses.

---

# UPSTREAM LLAMA.CPP

Also clone:

https://github.com/ggml-org/llama.cpp

We want both:

```text
third_party/llama.cpp
third_party/prism-llama.cpp
```

Pin exact commits in:

```text
third_party/versions.json
```

Use upstream llama.cpp for conventional baselines.

Use Prism's fork or the appropriate current upstream implementation for g128 ternary inference.

At runtime, automatically determine whether mainline llama.cpp has all required support.

Do not assume status from old documentation.

Inspect the source tree/current build.

---

# OPTIONAL BASELINE TOOL: AUTOROUND

Reference:

https://github.com/intel/auto-round

AutoRound currently supports useful comparison formats including:

- W2A16
- W3A16
- W4A16
- mixed-bit schemes
- group size 128
- multiple GGUF formats

It may require far more memory than available for full 27B optimization.

Therefore:

- use it only where it fits
- use it for small-layer experiments or comparison
- do not make the project dependent on successfully loading the entire 27B model into AutoRound

Our custom streaming/blockwise method is the primary strategy.

---

# TARGET OUTPUTS

Produce multiple checkpoints rather than one monolithic final model.

## Model A — Conventional baseline

Acquire/build at least:

```text
Q4 or IQ4 baseline
3-bit/IQ3 baseline
2.x-bit/IQ2 baseline
```

Prefer current well-maintained Qwen3.8-27B GGUF builds or generate locally if practical.

Record:

- exact quantization
- model size
- VRAM
- RAM
- prompt processing t/s
- decode t/s
- perplexity
- benchmark results

These become the control group.

---

# Model B — Hybrid v0.1

Target:

```text
Transformer block large matrices: ternary
Embeddings: Q4 or equivalent
LM head: Q4 or equivalent
Norms/small state tensors: FP16/FP32
Vision: omitted
MTP: omitted initially
```

Expected target footprint:

```text
~8-9 GB
```

This is the first serious milestone.

Do not sacrifice model quality simply to hit exactly 7.2 GB on the first attempt.

---

# Model C — Full-language ternary

Target:

```text
Embeddings: ternary
Attention matrices: ternary
DeltaNet matrices: ternary
MLP matrices: ternary
LM head: ternary
Norm/state tensors: appropriate higher precision where structurally required
```

Target deployed footprint:

```text
~7.0-7.5 GB
```

Preferred:

```text
~7.2 GB
```

---

# Model D — Adaptive mixed ternary

If some tensors/layers are highly sensitive, allow:

```text
ternary
3-bit
4-bit
FP16 for tiny tensors
```

but select precision empirically.

Primary goal:

**maximize quality per byte**, not ideological purity.

A 7.8 GB model scoring materially better than a 7.2 GB model is acceptable and may be preferable.

---

# STRETCH GOAL — Binary

Only attempt this after ternary works.

Reference:

```text
{-1, +1}
```

Expected deployed model:

```text
~3.9-4.5 GB
```

Do not spend substantial time on binary until ternary results justify it.

---

# PROJECT DIRECTORY STRUCTURE

Create:

```text
qwen38-ternary/
│
├── README.md
├── LICENSE
├── pyproject.toml
├── uv.lock
├── .gitignore
├── config/
│   ├── project.yaml
│   ├── hardware.yaml
│   ├── calibration.yaml
│   ├── quantization.yaml
│   ├── reconstruction.yaml
│   └── evaluation.yaml
│
├── scripts/
│   ├── bootstrap.ps1
│   ├── bootstrap.sh
│   ├── detect_hardware.py
│   ├── download_sources.py
│   ├── build_llamacpp.py
│   ├── baseline_models.py
│   ├── create_calibration.py
│   ├── teacher_cache.py
│   ├── sensitivity_scan.py
│   ├── ternary_init.py
│   ├── reconstruct_block.py
│   ├── reconstruct_all.py
│   ├── targeted_recovery.py
│   ├── pack_gguf.py
│   ├── validate_gguf.py
│   ├── run_benchmarks.py
│   ├── compare_models.py
│   └── run_pipeline.py
│
├── src/
│   └── q38ternary/
│       ├── __init__.py
│       ├── config.py
│       ├── hardware.py
│       ├── hf.py
│       ├── safetensors_io.py
│       ├── architecture.py
│       ├── streaming_model.py
│       ├── activation_cache.py
│       ├── calibration.py
│       ├── quant/
│       │   ├── ternary.py
│       │   ├── grouping.py
│       │   ├── scaling.py
│       │   ├── thresholds.py
│       │   ├── fake_quant.py
│       │   ├── ste.py
│       │   └── mixed_precision.py
│       ├── reconstruction/
│       │   ├── losses.py
│       │   ├── optimizer.py
│       │   ├── block_runner.py
│       │   ├── progressive.py
│       │   └── recovery.py
│       ├── gguf/
│       │   ├── writer.py
│       │   ├── prism_format.py
│       │   └── validation.py
│       ├── eval/
│       │   ├── perplexity.py
│       │   ├── logits.py
│       │   ├── tasks.py
│       │   ├── throughput.py
│       │   └── reporting.py
│       └── utils/
│           ├── logging.py
│           ├── memory.py
│           ├── checkpoints.py
│           └── manifest.py
│
├── tests/
│   ├── test_ternary.py
│   ├── test_grouping.py
│   ├── test_packing.py
│   ├── test_roundtrip.py
│   ├── test_layers.py
│   └── test_gguf.py
│
├── data/
│   ├── calibration/
│   └── evaluation/
│
├── cache/
│   ├── hf/
│   ├── activations/
│   └── teacher/
│
├── checkpoints/
│   ├── naive/
│   ├── reconstructed/
│   └── mixed/
│
├── models/
│   ├── source/
│   ├── baseline/
│   └── output/
│
├── artifacts/
│   ├── logs/
│   ├── reports/
│   └── plots/
│
└── third_party/
    ├── llama.cpp/
    └── prism-llama.cpp/
```

Large files must be ignored by Git.

---

# PYTHON ENVIRONMENT

Prefer Python:

```text
3.11 or 3.12
```

unless current Qwen3.8/Transformers requirements indicate otherwise.

Prefer `uv` for environment management if available.

Core dependencies likely include:

```text
torch
transformers
accelerate
safetensors
huggingface_hub
datasets
numpy
scipy
pandas
pyyaml
tqdm
psutil
rich
sentencepiece
einops
gguf
lm-eval
pytest
```

Install current versions compatible with Qwen3.8.

Do not pin obsolete versions simply because old examples use them.

After resolving working versions, lock them.

Record:

```text
torch version
transformers version
CUDA runtime
GPU name
driver
llama.cpp commit
Prism fork commit
Qwen model revision
```

---

# NVIDIA / PYTORCH VALIDATION

Run:

```python
import torch

print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_properties(0).total_memory)
```

Require:

```text
torch.cuda.is_available() == True
```

before GPU reconstruction.

If CUDA PyTorch is unavailable:

- install the appropriate official PyTorch CUDA build
- re-test
- do not silently fall back to CPU reconstruction for multi-day workloads

---

# DOWNLOAD STRATEGY

Use huggingface_hub snapshot_download.

Canonical repo:

```python
repo_id = "Qwen/Qwen3.8-27B"
```

Download into:

```text
models/source/Qwen3.8-27B/
```

Enable resume.

Use Xet/HF transfer mechanisms if current Hugging Face tooling recommends them.

Do not duplicate weights unnecessarily.

After download:

1. inspect safetensors index
2. create tensor -> shard mapping
3. inspect parameter names
4. calculate actual parameter counts by category
5. calculate theoretical footprint per quantization
6. save analysis

Produce:

```text
artifacts/model_inventory.json
artifacts/model_inventory.md
```

Inventory categories:

```text
embedding
lm_head
full_attention
gated_deltanet
mlp
norm
mtp
vision
other
```

---

# CRITICAL: ARCHITECTURE DISCOVERY

Do not hard-code old Qwen3.5/Qwen3.6 assumptions where avoidable.

Inspect:

```text
config.json
generation_config.json
processor config
modeling classes
safetensors keys
```

Generate a model map.

For each of 64 language blocks record:

```text
layer index
layer type
parameter count
weight matrices
shape
dtype
full attention vs DeltaNet
FFN weights
state tensors
norm tensors
```

Output:

```text
artifacts/architecture.json
artifacts/architecture.md
```

---

# STREAMING WEIGHT LOADER

Implement a safetensors shard-aware loader.

Requirements:

```python
load_tensor(name)
load_layer(layer_index)
release_layer(layer_index)
```

Avoid:

```python
AutoModel.from_pretrained(...).cuda()
```

for the full 27B BF16 model.

The pipeline must be capable of:

1. locating only shards needed for the current layer
2. loading those weights into CPU RAM
3. transferring the current layer to GPU
4. running it
5. unloading it
6. reclaiming GPU memory

Use mmap/safetensors capabilities wherever possible.

---

# BF16 TEACHER EXECUTION

We need reference activations from the original model.

Implement a streaming teacher executor capable of forwarding calibration samples through the language model while loading layers sequentially.

The design should account for Qwen3.8 hybrid attention.

Do not incorrectly treat DeltaNet layers as ordinary self-attention.

Use Qwen/Transformers model code as the reference for:

```text
position handling
attention masks
rotary embeddings
DeltaNet recurrent state
cache semantics
normalization
residual paths
thinking/chat template
```

Teacher outputs must be numerically validated against standard Transformers inference on a tiny test case where feasible.

For example:

- use a very short sequence
- CPU/offload standard model if necessary
- compare streaming executor logits
- require sufficiently small error before proceeding

---

# PHASE 0 — BASELINES

Before modifying weights, establish controls.

Test the best currently available conventional Qwen3.8-27B GGUFs around:

```text
~15 GB
~13 GB
~11 GB
~9-10 GB
```

Use maintained community quantizations or create equivalents locally.

Benchmark:

```text
model file size
peak VRAM
peak system RAM
prompt processing speed
generation speed
perplexity
output agreement
reasoning tasks
coding tasks
instruction following
structured output
tool calling
```

Save:

```text
artifacts/reports/baseline.json
artifacts/reports/baseline.md
```

This baseline is mandatory.

The ternary model must be judged relative to both:

```text
BF16 teacher
ordinary aggressive quantization
```

---

# CALIBRATION CORPUS

Build a text-only calibration set.

Do NOT use a single homogeneous corpus.

We want representative activations covering:

```text
general prose
encyclopedic text
multi-turn chat
reasoning
mathematics
coding
debugging
JSON
structured data
tool definitions
tool calls
function arguments
instruction following
longer documents
```

Start with:

```text
Pilot:
512 sequences
1024 tokens each

Standard:
2048 sequences
2048 tokens each

Deep:
4096 sequences
2048 tokens each
```

The pipeline should default to Pilot for initial experiments.

Only automatically escalate to Standard after the pilot passes quality gates.

Calibration data should be deterministic:

```text
seed = 42
```

Record exact dataset names, revisions and row IDs.

Avoid copyrighted/private data requiring authentication.

Public Hugging Face datasets are acceptable.

The exact corpus selection should be written to:

```text
data/calibration/manifest.json
```

---

# EVALUATION HOLDOUT

Calibration and evaluation must not be identical.

Create a separate holdout.

Include:

```text
general LM perplexity
math questions
reasoning
code generation
code debugging
instruction following
JSON validity
tool-call formatting
long-answer coherence
```

Do not optimize directly on the benchmark test items.

---

# PHASE 1 — IMPLEMENT TERNARY QUANTIZATION

For each eligible weight matrix, operate on groups of:

```text
128
```

weights.

For a group:

```math
w = (w_1, ..., w_128)
```

quantized representation:

```math
q_i \in \{-1,0,+1\}
```

and scale:

```math
\hat{w_i} = s q_i
```

Implement several initialization strategies and benchmark them.

---

# INITIALIZER A — ABSOLUTE THRESHOLD

Given threshold τ:

```math
q_i =
-1 if w_i < -τ
 0 if |w_i| <= τ
+1 if w_i > τ
```

For fixed q, optimal least-squares scale:

```math
s = (w^T q) / (q^T q)
```

Guard q^Tq = 0.

---

# INITIALIZER B — SEARCH THRESHOLD

Search candidate τ values per group.

Possible candidates:

```text
percentiles
mean(abs(w)) * scalar
std(w) * scalar
sorted magnitude breakpoints
```

Select τ and s minimizing:

```math
||W - sQ||²
```

---

# INITIALIZER C — ACTIVATION-WEIGHTED

Given representative input activation matrix X, minimize:

```math
||XW - XW_q||²
```

rather than raw weight MSE.

This is much more important.

Support per-channel/diagonal approximations where full multiplication is too expensive.

---

# INITIALIZER D — HESSIAN/SECOND-ORDER APPROXIMATION

Where computationally feasible estimate:

```math
H ≈ X^T X
```

and choose ternary assignment minimizing:

```math
(w - w_q)^T H (w - w_q)
```

A full Hessian is unnecessary.

Implement diagonal/block approximations.

This should be optional because of memory/compute cost.

---

# FAKE QUANTIZATION MODULE

Implement a differentiable fake ternary layer.

Store a latent FP16/BF16/FP32 weight proxy:

```text
W_latent
```

but forward with:

```text
W_q = ternary(W_latent)
```

Use a Straight-Through Estimator or an empirically superior alternative for backpropagation.

Do not export the latent weights.

Only export final:

```text
ternary codes
scales
metadata
```

---

# PHASE 2 — ONE-LAYER PILOT

DO NOT immediately launch all 64 layers.

First evaluate representative layers.

Suggested initial layer indices:

```text
0
3
15
31
47
63
```

Ensure sample includes both:

```text
DeltaNet layers
full-attention layers
```

For each selected layer:

1. Capture BF16 input activations.
2. Capture BF16 output activations.
3. Create naive ternary layer.
4. Measure degradation.
5. Reconstruct ternary layer.
6. Measure recovered degradation.
7. Report.

Metrics:

```text
weight MSE
output activation MSE
relative MSE
cosine similarity
max error
mean absolute error
downstream logit KL if available
```

Produce:

```text
artifacts/reports/layer_pilot.md
artifacts/reports/layer_pilot.json
```

---

# QUALITY GATE FOR PILOT

Do not launch the multi-day job unless reconstruction clearly improves over naive ternary.

At minimum, reconstructed layer must improve:

```text
activation relative MSE
and/or
cosine similarity
```

by a meaningful amount.

If reconstruction fails:

1. test threshold strategies
2. test scale initialization
3. adjust optimizer
4. adjust loss
5. increase calibration data
6. inspect architecture correctness
7. inspect activation caching correctness
8. test group size 64 versus 128 experimentally
9. test retaining sensitive projections at 3-bit
10. do not brute-force all 64 layers

---

# BLOCKWISE RECONSTRUCTION LOSS

For a block input X:

```text
Y_teacher = BF16_block(X)
Y_quant   = quant_block(X)
```

Base loss:

```math
L_hidden = MSE(Y_quant, Y_teacher)
```

Cosine component:

```math
L_cos = 1 - cosine(Y_quant, Y_teacher)
```

Optionally add intermediate losses for:

```text
attention output
DeltaNet output
FFN output
residual output
```

Where practical include downstream predictive loss.

Example:

```math
L =
1.0 * L_hidden
+ 0.1 * L_cos
+ 0.1 * L_intermediate
+ λ * L_logits
```

Tune empirically.

Do not blindly trust these exact coefficients.

---

# TEACHER-STUDENT ACTIVATIONS

Implement two modes.

## Mode A — Teacher input reconstruction

Quantized block receives the original BF16 block input.

Useful for stable local fitting.

## Mode B — Progressive student reconstruction

Quantized block receives the activations produced by already-quantized preceding blocks.

This captures accumulated quantization drift.

Full pipeline should use:

1. teacher-input fitting initially
2. progressive-student fitting as refinement

---

# OPTIMIZATION PARAMETERS

Initial pilot defaults:

```text
batch size: dynamically chosen
sequence length: 1024
steps per layer: 100-300
optimizer: AdamW or lower-memory alternative
learning rate: 1e-4 to 5e-3 search
weight decay: near zero unless shown useful
gradient clipping: enabled
mixed precision: enabled where stable
```

Do not allocate huge optimizer states for the entire model.

Only current block/tensor is trainable.

If block-level optimizer states exceed VRAM:

```text
optimize one matrix at a time
or
use 8-bit optimizer state if supported
or
keep scale/threshold optimization only
```

---

# WHAT MAY BE OPTIMIZED

Try, in increasing order of cost:

## Level 1

```text
scales only
```

## Level 2

```text
scales + thresholds
```

## Level 3

```text
scales + thresholds + discrete assignments
```

## Level 4

```text
latent weights with STE
```

## Level 5

```text
cross-layer/block reconstruction
```

Use the cheapest method that reaches quality targets.

---

# LAYER SENSITIVITY SCAN

Before final precision assignment, quantify every major tensor.

For each tensor compute:

```text
naive ternary weight error
activation-weighted error
block-output degradation
candidate 2-bit/ternary degradation
candidate 3-bit degradation
candidate 4-bit degradation
parameter count
estimated storage cost
```

Create a score approximately like:

```math
sensitivity = quality_loss / bytes_saved
```

Do not use that exact expression if a better metric emerges.

Produce:

```text
artifacts/reports/sensitivity.csv
artifacts/reports/sensitivity.md
```

Also generate plots.

---

# MIXED PRECISION ALLOCATION

Given a storage budget, assign each tensor to:

```text
ternary
3-bit
4-bit
FP16
```

using sensitivity measurements.

Budgets to evaluate:

```text
7.25 GB
7.5 GB
8.0 GB
8.5 GB
9.0 GB
```

Determine Pareto frontier:

```text
quality vs model size
```

Do not optimize only one arbitrary footprint.

---

# IMPORTANT SPECIAL TENSOR POLICY

Treat these separately:

```text
token embeddings
LM head
normalization parameters
DeltaNet state/control tensors
small biases
MTP
vision
```

Small tensors should not be quantized merely for ideological purity if doing so saves negligible space.

Keep tiny but sensitive tensors in FP16/FP32 if appropriate.

---

# FIRST MODEL POLICY

For `v0.1`:

```text
major transformer matrices -> ternary g128
embedding -> Q4
LM head -> Q4
norms -> FP16
small state tensors -> FP16 where appropriate
vision -> excluded
MTP -> excluded
```

Expected target:

```text
~8-9 GB
```

Calculate the exact predicted size from real tensor inventory before quantization.

---

# SECOND MODEL POLICY

For `v0.2`:

Quantize embedding and LM head to ternary if evaluation supports it.

Target:

```text
~7.0-7.5 GB
```

---

# ACTIVATION CACHE DESIGN

Activation caches can become huge.

Implement bounded caching.

Preferred:

```text
FP16 storage
memory-mapped files
chunked tensors
Zarr or safetensors shards if useful
```

Each cache file must encode:

```text
layer
sample ids
sequence length
dtype
shape
teacher revision
calibration revision
```

Do not create one gigantic fragile file.

---

# RESUMABILITY

Every stage must be restartable.

If reconstruction is interrupted at layer 37, the next execution should continue from layer 37/38 rather than starting over.

Checkpoint:

```text
quantizer state
layer index
tensor states
optimizer where worthwhile
metrics
RNG state
configuration hash
```

Every output must have a manifest.

---

# MEMORY MONITORING

During reconstruction poll:

```text
GPU allocated
GPU reserved
GPU free
system RAM
pagefile/swap
disk usage
temperature if available
```

If GPU usage exceeds safe threshold:

```text
reduce batch size automatically
```

If an OOM occurs:

1. catch it
2. clear cache
3. reduce batch size
4. retry
5. record retry

Do not crash the entire multi-day run on a recoverable OOM.

---

# THERMAL / LONG-RUN STABILITY

This is a workstation GPU.

Record temperatures using nvidia-smi.

Do not change GPU power settings without explicit user permission.

If sustained GPU temperature reaches an unsafe driver-defined operating region:

```text
pause/reduce workload
record event
```

Do not overclock.

---

# GGUF PACKING

Primary deployment format:

```text
GGUF
```

Desired ternary representation:

```text
Q2-style g128
{-1,0,+1}
FP16 scale per 128 weights
```

First inspect Prism's llama.cpp implementation.

Determine:

```text
GGML type
block layout
scale packing
2-bit code mapping
endianness
tensor alignment
GGUF metadata
architecture identifiers
```

Write automated roundtrip tests:

```text
FP values
-> ternary codes/scales
-> packed representation
-> unpack
-> reconstructed tensor
```

Require exact code/scales roundtrip where applicable.

---

# DO NOT ASSUME LLAMA-QUANTIZE CAN DIRECTLY CREATE OUR FILE

Inspect the current Prism and upstream quantizers.

If an existing supported conversion path accepts custom ternary tensors:

use it.

If not:

implement a purpose-built exporter or patch the quantizer.

Document exactly what was changed.

---

# GGUF VALIDATION

Before full-model packing, create a tiny synthetic GGUF using the target block representation.

Load it with the relevant ggml/llama.cpp code.

Then pack one actual Qwen tensor.

Then one layer.

Only after those pass, pack the full model.

---

# BUILD LLAMA.CPP CUDA

Automate a CUDA build.

Typical conceptual build:

```bash
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j
```

Adjust to actual host/compiler environment.

On Windows locate resulting executables.

Test:

```text
llama-cli
llama-server
llama-bench
llama-perplexity if available
```

Do not require Visual Studio UI interaction if a usable toolchain already exists.

---

# INFERENCE CONFIGURATION

Initial validation:

```text
context: 4096
GPU layers: maximum/full
vision: disabled
MTP: disabled
batch sizes: conservative
```

Once stable:

```text
8K
16K
32K
```

Do not waste time testing 262K context during early quantization research.

---

# PERPLEXITY / LOGIT EVALUATION

Track more than downstream benchmark scores.

Measure:

## Teacher/student logit KL

```math
D_KL(P_teacher || P_quant)
```

over a held-out token set.

## Top-1 agreement

Fraction of positions where:

```text
argmax teacher == argmax quant
```

## Top-k overlap

For example k=5 or k=10.

## Perplexity delta

Compare:

```text
BF16
Q4
3-bit
2-bit
naive ternary
reconstructed ternary
mixed ternary
```

These metrics are especially useful during development because they are cheaper than huge benchmark suites.

---

# FUNCTIONAL EVALUATION

Create a fixed internal suite.

At minimum:

## General reasoning

```text
50-100 deterministic questions
```

## Math

Use a modest GSM8K/MATH-style subset.

## Coding

Use HumanEval-style tasks or a current compatible coding evaluator.

## Instruction following

Include:

```text
strict formatting
constraints
multi-part directions
negative constraints
```

## JSON

Require exact parse validity.

## Tool calls

Supply tool schemas and check:

```text
tool name
arguments
JSON
required fields
no hallucinated parameters
```

## Long generation

Generate:

```text
2K-4K tokens
```

and detect:

```text
repetition
collapse
garbling
early EOS
looping
```

---

# SAMPLING CONSISTENCY

For deterministic comparison where possible:

```text
temperature = 0
```

For behavior evaluations use Qwen's recommended generation settings where appropriate.

Record all decoding parameters.

Do not compare models using different sampling settings.

---

# FULL 64-LAYER RECONSTRUCTION

Once pilot gates pass:

Run layers sequentially.

Conceptual pipeline:

```python
for layer_idx in range(64):
    teacher_inputs = obtain_teacher_inputs(layer_idx)

    teacher_outputs = run_teacher_layer(
        layer_idx,
        teacher_inputs,
    )

    qlayer = ternary_initialize(
        teacher_layer,
        group_size=128,
    )

    baseline_metrics = evaluate_layer(
        qlayer,
        teacher_inputs,
        teacher_outputs,
    )

    qlayer = reconstruct(
        qlayer,
        teacher_inputs,
        teacher_outputs,
    )

    final_metrics = evaluate_layer(
        qlayer,
        teacher_inputs,
        teacher_outputs,
    )

    save_quantized_layer(...)
    save_metrics(...)

    release_everything()
```

The real implementation must correctly handle Qwen3.8's layer classes/states.

---

# TARGETED RECOVERY

After the complete student exists:

Run end-to-end evaluation.

Rank layers/tensors by suspected contribution to degradation.

Candidates for recovery:

```text
lowest activation cosine
highest relative error
largest logit impact
most benchmark-sensitive
```

Automatically retry the worst:

```text
top 4
top 8
top 16
```

using:

```text
more calibration samples
more optimization steps
smaller learning rate
different threshold
higher precision
```

Only keep a recovery if evaluation improves.

---

# MIXED-BIT ESCALATION POLICY

If a layer remains bad after reasonable ternary reconstruction:

Test:

```text
3-bit
```

If still bad:

```text
4-bit
```

Do not repeatedly spend days forcing one layer to ternary when keeping it at 3-bit costs only tens/hundreds of MB.

Optimize globally.

---

# AUTOMATED SIZE ESTIMATOR

Implement:

```bash
python scripts/estimate_size.py
```

Given precision map, report:

```text
parameter count
effective bits/weight
scale overhead
metadata overhead
predicted GGUF size
predicted VRAM at multiple context sizes
```

Models to estimate:

```text
BF16
Q8
Q4
Q3
Q2
all ternary g128
hybrid v0.1
adaptive 7.5 GB
adaptive 8 GB
adaptive 9 GB
binary stretch
```

---

# EXPECTED SIZE TARGETS

Approximate only; compute exact values from the actual model.

```text
BF16:
~54 GB

ordinary Q4:
~15-18 GB

ordinary ~3-bit:
~12-14 GB

ordinary aggressive ~2-bit:
~9-11 GB

hybrid ternary:
~8-9 GB

full ternary deployed:
~7.2 GB

full ternary theoretical information content:
~5.9 GB

binary stretch:
~3.9-4.5 GB
```

Do not report theoretical ternary size as actual deployed GGUF size.

---

# SUCCESS CRITERIA

## Bronze

```text
size <= 9 GB
quality retention >= ~85%
stable generation
no catastrophic tool/JSON collapse
fully GPU resident
```

## Silver

```text
size <= 8.5 GB
quality retention >= ~90%
```

## Gold

```text
size roughly 7.0-7.5 GB
quality retention >= ~93%
```

## Aspirational

```text
~7.2 GB
~95% aggregate retention
```

Do not massage benchmark selection to achieve these numbers.

---

# QUALITY RETENTION DEFINITION

Do not define "95%" using one cherry-picked benchmark.

Build a composite containing:

```text
perplexity/logit preservation
general reasoning
math
coding
instruction following
tool use
structured output
long-generation stability
```

Report every component separately.

Then optionally report an aggregate normalized score.

---

# COMPARISON TABLE

Final report must include:

| Model | Size | BPW | VRAM 4K | VRAM 16K | Tok/s | PPL | Reasoning | Code | Tools | Overall |
|------|------|-----|---------|----------|-------|-----|-----------|------|-------|---------|
| BF16 | | | | | | | | | | |
| Q4 | | | | | | | | | | |
| 3-bit | | | | | | | | | | |
| aggressive 2-bit | | | | | | | | | | |
| naive ternary | | | | | | | | | | |
| reconstructed ternary | | | | | | | | | | |
| adaptive mixed | | | | | | | | | | |

---

# TIMING INSTRUMENTATION

Do not assume duration estimates.

Record actual wall-clock time per stage:

```text
download
conversion
teacher calibration
layer pilot
each layer reconstruction
GGUF pack
evaluation
targeted recovery
```

Calculate ETA based on completed layers.

Write:

```text
artifacts/progress.json
```

Fields:

```json
{
  "stage": "",
  "layer": 0,
  "layers_total": 64,
  "elapsed_seconds": 0,
  "estimated_remaining_seconds": 0,
  "gpu_memory_gb": 0,
  "system_memory_gb": 0
}
```

---

# EXPECTED ROUGH TIMELINE ON THIS MACHINE

Treat these as planning estimates only.

Possible range:

```text
bootstrap/download:
1-3 hours

baseline testing:
1-4 hours

pilot calibration:
1-4 hours

representative-layer pilot:
2-8 hours

full calibration:
4-16+ hours

first 64-layer reconstruction:
16-72+ hours

packing/validation:
1-4 hours

evaluation:
2-12 hours

targeted recovery:
8-48+ hours
```

A credible v0.1 may therefore take:

```text
~2-5 days
```

of unattended computation.

A deeply optimized version may take:

```text
~1 week or more
```

Do not promise these times. Measure them.

---

# RUN PIPELINE ENTRY POINT

Create one primary command.

Windows:

```powershell
.\scripts\bootstrap.ps1
python scripts/run_pipeline.py --mode auto
```

or ideally:

```powershell
.\scripts\run-all.ps1
```

The `auto` mode should:

1. detect environment
2. bootstrap dependencies
3. verify CUDA
4. check disk
5. clone dependencies
6. pin revisions
7. download model
8. inspect architecture
9. create baselines
10. build pilot calibration set
11. run layer pilot
12. evaluate pilot
13. choose reconstruction settings
14. run full calibration
15. sensitivity scan
16. construct hybrid precision map
17. reconstruct all layers
18. pack checkpoint
19. build/load GGUF
20. benchmark
21. target worst layers
22. repack
23. benchmark again
24. select best model
25. produce final report

---

# AUTOMATIC GATING

`--mode auto` MUST NOT blindly continue through a broken stage.

Examples:

If streaming teacher != reference teacher:

```text
STOP.
Debug architecture.
```

If naive ternary and reconstructed ternary are identical:

```text
STOP.
Debug optimizer/gradient path.
```

If reconstruction makes pilot layers worse:

```text
STOP.
Run parameter search.
```

If GGUF unpacked weights do not match pre-pack values:

```text
STOP.
Fix packer.
```

If generated output is gibberish:

```text
STOP.
Do not benchmark garbage for 12 hours.
```

---

# SEARCH/TUNING POLICY

For pilot layers, automatically sweep a modest grid:

```text
threshold initializer
learning rate
steps
loss weighting
group size if necessary
scale-only vs STE
```

Do not launch hundreds of expensive trials.

Use successive halving:

1. short cheap trials
2. eliminate bad configs
3. expand best 2-3
4. select winner

---

# DEBUG MODEL OPTION

Implement development support for running the same quantizer on a smaller compatible Qwen model if available.

This is for validating:

```text
packing
training
STE
GGUF loading
evaluation
```

before wasting days on 27B.

However:

The final target remains Qwen3.8-27B.

Do not quietly switch the project to a smaller model.

---

# CHECKPOINT FILE FORMAT

Before GGUF packaging, store reconstructed tensors in a transparent intermediate format.

Recommended:

```text
safetensors
+
JSON quantization metadata
```

For each tensor record:

```json
{
  "name": "...",
  "scheme": "ternary_g128",
  "shape": [],
  "group_size": 128,
  "scale_dtype": "float16",
  "codes_dtype": "packed2",
  "source_revision": "...",
  "reconstruction_loss": 0.0
}
```

Keep intermediate format independent of GGUF.

---

# TEST REQUIREMENTS

Tests must include:

## Quantization correctness

Known vector -> known ternary codes/scales.

## Scale optimum

Verify scale formula.

## Packing

Pack -> unpack -> equality.

## Shape edge cases

Matrices whose dimensions are not naturally divisible by group size.

## CUDA versus CPU

Fake quant outputs within tolerance.

## Layer reconstruction

Loss decreases on a tiny synthetic network.

## Qwen layer

One real Qwen3.8 block forwards successfully after fake quant.

## GGUF

Produced model loads.

---

# SOURCE CONTROL

Commit frequently.

Suggested commits:

```text
bootstrap environment
model inventory
streaming loader
teacher executor
calibration pipeline
ternary quantizer
activation reconstruction
pilot results
sensitivity analysis
mixed precision allocator
GGUF exporter
CUDA inference
evaluation suite
v0.1 model
targeted recovery
final report
```

Do not commit:

```text
model weights
activation caches
huge logs
HF cache
GGUF files
```

---

# LOGGING

Every command and experiment must be logged.

Use timestamped logs:

```text
artifacts/logs/YYYYMMDD-HHMM-stage.log
```

Also maintain:

```text
artifacts/experiment_registry.jsonl
```

Each experiment record should contain:

```text
git commit
config hash
model revision
calibration manifest
precision map
random seed
duration
metrics
output path
```

---

# REPORTS

Generate automatically:

```text
artifacts/reports/00_hardware.md
artifacts/reports/01_model_inventory.md
artifacts/reports/02_baselines.md
artifacts/reports/03_layer_pilot.md
artifacts/reports/04_sensitivity.md
artifacts/reports/05_full_reconstruction.md
artifacts/reports/06_targeted_recovery.md
artifacts/reports/07_final_comparison.md
```

And:

```text
FINAL_REPORT.md
```

---

# FINAL REPORT MUST ANSWER

1. How small did Qwen3.8-27B become?
2. What percentage reduction from BF16?
3. What is actual deployed bits/weight?
4. What is peak VRAM?
5. Does it fully reside on RTX 2000?
6. What context sizes work?
7. What is prompt-processing speed?
8. What is generation speed?
9. How much perplexity changed?
10. How closely do logits match BF16?
11. What reasoning ability was retained?
12. What coding ability was retained?
13. What tool-use ability was retained?
14. Which layers were most sensitive?
15. Were DeltaNet or full-attention layers more sensitive?
16. Which tensors required higher precision?
17. How much did reconstruction improve over naive ternary?
18. Is 7.2 GB worthwhile versus an ~10-11 GB conventional 2-bit quant?
19. What would additional GPU compute likely improve?
20. Is binary worth attempting next?

---

# PERFORMANCE TARGET

The finished model should ideally fit:

```text
weights
+ CUDA buffers
+ reasonable KV/recurrent state
+ 8K-16K context
```

inside approximately:

```text
16 GB VRAM
```

without CPU layer offload.

Measure rather than assume.

---

# IMPORTANT DISTINCTION: STORAGE VS MEMORY

Always report separately:

```text
GGUF file size
resident weight memory
runtime VRAM
KV/recurrent cache
CUDA scratch/buffers
```

Do not claim "7.2 GB VRAM" simply because GGUF is 7.2 GB.

---

# INITIAL PRIORITY ORDER

Do the project in exactly this conceptual order:

```text
1. Environment
2. Baselines
3. Architecture inventory
4. Streaming BF16 correctness
5. Calibration
6. Naive ternary
7. One-layer reconstruction
8. Multi-layer pilot
9. Sensitivity
10. Hybrid 8-9 GB model
11. Full evaluation
12. Targeted recovery
13. Full ~7.2 GB ternary attempt
14. Evaluation
15. Binary research only if justified
```

---

# DO NOT DO THESE THINGS

Do NOT:

```text
attempt full-model Adam training
allocate optimizer state for 27B parameters
load 54 GB BF16 directly into 16 GB VRAM
pretend raw Q2 rounding equals Prism's result
claim 95% retention without measuring it
optimize against benchmark test answers
download random community models as the teacher
make Docker mandatory
discard checkpoints
hard-code undocumented assumptions about Qwen3.8
silently fall back to CPU for massive training
spend days on all 64 layers before validating one-layer reconstruction
```

---

# POTENTIAL RESEARCH IMPROVEMENTS

Once baseline reconstruction works, investigate:

## GPTQ-like error propagation

After quantizing one column/group, compensate remaining weights using approximate second-order information.

## AWQ-style activation scaling

Protect salient channels before ternarization.

## OmniQuant-style transformations

Learn equivalent transformations that make low-bit quantization easier.

## Rotation

Investigate Hadamard/orthogonal transforms if compatible with runtime.

## Cross-block reconstruction

Optimize several adjacent layers together.

## Output distillation

Use teacher logits rather than only hidden state similarity.

## KL distillation

Match output distribution.

## Adaptive group sizes

For especially sensitive tensors try:

```text
g32
g64
g128
```

Balance scale overhead versus accuracy.

## Selective outliers

Test whether a very small sparse high-precision correction materially improves quality.

If used, report its real storage cost and do not hide it from the BPW calculation.

---

# OPTIONAL SPARSE RESIDUAL EXPERIMENT

Potentially represent:

```math
W ≈ W_ternary + R_sparse
```

where R contains only extreme reconstruction residuals.

Compare:

```text
pure ternary
ternary + 0.1% residual
ternary + 0.5% residual
ternary + 1% residual
```

Calculate total effective BPW.

This may produce a superior quality/size operating point.

Do not make this part of the first implementation.

---

# RESEARCH QUESTION: DELTANET VS ATTENTION

Qwen3.8's unusual hybrid architecture is important.

Explicitly analyze:

```text
DeltaNet layer sensitivity
vs
full-attention layer sensitivity
```

Aggregate metrics by type.

If one architecture type is consistently more sensitive, allocate bits accordingly.

This may be one of the most interesting outputs of the project.

---

# RESEARCH QUESTION: FFN VS ATTENTION

Also aggregate sensitivity across:

```text
MLP up/gate/down
Q
K
V
O
DeltaNet projections
embeddings
LM head
```

We want to determine where each additional bit provides the most value.

---

# STORAGE OPTIMIZATION OBJECTIVE

Eventually formulate precision assignment approximately as:

```math
maximize Quality(P)
subject to Size(P) <= B
```

where P is the tensor precision map.

Solve heuristically using measured layer sensitivity.

Evaluate multiple budgets B.

This turns the project from "make a ternary model" into:

**find the optimal low-bit representation of Qwen3.8-27B for a 16 GB GPU.**

That is the deeper objective.

---

# COMMAND-LINE UX

Examples:

```bash
python scripts/run_pipeline.py --mode pilot

python scripts/run_pipeline.py \
    --mode reconstruct \
    --target hybrid \
    --budget-gb 8.5

python scripts/run_pipeline.py \
    --mode reconstruct \
    --target ternary

python scripts/run_benchmarks.py \
    --model models/output/qwen38-ternary.gguf

python scripts/compare_models.py

python scripts/sensitivity_scan.py \
    --layers all
```

---

# CONFIG EXAMPLE

Create `config/project.yaml` roughly like:

```yaml
project:
  name: qwen38-ternary
  seed: 42

model:
  repo: Qwen/Qwen3.8-27B
  text_only: true
  include_vision: false
  include_mtp: false

hardware:
  gpu_memory_limit_gb: 14.0
  system_memory_limit_gb: 52.0
  minimum_free_disk_gb: 180

quantization:
  primary_scheme: ternary
  alphabet: [-1, 0, 1]
  group_size: 128
  scale_dtype: float16

reconstruction:
  pilot_layers: [0, 3, 15, 31, 47, 63]
  sequence_length: 1024
  initial_steps: 150
  progressive: true
  automatic_oom_recovery: true

calibration:
  pilot_sequences: 512
  pilot_length: 1024
  standard_sequences: 2048
  standard_length: 2048

output:
  initial_target_gb: 8.5
  full_ternary_target_gb: 7.5
```

Adapt based on actual measurements.

---

# AUTONOMOUS AGENT BEHAVIOR

This project is being handed to an autonomous coding agent.

Therefore:

Do not repeatedly ask the user basic implementation questions.

Use engineering judgment.

Automatically inspect the machine and repositories.

When there are multiple reasonable approaches:

1. choose the safest/reversible approach
2. document the decision
3. continue

Only stop for user input if:

```text
administrator credentials are absolutely required
a license requires explicit acceptance
storage is genuinely insufficient
required hardware is absent
a destructive operation would be required
a paid cloud resource would be required
```

Do not purchase or rent cloud resources automatically.

Everything up to that point should run locally.

---

# AGENT RESEARCH AUTHORITY

The agent is explicitly authorized to:

```text
read current Qwen source/configs
inspect Transformers source
inspect llama.cpp source
inspect PrismML public repositories
inspect GGUF format code
read public whitepapers/documentation
search for current Qwen3.8 compatibility issues
patch local source code
write custom CUDA/ggml integration if necessary
write Python/C++ tooling
run unit tests
run calibration
run quantization
run long experiments
benchmark outputs
restart failed jobs
modify the project based on measured results
```

Prefer primary sources:

```text
official Qwen
official Hugging Face repository
PrismML
llama.cpp
Intel AutoRound
published research
```

---

# FIRST ACTIONS

Begin by executing the following logical plan immediately:

## A.

Create repository and directory structure.

## B.

Detect hardware and write hardware report.

## C.

Verify available disk.

## D.

Verify Python/CUDA.

## E.

Clone:

```text
https://github.com/ggml-org/llama.cpp
https://github.com/PrismML-Eng/llama.cpp
https://github.com/PrismML-Eng/Bonsai-demo
https://github.com/intel/auto-round
```

## F.

Record exact commits.

## G.

Inspect:

```text
Qwen/Qwen3.8-27B
Prism ternary Q2 representation
Prism GGUF packing
current llama.cpp Qwen3.8 support
```

## H.

Download configuration/tokenizer first.

## I.

Write architecture inventory.

## J.

Only then download full BF16 model.

## K.

Build conventional baseline.

## L.

Implement/test ternary group quantizer independently.

## M.

Implement one-layer Qwen3.8 pilot.

## N.

Do not proceed to all layers until pilot report passes.

---

# DEFINITION OF "DONE"

The project is not done merely when a model file exists.

It is done when there is:

```text
a reproducible quantization pipeline
a validated ~7-9 GB low-bit Qwen3.8-27B
a working CUDA inference runtime
quality measurements
size measurements
performance measurements
comparison against conventional quants
layer sensitivity analysis
documented limitations
all scripts necessary to reproduce the result
```

The final repo should let another technically competent user with sufficient hardware run:

```text
bootstrap
-> quantize
-> pack
-> benchmark
```

without reverse-engineering undocumented manual steps.

---

# FINAL OBJECTIVE

The scientific question is:

> Can a 27B-class Qwen3.8 model be compressed from roughly 54 GB BF16 to approximately 7-9 GB on consumer/workstation hardware using ternary or adaptive mixed-bit weights, while preserving enough of the original reasoning, coding, instruction-following and agentic capability to materially outperform ordinary ultra-low-bit quantization at the same memory budget?

Do not optimize merely for producing the smallest file.

Optimize for:

```text
capability per GB
```

with a hard deployment target of one 16 GB RTX 2000 GPU.

Start now.
