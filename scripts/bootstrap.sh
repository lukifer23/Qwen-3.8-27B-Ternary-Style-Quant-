#!/usr/bin/env bash
# Create the local Python environment, install CUDA PyTorch, verify the machine.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

step() { printf '==> %s\n' "$1"; }

step "repo root: $ROOT"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install from https://github.com/astral-sh/uv and re-run." >&2
  exit 1
fi

step "pinning Python 3.11 via uv"
uv python install 3.11
uv python pin 3.11
uv venv --python 3.11 .venv

if [[ -x "$ROOT/.venv/bin/python" ]]; then
  VENV_PY="$ROOT/.venv/bin/python"
else
  VENV_PY="$ROOT/.venv/Scripts/python.exe"
fi

step "installing official CUDA PyTorch (trying cu128, then cu126)"
TORCH_OK=0
for INDEX in \
  https://download.pytorch.org/whl/cu128 \
  https://download.pytorch.org/whl/cu126 \
  https://download.pytorch.org/whl/cu124
do
  echo "    trying $INDEX"
  if uv pip install --python "$VENV_PY" torch --index-url "$INDEX"; then
    TORCH_OK=1
    break
  fi
done
if [[ "$TORCH_OK" -ne 1 ]]; then
  echo "Could not install a CUDA torch wheel." >&2
  exit 1
fi

step "installing project (editable) + pytest"
uv pip install --python "$VENV_PY" -e ".[dev]"

step "detecting hardware"
"$VENV_PY" "$ROOT/scripts/detect_hardware.py"

step "torch CUDA probe"
"$VENV_PY" - <<'PY'
import sys
import torch
print("torch", torch.__version__)
print("cuda_build", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
    print("total_memory", torch.cuda.get_device_properties(0).total_memory)
    sys.exit(0)
print("CUDA is not available in this venv. Do not start reconstruction.")
sys.exit(2)
PY

echo "bootstrap complete"
