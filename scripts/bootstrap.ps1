#Requires -Version 5.1
<#
.SYNOPSIS
    Create the local Python environment, install CUDA PyTorch, verify the machine.
#>
[CmdletBinding()]
param(
    [switch]$SkipTorch
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

function Write-Step([string]$Message) {
    Write-Host "==> $Message" -ForegroundColor Cyan
}

Write-Step "repo root: $Root"

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv is required. Install from https://github.com/astral-sh/uv and re-run."
}

Write-Step "pinning Python 3.11 via uv"
uv python install 3.11
uv python pin 3.11
uv venv --python 3.11 .venv

$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    throw "venv python missing at $VenvPython"
}

if (-not $SkipTorch) {
    Write-Step "installing official CUDA PyTorch (trying cu128, then cu126)"
    $torchOk = $false
    foreach ($index in @(
        "https://download.pytorch.org/whl/cu128",
        "https://download.pytorch.org/whl/cu126",
        "https://download.pytorch.org/whl/cu124"
    )) {
        Write-Host "    trying $index"
        & uv pip install --python $VenvPython torch --index-url $index
        if ($LASTEXITCODE -eq 0) {
            $torchOk = $true
            break
        }
    }
    if (-not $torchOk) {
        throw "Could not install a CUDA torch wheel. Install one manually from https://pytorch.org and re-run."
    }
}

Write-Step "installing project (editable) + pytest"
uv pip install --python $VenvPython -e ".[dev]"
if ($LASTEXITCODE -ne 0) {
    throw "uv pip install of the project failed"
}

Write-Step "detecting hardware"
& $VenvPython (Join-Path $Root "scripts\detect_hardware.py")
$detectCode = $LASTEXITCODE

Write-Step "torch CUDA probe"
& $VenvPython -c @"
import torch, sys
print('torch', torch.__version__)
print('cuda_build', torch.version.cuda)
print('cuda_available', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device', torch.cuda.get_device_name(0))
    print('total_memory', torch.cuda.get_device_properties(0).total_memory)
    sys.exit(0)
print('CUDA is not available in this venv. Do not start reconstruction.')
sys.exit(2)
"@
$cudaCode = $LASTEXITCODE

if ($detectCode -ne 0) {
    Write-Host "hardware detection reported issues (exit $detectCode). See artifacts/reports/00_hardware.md" -ForegroundColor Yellow
}
if ($cudaCode -ne 0) {
    throw "torch.cuda.is_available() is False after bootstrap. Fix the CUDA wheel before any reconstruction."
}

Write-Host "bootstrap complete" -ForegroundColor Green
exit 0
