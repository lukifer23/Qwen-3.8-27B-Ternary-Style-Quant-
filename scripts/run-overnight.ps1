#Requires -Version 5.1
<#
.SYNOPSIS
    Unattended overnight reconstruct + GGUF. Logs to artifacts/logs and the console.
#>
[CmdletBinding()]
param(
    [string]$Device = "cuda",
    [int]$Steps = 150,
    [int]$StartLayer = 0,
    [int]$EndLayer = 63,
    [switch]$Force,
    [switch]$Compile,
    [switch]$SkipGguf
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$env:PYTHONPATH = "src"
$env:PYTHONUNBUFFERED = "1"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"

$Python = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "Missing $Python — run scripts\bootstrap.ps1 first."
}

$argsList = @(
    "scripts\run_overnight.py",
    "--device", $Device,
    "--steps", "$Steps",
    "--start-layer", "$StartLayer",
    "--end-layer", "$EndLayer"
)
if ($Force) { $argsList += "--force" }
if ($Compile) { $argsList += "--compile" }
if ($SkipGguf) { $argsList += "--skip-gguf" }

Write-Host "Starting overnight job in $Root" -ForegroundColor Cyan
Write-Host "$Python -u $($argsList -join ' ')" -ForegroundColor Cyan
& $Python -u @argsList
exit $LASTEXITCODE
