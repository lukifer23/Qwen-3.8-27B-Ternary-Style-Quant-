#Requires -Version 5.1
<#
.SYNOPSIS
    Bootstrap the environment, then run the full gated pipeline.
#>
[CmdletBinding()]
param(
    [string]$Mode = "auto"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

& (Join-Path $PSScriptRoot "bootstrap.ps1")
if ($LASTEXITCODE -ne 0) {
    throw "bootstrap failed"
}

$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
& $VenvPython (Join-Path $Root "scripts\run_pipeline.py") --mode $Mode
exit $LASTEXITCODE
