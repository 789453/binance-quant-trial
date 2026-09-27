param(
    [string]$TrainEnd = "2025-01-01",
    [double]$FeeBps = 4.0,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = if ($env:QUANT_PYTHON) { $env:QUANT_PYTHON } else { (Get-Command python).Source }
$arguments = @("-m", "research.factor_combo_pipeline", "--train-end", $TrainEnd, "--fee-bps", $FeeBps)
if ($Force) { $arguments += "--force" }
Push-Location $projectRoot
try { & $python @arguments } finally { Pop-Location }
