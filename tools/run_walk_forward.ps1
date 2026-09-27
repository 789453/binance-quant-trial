param(
    [int]$TrainMonths = 6,
    [int]$StepMonths = 1,
    [double]$FeeBps = 4.0,
    [string]$Start,
    [switch]$ForceFactors
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = if ($env:QUANT_PYTHON) { $env:QUANT_PYTHON } else { (Get-Command python).Source }
$arguments = @("-m", "research.walk_forward", "--train-months", $TrainMonths,
               "--step-months", $StepMonths, "--fee-bps", $FeeBps)
if ($Start) { $arguments += @("--start", $Start) }
if ($ForceFactors) { $arguments += "--force-factors" }
Push-Location $projectRoot
try { & $python @arguments } finally { Pop-Location }
