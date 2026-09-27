param(
    [string[]]$Symbols,
    [ValidateSet("5m", "15m", "1h")][string[]]$Timeframes,
    [switch]$UseProxy
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = if ($env:QUANT_PYTHON) { $env:QUANT_PYTHON } else { (Get-Command python).Source }
$arguments = @("tools/update_research_data.py")
if ($Symbols) { $arguments += "--symbols"; $arguments += $Symbols }
if ($Timeframes) { $arguments += "--timeframes"; $arguments += $Timeframes }
if ($UseProxy) { $arguments += @("--proxy", "socks5h://127.0.0.1:7897") }
Push-Location $projectRoot
try { & $python @arguments } finally { Pop-Location }
