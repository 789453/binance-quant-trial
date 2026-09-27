param([int]$TrainMonths,[int]$StepMonths,[double]$FeeBps,[string]$Start,[string]$Config)
$ErrorActionPreference="Stop"
$root=Split-Path -Parent $PSScriptRoot
$argsList=@("-m","research.walk_forward_v2")
if($TrainMonths){$argsList+=@("--train-months",$TrainMonths)}
if($StepMonths){$argsList+=@("--step-months",$StepMonths)}
if($FeeBps){$argsList+=@("--fee-bps",$FeeBps)}
if($Start){$argsList+=@("--start",$Start)}
if($Config){$argsList+=@("--config",$Config)}
Push-Location $root
$python=if($env:QUANT_PYTHON){$env:QUANT_PYTHON}else{(Get-Command python).Source}
try{& $python @argsList}finally{Pop-Location}
