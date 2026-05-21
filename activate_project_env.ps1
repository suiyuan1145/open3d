Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned -Force

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvActivate = Join-Path $projectRoot ".venv\Scripts\Activate.ps1"

if (-not (Test-Path $venvActivate)) {
    Write-Error "Cannot find project virtual environment: $venvActivate"
    exit 1
}

& $venvActivate
Write-Host "Activated project environment: .venv"
