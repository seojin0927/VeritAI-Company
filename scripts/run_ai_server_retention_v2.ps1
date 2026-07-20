$ErrorActionPreference = "Stop"

$root = Resolve-Path (Join-Path $PSScriptRoot "..")
$python = Join-Path $root "ai\venv\Scripts\python.exe"
$main = Join-Path $root "ai\main.py"

if (-not (Test-Path $python)) {
    throw "AI virtualenv python not found: $python"
}

if (-not (Test-Path $main)) {
    throw "AI server entrypoint not found: $main"
}

$env:VERITAI_RETENTION_FEATURE_GUARD = "retention_precision_combo_v2"

Write-Host "Starting VeritAI AI server"
Write-Host "  retention guard: $env:VERITAI_RETENTION_FEATURE_GUARD"
Write-Host "  endpoint: http://127.0.0.1:8000/predict"
Write-Host "  health:   http://127.0.0.1:8000/"

Push-Location $root
try {
    & $python $main
}
finally {
    Pop-Location
}
