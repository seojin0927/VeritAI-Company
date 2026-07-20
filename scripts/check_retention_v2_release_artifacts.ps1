param(
    [string]$OutputDir = "images\eval_failure_reviews\service_retention"
)

$ErrorActionPreference = "Stop"

$root = Resolve-Path (Join-Path $PSScriptRoot "..")
$outputPath = Join-Path $root $OutputDir
New-Item -ItemType Directory -Force -Path $outputPath | Out-Null

$required = @(
    @{
        Name = "external_final_decision"
        Path = "images\eval_failure_reviews\service_retention\retention_precision_combo_v2_final_decision_summary_20260707.json"
    },
    @{
        Name = "internal_unlabeled_compare"
        Path = "images\eval_failure_reviews\service_retention\retention_precision_combo_v2_internal_unlabeled_compare_20260707.json"
    },
    @{
        Name = "upload_audit_0_100"
        Path = "images\upload_audit_runs\20260708_143842_retention_v2_smoke_0_100\summary.json"
    },
    @{
        Name = "upload_audit_100_300"
        Path = "images\upload_audit_runs\20260708_144247_retention_v2_upload_100_300\summary.json"
    },
    @{
        Name = "upload_audit_300_500"
        Path = "images\upload_audit_runs\20260708_145035_retention_v2_upload_300_500\summary.json"
    },
    @{
        Name = "upload_audit_500_700"
        Path = "images\upload_audit_runs\20260708_145746_retention_v2_upload_500_700\summary.json"
    },
    @{
        Name = "upload_audit_700_end"
        Path = "images\upload_audit_runs\20260708_150305_retention_v2_upload_700_end\summary.json"
    },
    @{
        Name = "spring_fastapi_smoke"
        Path = "images\eval_failure_reviews\service_retention\spring_fastapi_v2_smoke_20260708_153624.json"
    },
    @{
        Name = "v2_server_script"
        Path = "scripts\run_ai_server_retention_v2.ps1"
    },
    @{
        Name = "spring_smoke_script"
        Path = "scripts\smoke_spring_fastapi_v2.ps1"
    },
    @{
        Name = "decision_document"
        Path = "Readme0703_retention_precision_combo_v2_smoke.md"
        SearchName = "Readme0703_retention_precision_combo_v2_smoke.md"
    },
    @{
        Name = "release_decision_document"
        Path = "Readme0709_retention_v2_release_decision.md"
        SearchName = "Readme0709_retention_v2_release_decision.md"
    }
)

$items = foreach ($item in $required) {
    $absolutePath = Join-Path $root $item.Path
    $exists = Test-Path $absolutePath
    if (-not $exists -and $item.SearchName) {
        $exists = $null -ne (
            Get-ChildItem -Path $root -Recurse -Filter $item.SearchName -File -ErrorAction SilentlyContinue |
            Select-Object -First 1
        )
    }
    [pscustomobject]@{
        name = $item.Name
        path = $item.Path
        exists = $exists
    }
}

$missing = @($items | Where-Object { -not $_.exists })
$summary = [pscustomobject]@{
    generatedAt = (Get-Date).ToString("s")
    total = $items.Count
    present = @($items | Where-Object { $_.exists }).Count
    missing = $missing.Count
    ready = ($missing.Count -eq 0)
    items = $items
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$outputFile = Join-Path $outputPath "retention_v2_release_artifacts_check_$timestamp.json"
$summary | ConvertTo-Json -Depth 10 | Set-Content -Path $outputFile -Encoding UTF8
$summary | ConvertTo-Json -Depth 10

if ($missing.Count -gt 0) {
    throw "Missing $($missing.Count) required release artifact(s). Output: $outputFile"
}

Write-Host "Release artifact check output: $outputFile"
