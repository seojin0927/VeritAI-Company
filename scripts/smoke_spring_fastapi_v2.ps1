param(
    [string]$SpringBaseUrl = "http://127.0.0.1:8080",
    [string]$AiHealthUrl = "http://127.0.0.1:8000/",
    [string]$OutputDir = "images\eval_failure_reviews\service_retention",
    [int]$PollAttempts = 30,
    [int]$PollDelaySeconds = 1
)

$ErrorActionPreference = "Stop"

$root = Resolve-Path (Join-Path $PSScriptRoot "..")
$outputPath = Join-Path $root $OutputDir
New-Item -ItemType Directory -Force -Path $outputPath | Out-Null

function Invoke-JsonGet {
    param([string]$Url)
    $raw = curl.exe -s $Url
    if ([string]::IsNullOrWhiteSpace($raw)) {
        throw "Empty response from $Url"
    }
    return $raw | ConvertFrom-Json
}

function Invoke-DetectionUpload {
    param(
        [string]$FilePath,
        [string]$Url
    )
    $raw = curl.exe -s -X POST `
        -F "file=@$FilePath" `
        -F "analysisMode=full_image" `
        -F "clientType=spring-v2-smoke" `
        $Url
    if ([string]::IsNullOrWhiteSpace($raw)) {
        throw "Empty response from $Url"
    }
    return $raw | ConvertFrom-Json
}

function Wait-DetectionResult {
    param(
        [long]$RequestId,
        [string]$Url
    )
    for ($attempt = 0; $attempt -lt $PollAttempts; $attempt += 1) {
        $result = Invoke-JsonGet -Url "$Url/api/detections/$RequestId"
        if ($result.status -eq "DONE" -or $result.status -eq "FAILED") {
            return $result
        }
        Start-Sleep -Seconds $PollDelaySeconds
    }
    throw "Detection request $RequestId did not finish after $PollAttempts attempts."
}

$cases = @(
    @{
        Name = "frontal_pass"
        Path = "backend\backend_spring\uploads\0113fd0f-b25f-4dd5-b117-f979e167f691_capture.jpg"
    },
    @{
        Name = "profile_pass"
        Path = "backend\backend_spring\uploads\00a9b972-821f-44a2-ba69-3caf86e78ddb_capture.jpg"
    },
    @{
        Name = "no_face_warn"
        Path = "backend\backend_spring\uploads\01bffa9e-4b15-4c85-8ef2-5ee300f376b6_capture.jpg"
    },
    @{
        Name = "poor_quality_fail"
        Path = "backend\backend_spring\uploads\699111e1-4bd8-4cd7-939f-c86b235a10e3_capture.jpg"
    }
)

$aiHealth = Invoke-JsonGet -Url $AiHealthUrl
$queueBefore = Invoke-JsonGet -Url "$SpringBaseUrl/api/detections/queue"

$rows = foreach ($case in $cases) {
    $filePath = Join-Path $root $case.Path
    if (-not (Test-Path $filePath)) {
        throw "Smoke input not found: $filePath"
    }

    Write-Host "Uploading $($case.Name)"
    $created = Invoke-DetectionUpload -FilePath $filePath -Url "$SpringBaseUrl/api/detections"
    $completed = Wait-DetectionResult -RequestId $created.requestId -Url $SpringBaseUrl
    $face = $completed.result.faces | Select-Object -First 1

    [pscustomobject]@{
        case = $case.Name
        input = (Resolve-Path $filePath).Path
        requestId = $created.requestId
        initialStatus = $created.status
        finalStatus = $completed.status
        faceCount = $completed.result.faceCount
        pose = if ($null -ne $face) { $face.pose.label } else { $null }
        quality = if ($null -ne $face) { $face.quality.label } else { $null }
        detector = if ($null -ne $face) { $face.detector } else { $null }
        modelVersion = $completed.result.modelVersion
        analysisMode = $completed.result.analysisMode
        processingTimeMs = $completed.result.processingTimeMs
        message = $completed.result.message
    }
}

$queueAfter = Invoke-JsonGet -Url "$SpringBaseUrl/api/detections/queue"
$failedRows = @($rows | Where-Object { $_.finalStatus -ne "DONE" })
$summary = [pscustomobject]@{
    generatedAt = (Get-Date).ToString("s")
    springBaseUrl = $SpringBaseUrl
    aiHealthUrl = $AiHealthUrl
    aiHealth = $aiHealth
    queueBefore = $queueBefore
    queueAfter = $queueAfter
    caseCount = $rows.Count
    doneCount = @($rows | Where-Object { $_.finalStatus -eq "DONE" }).Count
    failedCount = $failedRows.Count
    cases = $rows
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$outputFile = Join-Path $outputPath "spring_fastapi_v2_smoke_$timestamp.json"
$summary | ConvertTo-Json -Depth 20 | Set-Content -Path $outputFile -Encoding UTF8

$summary | ConvertTo-Json -Depth 20

if ($failedRows.Count -gt 0) {
    throw "Spring/FastAPI smoke had $($failedRows.Count) failed case(s). Output: $outputFile"
}

Write-Host "Smoke output: $outputFile"
