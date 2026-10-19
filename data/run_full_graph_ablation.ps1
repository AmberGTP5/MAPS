param(
    [Parameter(Mandatory = $false)]
    [string]$Model = "google/gemini-2.5-flash",

    [Parameter(Mandatory = $false)]
    [string]$TaskFile = "graph_tasks_30.jsonl",

    [Parameter(Mandatory = $false)]
    [string]$OutputPrefix = "graph_benchmark_clean",

    [Parameter(Mandatory = $false)]
    [switch]$Resume,

    [Parameter(Mandatory = $false)]
    [switch]$CaptureStdout
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $TaskFile)) {
    throw "Task file not found: $TaskFile"
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$safeModel = ($Model -replace "[^a-zA-Z0-9]+", "_").Trim("_").ToLowerInvariant()
$baseOut = "${OutputPrefix}_${safeModel}_${timestamp}"

Write-Host "Model: $Model"
Write-Host "Task file: $TaskFile"
Write-Host "Base output dir: $baseOut"

$env:OPENROUTER_MODEL = $Model
$env:OPENROUTER_PLANNER_MODEL = $Model

$commonArgs = @(
    ".\run_graph_mode_benchmark.py",
    "--tasks-file", $TaskFile,
    "--modes", "greedy", "retry", "search",
    "--max-retries", "2"
)

if ($CaptureStdout) {
    $commonArgs += "--capture-stdout"
}
if ($Resume) {
    $commonArgs += "--resume"
}

$guidanceModes = @("none", "feasibility", "route_hint")

foreach ($guidance in $guidanceModes) {
    $outDir = "${baseOut}_${guidance}"
    Write-Host ""
    Write-Host "Running guidance_mode=$guidance -> $outDir"
    python @commonArgs --guidance-mode $guidance --output-dir $outDir
}

Write-Host ""
Write-Host "All runs completed."
Write-Host "Result dirs:"
foreach ($guidance in $guidanceModes) {
    Write-Host " - ${baseOut}_${guidance}"
}
