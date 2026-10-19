param(
    [Parameter(Mandatory = $false)]
    [string]$Model = "google/gemini-2.5-flash",

    [Parameter(Mandatory = $false)]
    [string]$TaskFile = "app_benchmark_tasks.json",

    [Parameter(Mandatory = $false)]
    [string]$OutputPrefix = "app_benchmark_clean",

    [Parameter(Mandatory = $false)]
    [int]$MaxRetries = 2,

    [Parameter(Mandatory = $false)]
    [int]$SearchIterations = 96,

    [Parameter(Mandatory = $false)]
    [int]$SearchDepth = 18,

    [Parameter(Mandatory = $false)]
    [double]$Temperature = 0.0,

    [Parameter(Mandatory = $false)]
    [switch]$CaptureStdout
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $TaskFile)) {
    throw "Task file not found: $TaskFile"
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$safeModel = ($Model -replace "[^a-zA-Z0-9]+", "_").Trim("_").ToLowerInvariant()
$outDir = "${OutputPrefix}_${safeModel}_${timestamp}"

Write-Host "Model: $Model"
Write-Host "Task file: $TaskFile"
Write-Host "Output dir: $outDir"
Write-Host "Modes: greedy retry search"
Write-Host "Search iterations/depth: $SearchIterations / $SearchDepth"
Write-Host "Temperature: $Temperature"

$env:OPENROUTER_MODEL = $Model
$env:OPENROUTER_PLANNER_MODEL = $Model
$env:OPENROUTER_EXECUTOR_MODEL = $Model
$env:OPENROUTER_TEMPERATURE = "$Temperature"

$argsList = @(
    ".\run_app_benchmark.py",
    "--tasks-file", $TaskFile,
    "--output-dir", $outDir,
    "--modes", "greedy", "retry", "search",
    "--max-retries", "$MaxRetries",
    "--search-iterations", "$SearchIterations",
    "--search-depth", "$SearchDepth",
    "--dry-run"
)

if ($CaptureStdout) {
    $argsList += "--capture-stdout"
}

python @argsList

Write-Host ""
Write-Host "Run complete."
Write-Host "Summary:"
Write-Host " - $outDir\benchmark_summary.md"
Write-Host " - $outDir\benchmark_results.json"
