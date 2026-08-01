param(
    [string]$UpstreamRepo = "",
    [string]$RuntimeRoot = "",
    [string]$DataRoot = "",
    [string]$PythonExe = "",
    [string]$ValidatedDriver = "591.86",
    [string]$MinDriver = "",
    [string]$MinComputeCapability = "6.1",
    [switch]$SummaryOnly,
    [switch]$RequireDeepSpeed
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RuntimeRoot)) {
    $RuntimeRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
}
if ([string]::IsNullOrWhiteSpace($UpstreamRepo)) {
    $UpstreamRepo = Join-Path $RuntimeRoot "app"
    if (-not (Test-Path -LiteralPath $UpstreamRepo)) {
        $UpstreamRepo = $RuntimeRoot
    }
}
if ([string]::IsNullOrWhiteSpace($DataRoot)) {
    $candidate = Join-Path $RuntimeRoot "data"
    if (Test-Path -LiteralPath $candidate) {
        $DataRoot = $candidate
    }
    else {
        $DataRoot = $RuntimeRoot
    }
}
if ([string]::IsNullOrWhiteSpace($PythonExe)) {
    $candidate = Join-Path $RuntimeRoot "runtime\python.exe"
    if (Test-Path -LiteralPath $candidate) {
        $PythonExe = $candidate
    }
    else {
        $candidate = Join-Path $RuntimeRoot "runtime\Scripts\python.exe"
        if (Test-Path -LiteralPath $candidate) {
            $PythonExe = $candidate
        }
        else {
            $PythonExe = Join-Path $UpstreamRepo ".venv\Scripts\python.exe"
        }
    }
}
$toolsPythonPath = Join-Path $RuntimeRoot "tools\windows_local"
if ($env:PYTHONPATH) {
    $env:PYTHONPATH = "$toolsPythonPath;$env:PYTHONPATH"
}
else {
    $env:PYTHONPATH = $toolsPythonPath
}

$scriptPath = Join-Path $RuntimeRoot "tools\windows_local\check_release_support.py"
$reportPath = Join-Path $DataRoot "outputs\release_support_report.json"
$cacheRoot = Join-Path $DataRoot "hf_cache"

Push-Location $DataRoot
try {
    $argsList = @(
        $scriptPath,
        "--report-json", $reportPath,
        "--validated-driver", $ValidatedDriver,
        "--min-compute-capability", $MinComputeCapability,
        "--cache-root", $cacheRoot
    )

    if (-not [string]::IsNullOrWhiteSpace($MinDriver)) {
        $argsList += @("--min-driver", $MinDriver)
    }
    if ($SummaryOnly) {
        $argsList += "--summary-only"
    }
    if ($RequireDeepSpeed) {
        $argsList += "--require-deepspeed"
    }

    & $PythonExe @argsList
}
finally {
    Pop-Location
}
