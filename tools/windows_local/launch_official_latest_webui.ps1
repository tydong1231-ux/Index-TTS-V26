param(
    [ValidateSet("auto", "deepspeed", "normal")]
    [string]$Mode = "auto",
    [string]$UpstreamRepo = "",
    [string]$RuntimeRoot = "",
    [string]$DataRoot = "",
    [string]$PythonExe = "",
    [string]$ModelDir = "checkpoints",
    [string]$BindHost = "127.0.0.1",
    [int]$Port = 7862,
    [switch]$OpenBrowser,
    [switch]$Fp16,
    [switch]$Verbose
)

$ErrorActionPreference = "Stop"

function Join-UnicodeChars {
    param([int[]]$Codes)
    return -join ($Codes | ForEach-Object { [char]$_ })
}

function Write-Banner {
    param([string]$ResolvedMode)

    $authorLine = Join-UnicodeChars @(20316,32773,32,47,32,25972,21512,21457,24067,58,32,29579,30693,39118)
    $groupLine = Join-UnicodeChars @(65,73,24037,20855,32676,65306,57,53,55,55,51,50,54,54,52)
    $waitLine = Join-UnicodeChars @(27491,22312,21551,21160,65292,35831,31245,20505,46,46,46)
    $modeLabel = Join-UnicodeChars @(21551,21160,27169,24335,58,32)
    $modeText = if ($ResolvedMode -eq "deepspeed") { "DeepSpeed" } else { "Normal" }

    Write-Host "==============================================="
    Write-Host "IndexTTS V26"
    Write-Host $authorLine
    Write-Host $groupLine
    Write-Host ($modeLabel + $modeText)
    Write-Host $waitLine
    Write-Host "==============================================="
}

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

$webuiPath = Join-Path $UpstreamRepo "webui.py"
$toolsPythonPath = Join-Path $RuntimeRoot "tools\windows_local"
if ($env:PYTHONPATH) {
    $env:PYTHONPATH = "$toolsPythonPath;$env:PYTHONPATH"
}
else {
    $env:PYTHONPATH = $toolsPythonPath
}

if ([System.IO.Path]::IsPathRooted($ModelDir)) {
    $resolvedModelDir = $ModelDir
}
else {
    $resolvedModelDir = Join-Path $DataRoot $ModelDir
}

$reportPath = Join-Path $DataRoot "outputs\release_support_report.json"
$numbaCache = Join-Path $DataRoot ".numba_cache"
$hfCache = Join-Path $DataRoot "hf_cache"
New-Item -ItemType Directory -Force -Path $numbaCache | Out-Null
New-Item -ItemType Directory -Force -Path $hfCache | Out-Null
$env:NUMBA_CACHE_DIR = $numbaCache
$env:HF_HOME = $hfCache
$env:HF_HUB_CACHE = $hfCache
$env:HUGGINGFACE_HUB_CACHE = $hfCache
$env:TRANSFORMERS_CACHE = ""
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"

$effectiveMode = $Mode
if ($Mode -eq "auto") {
    $checkArgs = @(
        (Join-Path $RuntimeRoot "tools\windows_local\check_release_support.py"),
        "--report-json", $reportPath,
        "--validated-driver", "591.86",
        "--min-compute-capability", "6.1",
        "--cache-root", $hfCache,
        "--summary-only",
        "--require-deepspeed"
    )

    Push-Location $DataRoot
    try {
        $precheckOutput = & $PythonExe @checkArgs 2>&1
        if ($LASTEXITCODE -eq 0) {
            $effectiveMode = "deepspeed"
        }
        else {
            $effectiveMode = "normal"
        }
    }
    finally {
        Pop-Location
    }

    foreach ($line in $precheckOutput) {
        $text = [string]$line
        if ($text.StartsWith("[precheck]")) {
            Write-Host $text
        }
    }
}

Write-Banner -ResolvedMode $effectiveMode

$argsList = @(
    $webuiPath,
    "--host", $BindHost,
    "--port", $Port,
    "--model_dir", $resolvedModelDir
)

if ($Fp16) {
    $argsList += "--fp16"
}
if ($Verbose) {
    $argsList += "--verbose"
}
if ($effectiveMode -eq "deepspeed") {
    $argsList += "--deepspeed"
}

# Ensure Gradio startup-events callback to localhost is not intercepted by proxy
$env:NO_PROXY = "127.0.0.1,localhost"
$env:no_proxy = "127.0.0.1,localhost"
$env:GRADIO_ANALYTICS_ENABLED = "False"

Push-Location $UpstreamRepo
try {
    & $PythonExe @argsList
}
finally {
    Pop-Location
}
