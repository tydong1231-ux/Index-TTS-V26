param(
    [string]$UpstreamRepo = "",
    [string]$RuntimeRoot = "",
    [string]$ReleaseRoot = "",
    [string]$ReleaseName = "index-tts-windows-cu128-deepspeed",
    [switch]$Clean
)

$ErrorActionPreference = "Stop"

function Require-Path([string]$PathValue, [string]$Label) {
    if (-not (Test-Path -LiteralPath $PathValue)) {
        throw "$Label not found: $PathValue"
    }
}

function Ensure-Dir([string]$PathValue) {
    New-Item -ItemType Directory -Force -Path $PathValue | Out-Null
}

function Mirror-Directory([string]$Source, [string]$Destination, [string[]]$ExcludeDirs = @(), [string[]]$ExcludeFiles = @()) {
    Require-Path $Source "Source directory"
    Ensure-Dir $Destination

    $args = @(
        $Source,
        $Destination,
        "/MIR",
        "/R:1",
        "/W:1",
        "/NFL",
        "/NDL",
        "/NP",
        "/NJH",
        "/NJS"
    )

    if ($ExcludeDirs.Count -gt 0) {
        $args += "/XD"
        $args += $ExcludeDirs
    }
    if ($ExcludeFiles.Count -gt 0) {
        $args += "/XF"
        $args += $ExcludeFiles
    }

    & robocopy @args | Out-Null
    $code = $LASTEXITCODE
    if ($code -ge 8) {
        throw "robocopy failed with exit code $code for $Source -> $Destination"
    }
}

function Write-Utf8NoBom([string]$PathValue, [string]$Content) {
    $encoding = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($PathValue, $Content, $encoding)
}

if ([string]::IsNullOrWhiteSpace($RuntimeRoot)) {
    $RuntimeRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
}
if ([string]::IsNullOrWhiteSpace($UpstreamRepo)) {
    throw "Pass -UpstreamRepo explicitly. This script no longer guesses an external source repo path."
}
if ([string]::IsNullOrWhiteSpace($ReleaseRoot)) {
    $ReleaseRoot = Join-Path $RuntimeRoot "dist\$ReleaseName"
}

Require-Path $UpstreamRepo "Upstream repo"
Require-Path $RuntimeRoot "Runtime root"
Require-Path (Join-Path $UpstreamRepo ".venv") "Project-local upstream runtime"
Require-Path (Join-Path $UpstreamRepo ".wheelhouse") "Project-local wheelhouse"
Require-Path (Join-Path $RuntimeRoot "checkpoints") "Checkpoint directory"
Require-Path (Join-Path $RuntimeRoot "hf_cache") "HF cache directory"
Require-Path (Join-Path $RuntimeRoot "examples") "Examples directory"

if ($Clean -and (Test-Path -LiteralPath $ReleaseRoot)) {
    Remove-Item -LiteralPath $ReleaseRoot -Recurse -Force
}

$appRoot = Join-Path $ReleaseRoot "app"
$runtimeDir = Join-Path $ReleaseRoot "runtime"
$wheelhouseDir = Join-Path $ReleaseRoot "wheelhouse"
$dataDir = Join-Path $ReleaseRoot "data"
$checkpointsDir = Join-Path $dataDir "checkpoints"
$hfCacheDir = Join-Path $dataDir "hf_cache"
$outputsDir = Join-Path $dataDir "outputs"
$toolsDir = Join-Path $ReleaseRoot "tools"
$docsDir = Join-Path $ReleaseRoot "docs"

Ensure-Dir $ReleaseRoot
Ensure-Dir $outputsDir

Write-Host "Mirroring official latest application source..."
Mirror-Directory `
    -Source $UpstreamRepo `
    -Destination $appRoot `
    -ExcludeDirs @(".git", ".uv-cache", ".vendor", ".wheelhouse", ".venv", "tmp", "__pycache__")

Write-Host "Overriding examples with project-local real sample assets..."
Mirror-Directory `
    -Source (Join-Path $RuntimeRoot "examples") `
    -Destination (Join-Path $appRoot "examples")

Write-Host "Mirroring project-local Python runtime..."
Mirror-Directory `
    -Source (Join-Path $UpstreamRepo ".venv") `
    -Destination $runtimeDir `
    -ExcludeDirs @("__pycache__")

Write-Host "Mirroring project-local wheelhouse..."
Mirror-Directory `
    -Source (Join-Path $UpstreamRepo ".wheelhouse") `
    -Destination $wheelhouseDir

Write-Host "Mirroring checkpoints..."
Mirror-Directory `
    -Source (Join-Path $RuntimeRoot "checkpoints") `
    -Destination $checkpointsDir `
    -ExcludeDirs @(".cache")

Write-Host "Mirroring Hugging Face cache..."
Mirror-Directory `
    -Source (Join-Path $RuntimeRoot "hf_cache") `
    -Destination $hfCacheDir

Write-Host "Mirroring Windows packaging tools..."
Mirror-Directory `
    -Source (Join-Path $RuntimeRoot "tools\windows_local") `
    -Destination (Join-Path $toolsDir "windows_local")

Write-Host "Mirroring packaging docs..."
Mirror-Directory `
    -Source (Join-Path $RuntimeRoot "docs") `
    -Destination $docsDir `
    -ExcludeFiles @("README_zh.md")

$launcherBat = @'
@echo off
setlocal
set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"
cd /d "%ROOT%"
set "NUMBA_CACHE_DIR=%ROOT%\data\.numba_cache"
set "HF_HOME=%ROOT%\data\hf_cache"
set "HF_HUB_CACHE=%ROOT%\data\hf_cache"
set "HUGGINGFACE_HUB_CACHE=%ROOT%\data\hf_cache"
set "TRANSFORMERS_CACHE="
powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%\tools\windows_local\launch_official_latest_webui.ps1" -UpstreamRepo "%ROOT%\app" -RuntimeRoot "%ROOT%" -DataRoot "%ROOT%\data" -PythonExe "%ROOT%\runtime\Scripts\python.exe" -Mode auto -Fp16
endlocal
'@

$checkBat = @'
@echo off
setlocal
set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"
cd /d "%ROOT%"
set "HF_HOME=%ROOT%\data\hf_cache"
set "HF_HUB_CACHE=%ROOT%\data\hf_cache"
set "HUGGINGFACE_HUB_CACHE=%ROOT%\data\hf_cache"
set "TRANSFORMERS_CACHE="
powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%\tools\windows_local\run_release_support_check.ps1" -UpstreamRepo "%ROOT%\app" -RuntimeRoot "%ROOT%" -DataRoot "%ROOT%\data" -PythonExe "%ROOT%\runtime\Scripts\python.exe" -SummaryOnly -RequireDeepSpeed
endlocal
'@

$readme = @'
# IndexTTS Windows Release

This release is frozen from:

- official latest application source
- project-local Python runtime
- project-local DeepSpeed wheelhouse
- project-local checkpoints and Hugging Face cache

## Layout

- app: official latest source tree used at runtime
- runtime: self-contained Python environment
- wheelhouse: local wheel cache, including DeepSpeed
- data\checkpoints: IndexTTS model checkpoints
- data\hf_cache: Hugging Face cache required by the official pipeline
- data\outputs: generated audio and reports

## Start

- Run start_webui_auto.bat to launch the WebUI.
- Run check_support.bat to emit a compatibility report before release or support work.

## Notes

- The package does not rely on a system Python installation.
- The package does not require a separate CUDA Toolkit installation on user machines.
- End users still need an NVIDIA GPU driver new enough for the packaged runtime baseline.
'@

Write-Utf8NoBom -PathValue (Join-Path $ReleaseRoot "start_webui_auto.bat") -Content $launcherBat
Write-Utf8NoBom -PathValue (Join-Path $ReleaseRoot "check_support.bat") -Content $checkBat
Write-Utf8NoBom -PathValue (Join-Path $ReleaseRoot "README.txt") -Content $readme

Write-Host ""
Write-Host "Windows release layout frozen at:"
Write-Host $ReleaseRoot
