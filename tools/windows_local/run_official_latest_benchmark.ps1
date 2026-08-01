param(
    [string]$UpstreamRepo = "",
    [string]$RuntimeRoot = "",
    [string]$PythonExe = "",
    [string]$Text = "Official latest normal mode and DeepSpeed mode benchmark.",
    [string]$SpeakerWav = "examples/voice_01.wav",
    [switch]$Fp16,
    [switch]$SkipNormal,
    [switch]$SkipDeepSpeed
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RuntimeRoot)) {
    $RuntimeRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
}
if ([string]::IsNullOrWhiteSpace($UpstreamRepo)) {
    throw "Pass -UpstreamRepo explicitly. This script no longer guesses an external source repo path."
}
if ([string]::IsNullOrWhiteSpace($PythonExe)) {
    $PythonExe = Join-Path $UpstreamRepo ".venv\Scripts\python.exe"
}

$scriptPath = Join-Path $RuntimeRoot "tools\windows_local\benchmark_official_latest.py"
$numbaCache = Join-Path $RuntimeRoot ".numba_cache"
New-Item -ItemType Directory -Force -Path $numbaCache | Out-Null
$env:NUMBA_CACHE_DIR = $numbaCache

$argsList = @(
    $scriptPath,
    "--upstream-repo", $UpstreamRepo,
    "--runtime-root", $RuntimeRoot,
    "--text", $Text,
    "--speaker-wav", $SpeakerWav
)

if ($Fp16) {
    $argsList += "--fp16"
}
if ($SkipNormal) {
    $argsList += "--skip-normal"
}
if ($SkipDeepSpeed) {
    $argsList += "--skip-deepspeed"
}

& $PythonExe @argsList
