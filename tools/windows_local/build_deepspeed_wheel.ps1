param(
    [string]$RepoRoot = (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)),
    [string]$DeepSpeedSource,
    [string]$PythonExe = "",
    [string]$WorkDir = "",
    [string]$Wheelhouse = "",
    [switch]$Clean
)

$ErrorActionPreference = "Stop"

function Require-Path([string]$PathValue, [string]$Label) {
    if (-not (Test-Path -LiteralPath $PathValue)) {
        throw "$Label not found: $PathValue"
    }
}

function Replace-ExactBlock {
    param(
        [string]$Path,
        [string]$Old,
        [string]$New
    )

    $text = [System.IO.File]::ReadAllText($Path)
    $normalized = $text.Replace("`r`n", "`n")
    $oldNormalized = $Old.Replace("`r`n", "`n")
    $newNormalized = $New.Replace("`r`n", "`n")
    if (-not $normalized.Contains($oldNormalized)) {
        throw "Expected block not found in $Path"
    }
    $normalized = $normalized.Replace($oldNormalized, $newNormalized)
    $finalText = $normalized.Replace("`n", "`r`n")
    [System.IO.File]::WriteAllText($Path, $finalText, (New-Object System.Text.UTF8Encoding($false)))
}

function Patch-DeepSpeedSource {
    param([string]$SourceRoot)

    $setupPath = Join-Path $SourceRoot "setup.py"
    $gitInfoPath = Join-Path $SourceRoot "deepspeed\git_version_info.py"

    $oldSetup = @"
install_ops = dict.fromkeys(ALL_OPS.keys(), False)
for op_name, builder in ALL_OPS.items():
    op_compatible = builder.is_compatible()

    # If op is requested but not available, throw an error.
    if op_enabled(op_name) and not op_compatible:
        env_var = op_envvar(op_name)
        if not is_env_set(env_var):
            builder.warning(f"Skip pre-compile of incompatible {op_name}; One can disable {op_name} with {env_var}=0")
        continue

    # If op is compatible but install is not enabled (JIT mode).
    if is_rocm_pytorch and op_compatible and not op_enabled(op_name):
        builder.hipify_extension()

    # If op install enabled, add builder to extensions.
    if op_enabled(op_name) and op_compatible:
        assert torch_available, f"Unable to pre-compile {op_name}, please first install torch"
        install_ops[op_name] = op_enabled(op_name)
        ext_modules.append(builder.builder())
"@

    $newSetup = @"
install_ops = dict.fromkeys(ALL_OPS.keys(), False)
for op_name, builder in ALL_OPS.items():
    if not op_enabled(op_name):
        continue

    op_compatible = builder.is_compatible()

    # If op is requested but not available, throw an error.
    if not op_compatible:
        env_var = op_envvar(op_name)
        if not is_env_set(env_var):
            builder.warning(f"Skip pre-compile of incompatible {op_name}; One can disable {op_name} with {env_var}=0")
        continue

    # If op install enabled, add builder to extensions.
    assert torch_available, f"Unable to pre-compile {op_name}, please first install torch"
    install_ops[op_name] = op_enabled(op_name)
    ext_modules.append(builder.builder())
"@

    $oldGitInfo = @"
compatible_ops = dict.fromkeys(ALL_OPS.keys(), False)
for op_name, builder in ALL_OPS.items():
    op_compatible = builder.is_compatible()
    compatible_ops[op_name] = op_compatible
    compatible_ops["deepspeed_not_implemented"] = False
"@

    $newGitInfo = @"
compatible_ops = dict.fromkeys(ALL_OPS.keys(), False)
for op_name, builder in ALL_OPS.items():
    try:
        op_compatible = builder.is_compatible()
    except Exception:
        op_compatible = False
    compatible_ops[op_name] = op_compatible
    compatible_ops["deepspeed_not_implemented"] = False
"@

    Replace-ExactBlock -Path $setupPath -Old $oldSetup -New $newSetup
    Replace-ExactBlock -Path $gitInfoPath -Old $oldGitInfo -New $newGitInfo
}

if ([string]::IsNullOrWhiteSpace($DeepSpeedSource)) {
    throw "Pass -DeepSpeedSource pointing to a DeepSpeed source tree."
}

if ([string]::IsNullOrWhiteSpace($PythonExe)) {
    $PythonExe = Join-Path $RepoRoot ".venv\Scripts\python.exe"
}
if ([string]::IsNullOrWhiteSpace($WorkDir)) {
    $WorkDir = Join-Path $RepoRoot ".vendor\deepspeed-src"
}
if ([string]::IsNullOrWhiteSpace($Wheelhouse)) {
    $Wheelhouse = Join-Path $RepoRoot ".wheelhouse"
}

Require-Path $RepoRoot "Repo root"
Require-Path $DeepSpeedSource "DeepSpeed source"
Require-Path $PythonExe "Python executable"

if ($Clean -and (Test-Path -LiteralPath $WorkDir)) {
    Remove-Item -LiteralPath $WorkDir -Recurse -Force
}

New-Item -ItemType Directory -Force -Path $WorkDir | Out-Null
New-Item -ItemType Directory -Force -Path $Wheelhouse | Out-Null

Write-Host "Copying DeepSpeed source into project-local workspace..."
Get-ChildItem -LiteralPath $DeepSpeedSource -Force | ForEach-Object {
    Copy-Item -LiteralPath $_.FullName -Destination $WorkDir -Recurse -Force
}

Write-Host "Patching DeepSpeed source for minimal Windows inference build..."
Patch-DeepSpeedSource -SourceRoot $WorkDir

$env:CUDA_HOME = if ($env:CUDA_PATH) { $env:CUDA_PATH } else { $env:CUDA_HOME }
$env:DISTUTILS_USE_SDK = "1"
$env:DS_SKIP_CUDA_CHECK = "1"
$env:DS_BUILD_OPS = "1"

$disabled = @(
    "DS_BUILD_AIO",
    "DS_BUILD_CPU_ADAGRAD",
    "DS_BUILD_CPU_ADAM",
    "DS_BUILD_CPU_LION",
    "DS_BUILD_CCL_COMM",
    "DS_BUILD_CUTLASS_OPS",
    "DS_BUILD_DEEP_COMPILE",
    "DS_BUILD_EVOFORMER_ATTN",
    "DS_BUILD_FP_QUANTIZER",
    "DS_BUILD_FUSED_ADAM",
    "DS_BUILD_FUSED_LAMB",
    "DS_BUILD_FUSED_LION",
    "DS_BUILD_GDS",
    "DS_BUILD_INFERENCE_CORE_OPS",
    "DS_BUILD_QUANTIZER",
    "DS_BUILD_RAGGED_DEVICE_OPS",
    "DS_BUILD_RAGGED_OPS",
    "DS_BUILD_RANDOM_LTD",
    "DS_BUILD_SHM_COMM",
    "DS_BUILD_SPARSE_ATTN",
    "DS_BUILD_SPATIAL_INFERENCE",
    "DS_BUILD_STOCHASTIC_TRANSFORMER",
    "DS_BUILD_TRANSFORMER",
    "DS_BUILD_UTILS"
)

foreach ($name in $disabled) {
    Set-Item -Path ("Env:\" + $name) -Value "0"
}

$env:DS_BUILD_TRANSFORMER_INFERENCE = "1"

if (-not $env:TORCH_CUDA_ARCH_LIST) {
    $env:TORCH_CUDA_ARCH_LIST = "6.1;7.0;7.5;8.0;8.6;8.9;9.0;12.0+PTX"
}

Push-Location $WorkDir
try {
    Write-Host "Building project-local DeepSpeed wheel..."
    & $PythonExe -m build --wheel --no-isolation
    Get-ChildItem -LiteralPath (Join-Path $WorkDir "dist") -Filter *.whl | ForEach-Object {
        Copy-Item -LiteralPath $_.FullName -Destination $Wheelhouse -Force
    }
}
finally {
    Pop-Location
}

Write-Host ""
Write-Host "DeepSpeed wheel build finished."
Write-Host "Wheelhouse: $Wheelhouse"
