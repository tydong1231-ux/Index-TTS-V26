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
powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%\tools\windows_local\launch_official_latest_webui.ps1" -UpstreamRepo "%ROOT%\app" -RuntimeRoot "%ROOT%" -DataRoot "%ROOT%\data" -PythonExe "%ROOT%\runtime\python.exe" -Mode normal -Fp16
pause
endlocal
