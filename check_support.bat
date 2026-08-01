@echo off
setlocal
set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"
cd /d "%ROOT%"
set "HF_HOME=%ROOT%\data\hf_cache"
set "HF_HUB_CACHE=%ROOT%\data\hf_cache"
set "HUGGINGFACE_HUB_CACHE=%ROOT%\data\hf_cache"
set "TRANSFORMERS_CACHE="
powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%\tools\windows_local\run_release_support_check.ps1" -UpstreamRepo "%ROOT%\app" -RuntimeRoot "%ROOT%" -DataRoot "%ROOT%\data" -PythonExe "%ROOT%\runtime\python.exe" -SummaryOnly -RequireDeepSpeed
if "%~1"=="" pause
endlocal
