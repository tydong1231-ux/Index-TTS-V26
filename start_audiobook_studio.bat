@echo off
setlocal
cd /d "%~dp0app"
if not exist "..\frontend-prototype\index.html" (
  echo Frontend files are missing.
  exit /b 1
)
python -m audiobook_server.main
endlocal
