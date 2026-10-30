@echo off
setlocal
cd /d "%~dp0.."
if not exist ".venv-live\Scripts\python.exe" (
  py -3 -m venv .venv-live
  if errorlevel 1 exit /b 1
)
".venv-live\Scripts\python.exe" -m pip install -r live_tavern\requirements.txt
if errorlevel 1 exit /b 1
if not exist "live_tavern\config.local.json" copy "live_tavern\config.example.json" "live_tavern\config.local.json"
echo Edit live_tavern\config.local.json and calibrate your client before pilot execution.
endlocal
