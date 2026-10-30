@echo off
setlocal
cd /d "%~dp0.."
if not exist ".venv-desktop\Scripts\python.exe" (
  py -3 -c "import sys; assert sys.version_info >= (3,10), 'Python 3.10+ required'"
  if errorlevel 1 exit /b 2
  py -3 -m venv .venv-desktop
  if errorlevel 1 exit /b 2
)
".venv-desktop\Scripts\python.exe" -m pip install -r desktop_tavern\requirements.txt
if errorlevel 1 exit /b 2
if not exist "desktop_tavern\config.windows.local.json" copy "desktop_tavern\config.windows.example.json" "desktop_tavern\config.windows.local.json"
echo Read docs\DESKTOP_V2.md. Configure account/log path, then calibrate this client.
endlocal
