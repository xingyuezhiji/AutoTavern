@echo off
setlocal
cd /d "%~dp0.."
if "%~1"=="" (
  echo Usage: start-windows.cmd "C:\path\tavern_bc_policy.pt" [--execute --allow-estimated-context]
  exit /b 2
)
if not exist ".venv-live\Scripts\python.exe" (
  echo Run live_tavern\setup-windows.cmd first.
  exit /b 2
)
if exist "live_tavern\STOP" (
  echo Remove live_tavern\STOP before starting again.
  exit /b 2
)
".venv-live\Scripts\python.exe" -m live_tavern --checkpoint "%~1" --output runs\live\decisions.jsonl watch %2 %3
endlocal
