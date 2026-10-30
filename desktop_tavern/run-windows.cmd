@echo off
setlocal
cd /d "%~dp0.."
if not exist ".venv-desktop\Scripts\python.exe" (
  echo Run desktop_tavern\setup-windows.cmd first.
  exit /b 2
)
if not defined TAVERN_CHECKPOINT set "TAVERN_CHECKPOINT=models\tavern_bc_policy.pt"
".venv-desktop\Scripts\python.exe" -m desktop_tavern --config desktop_tavern\config.windows.local.json --checkpoint "%TAVERN_CHECKPOINT%" %*
exit /b %errorlevel%
