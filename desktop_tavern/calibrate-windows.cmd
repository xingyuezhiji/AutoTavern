@echo off
call "%~dp0run-windows.cmd" calibrate %*
exit /b %errorlevel%
