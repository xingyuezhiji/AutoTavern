@echo off
call "%~dp0run-windows.cmd" reset %*
exit /b %errorlevel%
