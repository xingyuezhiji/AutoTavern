@echo off
call "%~dp0run-windows.cmd" doctor %*
exit /b %errorlevel%
