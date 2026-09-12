@echo off
setlocal
title CoverMorph Studio v0.5
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" goto no_venv
".venv\Scripts\python.exe" "app.py"
if errorlevel 1 goto run_error
exit /b 0
:no_venv
echo ERROR: Run INSTALL.bat first.
pause
exit /b 1
:run_error
echo ERROR: CoverMorph Studio stopped unexpectedly.
echo Check the logs folder for details.
pause
exit /b 1
