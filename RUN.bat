@echo off
setlocal EnableExtensions
title CoverMorph Studio v0.5.4
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" goto no_venv

".venv\Scripts\python.exe" -c "import sys; raise SystemExit(0 if (3, 11) <= sys.version_info[:2] <= (3, 12) else 1)"
if errorlevel 1 goto unsupported_python

".venv\Scripts\python.exe" "app.py"
if errorlevel 1 goto run_error
exit /b 0

:no_venv
echo ERROR: Run INSTALL.bat first.
pause
exit /b 1

:unsupported_python
echo ERROR: This virtual environment uses an unsupported Python version.
echo Recreate .venv with Python 3.11 or 3.12 by running INSTALL.bat.
pause
exit /b 1

:run_error
echo ERROR: CoverMorph Studio stopped unexpectedly.
echo Check the logs folder for details.
pause
exit /b 1
