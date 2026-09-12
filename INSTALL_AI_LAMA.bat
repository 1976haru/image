@echo off
setlocal EnableExtensions
title CoverMorph Studio v0.5.1 - LaMa Install
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" goto no_venv
".venv\Scripts\python.exe" -c "import sys; raise SystemExit(0 if (3, 11) <= sys.version_info[:2] <= (3, 12) else 1)"
if errorlevel 1 goto unsupported_python

echo Installing LaMa inpainting packages...
".venv\Scripts\python.exe" -m pip install -r "requirements_ai_lama.txt"
if errorlevel 1 goto install_error

echo SUCCESS: LaMa installation completed.
pause
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

:install_error
echo ERROR: LaMa installation failed. Review the messages above.
pause
exit /b 1
