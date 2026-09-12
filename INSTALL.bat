@echo off
setlocal EnableExtensions
title CoverMorph Studio v0.5.2 - Base Install
cd /d "%~dp0"

echo [1/5] Locating Python 3.11 or 3.12...
set "PYTHON_CMD="
py -3.11 -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)" >nul 2>nul
if not errorlevel 1 set "PYTHON_CMD=py -3.11"
if not defined PYTHON_CMD (
    py -3.12 -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)" >nul 2>nul
    if not errorlevel 1 set "PYTHON_CMD=py -3.12"
)
if not defined PYTHON_CMD (
    python -c "import sys; raise SystemExit(0 if (3, 11) <= sys.version_info[:2] <= (3, 12) else 1)" >nul 2>nul
    if not errorlevel 1 set "PYTHON_CMD=python"
)
if not defined PYTHON_CMD goto unsupported_python

%PYTHON_CMD% --version
if errorlevel 1 goto install_error

echo [2/5] Creating virtual environment...
%PYTHON_CMD% -m venv ".venv"
if errorlevel 1 goto venv_error
if not exist ".venv\Scripts\python.exe" goto venv_error

echo [3/5] Updating pip...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto install_error

echo [4/5] Installing base packages...
".venv\Scripts\python.exe" -m pip install -r "requirements.txt"
if errorlevel 1 goto install_error

echo [5/5] Verifying installation...
".venv\Scripts\python.exe" -c "import PIL, numpy, cv2, customtkinter; print('Base packages OK')"
if errorlevel 1 goto install_error
".venv\Scripts\python.exe" -m pip check
if errorlevel 1 goto install_error

echo SUCCESS: Base installation completed.
echo Run RUN.bat to start CoverMorph Studio.
pause
exit /b 0

:unsupported_python
echo ERROR: Python 3.11 or 3.12 was not found.
echo Install Python 3.11 or 3.12 from https://www.python.org/downloads/
pause
exit /b 1

:venv_error
echo ERROR: Could not create the Python virtual environment.
pause
exit /b 1

:install_error
echo ERROR: Installation failed. Review the messages above.
pause
exit /b 1
