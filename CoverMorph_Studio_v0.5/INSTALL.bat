@echo off
setlocal
title CoverMorph Studio v0.5 - Base Install
cd /d "%~dp0"
echo [1/4] Checking Python...
where py >nul 2>nul
if not errorlevel 1 goto use_py
where python >nul 2>nul
if not errorlevel 1 goto use_python
echo ERROR: Python was not found.
echo Install Python 3.11 from https://www.python.org/downloads/release/python-3119/
pause
exit /b 1
:use_py
py -3.11 -m venv ".venv"
if errorlevel 1 py -3 -m venv ".venv"
goto venv_done
:use_python
python -m venv ".venv"
:venv_done
if not exist ".venv\Scripts\python.exe" goto venv_error
echo [2/4] Updating pip...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto install_error
echo [3/4] Installing base packages...
".venv\Scripts\python.exe" -m pip install -r "requirements.txt"
if errorlevel 1 goto install_error
echo [4/4] Verifying installation...
".venv\Scripts\python.exe" -c "import PIL, numpy, cv2, customtkinter; print('Base packages OK')"
if errorlevel 1 goto install_error
echo SUCCESS: Base installation completed.
echo Run RUN.bat to start CoverMorph Studio.
pause
exit /b 0
:venv_error
echo ERROR: Could not create the Python virtual environment.
pause
exit /b 1
:install_error
echo ERROR: Package installation failed. Review the messages above.
pause
exit /b 1
