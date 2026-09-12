@echo off
setlocal EnableExtensions
title CoverMorph Studio v0.5.4 - LaMa Install
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" goto no_venv
".venv\Scripts\python.exe" -c "import sys; raise SystemExit(0 if (3, 11) <= sys.version_info[:2] <= (3, 12) else 1)"
if errorlevel 1 goto unsupported_python

echo Creating isolated LaMa environment...
if not exist ".venv_lama\Scripts\python.exe" ".venv\Scripts\python.exe" -m venv ".venv_lama"
if errorlevel 1 goto install_error
if exist ".venv_lama\READY" del ".venv_lama\READY"
".venv_lama\Scripts\python.exe" -m pip install "simple-lama-inpainting==0.1.2"
if errorlevel 1 goto install_error
".venv_lama\Scripts\python.exe" -m pip check
if errorlevel 1 goto install_error
".venv_lama\Scripts\python.exe" -c "from simple_lama_inpainting import SimpleLama; from PIL import Image; SimpleLama()(Image.new('RGB',(64,64)), Image.new('L',(64,64),255))"
if errorlevel 1 goto install_error
echo ready> ".venv_lama\READY"

echo SUCCESS: Isolated LaMa installation and smoke test completed.
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
