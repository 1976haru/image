@echo off
setlocal EnableExtensions
title CoverMorph Studio v0.5.3 - EXE Build
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" goto no_venv
".venv\Scripts\python.exe" -c "import sys; raise SystemExit(0 if (3, 11) <= sys.version_info[:2] <= (3, 12) else 1)"
if errorlevel 1 goto unsupported_python

echo [1/3] Installing build packages...
".venv\Scripts\python.exe" -m pip install -r "requirements_build.txt"
if errorlevel 1 goto build_error

echo [2/3] Building EXE...
".venv\Scripts\pyinstaller.exe" --noconfirm --clean --windowed --onedir --name "CoverMorphStudio_v0.5.3" --collect-all "customtkinter" --collect-all "tkinterdnd2" --add-data "presets;presets" --add-data "hub_manifest.json;." --add-data "tools\README.txt;tools" "app.py"
if errorlevel 1 goto build_error

echo [3/3] Done.
echo SUCCESS: Check "dist\CoverMorphStudio_v0.5.3"
echo Optional AI models stay external. Put Real-ESRGAN files in the EXE folder's tools directory.
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

:build_error
echo ERROR: EXE build failed. Review the messages above.
pause
exit /b 1
