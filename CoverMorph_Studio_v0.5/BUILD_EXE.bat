@echo off
setlocal
title CoverMorph Studio v0.5 - EXE Build
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" goto no_venv
".venv\Scripts\python.exe" -m pip install -r "requirements_build.txt"
if errorlevel 1 goto build_error
".venv\Scripts\pyinstaller.exe" --noconfirm --clean --windowed --onedir --name "CoverMorphStudio_v0.5" --collect-all customtkinter --add-data "presets;presets" --add-data "tools;tools" "app.py"
if errorlevel 1 goto build_error
echo SUCCESS: Check dist\CoverMorphStudio_v0.5
pause
exit /b 0
:no_venv
echo ERROR: Run INSTALL.bat first.
pause
exit /b 1
:build_error
echo ERROR: EXE build failed. Review the messages above.
pause
exit /b 1
