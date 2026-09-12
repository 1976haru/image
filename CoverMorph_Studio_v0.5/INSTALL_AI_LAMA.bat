@echo off
setlocal
title CoverMorph Studio v0.5 - LaMa Install
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" goto no_venv
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
:install_error
echo ERROR: LaMa installation failed. Review the messages above.
pause
exit /b 1
