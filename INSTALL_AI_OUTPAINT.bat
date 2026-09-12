@echo off
setlocal EnableExtensions
title CoverMorph Studio v0.5.2 - SDXL Outpaint Install
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" goto no_venv
".venv\Scripts\python.exe" -c "import sys; raise SystemExit(0 if (3, 11) <= sys.version_info[:2] <= (3, 12) else 1)"
if errorlevel 1 goto unsupported_python

echo [1/3] Detecting NVIDIA GPU...
where nvidia-smi >nul 2>nul
if errorlevel 1 goto install_cpu

echo NVIDIA GPU found. Installing CUDA PyTorch...
".venv\Scripts\python.exe" -m pip install torch torchvision --index-url "https://download.pytorch.org/whl/cu128"
if errorlevel 1 goto install_error
goto install_ai

:install_cpu
echo NVIDIA GPU was not found. Installing CPU PyTorch...
".venv\Scripts\python.exe" -m pip install torch torchvision --index-url "https://download.pytorch.org/whl/cpu"
if errorlevel 1 goto install_error

:install_ai
echo [2/3] Installing SDXL and person segmentation packages...
".venv\Scripts\python.exe" -m pip install -r "requirements_ai_outpaint.txt"
if errorlevel 1 goto install_error

echo [3/3] Verifying AI packages...
".venv\Scripts\python.exe" -c "import torch, diffusers, transformers, rembg; print('AI packages OK'); print('CUDA:', torch.cuda.is_available())"
if errorlevel 1 goto install_error

echo SUCCESS: SDXL outpainting installation completed.
echo The SDXL model downloads automatically on first use.
pause
exit /b 0

:no_venv
echo ERROR: Run INSTALL.bat first, then run this file again.
pause
exit /b 1

:unsupported_python
echo ERROR: This virtual environment uses an unsupported Python version.
echo Recreate .venv with Python 3.11 or 3.12 by running INSTALL.bat.
pause
exit /b 1

:install_error
echo ERROR: AI package installation failed. Review the messages above.
pause
exit /b 1
