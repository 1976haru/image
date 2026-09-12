@echo off
setlocal
title CoverMorph Studio v0.5 - SDXL Outpaint Install
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" goto no_venv
echo [1/3] Detecting NVIDIA GPU...
where nvidia-smi >nul 2>nul
if errorlevel 1 goto install_cpu
echo NVIDIA GPU found. Installing CUDA PyTorch...
".venv\Scripts\python.exe" -m pip install torch torchvision --index-url "https://download.pytorch.org/whl/cu128"
if errorlevel 1 goto install_cpu
goto install_ai
:install_cpu
echo Installing CPU PyTorch...
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
:install_error
echo ERROR: AI package installation failed. Review the messages above.
pause
exit /b 1
