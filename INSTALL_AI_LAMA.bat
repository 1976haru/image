@echo off
setlocal EnableExtensions
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" goto fail
if not exist ".venv_lama\Scripts\python.exe" ".venv\Scripts\python.exe" -m venv .venv_lama
if errorlevel 1 goto fail
if exist ".venv_lama\READY" del ".venv_lama\READY"
".venv_lama\Scripts\python.exe" -m pip install "simple-lama-inpainting==0.1.2"
if errorlevel 1 goto fail
".venv_lama\Scripts\python.exe" -m pip check
if errorlevel 1 goto fail
".venv_lama\Scripts\python.exe" -c "from simple_lama_inpainting import SimpleLama; from PIL import Image; SimpleLama()(Image.new('RGB',(64,64)), Image.new('L',(64,64),255))"
if errorlevel 1 goto fail
echo ready> ".venv_lama\READY"
echo SUCCESS: Isolated LaMa installed and inference verified.
pause
exit /b 0
:fail
echo ERROR: Installation or model test failed. Run INSTALL.bat first and check messages.
pause
exit /b 1
