@echo off
setlocal EnableExtensions
title CoverMorph Studio v1.0.0-rc1 - EXE Build
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" goto no_venv
".venv\Scripts\python.exe" -c "import sys; raise SystemExit(0 if (3, 11) <= sys.version_info[:2] <= (3, 12) else 1)"
if errorlevel 1 goto unsupported_python

echo [1/3] Installing build packages...
".venv\Scripts\python.exe" -m pip install -r "requirements_build.txt"
if errorlevel 1 goto build_error

echo [2/3] Building EXE...
".venv\Scripts\pyinstaller.exe" --noconfirm --clean --windowed --onedir --name "CoverMorphStudio" --collect-all "customtkinter" --collect-all "tkinterdnd2" --collect-data "cv2" --collect-submodules "diffusers" --hidden-import "diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl" --hidden-import "diffusers.schedulers.scheduling_euler_discrete" --hidden-import "transformers.models.clip.modeling_clip" --hidden-import "transformers.models.clip.tokenization_clip" --hidden-import "transformers.models.clip.tokenization_clip_fast" --hidden-import "transformers.models.clip.image_processing_clip" --copy-metadata "torch" --copy-metadata "diffusers" --copy-metadata "transformers" --copy-metadata "accelerate" --copy-metadata "safetensors" --copy-metadata "huggingface_hub" --copy-metadata "tokenizers" --copy-metadata "tqdm" --copy-metadata "regex" --copy-metadata "requests" --copy-metadata "packaging" --copy-metadata "filelock" --copy-metadata "numpy" --copy-metadata "pyyaml" --copy-metadata "pillow" --add-data "presets;presets" --add-data "scripts;scripts" --add-data "hub_manifest.json;." --add-data "tools\README.txt;tools" "app.py"
if errorlevel 1 goto build_error

echo [3/3] Done.
echo SUCCESS: Check "dist\CoverMorphStudio"
echo Optional AI models stay external: put them in the EXE folder's models directory or set COVERMORPH_MODELS_DIR.
echo Headless thumbnail bridge: CoverMorphStudio.exe --thumbnail-bridge-json  (JSON on stdin/stdout)
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
