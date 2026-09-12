@echo off
setlocal EnableExtensions
title CoverMorph Studio v0.5.1 - Tests
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" goto no_venv
".venv\Scripts\python.exe" -c "import sys; raise SystemExit(0 if (3, 11) <= sys.version_info[:2] <= (3, 12) else 1)"
if errorlevel 1 goto unsupported_python

echo [1/4] Installing test requirements...
".venv\Scripts\python.exe" -m pip install -r "requirements.txt"
if errorlevel 1 goto test_error

echo [2/4] Running pip check...
".venv\Scripts\python.exe" -m pip check
if errorlevel 1 goto test_error

echo [3/4] Running Ruff...
".venv\Scripts\python.exe" -m ruff check "."
if errorlevel 1 goto test_error

echo [4/4] Running pytest...
".venv\Scripts\python.exe" -m pytest
if errorlevel 1 goto test_error

echo SUCCESS: All tests passed.
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

:test_error
echo ERROR: Tests failed. Review the messages above.
pause
exit /b 1
