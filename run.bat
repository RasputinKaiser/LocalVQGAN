@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" goto :run

set "PYCMD="
for %%C in ("py -3.13" "py -3.12" "py -3.11" "py -3" "python") do (
    if not defined PYCMD (
        %%~C -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>&1
        if not errorlevel 1 set "PYCMD=%%~C"
    )
)

if not defined PYCMD (
    echo LocalVQGAN needs Python 3.11 or newer, and no matching interpreter was found.
    echo Install one from https://www.python.org/downloads/ ^(check "Add python.exe to PATH"
    echo during setup^), then run run.bat again.
    exit /b 1
)

for /f "delims=" %%V in ('%PYCMD% --version') do echo Using %%V
%PYCMD% -m venv .venv
if errorlevel 1 exit /b %errorlevel%

echo Installing LocalVQGAN and its dependencies (downloads ~1-2 GB, mostly PyTorch -- this can take a few minutes)...
".venv\Scripts\python.exe" -m pip install --quiet --upgrade pip
".venv\Scripts\python.exe" -m pip install -e .
if errorlevel 1 exit /b %errorlevel%

:run
".venv\Scripts\localvqgan.exe"
if errorlevel 1 (
    ".venv\Scripts\python.exe" -m localvqgan
)
