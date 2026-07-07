@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    py -3 -m venv .venv
    if errorlevel 1 exit /b %errorlevel%
    ".venv\Scripts\python.exe" -m pip install -e .
    if errorlevel 1 exit /b %errorlevel%
)

".venv\Scripts\localvqgan.exe"
if errorlevel 1 (
    ".venv\Scripts\python.exe" -m localvqgan
)
