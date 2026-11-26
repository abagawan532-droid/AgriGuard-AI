@echo off
title AgriGuard AI Platform
echo ======================================================================
echo                  AgriGuard AI Platform Starter
echo        200 Crops ^| 200 Tree Species ^| 200 Pathology Database
echo ======================================================================
echo.

cd /d "%~dp0"

IF EXIST ".venv\Scripts\python.exe" (
    echo [INFO] Found local virtual environment in .venv
    set "PYTHON_EXE=.venv\Scripts\python.exe"
) ELSE (
    echo [INFO] Using system Python
    set "PYTHON_EXE=python"
)

echo [INFO] Launching AgriGuard AI on http://127.0.0.1:5000 ...
start http://127.0.0.1:5000
%PYTHON_EXE% app.py

pause
