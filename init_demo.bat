@echo off
setlocal
title CYBER-NIDS - Initialize Benchmark Demo

echo ====================================================================
echo   CYBER-NIDS: AI-Powered Network Intrusion Detection System
echo   Benchmark Demo Initializer (CP-01 to CP-12)
echo ====================================================================
echo.

cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    set "PYTHON_EXE=.venv\Scripts\python.exe"
) else (
    set "PYTHON_EXE=python"
)

echo [INFO] Using Python interpreter: %PYTHON_EXE%
echo [INFO] Generating benchmark sample dataset and executing end-to-end pipeline...
echo.

"%PYTHON_EXE%" -m src.data.demo_dataset

if %ERRORLEVEL% equ 0 (
    echo.
    echo ====================================================================
    echo   [SUCCESS] All pipeline stages initialized successfully!
    echo   You can now run start.bat to explore all 8 populated views.
    echo ====================================================================
) else (
    echo.
    echo [ERROR] Pipeline initialization encountered an error (Code: %ERRORLEVEL%).
)

echo.
pause
