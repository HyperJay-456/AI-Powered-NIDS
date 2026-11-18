@echo off
setlocal enabledelayedexpansion

:: ============================================================================
:: AI-Powered Network Intrusion Detection System (NIDS) - Windows Launcher
:: ============================================================================

title AI-Powered NIDS - Server Launcher
cd /d "%~dp0"

echo ============================================================================
echo   AI-POWERED NETWORK INTRUSION DETECTION SYSTEM (NIDS)
echo   Local Dashboard Launcher [Streamlit]
echo ============================================================================
echo.

:: 1. Validate Python virtual environment presence
if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not detected at .venv\
    echo.
    echo Please initialize the environment before launching:
    echo   1. py -3.13 -m venv .venv
    echo   2. .venv\Scripts\activate
    echo   3. pip install -r requirements.txt
    echo.
    echo Press any key to exit...
    pause >nul
    exit /b 1
)

:: 2. Check for optional foreground execution flag (-f or --foreground)
if "%1"=="-f" goto RUN_FOREGROUND
if "%1"=="--foreground" goto RUN_FOREGROUND

:: 3. Check if server is already running on port 8501
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8501"') do (
    set EXISTING_PID=%%a
    if not "!EXISTING_PID!"=="" if not "!EXISTING_PID!"=="0" (
        echo [INFO] NIDS server is ALREADY ACTIVE on port 8501 [PID: !EXISTING_PID!].
        echo [INFO] Launching default browser at http://localhost:8501 ...
        start http://localhost:8501
        echo.
        echo To stop the server at any time, run: stop.bat
        echo.
        ping 127.0.0.1 -n 3 >nul
        exit /b 0
    )
)

:: 4. Launch Streamlit server in a dedicated window
echo [INFO] Starting Streamlit server on http://localhost:8501 ...
start "AI-Powered NIDS - Server Console" .venv\Scripts\python.exe -m streamlit run app.py --server.port 8501 --server.headless false

echo [INFO] Waiting for server initialization...
ping 127.0.0.1 -n 3 >nul

echo [SUCCESS] Server process initiated.
echo [INFO] Web interface accessible at: http://localhost:8501
echo [INFO] Server console is running in: "AI-Powered NIDS - Server Console"
echo.
echo ============================================================================
echo   To stop the server at any time, run: stop.bat
echo ============================================================================
echo.
ping 127.0.0.1 -n 4 >nul
exit /b 0

:RUN_FOREGROUND
echo [INFO] Starting Streamlit server in foreground mode...
echo [INFO] Press Ctrl+C in this console to stop the server.
echo.
.venv\Scripts\python.exe -m streamlit run app.py --server.port 8501 --server.headless false
exit /b %errorlevel%
