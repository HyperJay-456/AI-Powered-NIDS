@echo off
setlocal enabledelayedexpansion

:: ============================================================================
:: AI-Powered Network Intrusion Detection System (NIDS) - Shutdown Utility
:: ============================================================================

title AI-Powered NIDS - Server Shutdown
cd /d "%~dp0"

echo ============================================================================
echo   AI-POWERED NETWORK INTRUSION DETECTION SYSTEM (NIDS)
echo   Local Dashboard Shutdown Utility
echo ============================================================================
echo.

set KILLED_ANY=0

:: Check for processes listening on port 8501
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8501"') do (
    set TARGET_PID=%%a
    if not "!TARGET_PID!"=="" if not "!TARGET_PID!"=="0" (
        echo [INFO] Detected active NIDS server process [PID: !TARGET_PID!] on port 8501.
        echo [INFO] Terminating process tree for PID !TARGET_PID!...
        taskkill /F /T /PID !TARGET_PID! >nul 2>&1
        if !errorlevel! equ 0 (
            echo [SUCCESS] Terminated PID !TARGET_PID! successfully.
            set KILLED_ANY=1
        ) else (
            echo [WARNING] Failed to terminate PID !TARGET_PID!. You may need Administrator privileges.
        )
    )
)

if "!KILLED_ANY!"=="0" (
    echo [INFO] No active NIDS server process detected on port 8501.
    echo [INFO] Server is already stopped.
) else (
    echo.
    echo [SUCCESS] AI-Powered NIDS server has been completely shut down.
)

echo.
ping 127.0.0.1 -n 3 >nul
exit /b 0
