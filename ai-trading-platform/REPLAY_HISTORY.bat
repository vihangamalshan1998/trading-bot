@echo off
title REPLAY HISTORY - building training data from Binance history
cd /d "%~dp0"

echo ================================================================
echo   REPLAY HISTORY
echo   Builds training records from real Binance history, newest week
echo   first. Close this window any time; run it again to continue
echo   where it stopped (finished weeks are skipped).
echo   Needs Laragon MySQL running.
echo ================================================================
echo.

if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else (
    echo No 'venv' folder found. Using Global Python.
)
set PYTHONPATH=%~dp0

REM --days: how much history (ending 2 days ago). --workers: CPU cores used (lower = cooler laptop).
python -m apps.replay.batch --days 90 --workers 4

echo.
pause
