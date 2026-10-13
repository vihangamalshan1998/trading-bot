@echo off
if exist "venv\Scripts\python.exe" (
    venv\Scripts\python.exe scripts\show_brain_progress.py
) else (
    python scripts\show_brain_progress.py
)
echo.
pause
