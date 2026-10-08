@echo off
echo ===================================================
echo 🌟 EXTRACTING GOLDEN BATCH TO PERMANENT VAULT 🌟
echo ===================================================
echo.
echo Activating Virtual Environment...
call venv\Scripts\activate.bat
echo.
echo Launching Extraction Script...
python scratch\extract_golden_batch.py
echo.
pause
