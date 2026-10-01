@echo off
echo ===================================
echo AI Database Auto-Syncer
echo ===================================

echo [1] Opening secure SSH Tunnel to VPS Database (Port 3307)...
REM The -N flag means do not open a shell, just forward the port.
REM The /b flag runs it silently in the background.
start /b ssh -N -L 3307:127.0.0.1:3306 root@72.62.255.1

echo [2] Installing PyMySQL just in case...
pip install pymysql > nul 2>&1

echo [3] Starting Python Auto-Syncer!
echo Make sure Laragon MySQL is running.
python scripts\sync_db.py

pause
