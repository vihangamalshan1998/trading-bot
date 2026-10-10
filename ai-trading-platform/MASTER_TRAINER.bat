@echo off
echo ===================================================
echo 🧠 AI MASTER TRAINER (LOCAL LAPTOP -^> LIVE VPS)
echo ===================================================

echo [1] Opening Auto-Reconnecting SSH Tunnel to VPS Database and Redis...
start "SSH Tunnel" /min cmd /c "for /l %%x in () do (ssh -N -L 3307:127.0.0.1:3306 -L 6379:127.0.0.1:6379 -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -o ExitOnForwardFailure=yes root@72.62.255.1 & echo SSH dropped! Reconnecting... & timeout /t 3)"

echo [2] Checking for Python Virtual Environment...
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
    echo -^> Virtual Environment Activated!
) else (
    echo -^> No 'venv' folder found. Using Global Python.
)

echo [3] Launching AI Trainer at Maximum Power...
python -m apps.trainer.ppo

pause
