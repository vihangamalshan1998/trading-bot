@echo off
setlocal
title AI Trading Bot - SSH Key Setup

echo ===================================================
echo 🔑 AI TRADING BOT - SSH KEY SETUP
echo ===================================================
echo.
echo This script will generate a secure SSH key on your laptop
echo and install it onto your Hostinger VPS. 
echo.
echo This allows PyTorch to automatically upload the trained
echo AI Brains (.pt files) in the background without freezing!
echo.

set KEY_FILE=%USERPROFILE%\.ssh\id_ed25519
set VPS_USER=root
set VPS_IP=72.62.255.1

:: Step 1: Generate SSH Key if it doesn't exist
if not exist "%KEY_FILE%" (
    echo [1] Generating new SSH Key...
    ssh-keygen -t ed25519 -N "" -f "%KEY_FILE%"
) else (
    echo [1] SSH Key already exists on this laptop.
)

echo.
echo [2] Installing Key to your VPS...
echo ⚠️  IMPORTANT: You will be asked for your Hostinger VPS password ONE LAST TIME!
echo.

:: Step 2: Pipe the public key directly into the VPS authorized_keys file
type "%KEY_FILE%.pub" | ssh %VPS_USER%@%VPS_IP% "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 700 ~/.ssh && chmod 600 ~/.ssh/authorized_keys"

echo.
echo [3] Testing passwordless connection...
ssh -o BatchMode=yes %VPS_USER%@%VPS_IP% "echo ✔️ SUCCESS! Passwordless SSH is fully working!"

echo.
echo ===================================================
echo ✅ SETUP COMPLETE! 
echo Your MASTER_TRAINER.bat will now upload AI Brains automatically.
echo You may safely close this window.
echo ===================================================
pause
