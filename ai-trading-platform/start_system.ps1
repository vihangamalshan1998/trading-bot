# Launch script for the AI Trading Platform
# Starts the Data Collector and the Trading Bot

Write-Host "Starting AI Trading Platform..." -ForegroundColor Cyan

# Ensure virtual environment is activated
if (-not (Test-Path "venv\Scripts\Activate.ps1")) {
    Write-Host "Virtual environment not found! Please create it and run 'pip install -r requirements.txt'." -ForegroundColor Red
    exit
}

# Start Market Collector in a new PowerShell window
Write-Host "Launching Market Collector Service..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", ".\venv\Scripts\Activate.ps1; python apps/market_collector/main.py"

# Wait a few seconds to let Redis and REST APIs initialize
Start-Sleep -Seconds 5

# Start Trading Bot in a new PowerShell window
Write-Host "Launching Trading Bot Service..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", ".\venv\Scripts\Activate.ps1; python apps/trading_bot/main.py"

Write-Host "Both services have been launched in separate windows." -ForegroundColor Cyan
Write-Host "Press Ctrl+C in those windows to stop them."
