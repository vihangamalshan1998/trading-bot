@echo off
echo Uploading new AI Brain to VPS...

scp models\production\model_v1.pt root@72.62.255.1:/var/www/trading-bot/ai-trading-platform/models/production/model_v1.pt
scp models\production\version.json root@72.62.255.1:/var/www/trading-bot/ai-trading-platform/models/production/version.json
scp models\production\tournament_history.json root@72.62.255.1:/var/www/trading-bot/ai-trading-platform/models/production/tournament_history.json

echo.
echo Brain successfully uploaded! The VPS Trading Bot will auto-reload it on the next tick.
pause
