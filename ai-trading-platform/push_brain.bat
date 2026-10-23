@echo off
echo Uploading new AI Brain to VPS...

REM Upload into a temporary folder first, then move into place with ONE command: version.json (trade policy)
REM before model_v1.pt. The bot hot-reloads when model_v1.pt changes, so it never reads a half-uploaded brain
REM or a new brain with the old policy (same order as push_champion_to_vps in apps/trainer/ppo.py).
set VPS=root@72.62.255.1
set DIR=/var/www/trading-bot/ai-trading-platform/models/production

ssh %VPS% "mkdir -p %DIR%/.upload" || goto failed
scp models\production\version.json models\production\tournament_history.json models\production\model_v1.pt %VPS%:%DIR%/.upload/ || goto failed
ssh %VPS% "cd %DIR% && mv .upload/version.json version.json && mv .upload/tournament_history.json tournament_history.json && mv .upload/model_v1.pt model_v1.pt" || goto failed

echo.
echo Brain successfully uploaded! The VPS Trading Bot will auto-reload it on the next tick.
pause
exit /b 0

:failed
echo.
echo UPLOAD FAILED: the VPS keeps its previous brain. Check the SSH connection and try again.
pause
exit /b 1
