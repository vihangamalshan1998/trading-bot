# Binance Futures Testnet Operational Runbook

This operational runbook provides standard operating procedures (SOPs) for safely starting, monitoring, maintaining, and recovering the Binance Futures AI Trading Platform on the Testnet environment.

---

## 1. Safety Configuration Checks (Pre-Flight)

Before launching any service, verify that safe defaults are strictly active in `.env` and `core/config/settings.py`:

| Parameter | Required Safe Value | Purpose |
| :--- | :--- | :--- |
| `ALLOW_LIVE_TRADING` | `False` | **CRITICAL**: Completely blocks real-money exchange routing. |
| `ALLOW_TESTNET` | `True` (for Testnet) | Restricts REST & WebSocket routing to `testnet.binancefuture.com`. |
| `TRADING_ENABLED` | `False` (initial) | Ensures bot boots in read-only observation mode. |
| `DRY_RUN` | `True` (initial) | Simulates order placements without sending live API requests. |
| `EMERGENCY_STOP` | `False` (when running) | Set to `True` immediately to freeze all new orders. |

### Configuration Audit Command:
```powershell
python -c "from core.config.settings import settings; print('Testnet:', settings.allow_testnet, '| Live:', settings.allow_live_trading, '| DryRun:', settings.dry_run)"
```

---

## 2. Startup Procedures

### Local Development / Verification:
```powershell
# 1. Activate virtual environment
cd d:\laragon\www\Traning-bot\ai-trading-platform
.\venv\Scripts\Activate.ps1

# 2. Run full test verification suite (must pass 49/49)
python -m pytest tests/ -v

# 3. Start local trading bot in dry-run mode
python -m apps.trading_bot.main
```

### VPS Production Deployment (`72.62.255.1`):
```bash
# 1. SSH into VPS
ssh root@72.62.255.1

# 2. Navigate to project root and pull latest audited commits
cd /root/ai-trading-platform
git pull origin main

# 3. Restart PM2 processes safely
pm2 restart trading-bot
pm2 restart statarb-collector
pm2 restart whale-tracker

# 4. Confirm process status
pm2 status
```

---

## 3. Real-Time Health Checks & Observability

### A. PM2 Process Health:
```bash
pm2 status
pm2 logs trading-bot --lines 50
```

### B. Redis IPC & Streaming Channels:
```bash
redis-cli ping
# Verify active market ticker streams
redis-cli pubsub channels "market:*"
# Verify trade history length
redis-cli llen "dashboard:trade_history"
```

### C. MySQL Persistence Check:
```bash
mysql -u root -p -e "SELECT count(*) FROM trading_platform.experiences;"
mysql -u root -p -e "SELECT symbol, side, realized_pnl, created_at FROM trading_platform.trades ORDER BY id DESC LIMIT 5;"
```

### D. Model Tournament Log:
```bash
cat models/production/version.json
tail -n 25 models/production/tournament_history.json
```

---

## 4. Emergency Stop & Graceful Shutdown

### Emergency Stop (Instant Freeze of New Orders):
To halt new trading without killing the process:
```bash
# Via Redis kill switch
redis-cli set "kill_switch" "1"
```
Or set `emergency_stop = True` in settings / dashboard. The bot will:
1. Immediately cease all new order submissions.
2. Maintain active trailing stop-loss and take-profit management on existing open positions.

### Panic Market Close (Flatten All Open Positions):
```bash
redis-cli set "panic_flatten" "1"
```

### Graceful Process Shutdown:
```bash
pm2 stop trading-bot
# Verify no lingering zombie threads
ps aux | grep python
```

---

## 5. Rollback Procedure (Model Emergency)

If a newly promoted champion demonstrates unexpected behavior:
```python
from core.ai.registry import ModelRegistry

registry = ModelRegistry(model_dir="models/production")
success = registry.rollback()
if success:
    print("Successfully restored champion from model_v1_backup.pt!")
```
Or via terminal:
```bash
cp models/production/model_v1_backup.pt models/production/model_v1.pt
pm2 restart trading-bot
```

---

## 6. Testing & Regression Commands

### Run Full Test Suite:
```bash
python -m pytest tests/ -v
```

### Run Feature Engine & 200-Dim Parity Test:
```bash
python -m pytest tests/test_performance_intelligence_upgrade.py -k "feature_engine" -v
```

### Run Champion Safety & Promotion Gate Tests:
```bash
python -m pytest tests/test_performance_intelligence_upgrade.py -k "champion or candidate" -v
```

### Run Walk-Forward Research Simulation:
```bash
python apps/research/forward_test.py
```
