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

## 4. Emergency Stop, Passive Airbags & Graceful Shutdown

### Emergency Stop & Market Flatten (Cancel All Orders & Close to Cash):
To halt trading and immediately market-close all positions to USDT cash:
```bash
# Primary Redis Key (Recommended)
redis-cli set "system:kill_switch:active" "true"

# Legacy Redis Key (Also supported)
redis-cli set "kill_switch" "1"
```
Or via **Telegram Command Center**: tap the inline button **🚨 EMERGENCY FLAT ALL** or send `/flat_all`.
The bot will:
1. Immediately cancel all resting limit orders on Binance.
2. Market-close all active positions with `reduce_only=True` to 100% USDT Cash.
3. Halt new order evaluations until resumed via `redis-cli set "system:kill_switch:active" "false"` or Telegram **▶️ Resume Bot**.

### Temporary Trading Pause (1 Hour):
To suspend new entries without closing open positions (preserving trailing stops):
```bash
# Via Telegram
/pause
```
Or via Redis: `redis-cli set "system:pause_until" "<future_unix_timestamp>"`.

### Passive "Airbag" Safeguards (Zero Frequency Reduction):
The platform runs 4 passive airbags that stay dormant during normal trading and deploy only during catastrophic events:
1. **Airbag 1 (Loss Velocity Breaker)**: $\ge 3$ stop-losses in $< 120$s freezes all new entries for 15 minutes.
2. **Airbag 2 (WebSocket Heartbeat Watchdog)**: If ticks stall for $> 5.0$s while resting orders exist, automatically cancels them to prevent stale fills.
3. **Airbag 3 (Slippage Anomaly Guard)**: If executed fill price deviates $> 0.75\%$ from mid-price, quarantines the symbol for 15 minutes.
4. **Airbag 4 (Margin Cushion Trim)**: If free margin ratio $< 15\%$, gracefully auto-trims 25% of the most underwater position to avoid exchange auto-liquidation.

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
