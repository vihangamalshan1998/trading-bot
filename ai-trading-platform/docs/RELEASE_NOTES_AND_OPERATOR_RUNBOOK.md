# Release Notes & Operator Runbook: Verified Trading Platform Upgrade

## 1. Executive Summary & Root Causes Fixed

This release closes critical architectural and safety gaps across the AI trading platform:

| Component | Previous Behavior (Vulnerability) | Verified Fix (Hardened Gate) |
|---|---|---|
| **Model Promotion Gate** (`core/ai/registry.py`) | Permissive promotion allowed `save_model()` or `promote_candidate()` to overwrite the production champion without explicit tournament approval or metrics evidence. | **Strict Fail-Closed Promotion Gate**: Explicitly requires `metrics['approved'] is True`, validates all metrics are finite, verifies architecture compatibility (`SingleSymbolActorCritic(input_dim=200)`), checks for NaN/Inf weights, enforces a concurrency lock (`promotion.lock`), and backs up the champion before atomic replacement. |
| **Exchange Contract Validation** (`apps/trading_bot/main.py`) | When Binance Futures metadata was unavailable or empty, it fell back to accepting symbols if present in `settings.symbol_universe`. | **Fail-Closed Contract Validation**: Permissive fallback completely removed. New orders fail closed (`return False`) if exchange metadata is empty, unverified, delisted, non-USDT, or non-futures. |
| **Risk Defaults** (`core/config/settings.py`) | Unbounded defaults (`max_position_size=9999999.0`, `max_order_size=9999999.0`). | **Conservative Testnet Limits**: Calibrated to `max_position_size=50000.0` and `max_order_size=25000.0`. Safe startup defaults preserved (`trading_enabled=False`, `dry_run=True`, `allow_testnet=False`, `allow_live_trading=False`, `emergency_stop=True`). |
| **Offline/Online Training Pipeline** (`apps/trainer/ppo.py`) | Direct calls to `save_model()` could overwrite the champion model. | **Routed via `promote_candidate()`**: Candidate checkpoints are saved to `models/candidates/` and promoted only after passing `evaluate_candidate()` tournament benchmarks. |
| **Test Suite Resilience** (`tests/`) | Missing regression tests for promotion rejection gates and contract validation. | **49/49 Tests Passing**: Comprehensive tests added covering all promotion gates, empty registry fail-closed, NaN/Inf rejection, and dimension compatibility. |

---

## 2. Updated Configuration Guide (`core/config/settings.py`)

All settings default to maximum safety:
```ini
# Core Safety Defaults (Must be explicitly enabled for trading)
TRADING_ENABLED=false
DRY_RUN=true
ALLOW_TESTNET=false
ALLOW_LIVE_TRADING=false
EMERGENCY_STOP=true

# Calibrated Risk Limits
MAX_POSITION_SIZE=50000.0
MAX_ORDER_SIZE=25000.0
MAX_LEVERAGE=20
MAX_SYMBOL_EXPOSURE_PCT=5.0
MAX_PORTFOLIO_EXPOSURE_PCT=5.0
MAX_POSITION_MARGIN_PCT=0.25
MAX_DAILY_LOSS_PCT=0.05
MAX_DRAWDOWN_PCT=0.08
MAX_MARKET_DATA_AGE_SECONDS=60.0
CORRELATED_EXPOSURE_LIMIT_PCT=5.0
```

---

## 3. Step-by-Step Deployment Procedure

### Local Machine (Commit & Push)
```bash
cd d:\laragon\www\Traning-bot\ai-trading-platform
git add core/ai/registry.py apps/trading_bot/main.py apps/trainer/ppo.py core/config/settings.py tests/test_performance_intelligence_upgrade.py docs/RELEASE_NOTES_AND_OPERATOR_RUNBOOK.md
git commit -m "fix(core): enforce strict promotion gate, fail-closed contract validation, and conservative risk bounds"
git push origin main
```

### Remote VPS (`72.62.255.1`)
```bash
ssh root@72.62.255.1
cd /var/www/trading-bot/ai-trading-platform

# 1. Pull latest verified changes
git pull origin main

# 2. Run automated test suite to confirm VPS environment parity
venv/bin/pytest tests/ -v

# 3. Gracefully reload the trading bot
pm2 restart trading-bot

# 4. Confirm startup logs and health status
pm2 logs trading-bot --lines 50
```

---

## 4. Operator Runbook for Daily Operation

### Health Checks
- **Check Bot Process Status**:
  ```bash
  pm2 status
  ```
- **Inspect Live Logs**:
  ```bash
  pm2 logs trading-bot --raw
  ```
- **Verify Active Champion Model**:
  ```bash
  cat models/production/version.json
  ```
- **Verify Tournament History**:
  ```bash
  tail -n 20 models/production/tournament_history.json
  ```

### Managing Active Positions
- If a coin is excluded from the trading universe (e.g. `SUIUSDT`), the bot continues to reconcile and monitor open positions until closed by AI policy or stop-loss. It strictly blocks new entries.

---

## 5. Emergency Rollback Procedure

If unexpected behavior or degradation is observed in production:

### 1. Trigger Emergency Stop Immediately (via Redis or CLI)
```bash
# Via Redis CLI
redis-cli set EMERGENCY_STOP "true"

# Or via PM2
pm2 stop trading-bot
```

### 2. Roll Back Champion Model to Previous Backup
```bash
python -c "from core.ai.registry import ModelRegistry; registry = ModelRegistry(); print('Rolled back:', registry.rollback())"
```

### 3. Verify Rollback and Restart
```bash
cat models/production/version.json
pm2 restart trading-bot
```
