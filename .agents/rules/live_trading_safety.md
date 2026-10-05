# 🚨 LIVE TRADING SAFETY PROTOCOLS

This document outlines the mandatory safety mechanisms that **must** be implemented before the AI bot is allowed to trade with real funds on the Live VPS. 

AI models are powerful, but they can hallucinate or glitch during extreme market volatility (Regime Shifts). These hard-coded rules act as physical boundaries to prevent catastrophic account drain.

---

## 1. The "Shadow Mode" Switch (Max Daily Loss)
*As discussed, this prevents the bot from blowing the account during a market shift, while still allowing it to collect training data.*

**Implementation Logic:**
Inside your trade execution script, track the daily PNL. If it hits the limit, switch the bot from sending real API orders to simulating them.

```python
MAX_DAILY_LOSS = -50.00  # e.g., $50 max loss per day
current_daily_pnl = 0.00
trading_mode = "LIVE"    # "LIVE" or "SHADOW"

def execute_trade(action, symbol):
    global current_daily_pnl, trading_mode
    
    # 1. TRIGGER SAFETY LOCK
    if current_daily_pnl <= MAX_DAILY_LOSS and trading_mode == "LIVE":
        print("🚨 DAILY LOSS LIMIT HIT! Switching to SHADOW MODE.")
        trading_mode = "SHADOW"

    # 2. ROUTE THE ORDER
    if trading_mode == "LIVE":
        # Send REAL trade to exchange (Binance/Bybit)
        response = exchange.create_order(symbol, action)
    elif trading_mode == "SHADOW":
        # Simulate trade locally for training data only
        response = {"status": "simulated_fill", "price": get_current_price(symbol)}

    # 3. RECORD FOR CONTINUOUS LEARNING
    save_experience(action, response, mode=trading_mode)
```

---

## 2. API Key Security (The "Vault" Rules)
Never deploy standard API keys to a VPS. If the VPS is hacked, your funds will be drained.

* **Disable Withdrawals:** Your Exchange API Key MUST have the "Withdrawals" checkbox unchecked. The bot only needs "Read" and "Trade" permissions.
* **IP Whitelisting:** Bind the API key strictly to your Hostinger VPS IP address. If the key is leaked, it is completely useless on any other computer.
* **Environment Variables:** Never hardcode API keys in the Python files. Use the `.env` file (which is already set up in the `.gitignore`) so keys are never pushed to GitHub.

---

## 3. Hard-Coded Order Limits (Fat-Finger Protection)
Even if the AI outputs a confidence score of 99.9%, you must put a ceiling on the absolute size of any single trade to prevent a math bug from emptying your wallet in one click.

**Implementation:**
```python
MAX_ORDER_SIZE_USD = 100.00  # Never risk more than $100 per trade

def calculate_position_size(ai_confidence, account_balance):
    desired_size = account_balance * ai_confidence
    # Hard ceiling applied
    return min(desired_size, MAX_ORDER_SIZE_USD)
```

---

## 4. Slippage and Spread Protection
During flash crashes, the gap between the Bid and Ask price (the spread) can widen massively. If the bot executes a market order during this time, it will lose 5% instantly to slippage.

**Implementation:**
* Check the spread before firing a Market Order. If `(Ask Price - Bid Price) / Bid Price > 0.005` (0.5%), **Abort the Trade**.
* Alternatively, only use **Limit Orders** with a timeout. 

---

## 5. API Downtime / Exception Handling
Exchanges go down for maintenance or throw HTTP 502 errors. If your bot doesn't handle these gracefully, it will crash.

**Implementation:**
Wrap all exchange calls in a `try/except` block. If the API fails, the bot should default to `HOLD` and log the error, rather than crashing the script.

```python
try:
    response = exchange.create_order(...)
except ExchangeError as e:
    print(f"Exchange API Down! Defaulting to HOLD. Error: {e}")
    action = "HOLD"
```

---
**Deployment Checklist (REMINDER):**
- [ ] Implement Shadow Mode toggle in execution script.
- [ ] Set `MAX_DAILY_LOSS` variable.
- [ ] Generate VPS-specific API keys with IP binding.
- [ ] Set `MAX_ORDER_SIZE_USD` ceiling.
- [ ] Test the Exception Handling block by temporarily turning off internet on the VPS.

---

## 6. Official Safety Settings (From V1 Blueprint)
Currently, your `settings.py` has limits set to `9999999.0` (which is likely for testing/training). Before going live, you **MUST** revert the `core/config/settings.py` values back to these historically safe thresholds:

```python
    # ============================================================
    # Phase 1 RISK LIMITS (MANDATORY FOR LIVE)
    # ============================================================
    max_position_size: float = Field(default=10.0, gt=0)
    max_symbol_exposure_pct: float = Field(default=0.20, gt=0, le=1)
    max_portfolio_exposure_pct: float = Field(default=0.50, gt=0, le=1)

    max_leverage: int = Field(default=10, gt=0)
    max_order_size: float = Field(default=5.0, gt=0)
    max_open_positions: int = Field(default=5, gt=0)

    max_daily_loss_pct: float = Field(default=0.05, gt=0, le=1)
    max_drawdown_pct: float = Field(default=0.10, gt=0, le=1)

    max_market_data_age_seconds: float = Field(default=60.0, gt=0)
    correlated_exposure_limit_pct: float = Field(default=0.40, gt=0, le=1)
```

---

## 7. Re-Enable Hard Stop Loss (Real Money Safety)
During testnet, the Hard Stop Loss (originally set to -10% leveraged PNL) was disabled so the AI could learn how to manage exits itself without being choked out by market noise. 
However, **before deploying to Live Trading with real money**, you must re-enable this or set a wider stop loss (e.g., -30%) in `main.py` to protect against sudden market crashes if the AI freezes.

**Implementation:**
In `main.py` (around line 552), uncomment the safety block:
```python
if pnl_pct_leveraged <= -0.30: # -30% HARD STOP LOSS
    logger.warning(f"[{sym}] 🛑 HARD STOP LOSS TRIGGERED: Position is down {pnl_pct_leveraged*100:.2f}%. Overriding AI.")
    hard_stop_triggered = True
```

---

## 8. The Long-Game Strategy (Surviving Massive Waves in Live)
To allow the AI to successfully predict the "long game" and hold through massive market swings without getting liquidated or prematurely stopped out, adhere to this specific 3-part strategy:

* **A. "Catastrophe-Only" Stop Loss:** Do not disable the stop loss entirely in Live. Set it wide (e.g., `-40%` or `-50%`). This gives the AI the room to ride out standard market volatility while still providing a strict safety net against a "Black Swan" flash crash.
* **B. Lower Leverage (3x to 5x):** High leverage (e.g., 20x) is the enemy of the long game because a mere 5% swing will liquidate your account. By lowering your max leverage in `settings.py` to `3x` or `5x`, you give the bot a massive cushion (20%-33% swing) to survive dips before any exchange liquidation occurs.
* **C. Rely on Max Daily Loss (Shadow Mode):** Rely heavily on the Shadow Mode switch (Section 1). If the bot mispredicts the long game entirely and hits your daily loss limit (e.g., -$50), it will seamlessly switch to paper trading and prevent further damage.
