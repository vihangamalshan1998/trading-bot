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
