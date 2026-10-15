# 🏛️ Micro-Swing AI Trading Bot: Complete Operational Rulebook

This document defines the exact, end-to-end mathematical and systematic rules used by the **Production Micro-Swing Trading Bot** to open, manage, and close positions.

---

## 🟢 Part 1: Rules to OPEN a Trade

For the bot to open a new position on any symbol (`BTC`, `ETH`, `SOL`, `BNB`, `XRP`, `LINK`, `ADA`, `AVAX`, `NEAR`, `LTC`), it must pass **6 strict safety checkpoints in sequence**. If even a single condition fails, the trade is rejected:

```
[Market Ticks] ──► 1. AI Decision ──► 2. Cooldown Gate ──► 3. Confidence Gate ──► 4. Multi-Horizon Critic ──► 5. Macro Gate ──► 6. Risk Firewall ──► EXECUTE LIMIT ORDER
```

### 1. Deterministic AI Direction Proposal & Institutional Signal Smoothing
* The neural network inspects the 200-dimensional state vector (orderbook depth + whale volume + momentum).
* **3-Tick Temporal Window Smoothing:** Action values and confidence are smoothed across a 3-tick moving average window to eliminate high-frequency neural jitter and noisy 1-tick oscillations.
* **Hysteresis Deadband:**
  - **Flat Entry:** Requires $|action\_val| > 0.20$ to open a new Long or Short.
  - **Position Reversal / Exit:** Reversing or closing an existing position requires crossing an institutional deadband ($|action\_val| > 0.35$). Minor flutters within the position band are treated as **HOLD**.

### 2. Anti-Churn Cooldown Gate (180s Baseline + 120s Ironclad Rest)
* **120-Second Mandatory Rest:** Absolutely zero new trades are permitted within 120 seconds of closing a position on that symbol. This physically eliminates 5-second flip-flop churn traps.
* **Breakout Fast-Pass (120s to 180s):** If and only if 120s have elapsed, the bot can bypass the remaining 60s of cooldown if confidence is extreme ($\ge 78\%$) and the Critic forecasts a breakout ($\ge |0.85\%|$).

### 3. Spread Firewall ($\le 0.060\%$)
* Checks the live Binance orderbook bid/ask spread:
  - If $(ask - bid) / mid\_price > 0.0006$ (0.06% / 6 bps), it **blocks opening immediately**.
  - Prevents opening into thin books or news spikes where slippage drains capital.

### 4. Confidence Filter ($\ge 65\%$)
* The AI’s sigmoid probability must be at least **$0.65$ (65%)**.
* If confidence is $< 65\%$, it skips the trade and logs an authentic **HOLD experience (+0.05 patience reward)** into MySQL so the database learns flat-market discipline.

### 5. Multi-Horizon Critic Return Hurdle (Fee Protection)
* **5-Minute Forecast:** The Critic network must forecast at least a **$|0.40\%|$ spot price move** in the trade's direction (guarantees the move will easily beat Binance’s 0.10% round-trip fee).
* **1-Hour & 4-Hour Trend Alignment:** The 1-hour and 4-hour forecasts **cannot oppose the trade** (e.g., if opening a Long, the 1-hour Critic cannot be strongly negative $<-0.15\%$).

### 6. Macro Regime Filter (Anti-Whipsaw)
* Checks Bitcoin and Ethereum 15-minute momentum and 1-hour trend:
  - If both BTC and ETH are crashing ($\le -0.80\%$), it **blocks counter-trend LONGS**.
  - If both BTC and ETH are surging ($\ge +0.80\%$), it **blocks counter-trend SHORTS**.
  - *High-Conviction Fast Pass:* If the coin has $\ge 70\%$ confidence and $\ge |0.40\%|$ forecasted move, it can enter independently of minor BTC consolidation noise.

### 7. Institutional Dynamic Volatility & Conviction Margin Sizing
* Dynamically scales margin allocation between **6.0% and 15.0%** of account equity based on signal strength:
  - **Base Margin:** $10.0\%$ of account equity (~$450 margin on a $4,500 account).
  - **Conviction Multiplier:** $0.80\times$ to $1.30\times$ based on neural confidence.
  - **Critic Multi-Horizon Alignment:** $1.15\times$ if 5m and 1h agree, $0.85\times$ if divergent.
  - **Asset Beta Dampener:** $1.10\times$ for BTC/ETH; $1.00\times$ for SOL/BNB/LTC; $0.75\times$ for wild altcoins (ADA, XRP, NEAR, AVAX, LINK).
  - **Market Volatility Scaling:** Automatically trims position size by up to $30\%$ when market volatility $> 1.5\%$.
  - **Safe Clamping:** Dynamic margin is strictly bounded between **$6.0\%$ and $15.0\%$ of equity** (~$270 to ~$675 margin on a $4,500 account).
  - Minimum order size must be at least **$50 notional**.

👉 **If all 7 pass:** The bot places a **LIMIT ORDER** with price offset to minimize slippage.

---

## 🔴 Part 2: Rules to CLOSE a Trade (The 5 Scenarios)

Once in a trade, the bot continuously monitors the position every 5 seconds. It will exit **ONLY** under one of these 5 explicit scenarios:

```
                                      ┌──► Scenario 1: Hard Take-Profit Ceiling (+35.0% Leveraged) ──► Lock Peak Windfall!
                                      ├──► Scenario 2: Sudden Velocity Shock (Drops >= 3.5% in <15s) ──► Immediate Exit!
Active Position (PnL & Time Tracked) ─┼──► Scenario 3: Ratcheting Trailing Stop (Peak >= +8.0%) ──► Bank Net Green Cash!
                                      ├──► Scenario 4: Dynamic Critic Reversal Exit (Profit >= +3.5%) ──► Secure Gains!
                                      └──► Scenario 5: Protective Stop Loss (-15.0% Hard Stop / -12.0% Trap) ──► Cut Loss!
```

---

### Scenario 1: The Hard Take-Profit Ceiling (+35.0% Leveraged Gain)
* **Condition:** Position reaches $\ge \mathbf{+35.0\%}$ leveraged gain ($+1.75\%$ spot price move at 20x).
* **Action:** **Instant Market Close.** Snatches windfall profits immediately at the candle wick peak without waiting for any pullback!

### Scenario 2: The Sudden Velocity Shock Detector
* **Condition:** The trade was up $\ge +4.0\%$ leveraged, and suddenly drops by $\ge 3.5\%$ in less than 15 seconds, losing over 30% of its peak.
* **Action:** **Instant Market Close.** Catches flash dumps/pumps and locks in remaining profit before it turns negative.

### Scenario 3: The 3-Tier Ratcheting Trailing Stop
* **Tier 1 (Impulse Wave):**
  - Arms when peak profit reaches **$+8.0\%$ to $+15.0\%$ leveraged**.
  - Guarantees a minimum floor of **$+4.0\%$ leveraged** (+0.20% spot move).
  - *Math:* Covers the 0.10% Binance fee and guarantees **positive net cash in your account**.
* **Tier 2 (Developing Trend):**
  - Arms when peak profit reaches **$+15.0\%$ to $+30.0\%$ leveraged**.
  - Trailing floor: locks in at least **$+8.0\%$** or $75\%$ of the peak.
* **Tier 3 (Deep Runner):**
  - Arms when peak profit reaches **$\ge +30.0\%$ leveraged**.
  - Trailing floor: tight 15% trail (keeps $85\%$ of the highest peak reached).

### Scenario 4: Critic-Assisted Early Reversal Exit
* **Condition:** Position is in solid profit ($\ge +3.5\%$ leveraged), and the Critic network detects a confirmed adverse reversal (5m $\le -0.15\%$ or 1h $\le -0.25\%$).
* **Action:** Exits cleanly to protect green profits before the market pulls back.
* **Fee Protection Gate:** If profit is between $0.0\%$ and $+4.0\%$, the bot **skips closing** to prevent fee bleed, letting the swing ride unless a confirmed reversal hits.

### Scenario 5: Loss Protection & Hard Stop Loss
* **10-Minute Breathing Room:** If PnL is between $0.0\%$ and $-12.0\%$, the position **MUST hold for at least 600 seconds (10 minutes)** to survive normal crypto wicks.
* **Binance `updateTime` Memory Recovery:** Position entry timestamps are retrieved from Binance on service reboot, ensuring the 10-minute hold is never wiped by a restart.
* **Confirmed Trap Escape Door:** Cuts before 10 minutes **ONLY** if Critic forecasts a severe adverse trend breakdown ($\ge 0.60\%$ adverse move) with $\ge 70\%$ confidence.
* **Hard Stop Loss:** If PnL touches **$-15.0\%$ leveraged** (-0.75% spot move at 20x) $\to$ **Immediate Emergency Market Close** to protect total capital.

---

## 🎬 Part 3: Real-World Scenario Walkthrough

Let’s trace the exact life cycle of an **`ADAUSDT` SHORT Trade**:

1. **The Entry (1:34:31 PM):**
   - AI sees strong selling pressure in the orderbook. Action = $-0.85$ (Short), Confidence = $73\%$.
   - Critic forecasts a $-1.80\%$ drop in the next 5 minutes (well above the $0.40\%$ fee hurdle).
   - Risk Manager allocates $\$716$ margin (15% budget).
   - Order placed: Sell $64,670$ ADA at $\$0.2213$. Order fills instantly.

2. **The 10-Minute Breathing Room (1:34 to 1:44 PM):**
   - ADA wiggles up slightly to $\$0.2215$ (PnL is $-1.35\%$ leveraged).
   - In the old code: it would have panicked and closed after 2 minutes.
   - **In the new code:** It logs: `⏳ MICRO-SWING BREATHING: Minimum hold (600s) not met yet. Giving swing room to run!`
   - It survives the wick!

3. **The Momentum Drop (1:45 PM):**
   - ADA drops to $\$0.2200$ (PnL surges to **$+11.75\%$ leveraged**).
   - Tier 1 Trailing Stop arms: It locks in a minimum floor of **$+4.0\%$ leveraged** ($\$0.2208$).

4. **The Exit & Profit Lock (1:52 PM):**
   - ADA bounces back slightly to $\$0.2207$.
   - The trailing stop triggers: `🔒 MICRO-SWING IMPULSE TRAIL: Banking solid swing gain!`
   - Bot executes Market Buy to close.
   - Gross Gain: $+\$38.80$. Binance Fees: $-\$7.10$. **Net Cash Added to Account: $+\$31.70!$**
   - The trade is saved to MySQL, and the 180-second cooldown starts.
