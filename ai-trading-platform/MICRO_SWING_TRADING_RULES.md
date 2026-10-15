# 🏛️ Micro-Swing AI Trading Bot: Complete Operational Rulebook

This document defines the exact, end-to-end mathematical and systematic rules used by the **Production Micro-Swing Trading Bot** to open, manage, and close positions.

---

## 🟢 Part 1: Rules to OPEN a Trade

For the bot to open a new position on any symbol (`BTC`, `ETH`, `SOL`, `BNB`, `XRP`, `LINK`, `ADA`, `AVAX`, `NEAR`, `LTC`), it must pass **13 strict institutional safety checkpoints in sequence**. If even a single condition fails, the trade is rejected:

```
[Market Ticks] ──► 0. Circuit Breaker & Profit Ratchet ──► 0.5. Liquidity Clock (Session Regimes)
              ──► 1. AI Decision (Smoothed) ──► 2. Cooldown Gate (120s Rest)
              ──► 2.5. Portfolio Beta Governor (Max 3 Longs/Shorts, 30% Delta Cap)
              ──► 3. Spread Firewall (Session Scaled) ──► 4. Confidence Filter (Profit Ratchet Scaled)
              ──► 5. Multi-Horizon Critic (Session Hurdle) ──► 6. Macro Regime Alignment
              ──► 7. Lead-Lag Sniping (Crypto Echo) ──► 8. OFI Wall Defense ──► 9. StatArb Lead-Lag
              ──► 10. Whale Order Flow ──► 11. Funding Rate Squeeze Sniper ──► 12. Dynamic Margin ──► 13. POST-ONLY MAKER (GTX)
```

### 0. Portfolio 24-Hour 5% Drawdown Circuit Breaker & Daily High-Water Profit Ratchet

#### A. 5% Daily Loss Kill Switch ("The Circuit Breaker")
- **Ironclad Capital Preservation:** Tracks rolling 24-hour starting equity.
- If total account equity drops $\ge \mathbf{5.0\%}$ from the daily starting equity:
  - **Trips the emergency circuit breaker immediately.**
  - Pauses all new trade entries for **6 hours**.
  - All existing open positions continue to be safely managed (take-profit, trailing stop, stop-loss still execute cleanly).
  - Guarantees that even in a freak Black Swan event, **your account physically cannot lose more than 5% in a day**.

#### B. Daily High-Water Profit Ratchet ("Session Profit Lock")
- **The Hedge Fund Principle:** Eliminates giving back morning profits during choppy afternoon sessions.
- **Trigger Condition:** Once total daily profit reaches $\ge \mathbf{+4.0\%}$ ($+\$180$ on a $\$4,500$ account):
  - If current profit retraces by $\ge \mathbf{50\%}$ from the day's high-water peak (e.g. peak profit was $+\$220$ and drops to $+\$105$):
  - **Arms Capital Preservation Mode:**
    1. **Margin Defense:** Halves base margin allocation from $10.0\%$ to **$5.0\%$** (dynamic bounds clamped to $3.0\% - 8.0\%$).
    2. **Hurdle Elevation:** Raises the confidence hurdle to **$\mathbf{\ge 74\%}$**, allowing only highest-conviction "A+" setups.
  - Automatically disarms if equity recovers to $\ge 75\%$ of the peak, or resets cleanly every 24 hours.
  - **Result:** Permanently locks winning days into cash!

### 0.5. Institutional Liquidity Clock & Session Volatility Regimes

- Global crypto markets exhibit distinct liquidity and spread regimes based on the time of day:
  - **Weekend Low Liquidity (Saturday & Sunday UTC):** Institutional desks are offline, market depth drops by up to 50%, and spreads widen.
    - **Confidence Hurdle:** Raised to **$\mathbf{\ge 72\%}$**.
    - **Spread Firewall:** Tightened to **$\mathbf{\le 0.045\%}$** (skips wider books).
    - **Critic Fee Hurdle:** Raised to **$\mathbf{\ge 0.50\%}$** expected move.
  - **Asia Off-Peak Chop (00:00 - 08:00 UTC):** High mean-reversion, lower directional follow-through.
    - **Confidence Hurdle:** Raised to **$\mathbf{\ge 68\%}$**.
    - **Spread Firewall:** **$\mathbf{\le 0.050\%}$**.
    - **Critic Fee Hurdle:** **$\mathbf{\ge 0.45\%}$**.
  - **London Trend (08:00 - 13:30 UTC):** Morning trend establishment and European volume.
    - **Confidence Hurdle:** Standard **$\mathbf{\ge 65\%}$**.
    - **Spread Firewall:** Standard **$\mathbf{\le 0.060\%}$**.
    - **Critic Fee Hurdle:** Standard **$\mathbf{\ge 0.40\%}$**.
  - **US New York Peak (13:30 - 21:00 UTC):** Highest global volume, tightest bid/ask spreads, deepest breakout trends.
    - **Confidence Hurdle:** Standard **$\mathbf{\ge 65\%}$**.
    - **Spread Firewall:** Standard **$\mathbf{\le 0.060\%}$**.
    - **Critic Fee Hurdle:** Standard **$\mathbf{\ge 0.40\%}$**.

### 1. Deterministic AI Direction Proposal & Institutional Signal Smoothing

- The neural network inspects the 200-dimensional state vector (orderbook depth + whale volume + momentum).
- **3-Tick Temporal Window Smoothing:** Action values and confidence are smoothed across a 3-tick moving average window to eliminate high-frequency neural jitter and noisy 1-tick oscillations.
- **Hysteresis Deadband:**
  - **Flat Entry:** Requires $|action\_val| > 0.20$ to open a new Long or Short.
  - **Position Reversal / Exit:** Reversing or closing an existing position requires crossing an institutional deadband ($|action\_val| > 0.35$). Minor flutters within the position band are treated as **HOLD**.

### 2. Anti-Churn Cooldown Gate (180s Baseline + 120s Ironclad Rest)

- **120-Second Mandatory Rest:** Absolutely zero new trades are permitted within 120 seconds of closing a position on that symbol. This physically eliminates 5-second flip-flop churn traps.
- **Breakout Fast-Pass (120s to 180s):** If and only if 120s have elapsed, the bot can bypass the remaining 60s of cooldown if confidence is extreme ($\ge 78\%$) and the Critic forecasts a breakout ($\ge |0.85\%|$).

### 2.5. Net Portfolio Beta & Correlation Exposure Governor

- **Multi-Asset Factor Risk Defense:** In crypto, altcoins hold an $85\%–95\%$ correlation to Bitcoin. Without a portfolio governor, opening 5 Longs simultaneously exposes the account to catastrophic correlated drawdowns if BTC suddenly flushes.
- **Max Unidirectional Positions:** Limits simultaneous open trades in the same direction to a strict ceiling of **3 Longs** or **3 Shorts**.
- **Net Delta Exposure Cap:** Restricts net directional margin exposure ($\sum \text{Long Margin} - \sum \text{Short Margin}$) to **$\le \mathbf{30\%}$ of total account equity** (~$\$1,350$ on a $\$4,500$ account).
- **Market-Neutral Hedge Exemption:** If 3 Longs are open and a high-conviction Short signal appears, the Short is **approved** because it functions as an uncorrelated hedge that *reduces* net directional market exposure.

### 3. Spread Firewall ($\le 0.060\%$)

- Checks the live Binance orderbook bid/ask spread:
  - If $(ask - bid) / mid\_price > 0.0006$ (0.06% / 6 bps), it **blocks opening immediately**.
  - Prevents opening into thin books or news spikes where slippage drains capital.

### 4. Confidence Filter (Session Adaptive & Profit Ratchet Modulated)

- The AI’s sigmoid probability must meet the dynamic hurdle:
  - **Standard Session Hurdle:** $\ge 65\%$ (London / US NY), $\ge 68\%$ (Asia Chop), $\ge 72\%$ (Weekend).
  - **Profit Ratchet Mode Hurdle:** Elevated to **$\mathbf{\ge 74\%}$** to safeguard winning days.
- If confidence is below the hurdle, it skips the trade and logs an authentic **HOLD experience (+0.05 patience reward)** into MySQL so the database learns flat-market discipline.

### 5. Multi-Horizon Critic Return Hurdle (Fee Protection)

- **5-Minute Forecast:** The Critic network must forecast at least a **$|0.40\%|$ spot price move** in the trade's direction (guarantees the move will easily beat Binance’s 0.10% round-trip fee).
- **1-Hour & 4-Hour Trend Alignment:** The 1-hour and 4-hour forecasts **cannot oppose the trade** (e.g., if opening a Long, the 1-hour Critic cannot be strongly negative $<-0.15\%$).

### 6. Macro Regime Filter (Anti-Whipsaw)

- Checks Bitcoin and Ethereum 15-minute momentum and 1-hour trend:
  - If both BTC and ETH are crashing ($\le -0.80\%$), it **blocks counter-trend LONGS**.
  - If both BTC and ETH are surging ($\ge +0.80\%$), it **blocks counter-trend SHORTS**.
  - _High-Conviction Fast Pass:_ If the coin has $\ge 70\%$ confidence and $\ge |0.40\%|$ forecasted move, it can enter independently of minor BTC consolidation noise.

### 7. Cross-Asset Lead-Lag Sniping ("The Crypto Echo" Momentum Front-Running)

- **The Institutional Lead-Lag Dynamic:** In crypto, Bitcoin acts as the liquidity mothership. Altcoins follow BTC breakout impulses with a 30 to 180-second transmission lag ("the echo").
- **Flash Dump / Pump Firewall:**
  - **For Longs:** If BTC experiences a 1-minute flash dump ($\le \mathbf{-0.35\%}$), all altcoin Longs are **blocked immediately**, regardless of individual chart strength, preventing buying into an imminent correlated cascade.
  - **For Shorts:** If BTC experiences a 1-minute flash pump ($\ge \mathbf{+0.35\%}$), all altcoin Shorts are **blocked immediately**.
- **Echo Pulse Front-Running:**
  - If BTC is surging ($\ge +0.20\%$ in 1m) and the altcoin is lagging behind BTC, the bot **boosts conviction by $1.15\times$** to front-run the altcoin's catch-up explosion!
  - If BTC is dumping ($\le -0.20\%$ in 1m) and the altcoin is lagging, boosts Short conviction by $1.15\times$.

### 8. Level-2 Order Flow Imbalance (OFI / Book Depth Wall Defense)

- Checks the live top-5 orderbook depth ratio: $\text{Imbalance} = \frac{\text{Bid Depth} - \text{Ask Depth}}{\text{Total Depth}}$
  - **For Longs:** If Ask depth heavily outweighs Bid depth ($\text{Imbalance} < -0.30$), it **blocks Long entries** to avoid buying into massive institutional sell walls.
  - **For Shorts:** If Bid depth heavily outweighs Ask depth ($\text{Imbalance} > +0.30$), it **blocks Short entries** to avoid shorting into buy walls.

### 9. Cross-Exchange StatArb Confirmation (OKX & Bybit Lead-Lag Filter)

- Cross-references live price premiums from OKX and Bybit via CCXT (`apps/statarb_collector`):
  - **For Longs:** If OKX or Bybit are trading at a discount ($< -0.15\%$), it **blocks the Long** because Binance is experiencing a local fakeout wick that will crash back to global prices.
  - **For Shorts:** If OKX or Bybit are trading at a premium ($> +0.15\%$), it **blocks the Short**.
  - _Fail-Open Resilience:_ If external exchange feeds lag (> 15s), it falls back to Binance data so execution is never stalled.

### 10. Whale Market Order Flow Momentum (Whale Tracker Integration)

- Streams live institutional $>\$100,000$ market order sweeps via `apps/whale_tracker`:
  - **For Longs:** If whales executed $\ge \$300,000$ in market sells and selling is $2\times$ greater than buying in the last 60 seconds, **blocks Long entries** (never buy into an institutional dump).
  - **For Shorts:** If whales executed $\ge \$300,000$ in market buys and buying is $2\times$ greater than selling, **blocks Short entries**.
  - **Whale Conviction Boost:** If whales are buying heavily ($\ge \$200,000$) in alignment with a Long signal, boosts margin conviction multiplier by $+10\%$.

### 11. Funding Rate Asymmetry & Short-Squeeze Sniping

- Reads live 8-hour perpetual funding rates:
  - **Short Squeeze Sniping:** When funding is deep negative ($\le -0.03\%$), retail shorts are trapped paying longs. **Blocks Short trades** (prevents paying funding drain and getting short-squeezed) and flags Long setups for explosive short-squeeze upside!
  - **Overheated Long Defense:** When funding is frothy positive ($\ge +0.05\%$), **blocks Long trades** to avoid paying heavy funding fees to short sellers.

### 12. Institutional Dynamic Volatility & Conviction Margin Sizing

- Dynamically scales margin allocation based on conviction, market volatility, and risk regime:
  - **Base Margin:** $10.0\%$ standard (~$450 on a $4,500 account); automatically halved to **$5.0\%$** when Profit Ratchet is active.
  - **Conviction Multiplier:** $0.80\times$ to $1.30\times$ based on neural confidence relative to effective hurdle.
  - **Critic Multi-Horizon Alignment:** $1.15\times$ if 5m and 1h agree, $0.85\times$ if divergent.
  - **Lead-Lag Echo Multiplier:** $1.15\times$ when front-running BTC breakout impulses.
  - **Whale Flow Boost:** $1.10\times$ if institutional whale flow is actively buying with the trade.
  - **Asset Beta Dampener:** $1.10\times$ for BTC/ETH; $1.00\times$ for SOL/BNB/LTC; $0.75\times$ for wild altcoins (ADA, XRP, NEAR, AVAX, LINK).
  - **Market Volatility Scaling:** Automatically trims position size by up to $30\%$ when market volatility $> 1.5\%$.
  - **Safe Clamping:** Dynamic margin is strictly bounded between **$6.0\%$ and $15.0\%$ of equity** ($3.0\% - 8.0\%$ in Profit Ratchet Mode).
  - Minimum order size must be at least **$50 notional**.

### 13. Post-Only Maker Execution (`GTX`) & Active Pegging

- **Zero Taker Fees:** All entry orders are sent with `time_in_force="GTX"` (Good-Till-Crossing / Post-Only).
- **Top-of-Book Pegging:** Longs peg to `best_bid`; Shorts peg to `best_ask`.
- **Binance `-5022` Protection:** If an order would cross the spread, it automatically re-pegs 1 tick deeper to guarantee Maker status (0.02% fee instead of 0.05% taker fee).
- **30-Second Pegging Timeout:** If unfilled after 30 seconds, it cancels cleanly to release margin.

👉 **If all 13 checkpoints pass:** The bot places an authentic **POST-ONLY MAKER ORDER (GTX)** on the book!

---

## 🔴 Part 2: Rules to CLOSE a Trade (The 7 Scenarios)

Once in a trade, the bot continuously monitors the position every 5 seconds. It will exit **ONLY** under one of these 7 explicit scenarios:

```
                                      ┌──► Scenario 1: Hard Take-Profit Ceiling (+35.0% Leveraged) ──► Lock Peak Windfall!
                                      ├──► Scenario 2: Dynamic 50% Scale-Out (TP1 at +12.0%) ──► Bank Cash & Arm Free-Roll!
                                      ├──► Scenario 3: Free-Roll Breakeven Stop (+0.25% Floor) ──► Zero-Risk Guaranteed Win!
Active Position (PnL & Time Tracked) ─┼──► Scenario 4: Sudden Velocity Shock (Drops >= 3.5% in <15s) ──► Immediate Exit!
                                      ├──► Scenario 5: Ratcheting Trailing Stop (Peak >= +8.0%) ──► Bank Net Green Cash!
                                      ├──► Scenario 6: Dynamic Critic Reversal Exit (Profit >= +3.5%) ──► Secure Gains!
                                      ├──► Scenario 7: Protective Stop Loss (-15.0% Hard Stop / -12.0% Trap) ──► Cut Loss!
                                      └──► Scenario 8: Stale Position Pruner (Held >= 90m, |PnL| < 2.0%) ──► Recycle Capital!
```

---

### Scenario 1: The Hard Take-Profit Ceiling (+35.0% Leveraged Gain)

- **Condition:** Position reaches $\ge \mathbf{+35.0\%}$ leveraged gain ($+1.75\%$ spot price move at 20x).
- **Action:** **Instant Market Close.** Snatches windfall profits immediately at the candle wick peak without waiting for any pullback!

### Scenario 2: Dynamic 50% Scale-Out (TP1 at +12.0% Leveraged Gain)

- **Condition:** Position surges to $\ge \mathbf{+12.0\%}$ leveraged gain ($+0.60\%$ spot price move at 20x).
- **Action:** **50% Partial Market Close (`reduce_only=True`).**
  - Banks realized cash directly into your account wallet, cleanly paying all exchange fees with net profit.
  - Immediately arms **Free-Roll Mode** on the remaining 50% quantity.
  - Dashboard logs `PARTIAL_CLOSE_LONG` / `PARTIAL_CLOSE_SHORT` with green realized PnL.

### Scenario 3: Free-Roll Breakeven Stop Loss (+0.25% Leveraged Floor)

- **Condition:** Active **ONLY** after Scenario 2 has already banked the 50% partial take-profit.
- If the market retraces and leveraged PnL pulls back to $\le \mathbf{+0.25\%}$ (the exchange round-trip fee floor):
- **Action:** **Instant Market Close on Remaining 50%.**
  - Exits the remaining half at breakeven with zero loss.
  - Guarantees the overall trade ends with net green profit in the wallet.

### Scenario 4: The Sudden Velocity Shock Detector

- **Condition:** The trade was up $\ge +4.0\%$ leveraged, and suddenly drops by $\ge 3.5\%$ in less than 15 seconds, losing over 30% of its peak.
- **Action:** **Instant Market Close.** Catches flash dumps/pumps and locks in remaining profit before it turns negative.

### Scenario 5: ATR-Adaptive Volatility Trailing Stops (Coin Beta & Volatility Scaled)

- **The Problem with Rigid Percentage Trails:** Rigid percentage trailing stops treat low-volatility majors (`BTC`, `ETH`) identically to explosive high-beta altcoins (`NEAR`, `SOL`, `ADA`). This causes BTC to give back too much profit on pullbacks, while premature stop-outs constantly shake out alts on normal wicks before a $+30\%$ to $+50\%$ runner develops.
- **The Institutional Solution:** Trailing stop floors dynamically calibrate to each symbol's live ATR / volatility (`vol_factor`):
  - **Low-Volatility Assets (BTC, ETH ~0.8% vol):**
    - `vol_factor` = $-0.028$.
    - **Tier 1 (Impulse Wave, Peak $+8\%$ to $+15\%$):** Locks in **$73.0\%$** of peak (min $+4.0\%$ floor) to bank cash fast without unnecessary giveback.
    - **Tier 2 (Developing Swing, Peak $+15\%$ to $+30\%$):** Locks in **$78.0\%$** of peak (min $+8.0\%$ floor).
    - **Tier 3 (Deep Runner, Peak $\ge +30\%$):** Locks in **$86.5\%$** of peak.
  - **High-Volatility Assets (NEAR, SOL, ADA, AVAX ~3.0% vol):**
    - `vol_factor` = $+0.060$.
    - **Tier 1 (Impulse Wave, Peak $+8\%$ to $+15\%$):** Locks in **$64.0\%$** of peak (min $+4.0\%$ floor), providing the wider breathing room necessary to survive alts wicks.
    - **Tier 2 (Developing Swing, Peak $+15\%$ to $+30\%$):** Locks in **$69.0\%$** of peak (min $+8.0\%$ floor).
    - **Tier 3 (Deep Runner, Peak $\ge +30\%$):** Locks in **$82.0\%$** of peak to let runners reach $+50\%$ or $+100\%$.
- **Safety Floors:** Always guarantees at least $+4.0\%$ leveraged gain (Tier 1) and $+8.0\%$ leveraged gain (Tier 2), ensuring all round-trip exchange fees and slippage are completely covered with positive net cash added to your wallet!

### Scenario 6: Critic-Assisted Early Reversal Exit

- **Condition:** Position is in solid profit ($\ge +3.5\%$ leveraged), and the Critic network detects a confirmed adverse reversal (5m $\le -0.15\%$ or 1h $\le -0.25\%$).
- **Action:** Exits cleanly to protect green profits before the market pulls back.
- **Fee Protection Gate:** If profit is between $0.0\%$ and $+4.0\%$, the bot **skips closing** to prevent fee bleed, letting the swing ride unless a confirmed reversal hits.

### Scenario 7: Loss Protection & Hard Stop Loss

- **10-Minute Breathing Room:** If PnL is between $0.0\%$ and $-12.0\%$, the position **MUST hold for at least 600 seconds (10 minutes)** to survive normal crypto wicks.
- **Binance `updateTime` Memory Recovery:** Position entry timestamps are retrieved from Binance on service reboot, ensuring the 10-minute hold is never wiped by a restart.
- **Confirmed Trap Escape Door:** Cuts before 10 minutes **ONLY** if Critic forecasts a severe adverse trend breakdown ($\ge 0.60\%$ adverse move) with $\ge 70\%$ confidence.
- **Hard Stop Loss:** If PnL touches **$-15.0\%$ leveraged** (-0.75% spot move at 20x) $\to$ **Immediate Emergency Market Close** to protect total capital.

### Scenario 8: Institutional Stale Position Pruner & Capital Recycling

- **Condition:** Position has been open for $\ge \mathbf{90\text{ minutes}}$ ($5,400\text{ seconds}$) and remains completely flat / stagnant ($|PnL| < \mathbf{2.0\%}$ leveraged).
- **Rationale:** Capital stuck in rangebound chop generates zero alpha while paying funding fees and blocking other high-conviction breakout setups.
- **Action:** **Market Close (`reduce_only=True`).** Prunes the dormant trade, clears orderbook locks, and recycles margin back to `free_margin` for explosive breakout runners!

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
