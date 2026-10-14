# Comprehensive System Trading Rules & Safeguards

This document provides a complete, authoritative specification of all trading rules, entry filters, risk management constraints, exit mechanisms, and model safety gates implemented in the AI Trading Platform.

---

## 1. Trade Entry Rules (Opening Gates)

Every new order must pass five consecutive safety filters before execution. If any filter fails, the trade is rejected and logged.

### Guard 1: Anti-Churn Cooldown & High-Conviction Fast-Pass
* **Baseline Cooldown:** **180 seconds (3 minutes)** per symbol after closing any position.
* **Objective:** Prevents rapid re-entry churn on the same asset after an exit, eliminating micro-fee burn.
* **High-Conviction Fast-Pass Exception:**
  * Cooldown is bypassed **ONLY** if:
    1. AI Actor Confidence $\ge 0.90$ ($90\%$), **AND**
    2. Critic 5-Minute Return Forecast $|\text{pred\_ret\_5m}| \ge 0.75\%$ in the trade direction ($+0.75\%$ for Long, $-0.75\%$ for Short).

### Guard 2: Minimum Hold Duration (Anti-Noise Filter)
* **Hold Window:** Positions must be held for a minimum of **90 seconds** before non-emergency closure.
* **Objective:** Prevents closing on sub-minute market noise before price moves can clear the exchange fee hurdle.
* **Emergency AI Conviction Bailout:**
  * Bypasses the 90s timer if the AI Actor detects market breakdown and issues an opposing exit with extreme conviction ($\text{Confidence} \ge 0.85$).

### Guard 3: Directional Confidence Hurdle
* **Open Threshold:** $\text{Confidence} \ge 0.75$ ($75\%$).
* **Objective:** Eliminates low-probability coin-toss setups and requires strong mathematical conviction.

### Guard 4: Critic Multi-Horizon Return Check (Fee Hurdle)
* **Minimum Expected Move:** $\ge 0.40\%$ price return.
* **Directional Alignment:**
  * **OPEN_LONG:** Requires $\text{pred\_ret\_5m} \ge +0.40\%$.
  * **OPEN_SHORT:** Requires $\text{pred\_ret\_5m} \le -0.40\%$.
* **Adverse Contradiction Rejection:**
  * If the Actor wants to SHORT but the Critic predicts a pump ($> -0.40\%$, e.g., $+0.64\%$), the entry is blocked.
  * If the Actor wants to LONG but the Critic predicts a dump ($< +0.40\%$, e.g., $-0.20\%$), the entry is blocked.
* **Objective:** Ensures every entry has at least $4\times$ the round-trip fee coverage ($\sim 0.09\%$ round-trip fee vs $\ge 0.40\%$ expected price gain).

### Guard 5: Margin Budgeting & Exposure Limits
* **Max Margin per Coin:** Capped at $15\%$ of total portfolio equity (`max_position_margin_pct = 0.15`).
* **Leverage Setting:** Enforced at $20\text{x}$ (`max_leverage = 20`).
* **Correlated Exposure Guard:** Limits cumulative exposure across correlated crypto assets to prevent portfolio-wide drawdowns during broad market swings.

---

## 2. Trade Exit Rules (Closing & Profit Locking)

Position closures prioritize capital preservation and guaranteed execution over passive price matching.

### 1. Guaranteed Market Exits
* **Execution Type:** All closures (`CLOSE_LONG`, `CLOSE_SHORT`, Trailing Stop, Sudden Shock, Stop-Loss) are executed as **`MARKET` orders with `reduce_only=True`**.
* **Pre-Exit Cleanup:** The bot calls `cancel_all_orders(symbol)` before placing the market exit to cancel any pending or lingering limit orders, preventing phantom fills.

### 2. Hard Stop-Loss (-25% Leveraged)
* **Trigger:** Leveraged $\text{PnL} \le -25.0\%$ (equivalent to a $-1.25\%$ price move against the position at $20\text{x}$ leverage).
* **Action:** Overrides AI policy and forces an immediate emergency market closure.

### 3. Sudden Velocity Shock Detector (<15s Window)
* **Trigger Window:** Inspects rolling PnL history over the last 15 seconds (`self.pnl_history`).
* **Condition:** If the position had reached a peak $\ge +6.0\%$ leveraged, and PnL drops by $\ge 5.0\%$ leveraged in under 15 seconds while losing $>30\%$ of the recent peak.
* **Action:** Triggers instant market exit to lock in the remaining profit before flash spikes or wick reversals wipe out the gain.

### 4. AI Critic Adverse Regime Shock Detector
* **Trigger:** Triggers when the Critic neural network predicts a sudden $\pm 0.40\%$ price shock opposing the active position:
  * Active **SHORT**, but Critic predicts pump $\ge +0.40\%$.
  * Active **LONG**, but Critic predicts dump $\le -0.40\%$.
* **Action:** Immediate emergency market exit before the adverse price move occurs.

### 5. Dynamic Adaptive Ratcheting Trailing Stop (Infinite Runner)
* **Concept:** No fixed take-profit ceiling. Allows trends to run indefinitely while ratcheting stop floors upward.
* **Tier 1 (Peak +6% to +10% Leveraged):**
  * Locks in at $+2.5\%$ leveraged PnL floor.
  * Guarantees positive net profit after clearing all round-trip exchange fees.
* **Tier 2 (Peak +10% to +25% Leveraged — Developing Trend):**
  * Trailing floor set to $\max(+4.0\%, \text{prev\_peak} \times 0.75)$.
  * Provides $25\%$ breathing room for normal trend oscillations.
* **Tier 3 (Peak $\ge +25\%$ Leveraged — Deep Trend Runner):**
  * Trailing floor tightens to $\text{prev\_peak} \times 0.85$.
  * Harvests and banks $85\%$ of peak gains on massive runs (e.g., $+50\%$, $+100\%+$).

### 6. Strict Net-Profit Gate & Spread Noise Breathing Room
* **Winning Profit Protection:** The bot will **NEVER close a winning trade prematurely** if leveraged PnL is between $0.0\%$ and $+5.0\%$ ($0.0 \le \text{PnL} < +5.0\%$). It forces the trade to ride until target or trailing stop.
* **Spread Noise Protection:** If slightly negative between $-3.0\%$ and $0.0\%$ leveraged (e.g. $-\$0.88$ normal bid-ask spread), panic-closing is **completely blocked**. The trade is given breathing room.
* **Emergency Bailout Condition:** Emergency bailout to bypass the 90s hold requires a **REAL loss** ($\text{PnL} \le -5.0\%$ leveraged) with extreme conviction ($\text{Confidence} \ge 0.85$). Minor wiggles are never treated as emergencies.
* **Solid Profit ($\ge +5.0\%$ leveraged):** AI is allowed to take profit with moderate confidence ($\text{Confidence} \ge 0.40$), guaranteeing large net gains after clearing all round-trip exchange fees.
* **Real Loss ($\le -3.0\%$ leveraged):** Allowed to cut the trade if the AI has high conviction ($\text{Confidence} \ge 0.85$).

---

## 3. Experience & Pipeline Integrity

* **Fill-Confirmed Experience Emission:** Trade experiences are published to Redis (`experience:completed`) and dashboard history **ONLY after Binance confirms the order is filled**. Cancelled or timed-out orders never pollute the training replay buffer.
* **Patience Reward on HOLD:** The bot logs high-quality HOLD experiences (sampled at $15\%$) with a $+0.05$ patience reward for flat states, training the neural network that preserving capital in cash is superior to fee churning.

---

## 4. AI Brain Evolution & Tournament Rules

* **A/B Champion Benchmark:** Candidate neural networks are evaluated against the live production Champion on the exact same 1,000-sample slice from the active replay buffer.
* **Promotion Criteria:** A Challenger is promoted to the live VPS **ONLY** if:
  1. Composite Decision Score $>$ Champion Score, OR
  2. Critic MSE is lower (more accurate price forecast) with non-degraded simulated PnL.
* **Safety Checks:** Rejects candidates with exploding loss ($> 100.0$), NaN weights, or mode-collapsed policy entropy ($< 1\text{e-}4$).

---

## 5. Quick Reference Summary Table

| Rule / Guard | Parameter | Objective |
| :--- | :--- | :--- |
| **Normal Entry Critic Hurdle** | $\ge \pm 0.40\%$ | Eliminates fee bleed; requires $>4\times$ fee coverage |
| **Normal Entry Confidence** | $\ge 0.75$ | Ensures high directional conviction |
| **Anti-Churn Cooldown** | $180\text{s}$ | Prevents rapid back-to-back re-entry on same coin |
| **Fast-Pass Cooldown Override** | Conf $\ge 0.90$ & Critic $\ge 0.75\%$ | Catches massive breakout momentum instantly |
| **Minimum Hold Duration** | $90\text{s}$ | Gives trades room to expand beyond fees |
| **Emergency AI Bailout** | Conf $\ge 0.85$ (opposing) | Bypasses 90s hold to cut severe losses early |
| **Hard Stop-Loss** | $-25.0\%$ leveraged | Absolute risk cap; immediate market exit |
| **Velocity Shock Detector** | Drop $\ge 5\%$ in $<15\text{s}$ | Protects profits against flash wicks |
| **Adverse Critic Shock** | Opposing Critic $\ge \pm 0.40\%$ | Exits immediately before predicted crash/pump |
| **Tier 1 Breakeven Stop** | Floor at $+2.5\%$ leveraged | Guarantees net profit after fees if peak reached $+6\%$ |
| **Tier 2 Trailing Stop** | $25\%$ giveback room | Allows developing trends room to ride |
| **Tier 3 Trailing Stop** | $15\%$ giveback room (banks $85\%$) | Locks in massive profits on large trend runners |
| **Order Execution on Exits** | `MARKET` (`reduce_only=True`) | Guarantees zero missed exits or timed-out limit orders |
| **Max Position Margin** | $15\%$ of total equity | Diversifies capital; prevents single-coin liquidation |
