# Institutional Brain Push Protocol (3-Session Cadence)

## 1. Executive Summary & Core Principle

* **Daily Target**: **Exactly 3 Brain Pushes per 24-hour cycle (1 Push per Global Market Session)**.
* **Core Philosophy**: **Recency with Stability**. Neural network policies require execution consistency across a 6 to 8-hour window to allow micro-swings to develop, while remaining fresh enough to adapt to macro regime shifts.

---

## 2. Daily Master Trainer Schedule (IST & UTC)

| Run | Local Time (IST) | UTC Time | Market Session & Goal |
| :--- | :--- | :--- | :--- |
| **Run 1: Afternoon** | **12:30 PM – 1:30 PM IST** | 07:00 – 08:00 UTC | **London Europe Open**: Trains on Asian overnight data and deploys for the European momentum trend. |
| **Run 2: Evening** | **7:00 PM – 8:30 PM IST** | 13:30 – 15:00 UTC | **Wall Street New York Peak**: Trains on European data and deploys for peak US ETF & institutional breakout volume. |
| **Run 3: Late Night** | **2:30 AM – 4:00 AM IST** *(or before bed ~11:30 PM)* | 21:00 – 22:30 UTC | **Asian Midnight Hand-off**: Trains on US close & funding settlements, deploying for overnight Asian ranges. |

---

## 3. Quantitative Rationale: Why Exactly 3 Pushes?

### A. The "Micro-Chop Overfitting" Trap (Why NOT 10+ pushes/day)
* If models are pushed every 60–90 minutes, the neural net **overfits to the last hour of price noise**.
* In a sideways weekend or midnight range, an hourly model learns that *"breakouts always fail"*, aggressively shorting the local range bottom.
* When a real trend expansion begins, that overfitted model suffers consecutive cut-losses.

### B. Eliminating Policy Flip-Flop
* Micro-swing trades require **15 minutes to 2 hours** to reach their full ATR profit targets.
* Pushing too frequently creates "policy jitter"—a model pushed at 14:00 might hold Shorts, while a model pushed at 15:00 flips bullish, forcing premature exits and churning commissions.
* A 6 to 8-hour holding window per brain gives established trades time to complete their profit cycle.

### C. Preventing Model Stagnation (Why NOT once a week)
* Crypto perpetual futures environments evolve rapidly: funding rates flip from negative to positive, open interest cascades, and volatility regimes compress or expand.
* 3 pushes per day ensures the network's 200-dimensional memory is never more than 8 hours behind live market structure.

---

## 4. Tournament Gatekeeper Standards

Before any challenger model is promoted to production, the `apps/trainer/tournament.py` engine strictly evaluates it against the reigning Champion:

1. **Superiority Threshold**:
   * Challenger Score must exceed Champion Score:
     $$\Delta \text{Score} = \text{Score}_{\text{Challenger}} - \text{Score}_{\text{Champion}} > 0.5$$
2. **Critic Precision**:
   * Critic Mean Squared Error (MSE) on predicted returns must not degrade significantly compared to the Champion.
3. **Simulated PnL**:
   * Out-of-sample forward simulation must show positive net expectancy after accounting for 0.04% round-trip trading fees.
4. **Entropy Floor**:
   * Policy entropy must stay between **0.70 and 0.90** to prevent policy collapse into single-direction bias (e.g. permanent shorting).

> **Golden Rule**: If a challenger is rejected or only beats the champion by marginal noise ($< 0.3$), **KEEP THE CHAMPION RUNNING**. Never force a push just to fulfill a quota.

---

## 5. Deployment Checklist (Per Push)

When a model qualifies at the end of a session:
1. Verify tournament approval in `models/production/version.json`.
2. Push model weights to VPS via SSH:
   ```bash
   scp models/production/model_v1.pt root@72.62.255.1:/var/www/trading-bot/ai-trading-platform/models/production/model_v1.pt
   scp models/production/version.json root@72.62.255.1:/var/www/trading-bot/ai-trading-platform/models/production/version.json
   scp models/production/tournament_history.json root@72.62.255.1:/var/www/trading-bot/ai-trading-platform/models/production/tournament_history.json
   ```
3. The live trading bot detects the new model version via `ModelRegistry` and hot-reloads the neural weights dynamically without interrupting active positions.
