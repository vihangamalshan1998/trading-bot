# 72-Hour Live Experiment Baseline & Forensic Protocol

## 1. Executive Agreement & Absolute Code Freeze

* **Experiment Window**: **Sunday, Oct 18, 2026 @ 09:06:52 IST (03:36:52 UTC) $\rightarrow$ Wednesday, Oct 21, 2026 @ 09:00:00 IST (03:30:00 UTC)**.
* **Duration**: Exactly **72 Hours (3 Global Trading Days: Asia, London, New York)**.
* **Policy**: **ABSOLUTE CODE FREEZE**.
  * Zero code edits to `main.py` or risk managers.
  * Zero threshold tampering (confidence, spreads, leverage stay untouched).
  * The system runs continuously and autonomously on the VPS to capture pure, uncorrupted statistical expectancy.

---

## 2. Ground Truth Benchmark Starting State

The following figures were captured directly from the live Binance Testnet ledger at the moment of code freeze:

| Metric | Ground Truth Value |
| :--- | :--- |
| **Benchmark Start Time (UTC)** | `2026-10-18 03:36:52 UTC` |
| **Benchmark Start Time (IST)** | `2026-10-18 09:06:52 IST` |
| **Benchmark Epoch Timestamp** | `1792294612528 ms` |
| **Starting Wallet Balance** | **`$4,440.1045 USDT`** |
| **Starting Available Balance** | **`$4,440.1045 USDT`** |
| **Starting Unrealized PnL** | **`$0.0000 USDT`** |
| **Starting Net Equity** | **`$4,440.1045 USDT`** |
| **Open Positions at Start** | **`0` (Completely flat liquid cash)** |
| **Active Open Orders** | **`0`** |
| **Last Income Transaction ID** | `90101174260245` (2026-10-18 00:36:35 UTC) |

---

## 3. Active Architecture Under Test

* **VPS Git Commit**: `1524378` (includes `77255bc`).
* **Active Production Brain**: `v1_push_79` (Champion Score: 20.75, Critic MSE: 2.39).
* **Symbol Universe (18 Pairs)**: `BTCUSDT`, `ETHUSDT`, `SOLUSDT`, `BNBUSDT`, `XRPUSDT`, `LINKUSDT`, `ADAUSDT`, `AVAXUSDT`, `NEARUSDT`, `LTCUSDT`, `SUIUSDT`, `DOGEUSDT`, `RENDERUSDT`, `1000PEPEUSDT`, `APTUSDT`, `TAOUSDT`, `WIFUSDT`, `INJUSDT`.
* **Fee Reduction Engine**: **Hybrid Maker Exit Active (0.02% Post-Only GTX)** with 15-second immediate market close fallback.
* **Risk Firewall**:
  * Airbag Hard Stop: `-15.0%` leveraged.
  * Daily Circuit Breaker Handbrake: `5.0%` drawdown.
  * Net Portfolio Beta Governor: Max 5 concurrent unidirectional trades.
  * Spread Firewall: `0.045%` weekend, `0.060%` weekdays.
* **Anti-Churn Cooldown**: `180s` baseline, `120s` mandatory rest.

---

## 4. Evaluation Criteria for Wednesday Audit

On Wednesday, October 21 at 09:00 AM IST, the 72-hour test will be evaluated against these quantitative standards:

1. **Fee Reduction Ratio**:
   * Verification that profitable exits successfully executed as **Maker (0.02% fee)** instead of Taker (0.05%), cutting total fee bleed by $\ge 50\%$.
2. **Gross vs Net Expectancy**:
   * Evaluation of gross directional accuracy (model win-rate and profit factor) across Monday–Tuesday peak institutional liquidity.
3. **Net Capital Preservation**:
   * Net account balance relative to the **`$4,440.1045 USDT`** baseline.

---

## 5. How to Run the Forensic Audit on Wednesday

On Wednesday morning at 09:00 AM IST, run this single command to produce the complete forensic report:

```bash
venv\Scripts\python scratch/audit_72h_window.py
```

This script is pre-programmed with the exact starting timestamp (`1792294612528 ms`) and baseline balance (`$4,440.1045 USDT`), calculating exact gross PnL, commissions paid, funding fees, win rate, and net ROI over the 72-hour test.
