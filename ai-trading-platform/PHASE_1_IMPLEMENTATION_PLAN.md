# Phase 1: Full Technical Implementation Plan
*WARNING: Do not execute these steps today. Save this document. Let V3 train for 2-4 weeks to prove profitability before tearing the code apart to implement this.*

## A-Z Risk Assessment (What Will Break)
Before writing any code, here are the absolute failure points you must watch out for when implementing Phase 1:

1. **The Msgpack Crash:** If you change `main.py` to publish in `msgpack`, but forget to update `apps/experience/storage.py` to unpack in `msgpack`, the storage service will crash instantly trying to parse binary data as JSON. You will silently lose all trade data.
2. **The `torch.stack` Death Spiral:** In Batched Inference, you will group all 6 coins into one Tensor. If *just one* coin is missing a data point (e.g., it only has 299 rows instead of 300), `torch.stack` will throw a dimension mismatch error and crash the **entire** bot, taking down the other 5 healthy coins with it.
3. **The Hallucination Wipe:** Implementing Smart Memory changes how time works. If you do not run `python scripts/reset_v3_db.py` the exact second you push the Smart Memory code, the AI will mix old time-based data with new tick-based data and mathematically hallucinate, destroying your Win Rate.

---

## ✅ COMPLETED IMPLEMENTATIONS

The following items have been fully hardcoded and deployed into the current architecture.

### 1. Vector Compression (Msgpack) [✅ COMPLETED]
Replaced bloated JSON strings in Redis Pub/Sub with `msgpack` binary serialization in `main.py` and `storage.py`.

### 2. Batched Live Inference [✅ COMPLETED]
`main.py` now collects all coin tensors and executes a single batched `torch.stack` inference pass, slashing PyTorch CPU load.

### 3. The "Double-Counting" Math Bug [✅ COMPLETED]
Updated `apps/trainer/ppo.py` to stop applying recursive Temporal Difference (TD) learning since the database already calculates the exact future multi-horizon returns perfectly.

### 4. The LSTM "Amnesia" Bug [✅ COMPLETED]
Updated `apps/research/model.py` and `apps/trading_bot/main.py` to retain and pass the LSTM `hidden_state` per-symbol on every tick, giving the AI true continuous memory.

### 5. Priority Experience Replay (PER) SumTree [✅ COMPLETED]
Created `core/ai/sumtree.py` and updated `core/ai/replay_buffer.py` to sample proportional to the absolute value of rewards, forcing the AI to focus on its biggest mistakes.


## ⏳ PENDING UPGRADES (Phase 1)

### 🛠️ Pending Task 1: Implementing Smart Memory (Volatility Triggers)

**Update `main.py` (Market Data Loop):**
We abandon the 1-second timer and only record data if price moves.

```python
# ADD TO __init__:
self.last_price = {sym: 0.0 for sym in SYMBOLS}

# IN THE DATA LOOP:
current_price = market.mid_price
last_price = self.last_price[sym]

# Calculate % change
if last_price == 0.0:
    price_change = 1.0 # Force first record
else:
    price_change = abs((current_price - last_price) / last_price) * 100

# Only append to history if price moved more than 0.1%
if price_change >= 0.1:
    self.state_history[sym].append(features)
    self.last_price[sym] = current_price # Reset trigger
```
*CRITICAL: You must wipe the database and `.pt` file immediately after pushing this code.*
