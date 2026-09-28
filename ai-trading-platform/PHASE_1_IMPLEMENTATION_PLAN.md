# Phase 1: Full Technical Implementation Plan
*WARNING: Do not execute these steps today. Save this document. Let V3 train for 2-4 weeks to prove profitability before tearing the code apart to implement this.*

## A-Z Risk Assessment (What Will Break)
Before writing any code, here are the absolute failure points you must watch out for when implementing Phase 1:

1. **The Msgpack Crash:** If you change `main.py` to publish in `msgpack`, but forget to update `apps/experience/storage.py` to unpack in `msgpack`, the storage service will crash instantly trying to parse binary data as JSON. You will silently lose all trade data.
2. **The `torch.stack` Death Spiral:** In Batched Inference, you will group all 6 coins into one Tensor. If *just one* coin is missing a data point (e.g., it only has 299 rows instead of 300), `torch.stack` will throw a dimension mismatch error and crash the **entire** bot, taking down the other 5 healthy coins with it.
3. **The Hallucination Wipe:** Implementing Smart Memory changes how time works. If you do not run `python scripts/reset_v3_db.py` the exact second you push the Smart Memory code, the AI will mix old time-based data with new tick-based data and mathematically hallucinate, destroying your Win Rate.

---

## 🛠️ Step 1: Implementing Vector Compression (Msgpack)

**1. Install Dependency:**
```bash
pip install msgpack-python
```

**2. Update `main.py` (Publishing):**
Find the line where the bot sends the experience:
```python
# OLD
await redis_manager.redis.publish("experience:completed", json.dumps(exp_data))

# NEW
import msgpack
await redis_manager.redis.publish("experience:completed", msgpack.packb(exp_data, use_bin_type=True))
```

**3. Update `apps/experience/storage.py` (Subscribing):**
Find the line where the storage reads the Redis message:
```python
# OLD
exp_data = json.loads(message['data'])

# NEW
import msgpack
exp_data = msgpack.unpackb(message['data'], raw=False)
```

---

## 🛠️ Step 2: Implementing Batched Live Inference

**Update `main.py` (Inference Loop):**
Instead of calling `self.model(seq_tensor)` inside the loop, we collect them all and execute once.

```python
# 1. Collect all valid sequences
batch_tensors = []
batch_symbols = []

for sym in SYMBOLS:
    # ... logic to check if sequence is ready ...
    if len(self.state_history[sym]) == 300:
        seq_tensor = torch.tensor(list(self.state_history[sym]), dtype=torch.float32)
        batch_tensors.append(seq_tensor)
        batch_symbols.append(sym)

# 2. Execute Batch Inference (Only if we have symbols to process)
if len(batch_tensors) > 0:
    # Stack them into shape [N, 300, 200]
    final_batch = torch.stack(batch_tensors).to(self.device)
    
    with torch.no_grad():
        # 1 single PyTorch execution for all coins!
        action_preds, values = self.model(final_batch)
        
    # 3. Distribute results back to symbols
    for i, sym in enumerate(batch_symbols):
        pred = action_preds[i]
        # ... proceed to Risk Manager logic ...
```

---

## 🛠️ Step 3: Implementing Smart Memory (Volatility Triggers)

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

---

## 🛠️ Step 4: Implementing PER SumTree

**1. Create `core/ai/sumtree.py`:**
```python
import numpy as np

class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write_idx = 0

    def update(self, idx, priority):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        while idx != 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change

    def add(self, priority, data):
        idx = self.write_idx + self.capacity - 1
        self.data[self.write_idx] = data
        self.update(idx, priority)
        self.write_idx += 1
        if self.write_idx >= self.capacity:
            self.write_idx = 0
```

**2. Update `ppo.py`:**
Remove random sampling and sample proportionally from the SumTree based on the absolute value of the reward `abs(reward)`.
