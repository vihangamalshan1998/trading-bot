import torch
import numpy as np
import json
import redis as py_redis
from apps.trading_bot.main import ProductionTradingBot
from core.schemas.state_schema import MarketState

symbols = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT",
    "ADAUSDT", "1000PEPEUSDT", "NEARUSDT", "AVAXUSDT", "LINKUSDT",
    "LTCUSDT", "APTUSDT", "RENDERUSDT", "TAOUSDT", "INJUSDT", "WIFUSDT"
]

bot = ProductionTradingBot(symbols=symbols)
r = py_redis.Redis()

for sym in symbols:
    raw = r.get(f"market:state:{sym}")
    if raw:
        bot.market_states[sym] = MarketState(**json.loads(raw))

print(f"Loaded market states for {len(bot.market_states)} symbols.")

# Run the exact code from main.py:
batch_tensors = []
batch_syms = []
precomputed_seqs = {}
precomputed_actions = {}
precomputed_critics = {}

for sym in bot.symbols:
    if sym not in bot.market_states:
        continue
    sv, mf = bot._build_state_vector(sym)
    sv_list = sv.squeeze(0).tolist()
    seq = list(bot.state_history[sym]) + [sv_list]
    while len(seq) < bot.sequence_length:
        seq.insert(0, seq[0])
    seq = seq[-bot.sequence_length:]
    
    st = torch.tensor(seq, dtype=torch.float32)
    if st.shape == (bot.sequence_length, 200):
        batch_tensors.append(st)
        batch_syms.append(sym)
        precomputed_seqs[sym] = (sv, seq)

final_batch = torch.stack(batch_tensors)
final_batch = torch.nan_to_num(final_batch, nan=0.0, posinf=1.0, neginf=-1.0)
print(f"final_batch shape: {final_batch.shape}")

with torch.no_grad():
    batch_logits, batch_critic = bot.model(final_batch, hidden_state=None, return_hidden=False)

print("\n=== BATCH MODEL OUTPUTS ===")
for i, sym in enumerate(batch_syms):
    act = batch_logits[i].numpy()
    crit = batch_critic[i].numpy()
    conf = (act[1] + 1.0) / 2.0
    side = "LONG" if act[0] > 0.2 else ("SHORT" if act[0] < -0.2 else "HOLD")
    print(f"[{sym:12s}] Side={side:5s} | Conf={conf*100:5.1f}% | Critic 5m={crit[0]:+.4f}%, 1h={crit[1]:+.4f}%, 4h={crit[2]:+.4f}%")

print("\n=== COMPARE TO REDIS ai:state:* RIGHT NOW ===")
for sym in ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'RENDERUSDT']:
    redis_val = r.get(f"ai:state:{sym}")
    print(f"[{sym:12s}] in Redis -> {redis_val.decode() if redis_val else 'None'}")
