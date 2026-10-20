import sys
import torch
import numpy as np
import json
import redis as py_redis
from apps.trading_bot.main import ProductionTradingBot
from core.schemas.state_schema import MarketState

symbols = ['BTCUSDT', 'ETHUSDT', 'DOGEUSDT']
bot = ProductionTradingBot(symbols=symbols)
r = py_redis.Redis()

for sym in symbols:
    raw = r.get(f"market:state:{sym}")
    if raw:
        bot.market_states[sym] = MarketState(**json.loads(raw))

print("=== Simulating state_history accumulation over 300 ticks ===")
for tick in range(1, 301):
    for sym in symbols:
        sv, mf = bot._build_state_vector(sym)
        bot.state_history[sym].append(sv.squeeze(0).tolist())
        
    if tick in [1, 2, 5, 10, 50, 100, 200, 300]:
        print(f"\n--- TICK {tick} (history len: {len(bot.state_history['BTCUSDT'])}) ---")
        batch_tensors = []
        for sym in symbols:
            seq = list(bot.state_history[sym])
            while len(seq) < bot.sequence_length:
                seq.insert(0, seq[0])
            seq = seq[-bot.sequence_length:]
            batch_tensors.append(torch.tensor(seq, dtype=torch.float32))
            
        final_batch = torch.stack(batch_tensors)
        with torch.no_grad():
            batch_logits, batch_critic = bot.model(final_batch, hidden_state=None)
            
        for i, sym in enumerate(symbols):
            c0 = batch_critic[i, 0].item()
            c1 = batch_critic[i, 1].item()
            conf = (batch_logits[i, 1].item() + 1.0) / 2.0
            print(f"  [{sym:10s}] Conf: {conf*100:.1f}% | Critic 5m: {c0:+.5f}%, 1h: {c1:+.5f}%")
