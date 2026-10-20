"""Reproduce live collapse: accumulate REAL changing frames for 330 ticks (like the bot),
compare model outputs with raw inputs (current live bot) vs trainer-normalized inputs."""
import time, json, asyncio
import torch
import redis as py_redis
from apps.trading_bot.main import ProductionTradingBot
from apps.trainer.ppo import normalize_state_tensor
from core.schemas.state_schema import MarketState

symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'DOGEUSDT', '1000PEPEUSDT', 'LINKUSDT']
bot = ProductionTradingBot(symbols=symbols)
r = py_redis.Redis()

def run_batch(normalize):
    seqs = []
    for sym in symbols:
        seq = list(bot.state_history[sym])
        while len(seq) < bot.sequence_length:
            seq.insert(0, seq[0])
        seqs.append(torch.tensor(seq[-bot.sequence_length:], dtype=torch.float32))
    b = torch.nan_to_num(torch.stack(seqs), nan=0.0, posinf=1.0, neginf=-1.0)
    if normalize:
        b = normalize_state_tensor(b)
    with torch.no_grad():
        a, c = bot.model(b, hidden_state=None, return_hidden=False)
    return a, c, b

for tick in range(1, 331):
    for sym in symbols:
        raw = r.get(f"market:state:{sym}")
        if raw:
            bot.market_states[sym] = MarketState(**json.loads(raw))
        sv, _ = bot._build_state_vector(sym)
        bot.state_history[sym].append(sv.squeeze(0).tolist())
    if tick in (1, 30, 100, 200, 300, 330):
        print(f"\n=== TICK {tick} ===", flush=True)
        for norm in (False, True):
            a, c, b = run_batch(norm)
            print(f"  {'NORMALIZED' if norm else 'RAW (live bot)'}  max|input|={b.abs().max().item():.1f}", flush=True)
            for i, sym in enumerate(symbols):
                conf = (a[i, 1].item() + 1) / 2
                print(f"    {sym:13s} dir={a[i,0].item():+.3f} conf={conf*100:5.1f}% c5m={c[i,0].item():+.5f}", flush=True)
        if tick == 330:
            _, _, b = run_batch(False)
            big = (b.abs() > 5.0).any(dim=0).any(dim=0).nonzero().flatten().tolist()
            print(f"  Slots exceeding |5| in raw input: {big}")
            for s in big[:20]:
                print(f"    slot {s}: min={b[...,s].min().item():.2f} max={b[...,s].max().item():.2f}")
    time.sleep(1.0)

asyncio.run(bot.binance.close())
