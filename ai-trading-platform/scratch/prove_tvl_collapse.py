"""Proof: load REAL macro state from Redis (contains raw defi_tvl ~4.95e10) and compare outputs."""
import json, asyncio, torch, redis as py_redis
from apps.trading_bot.main import ProductionTradingBot
from core.schemas.state_schema import MarketState

symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'DOGEUSDT', '1000PEPEUSDT', 'LINKUSDT']
bot = ProductionTradingBot(symbols=symbols)
r = py_redis.Redis()
for s in symbols:
    raw = r.get(f"market:state:{s}")
    if raw:
        bot.market_states[s] = MarketState(**json.loads(raw))
macro = json.loads(r.get("dashboard:macro_state"))
raw_tvl = macro["defi_tvl_momentum"]
bot.macro_state.defi_tvl_momentum = raw_tvl
print("defi_tvl_momentum in Redis:", raw_tvl)

def run(inject_raw):
    seqs = []
    for s in symbols:
        sv, _ = bot._build_state_vector(s)
        if inject_raw:
            sv[0, 138] = raw_tvl   # what the OLD code fed the model
        seqs.append(sv.repeat(300, 1))
    with torch.no_grad():
        a, c = bot.model(torch.stack(seqs), hidden_state=None, return_hidden=False)
    for i, s in enumerate(symbols):
        print(f"  {s:13s} conf={(a[i,1].item()+1)/2*100:6.2f}%  c5m={c[i,0].item():+.6f}")

print("\nOLD behaviour (raw TVL in slot 138):"); run(True)
print("\nFIXED (clamped input):"); run(False)
asyncio.run(bot.binance.close())
