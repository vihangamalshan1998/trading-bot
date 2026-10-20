import asyncio, sys, torch
sys.path.insert(0, '.')
from apps.trading_bot.main import ProductionTradingBot
from core.config.settings import settings

async def main():
    bot = ProductionTradingBot(settings.symbol_universe)
    await bot.redis.connect()
    
    # Load market states from Redis into bot.market_states
    for sym in bot.symbols:
        raw_m = await bot.redis.redis.get(f"market:state:{sym}")
        if raw_m:
            import json
            from core.schemas.state_schema import MarketState
            try:
                bot.market_states[sym] = MarketState(**json.loads(raw_m))
            except Exception as e:
                print(f"Error parsing {sym}: {e}")
                
    print(f"Loaded {len(bot.market_states)} market states into bot.")
    
    # Build state vector for each symbol and run model
    for sym in ["BTCUSDT", "ETHUSDT", "SOLUSDT", "NEARUSDT", "APTUSDT"]:
        if sym in bot.market_states:
            sv, mf = bot._build_state_vector(sym)
            vec = sv[0].numpy()
            print(f"\n[{sym}]")
            print(f"  Portfolio (dims 0-3): {vec[:4]}")
            print(f"  Position (dims 10-14): {vec[10:15]}")
            print(f"  Market (dims 20-24): {vec[20:25]}")
            print(f"  Non-zero market dims count: {(vec[20:110] != 0).sum()}")

            # Single inference
            seq = [vec.tolist()] * 300
            t = torch.tensor([seq], dtype=torch.float32)
            with torch.no_grad():
                a, c = bot.model(t)
            act = a[0].numpy()
            crit = c[0].numpy()
            conf_sigmoid = 1.0 / (1.0 + torch.exp(-torch.tensor(act[1])).item())
            conf_linear = (act[1] + 1.0) / 2.0
            print(f"  Model output:")
            print(f"    act raw: {act}")
            print(f"    critic:  {crit}")
            print(f"    conf (current sigmoid): {conf_sigmoid:.4f}")
            print(f"    conf (linear unscaled): {conf_linear:.4f}")

if __name__ == '__main__':
    asyncio.run(main())
