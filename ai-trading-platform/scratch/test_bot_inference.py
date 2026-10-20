import asyncio
import time
import torch
import numpy as np
from apps.trading_bot.main import ProductionTradingBot

async def test():
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'DOGEUSDT', '1000PEPEUSDT', 'ADAUSDT', 'RENDERUSDT']
    bot = ProductionTradingBot(symbols=symbols)
    await bot.connect_redis()
    # Populate market states from Redis
    for sym in symbols:
        raw = await bot.redis.redis.get(f"market:state:{sym}")
        if raw:
            from core.database.models.market import MarketState
            import json
            data = json.loads(raw)
            bot.market_states[sym] = MarketState(**data)
            
    print(f"Bot initialized. Total symbols: {len(bot.symbols)}")
    print(f"Loaded model version: {bot.registry.get_current_version()}")
    
    # Check 3 symbols: BTCUSDT, DOGEUSDT, 1000PEPEUSDT
    for sym in ['BTCUSDT', 'DOGEUSDT', '1000PEPEUSDT']:
        sv, mf = bot._build_state_vector(sym)
        arr = sv.squeeze(0).numpy()
        print(f"\n--- {sym} ---")
        print(f"State Vector: shape={sv.shape}, min={np.min(arr):.4f}, max={np.max(arr):.4f}, mean={np.mean(arr):.4f}, std={np.std(arr):.4f}")
        
        # Test model with h=None
        seq = [arr.tolist()] * 30
        st = torch.tensor([seq], dtype=torch.float32)
        with torch.no_grad():
            a, c = bot.model(st)
        raw_act = a[0, 0].item()
        raw_conf = (a[0, 1].item() + 1.0) / 2.0
        c5m = c[0, 0].item()
        print(f"Inference with h=None -> Act: {raw_act:+.3f}, Conf: {raw_conf*100:.1f}%, Critic: {c5m:+.4f}%")
        
        # Test model with carryover hidden state (simulate 10 ticks)
        h = None
        for i in range(10):
            with torch.no_grad():
                a, c, h = bot.model(st, hidden_state=h, return_hidden=True)
        raw_act_10 = a[0, 0].item()
        raw_conf_10 = (a[0, 1].item() + 1.0) / 2.0
        c5m_10 = c[0, 0].item()
        print(f"Inference with 10 ticks of h carryover -> Act: {raw_act_10:+.3f}, Conf: {raw_conf_10*100:.1f}%, Critic: {c5m_10:+.4f}%")
        
    await bot.binance.close()
    if bot.redis.redis:
        await bot.redis.redis.close()

if __name__ == "__main__":
    asyncio.run(test())
