import sys
import torch
import numpy as np
from apps.trading_bot.main import ProductionTradingBot
from apps.trainer.ppo import normalize_state_tensor

symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'DOGEUSDT', '1000PEPEUSDT', 'RENDERUSDT', 'LINKUSDT']
bot = ProductionTradingBot(symbols=symbols)

import asyncio
from core.schemas.state_schema import MarketState
import json
import redis as py_redis
r_client = py_redis.Redis(host='localhost', port=6379, db=0)

async def run():
    for sym in symbols:
        raw = r_client.get(f"market:state:{sym}")
        if raw:
            bot.market_states[sym] = MarketState(**json.loads(raw))
            
    print("Testing model on current live market states:")
    for sym in symbols:
        sv, mf = bot._build_state_vector(sym)
        sv_list = sv.squeeze(0).tolist()
        
        # Test 1: Single frame repeated 300 times (how bot starts)
        st_repeated = torch.tensor([[sv_list] * 300], dtype=torch.float32)
        with torch.no_grad():
            a_rep, c_rep = bot.model(st_repeated, hidden_state=None)
        conf_rep = (a_rep[0, 1].item() + 1.0) / 2.0
        crit_rep = c_rep[0, 0].item()
        
        print(f"[{sym:12s}] Repeated Frame -> Dir: {a_rep[0,0].item():+.3f} | Conf: {conf_rep*100:.1f}% | Critic: {crit_rep:+.5f}% (1h: {c_rep[0,1].item():+.4f}%)")
        
    await bot.binance.close()

if __name__ == "__main__":
    asyncio.run(run())
