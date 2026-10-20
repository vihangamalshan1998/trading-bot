import redis, json, torch, sys
sys.path.insert(0, '.')
from apps.research.model import SingleSymbolActorCritic
from core.schemas.state_schema import MarketState
from apps.trading_bot.main import ProductionTradingBot
import asyncio

async def run():
    bot = ProductionTradingBot(symbols=['BTCUSDT', 'ETHUSDT', 'AVAXUSDT', 'SOLUSDT', 'RENDERUSDT', 'LINKUSDT'])
    await bot.redis.connect()
    
    # Load market states from redis
    for sym in bot.symbols:
        raw_m = await bot.redis.redis.get(f'market:state:{sym}')
        if raw_m:
            data = json.loads(raw_m)
            bot.market_states[sym] = MarketState(**data)
            
    # Load portfolio from redis
    p_raw = await bot.redis.redis.get('dashboard:portfolio')
    if p_raw:
        p_data = json.loads(p_raw)
        bot.portfolio_state.equity = p_data.get('equity', 5000.0)
        bot.portfolio_state.wallet_balance = p_data.get('equity', 5000.0)
        bot.portfolio_state.free_margin = p_data.get('equity', 5000.0)
    
    print("\n=== COMPARING OBS VECTORS ACROSS SYMBOLS ===")
    for sym in ['BTCUSDT', 'ETHUSDT', 'AVAXUSDT', 'SOLUSDT', 'RENDERUSDT', 'LINKUSDT']:
        sv, mf = bot._build_state_vector(sym)
        sv_np = sv.squeeze(0).numpy()
        print(f"[{sym:10s}] sv min={sv_np.min():.3f}, max={sv_np.max():.3f}, mean={sv_np.mean():.4f}, norm={np.linalg.norm(sv_np):.3f}")
        print(f"  Portfolio (0:10):  {sv_np[0:10].round(3)}")
        print(f"  Position  (10:20): {sv_np[10:20].round(3)}")
        print(f"  Feats     (20:25): {sv_np[20:25].round(4)}")
        print(f"  Whale     (130:135): {sv_np[130:135].round(3)}")
        
        # Test model output directly on this symbol's sequence
        seq = [sv_np.tolist()] * 300
        st = torch.tensor([seq], dtype=torch.float32)
        with torch.no_grad():
            act, crit = bot.model(st)
        act_np = act[0].numpy()
        crit_np = crit[0].numpy()
        print(f"  DIRECT MODEL: act={act_np[0]:+.3f}, conf={(act_np[1]+1)/2:.3f}, size={(act_np[2]+1)/2:.3f} | critic_5m={crit_np[0]:+.4f}, 1h={crit_np[1]:+.4f}, 4h={crit_np[2]:+.4f}")
    
    pass

if __name__ == '__main__':
    import numpy as np
    asyncio.run(run())
