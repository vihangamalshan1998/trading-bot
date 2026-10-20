import redis, json, torch, sys
sys.path.insert(0, '.')
from apps.trading_bot.main import ProductionTradingBot
from core.config.settings import settings
from core.schemas.state_schema import MarketState
import asyncio

async def test_hidden_effect():
    bot = ProductionTradingBot(symbols=settings.symbol_universe)
    await bot.redis.connect()
    
    # Load market states
    for sym in bot.symbols:
        raw_m = await bot.redis.redis.get(f'market:state:{sym}')
        if raw_m:
            bot.market_states[sym] = MarketState(**json.loads(raw_m))

    batch_tensors = []
    batch_syms = []
    for sym in bot.symbols:
        if sym not in bot.market_states:
            continue
        sv, mf = bot._build_state_vector(sym)
        sv_list = sv.squeeze(0).tolist()
        seq = [sv_list] * 300
        st = torch.tensor(seq, dtype=torch.float32)
        batch_tensors.append(st)
        batch_syms.append(sym)

    final_batch = torch.stack(batch_tensors)
    
    print("\nSimulating 10 ticks with hidden state carryover:")
    hidden = None
    for tick in range(10):
        with torch.no_grad():
            batch_logits, batch_critic, hidden = bot.model(final_batch, hidden_state=hidden, return_hidden=True)
        btc_i = batch_syms.index('BTCUSDT')
        act = batch_logits[btc_i].numpy()
        crit = batch_critic[btc_i].numpy()
        print(f"Tick {tick:02d}: act={act[0]:+.3f}, conf={(act[1]+1)/2*100:.1f}%, critic_5m={crit[0]:+.4f}%, h_norm={hidden[0].norm():.2f}")

if __name__ == '__main__':
    asyncio.run(test_hidden_effect())
