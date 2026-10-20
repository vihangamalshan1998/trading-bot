import redis, json, torch, sys
sys.path.insert(0, '.')
from apps.trading_bot.main import ProductionTradingBot
from core.config.settings import settings
from core.schemas.state_schema import MarketState
import asyncio

async def test_full_universe():
    bot = ProductionTradingBot(symbols=settings.symbol_universe)
    await bot.redis.connect()
    
    # Load market states
    for sym in bot.symbols:
        raw_m = await bot.redis.redis.get(f'market:state:{sym}')
        if raw_m:
            bot.market_states[sym] = MarketState(**json.loads(raw_m))
            
    # Load portfolio
    p_raw = await bot.redis.redis.get('dashboard:portfolio')
    if p_raw:
        p_data = json.loads(p_raw)
        bot.portfolio_state.equity = p_data.get('equity', 5000.0)
        bot.portfolio_state.wallet_balance = p_data.get('equity', 5000.0)
        bot.portfolio_state.free_margin = p_data.get('equity', 5000.0)

    # Now run the exact inference loop batch pass
    batch_tensors = []
    batch_syms = []
    precomputed_seqs = {}
    for sym in bot.symbols:
        if sym not in bot.market_states:
            print(f"[{sym}] NOT in market_states!")
            continue
        sv, mf = bot._build_state_vector(sym)
        sv_list = sv.squeeze(0).tolist()
        seq = list(bot.state_history[sym]) + [sv_list]
        while len(seq) < bot.sequence_length:
            seq.insert(0, seq[0])
        seq = seq[-bot.sequence_length:]
        st = torch.tensor(seq, dtype=torch.float32)
        batch_tensors.append(st)
        batch_syms.append(sym)
        precomputed_seqs[sym] = (sv, seq)

    final_batch = torch.stack(batch_tensors)
    with torch.no_grad():
        batch_logits, batch_critic, new_hidden_batch = bot.model(final_batch, hidden_state=None, return_hidden=True)

    print("\n=== OUTPUTS OF BATCH PASS WITH REAL REDIS MARKET STATES ===")
    for i, sym in enumerate(batch_syms):
        act = batch_logits[i].numpy()
        crit = batch_critic[i].numpy()
        side = "LONG" if act[0] > 0.2 else ("SHORT" if act[0] < -0.2 else "HOLD")
        print(f"[{sym:12s}] act={act[0]:+.3f} ({side:5s}) | conf={(act[1]+1)/2*100:.1f}% | size={(act[2]+1)/2*100:.1f}% | critic 5m={crit[0]:+.4f}%, 1h={crit[1]:+.4f}%, 4h={crit[2]:+.4f}%")

if __name__ == '__main__':
    asyncio.run(test_full_universe())
