import asyncio, sys, torch, json
sys.path.insert(0, '.')
from apps.trading_bot.main import ProductionTradingBot
from core.config.settings import settings
from core.db.redis import redis_manager
from core.schemas.state_schema import MarketState

async def test():
    await redis_manager.connect()
    r = redis_manager.redis
    bot = ProductionTradingBot(symbols=settings.symbol_universe)
    await bot.sync_dashboard()
    
    # Load all live market states from Redis
    for sym in bot.symbols:
        raw_m = await r.get(f"market:state:{sym}")
        if raw_m:
            try:
                m_dict = json.loads(raw_m)
                ms = MarketState(**m_dict)
                bot.market_states[sym] = ms
            except Exception as e:
                pass
            
    print(f"Loaded {len(bot.market_states)} market states.")
    print("Equity:", bot.portfolio_state.equity, "Free margin:", bot.portfolio_state.free_margin)

    # Now let's build batch tensors exactly as main.py does in lines 872-902
    batch_tensors = []
    batch_syms = []
    precomputed_seqs = {}
    for sym in bot.symbols:
        if sym not in bot.market_states:
            continue
        sv, mf = bot._build_state_vector(sym)
        sv_list = sv.squeeze(0).tolist()
        seq = [sv_list] * 300
        st = torch.tensor(seq, dtype=torch.float32)
        if st.shape == (300, 200):
            batch_tensors.append(st)
            batch_syms.append(sym)
            precomputed_seqs[sym] = (sv, seq)

    final_batch = torch.stack(batch_tensors)
    with torch.no_grad():
        batch_logits, batch_critic, _ = bot.model(final_batch, return_hidden=True)

    print("\n=== LIVE INFERENCE FROM EXACT BOT LOGIC ===")
    for i, sym in enumerate(batch_syms):
        act_logits = batch_logits[i].numpy()
        critic = batch_critic[i].numpy()
        
        # Linear unscaling currently in main.py:
        conf_linear = float(torch.clamp(torch.tensor((act_logits[1] + 1.0) / 2.0), 0.05, 0.99))
        size_linear = float(torch.clamp(torch.tensor((act_logits[2] + 1.0) / 2.0), 0.25, 1.0))
        
        print(f"[{sym:12s}] logits: [dir={act_logits[0]:+6.3f}, c_raw={act_logits[1]:+6.3f}, s_raw={act_logits[2]:+6.3f}] | Conf: {conf_linear*100:5.1f}% | Size: {size_linear*100:5.1f}% | Critic: {critic[0]:+.3f}%")

if __name__ == '__main__':
    asyncio.run(test())
