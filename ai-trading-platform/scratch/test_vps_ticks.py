import redis, json, torch, sys
sys.path.insert(0, '.')
from apps.trading_bot.main import ProductionTradingBot
from core.schemas.state_schema import MarketState
import asyncio

async def test_ticks():
    bot = ProductionTradingBot(symbols=['BTCUSDT', 'ETHUSDT', 'AVAXUSDT', 'SOLUSDT'])
    await bot.redis.connect()
    for sym in bot.symbols:
        raw_m = await bot.redis.redis.get(f'market:state:{sym}')
        if raw_m:
            bot.market_states[sym] = MarketState(**json.loads(raw_m))
            
    print("Testing 15 simulated consecutive ticks in main.py loop...")
    for tick in range(15):
        # Exactly replicate lines 903-949
        batch_tensors = []
        batch_syms = []
        for sym in bot.symbols:
            sv, mf = bot._build_state_vector(sym)
            seq = list(bot.state_history[sym]) + [sv.squeeze(0).tolist()]
            while len(seq) < bot.sequence_length:
                seq.insert(0, seq[0])
            seq = seq[-bot.sequence_length:]
            st = torch.tensor(seq, dtype=torch.float32)
            batch_tensors.append(st)
            batch_syms.append(sym)
            bot.state_history[sym].append(sv.squeeze(0).tolist())
            
        final_batch = torch.stack(batch_tensors)
        
        h_list, c_list = [], []
        has_hidden = False
        hidden_dim = 256
        for sym in batch_syms:
            h = bot.hidden_states.get(sym)
            if h is not None:
                h_list.append(h[0])
                c_list.append(h[1])
                has_hidden = True
            else:
                h_list.append(torch.zeros(1, 1, hidden_dim))
                c_list.append(torch.zeros(1, 1, hidden_dim))
        batch_hidden = (torch.cat(h_list, dim=1), torch.cat(c_list, dim=1)) if has_hidden else None
        
        with torch.no_grad():
            batch_logits, batch_critic, new_hidden_batch = bot.model(final_batch, hidden_state=batch_hidden, return_hidden=True)
            
        for i, sym in enumerate(batch_syms):
            bot.hidden_states[sym] = (new_hidden_batch[0][:, i:i+1, :].detach(), new_hidden_batch[1][:, i:i+1, :].detach())
            
        if tick in [0, 1, 2, 5, 9, 14]:
            btc_i = batch_syms.index('BTCUSDT')
            btc_act = batch_logits[btc_i].numpy()
            btc_crit = batch_critic[btc_i].numpy()
            print(f"Tick {tick:02d} [BTCUSDT]: act_dir={btc_act[0]:+.3f}, conf={(btc_act[1]+1)/2:.3f}, critic_5m={btc_crit[0]:+.4f}, critic_4h={btc_crit[2]:+.4f}")

if __name__ == '__main__':
    asyncio.run(test_ticks())
