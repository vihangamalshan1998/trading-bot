import redis
import json
import torch
from apps.trading_bot.main import ProductionTradingBot
from core.schemas.state_schema import MarketState

bot = ProductionTradingBot(symbols=['BTCUSDT'])
r = redis.Redis()
raw = r.get('market:state:BTCUSDT')
if raw:
    bot.market_states['BTCUSDT'] = MarketState(**json.loads(raw))

sv, mf = bot._build_state_vector('BTCUSDT')
arr = sv.squeeze(0).tolist()
print('Full 200-dim vector for BTCUSDT:')
for i in range(0, 200, 10):
    chunk = [f'{x:7.2f}' for x in arr[i:i+10]]
    print(f'{i:3d}-{i+9:3d}:', ' '.join(chunk))

# Now test this exact sv on bot.model
st = torch.tensor([[arr] * 300], dtype=torch.float32)
with torch.no_grad():
    a, c = bot.model(st, hidden_state=None)
print('\nInference on this single repeated frame:')
print('  Actor:', a.detach().numpy())
print('  Critic:', c.detach().numpy())
