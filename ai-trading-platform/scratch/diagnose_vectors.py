import sys, torch
sys.path.insert(0, '/var/www/trading-bot/ai-trading-platform')
from apps.trading_bot.main import ProductionTradingBot

bot = ProductionTradingBot(symbols=['BTCUSDT', 'SOLUSDT', 'BNBUSDT'])
# Check if market states exist
for sym in ['BTCUSDT', 'SOLUSDT', 'BNBUSDT']:
    print(f"{sym} in market_states:", sym in bot.market_states)

v_btc, _ = bot._build_state_vector('BTCUSDT')
v_sol, _ = bot._build_state_vector('SOLUSDT')

print("BTC vector non-zero count:", (v_btc != 0).sum().item())
print("SOL vector non-zero count:", (v_sol != 0).sum().item())
print("Abs diff between BTC and SOL:", (v_btc - v_sol).abs().sum().item())

# Test model on both
bot.model.eval()
with torch.no_grad():
    a_btc, c_btc = bot.model(v_btc.unsqueeze(1))
    a_sol, c_sol = bot.model(v_sol.unsqueeze(1))

print("BTC action:", a_btc.numpy())
print("SOL action:", a_sol.numpy())
print("BTC critic:", c_btc.numpy())
print("SOL critic:", c_sol.numpy())
