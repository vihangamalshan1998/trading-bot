import asyncio
import os
import sys
import time
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.exchange.binance_client import BinanceFuturesAdapter
from core.config.settings import settings
from apps.research.model import SingleSymbolActorCritic

async def test_live_market_conditions():
    client = BinanceFuturesAdapter()
    await client.connect()
    
    print("=" * 75)
    print("      DEEP FORENSIC INVESTIGATION: WHY NO TRADES ARE OPENING       ")
    print("=" * 75)
    
    # 1. Fetch 24hr tickers from Binance
    tickers = await client.get_24hr_ticker()
    ticker_map = {t["symbol"]: t for t in tickers if t.get("symbol") in settings.symbol_universe}
    
    print(f"\n1. MARKET CONDITION CHECK (Total Symbols Tracked: {len(ticker_map)})")
    print(f"Current UTC Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())}")
    current_utc_hour = time.gmtime().tm_hour
    print(f"Current Session: {'ASIA CHOP (00:00-08:00 UTC)' if 0 <= current_utc_hour < 8 else 'LONDON/NY'}")
    
    # Analyze market volatility and 24h price changes
    changes = []
    for sym in settings.symbol_universe:
        t = ticker_map.get(sym)
        if t:
            pct_change = float(t.get("priceChangePercent", 0.0))
            high = float(t.get("highPrice", 0.0))
            low = float(t.get("lowPrice", 0.0))
            last = float(t.get("lastPrice", 0.0))
            vol_range = ((high - low) / low * 100) if low > 0 else 0.0
            changes.append((sym, pct_change, vol_range, last))
            
    print("\nSymbol Price Performance & 24h Range:")
    for sym, pct, v_range, last in changes[:10]:
        print(f"  {sym:<12}: 24h Change = {pct:>+6.2f}%, 24h High-Low Range = {v_range:>5.2f}%, Price = ${last:.4f}")
        
    # Check BTC specifically (the market leader)
    btc_t = ticker_map.get("BTCUSDT", {})
    btc_change = float(btc_t.get("priceChangePercent", 0.0))
    btc_vol_range = ((float(btc_t.get("highPrice", 1)) - float(btc_t.get("lowPrice", 1))) / float(btc_t.get("lowPrice", 1))) * 100
    print(f"\n  >> BTC 24h Change: {btc_change:+.2f}%, Range: {btc_vol_range:.2f}%")
    
    # 2. Check Model Weights & Checkpoint
    checkpoint_path = "models/production/model_v1.pt"
    has_checkpoint = os.path.exists(checkpoint_path)
    print(f"\n2. BRAIN CHECK (Model Checkpoint)")
    print(f"  Checkpoint File: {checkpoint_path}")
    print(f"  Exists: {has_checkpoint}")
    if has_checkpoint:
        mtime = os.path.getmtime(checkpoint_path)
        print(f"  Last Modified: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(mtime))}")
        file_size = os.path.getsize(checkpoint_path)
        print(f"  File Size: {file_size / 1024:.1f} KB")
        
        # Load and inspect model outputs
        model = SingleSymbolActorCritic(input_dim=200)
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
        
        # Test synthetic sample to see baseline activation
        sample_input = torch.zeros((1, 300, 200), dtype=torch.float32)
        with torch.no_grad():
            act, exp_ret = model(sample_input)
            raw_act = float(act[0, 0])
            raw_conf = float(1.0 / (1.0 + np.exp(-float(act[0, 1]))))
            exp_5m = float(exp_ret[0, 0])
            print(f"  Model Zero-Baseline Response: Action={raw_act:+.3f}, Conf={raw_conf:.3f}, ExpReturn={exp_5m:+.3f}%")
            
    # 3. Check Recent Account Trades & Activity from Binance
    print(f"\n3. ACCOUNT EXECUTION HISTORY (Recent Fills on Binance)")
    try:
        incomes = await client._request("GET", "/fapi/v1/income", signed=True, params={"limit": 20})
        realized = [i for i in incomes if i.get("incomeType") == "REALIZED_PNL"]
        print(f"  Total Recent Realized PnL Events Found: {len(realized)}")
        for r in realized[:5]:
            t_str = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(float(r.get('time', 0))/1000))
            print(f"    {t_str} | {r.get('symbol'):<10} | PnL: ${float(r.get('income', 0)):>+7.4f}")
    except Exception as e:
        print(f"  Could not fetch income history: {e}")
        
    await client.close()

if __name__ == "__main__":
    asyncio.run(test_live_market_conditions())
