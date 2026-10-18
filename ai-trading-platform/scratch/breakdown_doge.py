import asyncio
import sys
import datetime

sys.path.insert(0, ".")
from core.exchange.binance_client import BinanceFuturesAdapter

BENCHMARK_START_MS = 1792294612528 # Oct 18 03:36:52 UTC

async def main():
    client = BinanceFuturesAdapter()
    await client.connect()
    
    incomes = await client._request('GET', '/fapi/v1/income', signed=True, params={'symbol': 'DOGEUSDT', 'limit': 1000})
    
    bench_pnl, bench_fee = 0.0, 0.0
    pre_bench_pnl, pre_bench_fee = 0.0, 0.0
    
    print("=== ALL DOGE TRADES BREAKDOWN ===")
    for inc in incomes:
        itype = inc.get('incomeType')
        val = float(inc.get('income', 0.0))
        t_ms = int(inc.get('time', 0))
        dt = datetime.datetime.fromtimestamp(t_ms / 1000, tz=datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        
        is_bench = t_ms >= BENCHMARK_START_MS
        
        if itype == 'REALIZED_PNL':
            if is_bench:
                bench_pnl += val
                print(f"[CURRENT BENCHMARK {dt}] Realized PnL: {val:+.4f} USDT")
            else:
                pre_bench_pnl += val
        elif itype == 'COMMISSION':
            if is_bench:
                bench_fee += val
            else:
                pre_bench_fee += val
                
    print("\n--- SUMMARY ---")
    print(f"Current 72h Benchmark DOGE Realized PnL: {bench_pnl:+.4f} USDT")
    print(f"Current 72h Benchmark DOGE Fees:         {bench_fee:+.4f} USDT")
    print(f"Current 72h Benchmark DOGE Net:          {bench_pnl + bench_fee:+.4f} USDT")
    print(f"Historical (Pre-Upgrade) DOGE Net:       {pre_bench_pnl + pre_bench_fee:+.4f} USDT")

if __name__ == "__main__":
    asyncio.run(main())
