import asyncio
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.getcwd())
from core.exchange.binance_client import BinanceFuturesAdapter

async def run():
    adapter = BinanceFuturesAdapter()
    await adapter.connect()
    # Start from Oct 14 11:30 UTC (17:00 Sri Lanka time)
    start_dt = datetime(2026, 10, 14, 11, 30, 0)
    start_ms = int(start_dt.timestamp() * 1000)
    records = await adapter._request('GET', '/fapi/v1/income', signed=True, params={'startTime': start_ms, 'limit': 1000})
    if not isinstance(records, list):
        print('Error:', records)
        return
    print(f'Total income records retrieved: {len(records)}')
    
    hourly = {}
    for r in records:
        ts = r['time'] / 1000 + 5.5 * 3600 # Sri Lanka time (UTC+5.30)
        hr = datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:00')
        itype = r['incomeType']
        amt = float(r['income'])
        if hr not in hourly:
            hourly[hr] = {'wins': 0.0, 'losses': 0.0, 'comm': 0.0, 'win_count': 0, 'loss_count': 0, 'transfers': 0.0}
        if itype == 'REALIZED_PNL':
            if amt > 0:
                hourly[hr]['wins'] += amt
                hourly[hr]['win_count'] += 1
            else:
                hourly[hr]['losses'] += amt
                hourly[hr]['loss_count'] += 1
        elif itype == 'COMMISSION':
            hourly[hr]['comm'] += amt
        elif itype == 'TRANSFER':
            hourly[hr]['transfers'] += amt

    print(f"{'Hour (SL Time)':<18} | {'Wins ($)':<15} | {'Losses ($)':<16} | {'Comm ($)':<10} | {'Net ($)':<10} | {'Transfers ($)':<12}")
    print('-' * 95)
    for hr in sorted(hourly.keys()):
        h = hourly[hr]
        net = h['wins'] + h['losses'] + h['comm']
        w_str = f"+{h['wins']:.2f} ({h['win_count']}w)"
        l_str = f"{h['losses']:.2f} ({h['loss_count']}l)"
        print(f"{hr:<18} | {w_str:<15} | {l_str:<16} | {h['comm']:<10.2f} | {net:<10.2f} | {h['transfers']:<12.2f}")
    if hasattr(adapter, 'close'):
        await adapter.close()

if __name__ == '__main__':
    asyncio.run(run())
