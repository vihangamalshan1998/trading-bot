import urllib.request
import json
import time

url = 'https://fapi.binance.com/fapi/v1/ticker/24hr'
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
try:
    with urllib.request.urlopen(req, timeout=8) as resp:
        data = json.loads(resp.read().decode())

    tracked = {'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT', 'DOGEUSDT', 'NEARUSDT', 'LINKUSDT', 'AVAXUSDT', 'ADAUSDT', 'LTCUSDT'}
    print(f"LIVE MARKET STATE AT {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}:")
    for t in data:
        if t['symbol'] in tracked:
            pct = float(t['priceChangePercent'])
            h = float(t['highPrice'])
            l = float(t['lowPrice'])
            rng = ((h - l) / l) * 100
            print(f"{t['symbol']:<10}: 24h Change={pct:>+6.2f}%, 24h Range={rng:>5.2f}%, Last=${float(t['lastPrice']):.4f}")
except Exception as e:
    print(f"Error fetching live market: {e}")
