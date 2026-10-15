import asyncio
from core.exchange.binance_client import BinanceFuturesAdapter

async def test_symbols():
    client = BinanceFuturesAdapter()
    info = await client._request("GET", "/fapi/v1/exchangeInfo")
    active = {s["symbol"] for s in info["symbols"] if s.get("status") == "TRADING" and s.get("contractType") == "PERPETUAL"}
    candidates = [
        "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
        "LINKUSDT", "ADAUSDT", "AVAXUSDT", "NEARUSDT", "LTCUSDT",
        "DOTUSDT", "SUIUSDT"
    ]
    print("Available candidates on Binance Futures Testnet:")
    for c in candidates:
        print(f"  {c}: {'ACTIVE' if c in active else 'MISSING'}")
    await client.close()

if __name__ == "__main__":
    asyncio.run(test_symbols())
