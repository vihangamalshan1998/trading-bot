import asyncio
import websockets

async def test_stream(url):
    print(f"Connecting to {url}")
    try:
        async with websockets.connect(url) as ws:
            print(f"Connected to {url}!")
            msg = await asyncio.wait_for(ws.recv(), timeout=2.0)
            print(f"Received: {msg}")
    except Exception as e:
        print(f"Failed {url}: {e}")

async def main():
    urls = [
        "wss://testnet.binance.vision/ws/btcusdt@trade",
        "wss://stream.binance.vision:9443/ws/btcusdt@trade",
        "wss://testnet.binance.vision/stream?streams=btcusdt@trade",
        "wss://stream.binance.com:9443/ws/btcusdt@trade"
    ]
    for url in urls:
        await test_stream(url)

if __name__ == "__main__":
    asyncio.run(main())
