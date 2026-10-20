import asyncio, sys, uuid
sys.path.insert(0, '.')
from core.exchange.binance_client import BinanceFuturesAdapter
from core.universe.screener import registry

async def close_btc_hedge():
    b = BinanceFuturesAdapter()
    await b.connect()
    positions = await b.get_positions()
    btc_pos = [p for p in positions if p.get('symbol') == 'BTCUSDT']
    if not btc_pos:
        print("No BTCUSDT position found.")
        await b.close()
        return
        
    amt = float(btc_pos[0].get('positionAmt', 0))
    print(f"Current BTC position: {amt}")
    if abs(amt) > 0:
        side = "BUY" if amt < 0 else "SELL"
        qty = abs(amt)
        sym_config = registry.get_symbol("BTCUSDT")
        fmt_qty = float(sym_config.format_quantity(qty)) if sym_config else round(qty, 3)
        print(f"Closing BTC hedge: {side} {fmt_qty} BTCUSDT (MARKET, reduceOnly=True)...")
        res = await b.create_order(
            symbol="BTCUSDT",
            side=side,
            quantity=fmt_qty,
            client_order_id=f"close_hedge_{uuid.uuid4().hex[:8]}",
            order_type="MARKET",
            reduce_only=True
        )
        print("Order result:", res)
    else:
        print("BTC position is already 0.")
    await b.close()

if __name__ == '__main__':
    asyncio.run(close_btc_hedge())
