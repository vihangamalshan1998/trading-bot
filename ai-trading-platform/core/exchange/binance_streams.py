"""
Binance USD-M Futures websocket routing.

Binance split the mainnet futures market-data websocket by stream type (verified 2026-10-21):
  wss://fstream.binance.com/market/...  aggTrade, trade, markPrice, ticker, miniTicker, forceOrder, kline
  wss://fstream.binance.com/public/...  depth (diff and partial), bookTicker
On the legacy root (wss://fstream.binance.com/stream) only depth / bookTicker still deliver; the other
stream types connect fine but stay SILENT. That silently zeroed whale flow, Binance liquidations,
trade volume and mark-price premium. Testnet (stream.binancefuture.com) still serves everything on
one endpoint.
"""
from typing import Dict, List

MAINNET_MARKET = "wss://fstream.binance.com/market"
MAINNET_PUBLIC = "wss://fstream.binance.com/public"
TESTNET = "wss://stream.binancefuture.com"

PUBLIC_STREAM_TYPES = ("depth", "bookTicker")
MAX_STREAMS_PER_CONNECTION = 200


def stream_type(name: str) -> str:
    """'btcusdt@depth20@100ms' -> 'depth20', '!forceOrder@arr' -> 'forceOrder'."""
    part = name.split("@")[0] if name.startswith("!") else name.split("@")[1]
    return part.lstrip("!")


def endpoint_for(name: str, mainnet: bool) -> str:
    if not mainnet:
        return TESTNET
    return MAINNET_PUBLIC if stream_type(name).startswith(PUBLIC_STREAM_TYPES) else MAINNET_MARKET


def combined_stream_urls(stream_names: List[str], mainnet: bool) -> List[str]:
    """Groups streams by the endpoint that actually serves them; one combined URL per group/chunk."""
    groups: Dict[str, List[str]] = {}
    for name in stream_names:
        groups.setdefault(endpoint_for(name, mainnet), []).append(name)
    urls = []
    for base, names in groups.items():
        for i in range(0, len(names), MAX_STREAMS_PER_CONNECTION):
            urls.append(f"{base}/stream?streams={'/'.join(names[i:i + MAX_STREAMS_PER_CONNECTION])}")
    return urls


def single_stream_url(name: str, mainnet: bool) -> str:
    return f"{endpoint_for(name, mainnet)}/ws/{name}"
