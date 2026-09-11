import json
import time
import asyncio
from typing import Dict, Any, Callable, Awaitable
import httpx
import websockets

from core.exchange.base import ExchangeAdapter
from core.config.settings import settings
from core.logging.logger import logger

class BinanceFuturesAdapter(ExchangeAdapter):
    """
    Adapter for Binance Futures Testnet (or Mainnet).
    """
    def __init__(self):
        self.is_testnet = settings.binance_testnet
        
        # Futures Base URLs
        if self.is_testnet:
            self.rest_url = "https://testnet.binancefuture.com"
            self.ws_url = "wss://stream.binancefuture.com/ws"
        else:
            self.rest_url = "https://fapi.binance.com"
            self.ws_url = "wss://fstream.binance.com/ws"
            
        self.api_key = settings.binance_api_key
        self.api_secret = settings.binance_api_secret
        
        self.client = httpx.AsyncClient(
            base_url=self.rest_url,
            headers={"X-MBX-APIKEY": self.api_key}
        )
        self.ws_connection = None
        
    async def get_exchange_info(self) -> Dict[str, Any]:
        """Fetch symbol configuration and exchange rules."""
        response = await self.client.get("/fapi/v1/exchangeInfo")
        response.raise_for_status()
        return response.json()

    async def get_account(self) -> Dict[str, Any]:
        """Fetch account data (not implemented with signature yet)."""
        # Note: Requires HMAC SHA256 signature in production
        return {}

    async def get_positions(self) -> list[Dict[str, Any]]:
        """Fetch all open futures positions."""
        # Note: Requires HMAC SHA256 signature in production
        return []

    async def get_margin(self) -> Dict[str, Any]:
        """Fetch margin balance."""
        # Note: Requires HMAC SHA256 signature in production
        return {}

    async def set_leverage(self, symbol: str, leverage: int) -> Dict[str, Any]:
        """Set leverage for a symbol."""
        # Note: Requires HMAC SHA256 signature in production
        return {}

    async def place_order(self, symbol: str, side: str, order_type: str, quantity: float, price: float = None) -> Dict[str, Any]:
        """Submit a new futures order."""
        # Note: Requires HMAC SHA256 signature in production
        return {}

    async def cancel_order(self, symbol: str, order_id: str) -> Dict[str, Any]:
        """Cancel an existing futures order."""
        return {}

    async def subscribe_market_data(self, symbols: list[str], streams: list[str], callback: Callable[[Dict[str, Any]], Awaitable[None]]):
        """
        Subscribe to public WebSocket streams for multiple symbols.
        For multiple streams on futures, we use the combined stream format:
        wss://stream.binancefuture.com/stream?streams=btcusdt@depth/btcusdt@aggTrade
        """
        all_params = []
        for sym in symbols:
            for stream in streams:
                all_params.append(f"{sym.lower()}@{stream}")
                
        stream_params = "/".join(all_params)
        combined_url = f"{self.ws_url.replace('/ws', '/stream')}?streams={stream_params}"
        
        logger.info(f"Connecting to Futures WS: {combined_url}")
        
        try:
            async with websockets.connect(combined_url) as ws:
                self.ws_connection = ws
                while True:
                    message = await ws.recv()
                    data = json.loads(message)
                    await callback(data)
        except asyncio.CancelledError:
            logger.info("Futures WebSocket subscription cancelled.")
        except Exception as e:
            logger.error(f"Futures WebSocket error: {e}", exc_info=True)
            
    async def subscribe_user_data(self, callback: Callable[[Dict[str, Any]], Awaitable[None]]):
        """Subscribe to private user data WebSocket stream."""
        pass
        
    async def close(self):
        """Close HTTP client and WebSockets."""
        await self.client.aclose()
        if self.ws_connection:
            await self.ws_connection.close()
