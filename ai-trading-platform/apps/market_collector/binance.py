import time
import hmac
import hashlib
import json
import asyncio
import httpx
import websockets
from urllib.parse import urlencode
from typing import Dict, Any, Callable, Awaitable

from core.exchange.base import ExchangeAdapter
from core.config.settings import settings
from core.logging.logger import logger

class BinanceSpotAdapter(ExchangeAdapter):
    def __init__(self):
        self.api_key = settings.binance_api_key
        self.api_secret = settings.binance_api_secret
        self.testnet = settings.binance_testnet
        
        if self.testnet:
            self.rest_url = "https://testnet.binance.vision"
            self.ws_url = "wss://stream.testnet.binance.vision/ws"
        else:
            self.rest_url = "https://api.binance.com"
            self.ws_url = "wss://stream.binance.com:9443/ws"
            
        self.client = httpx.AsyncClient(base_url=self.rest_url)

    def _generate_signature(self, params: Dict[str, Any]) -> str:
        query_string = urlencode(params)
        return hmac.new(
            self.api_secret.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()

    async def _request(self, method: str, endpoint: str, signed: bool = False, **kwargs) -> Dict[str, Any]:
        params = kwargs.get("params", {})
        if signed:
            params["timestamp"] = int(time.time() * 1000)
            params["signature"] = self._generate_signature(params)
            kwargs["params"] = params
            headers = kwargs.get("headers", {})
            headers["X-MBX-APIKEY"] = self.api_key
            kwargs["headers"] = headers

        try:
            response = await self.client.request(method, endpoint, **kwargs)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error("HTTP error", extra={"endpoint": endpoint, "status": e.response.status_code, "text": e.response.text})
            raise
        except Exception as e:
            logger.error("Request failed", extra={"endpoint": endpoint, "error": str(e)})
            raise

    async def get_exchange_info(self) -> Dict[str, Any]:
        return await self._request("GET", "/api/v3/exchangeInfo")

    async def get_depth_snapshot(self, symbol: str, limit: int = 1000) -> Dict[str, Any]:
        params = {"symbol": symbol, "limit": limit}
        return await self._request("GET", "/api/v3/depth", params=params)

    async def get_account(self) -> Dict[str, Any]:
        return await self._request("GET", "/api/v3/account", signed=True)

    async def place_order(self, symbol: str, side: str, order_type: str, quantity: float, price: float = None) -> Dict[str, Any]:
        params = {
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "quantity": quantity,
        }
        if price is not None:
            params["price"] = price
            
        return await self._request("POST", "/api/v3/order", signed=True, params=params)

    async def cancel_order(self, symbol: str, order_id: str) -> Dict[str, Any]:
        params = {"symbol": symbol, "orderId": order_id}
        return await self._request("DELETE", "/api/v3/order", signed=True, params=params)

    async def subscribe_market_data(self, symbol: str, streams: list[str], callback: Callable[[Dict[str, Any]], Awaitable[None]]):
        """
        Connects to multiple streams for a single symbol using combined streams.
        E.g. streams = ['depth@100ms', 'aggTrade']
        """
        symbol_lower = symbol.lower()
        stream_names = [f"{symbol_lower}@{stream}" for stream in streams]
        combined_path = "/".join(stream_names)
        
        # In binance, combined streams are at /stream?streams=
        url = f"{self.ws_url.replace('/ws', '/stream')}?streams={combined_path}"
        
        while True:
            try:
                logger.info("Connecting to market data stream", extra={"url": url})
                async with websockets.connect(url) as websocket:
                    logger.info("Connected to market data stream")
                    async for message in websocket:
                        data = json.loads(message)
                        await callback(data)
            except websockets.ConnectionClosed as e:
                logger.warning("Market data connection closed, reconnecting...", extra={"code": e.code, "reason": e.reason})
                await asyncio.sleep(5)
            except Exception as e:
                logger.error("Market data stream error", extra={"error": str(e)}, exc_info=True)
                await asyncio.sleep(5)

    async def subscribe_user_data(self, callback: Callable[[Dict[str, Any]], Awaitable[None]]):
        # Obtain listen key
        listen_key_resp = await self._request("POST", "/api/v3/userDataStream")
        listen_key = listen_key_resp["listenKey"]
        
        url = f"{self.ws_url}/{listen_key}"
        
        # User data stream requires a keepalive ping every 30 mins, will implement later.
        while True:
            try:
                logger.info("Connecting to user data stream")
                async with websockets.connect(url) as websocket:
                    logger.info("Connected to user data stream")
                    async for message in websocket:
                        data = json.loads(message)
                        await callback(data)
            except Exception as e:
                logger.error("User data stream error", extra={"error": str(e)})
                await asyncio.sleep(5)

    async def close(self):
        await self.client.aclose()
