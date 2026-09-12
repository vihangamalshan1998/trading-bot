import aiohttp
import asyncio
import json
import hashlib
import hmac
import time
import urllib.parse
from typing import Dict, Any, List, Optional
from core.config.settings import settings
from core.logging.logger import logger
from core.exchange.base_adapter import ExchangeAdapter

class BinanceFuturesAdapter(ExchangeAdapter):
    """
    Phase 2: Consolidated Binance USD-M Futures Testnet Adapter.
    Implements signed REST endpoints and User Data WebSocket for real-time execution updates.
    """
    def __init__(self):
        # We use testnet as dictated by the prompt requirements (Safety Rule)
        self.base_url = "https://testnet.binancefuture.com" if settings.binance_testnet else "https://fapi.binance.com"
        self.ws_base_url = "wss://stream.binancefuture.com" if settings.binance_testnet else "wss://fstream.binance.com"
        
        self.api_key = settings.binance_api_key
        self.api_secret = settings.binance_api_secret
        self.session: Optional[aiohttp.ClientSession] = None
        
        self.listen_key = None
        self.ws_connection = None
        
    async def connect(self):
        if self.session is None:
            self.session = aiohttp.ClientSession(
                headers={"X-MBX-APIKEY": self.api_key}
            )
            logger.info(f"Connected to Binance Futures Adapter (Testnet={settings.binance_testnet})")
            
    async def close(self):
        if self.ws_connection:
            await self.ws_connection.close()
        if self.session:
            await self.session.close()
            self.session = None
            
    def _generate_signature(self, query_string: str) -> str:
        return hmac.new(
            self.api_secret.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()
        
    async def _request(self, method: str, endpoint: str, signed: bool = False, params: Dict[str, Any] = None) -> Dict[str, Any]:
        if not self.session:
            await self.connect()
            
        if params is None:
            params = {}
            
        if signed:
            params["timestamp"] = int(time.time() * 1000)
            query_string = urllib.parse.urlencode(params)
            signature = self._generate_signature(query_string)
            params["signature"] = signature
            
        url = f"{self.base_url}{endpoint}"
        
        try:
            async with self.session.request(method, url, params=params) as response:
                data = await response.json()
                if response.status != 200:
                    logger.error(f"Binance API Error: {data}")
                    response.raise_for_status()
                return data
        except Exception as e:
            logger.error(f"Request failed: {method} {url} - {e}")
            raise
            
    # Public Endpoints
    async def get_exchange_info(self) -> Dict[str, Any]:
        """Fetches trading rules and symbol limits."""
        return await self._request("GET", "/fapi/v1/exchangeInfo")
        
    async def get_24hr_ticker(self) -> List[Dict[str, Any]]:
        """Fetches 24hr ticker data for all symbols."""
        return await self._request("GET", "/fapi/v1/ticker/24hr")
        
    # Private Endpoints
    async def get_account_balance(self) -> float:
        """Returns total wallet balance in USDT."""
        data = await self._request("GET", "/fapi/v2/balance", signed=True)
        for asset in data:
            if asset["asset"] == "USDT":
                return float(asset["balance"])
        return 0.0
        
    async def get_positions(self) -> List[Dict[str, Any]]:
        """Returns all open positions."""
        data = await self._request("GET", "/fapi/v2/positionRisk", signed=True)
        return [p for p in data if float(p.get("positionAmt", 0)) != 0]
        
    async def create_order(self, symbol: str, side: str, quantity: float, client_order_id: str) -> Dict[str, Any]:
        """Places a live order with idempotency using client_order_id."""
        params = {
            "symbol": symbol,
            "side": side.upper(),
            "type": "MARKET",
            "quantity": f"{quantity}",
            "newClientOrderId": client_order_id
        }
        logger.info(f"Placing Order: {params}")
        return await self._request("POST", "/fapi/v1/order", signed=True, params=params)

    async def cancel_order(self, symbol: str, order_id: str) -> Dict[str, Any]:
        """Cancels an active futures order."""
        params = {
            "symbol": symbol,
            "orderId": order_id
        }
        logger.info(f"Cancelling Order: {params}")
        return await self._request("DELETE", "/fapi/v1/order", signed=True, params=params)

    async def set_leverage(self, symbol: str, leverage: int) -> Dict[str, Any]:
        """Sets the leverage for a symbol."""
        params = {
            "symbol": symbol,
            "leverage": leverage
        }
        logger.info(f"Setting Leverage: {params}")
        return await self._request("POST", "/fapi/v1/leverage", signed=True, params=params)

    # User Data Stream
    async def _start_user_data_stream(self):
        """Creates a listen key for the websocket."""
        res = await self._request("POST", "/fapi/v1/listenKey")
        self.listen_key = res.get("listenKey")
        return self.listen_key
        
    async def _keepalive_user_data_stream(self):
        """Keeps the listen key alive. Must be called every 30 mins."""
        if not self.listen_key: return
        await self._request("PUT", "/fapi/v1/listenKey")
        
    async def listen_user_data(self, callback):
        """
        Connects to the User Data Stream via WebSocket and listens for execution updates.
        Passes parsed events to the provided callback function for reconciliation.
        """
        await self._start_user_data_stream()
        ws_url = f"{self.ws_base_url}/ws/{self.listen_key}"
        
        if not self.session:
            await self.connect()
            
        logger.info(f"Connecting to Binance User Data Stream...")
        try:
            async with self.session.ws_connect(ws_url) as ws:
                self.ws_connection = ws
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        data = json.loads(msg.data)
                        event_type = data.get("e")
                        if event_type == "ORDER_TRADE_UPDATE": # Execution Update
                            await callback(data)
                        elif event_type == "ACCOUNT_UPDATE": # Balance Update
                            await callback(data)
                    elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                        break
        except Exception as e:
            logger.error(f"User Data Stream disconnected: {e}")
        finally:
            self.ws_connection = None
