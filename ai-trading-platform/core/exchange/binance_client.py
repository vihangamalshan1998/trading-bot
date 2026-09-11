import aiohttp
import hashlib
import hmac
import time
import urllib.parse
from typing import Dict, Any, Optional
from core.config.settings import settings
from core.logging.logger import logger

class BinanceFuturesClient:
    """
    Asynchronous REST client for Binance USD-M Futures Testnet.
    """
    def __init__(self):
        # We use testnet as dictated by the prompt requirements
        self.base_url = "https://testnet.binancefuture.com" if settings.binance_testnet else "https://fapi.binance.com"
        self.api_key = settings.binance_api_key
        self.api_secret = settings.binance_api_secret
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def connect(self):
        if self.session is None:
            self.session = aiohttp.ClientSession(
                headers={"X-MBX-APIKEY": self.api_key}
            )
            logger.info(f"Connected to Binance Futures (Testnet={settings.binance_testnet})")
            
    async def close(self):
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
        """Phase 15: Fetches 24hr ticker data for all symbols."""
        return await self._request("GET", "/fapi/v1/ticker/24hr")
        
    # Private Endpoints
    async def get_account_balance(self) -> float:
        """Returns total wallet balance in USDT."""
        data = await self._request("GET", "/fapi/v2/balance", signed=True)
        for asset in data:
            if asset["asset"] == "USDT":
                return float(asset["balance"])
        return 0.0
        
    async def get_position_risk(self, symbol: str) -> Dict[str, Any]:
        """Returns position information for a specific symbol."""
        data = await self._request("GET", "/fapi/v2/positionRisk", signed=True, params={"symbol": symbol})
        if isinstance(data, list) and len(data) > 0:
            return data[0]
        return {}
        
    async def create_order(self, symbol: str, side: str, quantity: float, order_type: str = "MARKET") -> Dict[str, Any]:
        """Places a live order."""
        params = {
            "symbol": symbol,
            "side": side.upper(),
            "type": order_type.upper(),
            "quantity": f"{quantity}", # String precision is critical here
        }
        logger.info(f"Placing Order: {params}")
        return await self._request("POST", "/fapi/v1/order", signed=True, params=params)
