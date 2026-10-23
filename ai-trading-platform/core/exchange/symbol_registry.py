from typing import Dict, List, Optional
from datetime import datetime
from core.logging.logger import logger
from core.exchange.binance_client import BinanceFuturesAdapter

import numpy as np
from decimal import Decimal, ROUND_DOWN

class SymbolMetadata:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.sector = "UNKNOWN"
        self.launch_date = "2020-01-01"
        self.max_leverage = 20
        self.risk_category = "MEDIUM"

class SymbolEmbedding:
    def __init__(self, embedding_dim: int = 16):
        self.embedding_dim = embedding_dim
        self.embedding_vector = np.random.randn(embedding_dim).astype(np.float32)

class SymbolConfig:
    """Represents the exchange configuration and rules for a single trading pair."""
    def __init__(self, symbol: str, base_asset: str = "UNKNOWN", quote_asset: str = "USDT"):
        self.symbol = symbol
        self.market_type = "FUTURES"
        self.base_asset = base_asset
        self.quote_asset = quote_asset
        self.status = "ENABLED"
        self.trading_enabled = True
        
        # Exchange enforced limits (populated by exchangeInfo)
        self.quantity_precision: int = 3
        self.price_precision: int = 2
        self.min_quantity: float = 0.001
        self.max_quantity: float = 1000.0
        self.min_notional: float = 5.0
        self.tick_size: float = 0.01
        self.step_size: Optional[float] = None  # LOT_SIZE stepSize; None = 10^-quantity_precision
        
        # New additions for Phase 2
        self.metadata = SymbolMetadata(symbol)
        self.embedding = SymbolEmbedding()
        
    @staticmethod
    def _floor_to_step(value: float, step: Decimal) -> Decimal:
        """Largest multiple of `step` <= value, in exact decimal arithmetic. Binary floats lost a whole step
        (0.29 * 100 = 28.999999999999996 -> 0.28), so closing a 0.29 position left 0.01 open on Binance."""
        v = Decimal(repr(float(value)))
        return (v / step).to_integral_value(rounding=ROUND_DOWN) * step

    def format_quantity(self, qty: float) -> str:
        """Truncates quantity to the exchange step size (Binance rejects rounding up)."""
        step = Decimal(str(self.step_size)) if self.step_size else Decimal(1).scaleb(-self.quantity_precision)
        return f"{self._floor_to_step(qty, step):.{self.quantity_precision}f}"

    def format_price(self, price: float) -> str:
        """Truncates price to the exchange tick size."""
        return f"{self._floor_to_step(price, Decimal(str(self.tick_size))):.{self.price_precision}f}"

class SymbolRegistry:
    """
    Centralized registry for managing the universe of tradable assets and formatting
    neural network outputs into valid API string formats.
    """
    def __init__(self):
        self._symbols: Dict[str, SymbolConfig] = {}
        
    async def initialize_from_exchange(self, client: 'BinanceFuturesAdapter'):
        """Fetches /fapi/v1/exchangeInfo and dynamically loads precision limits."""
        logger.info("Fetching exchangeInfo to build Symbol Registry...")
        try:
            info = await client.get_exchange_info()
            for s in info.get("symbols", []):
                if s["contractType"] != "PERPETUAL" or s["status"] != "TRADING":
                    continue
                    
                sym = s["symbol"]
                config = SymbolConfig(sym, s["baseAsset"], s["quoteAsset"])
                config.quantity_precision = s["quantityPrecision"]
                config.price_precision = s["pricePrecision"]
                
                # Parse filters for min qty / min notional
                for f in s.get("filters", []):
                    if f["filterType"] == "LOT_SIZE":
                        config.min_quantity = float(f["minQty"])
                        config.max_quantity = float(f["maxQty"])
                        config.step_size = float(f["stepSize"]) if float(f.get("stepSize") or 0) > 0 else None
                    elif f["filterType"] == "MIN_NOTIONAL":
                        config.min_notional = float(f["notional"])
                    elif f["filterType"] == "PRICE_FILTER":
                        config.tick_size = float(f["tickSize"])
                        
                self._symbols[sym] = config
                
            logger.info(f"Symbol Registry initialized with {len(self._symbols)} active perpetual contracts.")
        except Exception as e:
            logger.error(f"Failed to initialize Symbol Registry: {e}")
            raise
            
    def get_symbol(self, symbol: str) -> Optional[SymbolConfig]:
        return self._symbols.get(symbol)
        
    def get_active_symbols(self) -> List[SymbolConfig]:
        return [sym for sym in self._symbols.values() if sym.trading_enabled]
        
    def add_symbol(self, config: SymbolConfig):
        self._symbols[config.symbol] = config
        
    def update_status(self, symbol: str, status: str):
        if symbol in self._symbols:
            self._symbols[symbol].status = status
            self._symbols[symbol].trading_enabled = (status == "ENABLED")

# Singleton registry instance
registry = SymbolRegistry()
