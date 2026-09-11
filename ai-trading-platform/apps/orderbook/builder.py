from typing import Dict, List, Any
from decimal import Decimal
import asyncio
from core.logging.logger import logger

class OrderBookBuilder:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.bids: Dict[Decimal, Decimal] = {}
        self.asks: Dict[Decimal, Decimal] = {}
        self.last_update_id = 0
        self.initialized = False
        self._buffer: List[Dict[str, Any]] = []

    def _process_level(self, level_list: List[List[str]], book_side: Dict[Decimal, Decimal]):
        for price_str, qty_str in level_list:
            price = Decimal(price_str)
            qty = Decimal(qty_str)
            if qty == 0:
                book_side.pop(price, None)
            else:
                book_side[price] = qty

    def initialize_book(self, snapshot: Dict[str, Any]):
        """Initialize the order book from a REST snapshot."""
        self.last_update_id = snapshot["lastUpdateId"]
        self.bids.clear()
        self.asks.clear()
        
        self._process_level(snapshot.get("bids", []), self.bids)
        self._process_level(snapshot.get("asks", []), self.asks)
        
        self.initialized = True
        logger.info(f"Order book initialized for {self.symbol} with update id {self.last_update_id}")
        
        # Process any buffered events
        for event in self._buffer:
            self.process_update(event)
        self._buffer.clear()

    def process_update(self, event: Dict[str, Any]):
        """Process a WebSocket depth update event."""
        if not self.initialized:
            self._buffer.append(event)
            return

        # Binance depth update format: 
        # U = first update ID in event, u = final update ID in event
        # We only process events where u > last_update_id
        if event["u"] <= self.last_update_id:
            return
            
        self._process_level(event.get("b", []), self.bids)
        self._process_level(event.get("a", []), self.asks)
        
        self.last_update_id = event["u"]

    def get_best_bid(self) -> float:
        if not self.bids: return 0.0
        return float(max(self.bids.keys()))

    def get_best_ask(self) -> float:
        if not self.asks: return 0.0
        return float(min(self.asks.keys()))

    def get_spread_bps(self) -> float:
        bid = self.get_best_bid()
        ask = self.get_best_ask()
        if bid == 0.0 or ask == 0.0: return 0.0
        mid = (ask + bid) / 2
        return ((ask - bid) / mid) * 10000

    def get_mid_price(self) -> float:
        bid = self.get_best_bid()
        ask = self.get_best_ask()
        if bid == 0.0 or ask == 0.0: return 0.0
        return (ask + bid) / 2
