import pandas as pd
import numpy as np
from typing import Dict, Any, List
from apps.orderbook.builder import OrderBookBuilder

class FeatureEngine:
    def __init__(self):
        self.trades_buffer: List[Dict[str, Any]] = []
        
    def add_trade(self, trade_event: Dict[str, Any]):
        """
        Add an aggTrade event to the buffer.
        """
        self.trades_buffer.append({
            "timestamp": trade_event["E"],
            "price": float(trade_event["p"]),
            "quantity": float(trade_event["q"]),
            "is_buyer_maker": trade_event["m"]
        })

    def _get_vwap(self) -> float:
        if not self.trades_buffer:
            return 0.0
        df = pd.DataFrame(self.trades_buffer)
        return float((df['price'] * df['quantity']).sum() / df['quantity'].sum())

    def extract_features(self, order_book: OrderBookBuilder) -> Dict[str, float]:
        """
        Extract features from the current order book and recent trades.
        """
        best_bid = order_book.get_best_bid()
        best_ask = order_book.get_best_ask()
        
        # Calculate Imbalance
        bid_qty = float(order_book.bids.get(best_bid, 0)) if best_bid else 0.0
        ask_qty = float(order_book.asks.get(best_ask, 0)) if best_ask else 0.0
        
        total_qty = bid_qty + ask_qty
        imbalance = (bid_qty - ask_qty) / total_qty if total_qty > 0 else 0.0
        
        # Micro Price
        if total_qty > 0:
            micro_price = (best_bid * ask_qty + best_ask * bid_qty) / total_qty
        else:
            micro_price = order_book.get_mid_price()
            
        features = {
            "spread_bps": order_book.get_spread_bps(),
            "mid_price": order_book.get_mid_price(),
            "micro_price": micro_price,
            "imbalance": imbalance,
            "vwap_recent": self._get_vwap()
        }
        
        # Optionally clear the trades buffer periodically
        if len(self.trades_buffer) > 1000:
            self.trades_buffer = self.trades_buffer[-500:]
            
        return features
