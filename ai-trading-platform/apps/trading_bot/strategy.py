from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import os
import torch

from core.logging.logger import logger
from apps.research.model import TradingNet

class BaseStrategy(ABC):
    @abstractmethod
    def evaluate(self, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Evaluate the market state and return an order dictionary if a trade should be placed.
        Return None if no trade should be placed.
        Format: {"side": "BUY", "quantity": 0.001, "price": 50000.0}
        """
        pass

class DummySpreadStrategy(BaseStrategy):
    """
    A placeholder strategy that buys if order book imbalance is highly positive 
    and spread is tight. It is purely for testing the execution pipeline.
    """
    def __init__(self, symbol: str, trade_qty: float = 0.001):
        self.symbol = symbol
        self.trade_qty = trade_qty
        
    def evaluate(self, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        imbalance = state.get("imbalance", 0.0)
        spread_bps = state.get("spread_bps", 100.0)
        best_ask = state.get("best_ask", 0.0)
        
        if best_ask > 0 and imbalance > 0.5 and spread_bps < 5.0:
            return {
                "side": "BUY",
                "quantity": self.trade_qty,
                "price": best_ask # Marketable limit order
            }
            
        return None

class AITradingStrategy(BaseStrategy):
    """
    A live trading strategy driven by the trained PyTorch Reinforcement Learning model.
    """
    def __init__(self, symbol: str, model_path: str = "models/best_model.pth", trade_qty: float = 0.001):
        self.symbol = symbol
        self.trade_qty = trade_qty
        self.model = TradingNet(input_dim=6, hidden_dim=64, output_dim=3)
        
        # Load weights
        if os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
                self.model.eval()
                logger.info(f"Successfully loaded AI model weights from {model_path}")
                self.is_ready = True
            except Exception as e:
                logger.error(f"Failed to load AI model weights: {e}")
                self.is_ready = False
        else:
            logger.warning(f"Model path {model_path} not found. Strategy will not generate signals.")
            self.is_ready = False
            
    def evaluate(self, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not self.is_ready:
            return None
            
        # The model expects [balance, position, spread, mid, imbalance, vwap]
        # Since we are live, we use a placeholder or tracked balance if needed.
        # For a truly advanced setup, we would read the real Binance balance.
        # Here we assume a normalized or standard input shape for the model.
        features = [
            1000.0, # Dummy balance
            0.0,    # Dummy position
            state.get("spread_bps", 0.0),
            state.get("mid_price", 0.0),
            state.get("imbalance", 0.0),
            state.get("vwap_recent", 0.0)
        ]
        
        # Forward pass
        with torch.no_grad():
            state_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            logits = self.model(state_tensor)
            action = torch.argmax(logits, dim=1).item()
            
        # action 0=Hold, 1=Buy, 2=Sell
        best_bid = state.get("best_bid", 0.0)
        best_ask = state.get("best_ask", 0.0)
        
        if action == 1 and best_ask > 0:
            logger.info("AI Model Signal: BUY")
            return {
                "side": "BUY",
                "quantity": self.trade_qty,
                "price": best_ask
            }
        elif action == 2 and best_bid > 0:
            logger.info("AI Model Signal: SELL")
            return {
                "side": "SELL",
                "quantity": self.trade_qty,
                "price": best_bid
            }
            
        return None
