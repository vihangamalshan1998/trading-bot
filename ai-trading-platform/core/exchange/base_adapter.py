from abc import ABC, abstractmethod
from typing import Dict, Any, List

class ExchangeAdapter(ABC):
    """
    Phase 2: Abstract Exchange Adapter defining the canonical interface.
    """
    @abstractmethod
    async def connect(self):
        pass
        
    @abstractmethod
    async def close(self):
        pass
        
    @abstractmethod
    async def get_exchange_info(self) -> Dict[str, Any]:
        pass
        
    @abstractmethod
    async def get_account_details(self) -> Dict[str, Any]:
        pass
        
    @abstractmethod
    async def get_positions(self) -> List[Dict[str, Any]]:
        pass
        
    @abstractmethod
    async def create_order(self, symbol: str, side: str, quantity: float, client_order_id: str) -> Dict[str, Any]:
        pass
        
    @abstractmethod
    async def cancel_order(self, symbol: str, order_id: str) -> Dict[str, Any]:
        pass
        
    @abstractmethod
    async def set_leverage(self, symbol: str, leverage: int) -> Dict[str, Any]:
        pass
        
    @abstractmethod
    async def listen_user_data(self, callback):
        """Listen to the user data stream for execution and balance updates."""
        pass
