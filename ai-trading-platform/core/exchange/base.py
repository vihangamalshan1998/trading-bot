from abc import ABC, abstractmethod
from typing import Dict, Any, Callable, Awaitable

class ExchangeAdapter(ABC):
    """
    Abstract base class for all exchange implementations.
    Ensures that the core trading system is not tied to a single exchange's API.
    """

    @abstractmethod
    async def get_exchange_info(self) -> Dict[str, Any]:
        """Fetch general exchange information (symbols, limits, etc.)."""
        pass

    @abstractmethod
    async def get_account(self) -> Dict[str, Any]:
        """Fetch account balances and status."""
        pass

    @abstractmethod
    async def place_order(self, symbol: str, side: str, order_type: str, quantity: float, price: float = None) -> Dict[str, Any]:
        """Submit a new order."""
        pass

    @abstractmethod
    async def cancel_order(self, symbol: str, order_id: str) -> Dict[str, Any]:
        """Cancel an existing order."""
        pass

    @abstractmethod
    async def subscribe_market_data(self, symbols: list[str], streams: list[str], callback: Callable[[Dict[str, Any]], Awaitable[None]]):
        """
        Subscribe to public WebSocket streams (e.g., depth, aggTrade).
        """
        pass

    @abstractmethod
    async def subscribe_user_data(self, callback: Callable[[Dict[str, Any]], Awaitable[None]]):
        """
        Subscribe to private user data WebSocket stream.
        """
        pass

    @abstractmethod
    async def get_positions(self) -> list[Dict[str, Any]]:
        """Fetch all open futures positions."""
        pass

    @abstractmethod
    async def get_margin(self) -> Dict[str, Any]:
        """Fetch account margin details (wallet balance, margin balance, available balance)."""
        pass

    @abstractmethod
    async def set_leverage(self, symbol: str, leverage: int) -> Dict[str, Any]:
        """Set the leverage for a specific symbol."""
        pass
