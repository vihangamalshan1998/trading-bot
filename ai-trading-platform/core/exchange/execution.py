from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional, Dict

class OrderState(str, Enum):
    PROPOSED = "PROPOSED"
    RISK_CHECKED = "RISK_CHECKED"
    SUBMITTED = "SUBMITTED"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"
    UNKNOWN = "UNKNOWN"

class ExecutionRecord(BaseModel):
    """
    Tracks the lifecycle of an order to ensure idempotent, safe execution.
    """
    client_order_id: str
    symbol: str
    side: str
    requested_quantity: float
    filled_quantity: float = 0.0
    average_fill_price: float = 0.0
    state: OrderState = OrderState.PROPOSED
    exchange_order_id: Optional[str] = None
    created_at: float
    updated_at: float
    error_message: Optional[str] = None
    
class ExecutionTracker:
    """
    Maintains the state machine for all live orders to prevent duplicates.
    """
    def __init__(self):
        self.active_orders: Dict[str, ExecutionRecord] = {}
        
    def propose_order(self, client_order_id: str, symbol: str, side: str, quantity: float, timestamp: float) -> ExecutionRecord:
        if client_order_id in self.active_orders:
            raise ValueError("Duplicate client_order_id proposed.")
            
        record = ExecutionRecord(
            client_order_id=client_order_id,
            symbol=symbol,
            side=side,
            requested_quantity=quantity,
            state=OrderState.PROPOSED,
            created_at=timestamp,
            updated_at=timestamp
        )
        self.active_orders[client_order_id] = record
        return record
        
    def transition(self, client_order_id: str, new_state: OrderState, exchange_order_id: str = None, filled_qty: float = 0.0, avg_price: float = 0.0, error: str = None):
        if client_order_id not in self.active_orders:
            return
            
        record = self.active_orders[client_order_id]
        record.state = new_state
        if exchange_order_id:
            record.exchange_order_id = exchange_order_id
        if filled_qty > 0:
            record.filled_quantity = filled_qty
        if avg_price > 0:
            record.average_fill_price = avg_price
        if error:
            record.error_message = error
            
        # Clean up finished orders from active tracking to prevent memory leaks,
        # but in a real system they'd be pushed to MySQL first.
        if new_state in [OrderState.FILLED, OrderState.REJECTED, OrderState.CANCELLED, OrderState.EXPIRED]:
            # self.active_orders.pop(client_order_id) # Deferring cleanup to higher level for logging
            pass
