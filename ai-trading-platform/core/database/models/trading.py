from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Enum, JSON
from core.database.base import Base
from datetime import datetime
import enum

class SymbolStatus(enum.Enum):
    TRADING = "TRADING"
    HALTED = "HALTED"
    BREAK = "BREAK"

class Symbol(Base):
    __tablename__ = "symbols"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), unique=True, index=True, nullable=False)
    base_asset = Column(String(10), nullable=False)
    quote_asset = Column(String(10), nullable=False)
    exchange = Column(String(50), default="BinanceSpot")
    status = Column(Enum(SymbolStatus), default=SymbolStatus.TRADING)
    trading_enabled = Column(Boolean, default=True)
    
    # Filters
    min_quantity = Column(Float)
    max_quantity = Column(Float)
    quantity_step = Column(Float)
    min_price = Column(Float)
    max_price = Column(Float)
    price_tick = Column(Float)
    min_notional = Column(Float)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class Order(Base):
    __tablename__ = "orders"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), index=True, nullable=False)
    client_order_id = Column(String(100), unique=True, index=True)
    exchange_order_id = Column(String(100), index=True)
    
    side = Column(String(10), nullable=False) # BUY / SELL
    type = Column(String(20), nullable=False) # MARKET / LIMIT
    
    price = Column(Float)
    quantity = Column(Float, nullable=False)
    executed_quantity = Column(Float, default=0.0)
    
    status = Column(String(20), nullable=False) # NEW, PARTIALLY_FILLED, FILLED, CANCELED, REJECTED
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
