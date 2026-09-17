from sqlalchemy import Column, Integer, String, Float, DateTime, JSON
from core.database.base import Base
from datetime import datetime

class Candle(Base):
    __tablename__ = "candles"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), index=True, nullable=False)
    interval = Column(String(10), nullable=False) # e.g. 1m, 5m, 1h
    
    timestamp = Column(DateTime, index=True, nullable=False)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    
    created_at = Column(DateTime, default=datetime.utcnow)

class MarketFeature(Base):
    __tablename__ = "market_features"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), index=True, nullable=False)
    timestamp = Column(DateTime, index=True, nullable=False)
    
    # Store features as JSON for flexibility, or we could have explicit columns for core features
    features = Column(JSON, nullable=False)
    
    created_at = Column(DateTime, default=datetime.utcnow)
