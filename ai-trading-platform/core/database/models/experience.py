from sqlalchemy import Column, Integer, String, Float, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Experience(Base):
    """
    Phase 3: SQLAlchemy model for persistent Replay Buffer experience storage.
    Stores complete context to reproduce and train on past decisions.
    """
    __tablename__ = "experiences"

    experience_id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(Float, index=True)
    symbol = Column(String(50), index=True)
    
    # State vectors serialized as JSON lists
    market_state = Column(JSON)
    portfolio_state = Column(JSON)
    position_state = Column(JSON)
    macro_state = Column(JSON)
    event_context = Column(JSON)
    
    # Action taken
    action_type = Column(String(50))
    confidence = Column(Float)
    requested_size = Column(Float)
    approved_size = Column(Float)
    
    # Outcomes
    entry_price = Column(Float)
    exit_price = Column(Float)
    reward = Column(Float)
    fees = Column(Float)
    realized_pnl = Column(Float)
    drawdown = Column(Float)
    
    # Target / Next State
    next_state = Column(JSON)
    done = Column(Boolean, default=False)
    
    # Metadata
    model_version = Column(String(100))
    data_quality = Column(Float, default=1.0)
    regime = Column(Float, default=0.0)
