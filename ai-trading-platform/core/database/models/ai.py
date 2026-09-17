from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, ForeignKey, BigInteger
from core.database.base import Base
from datetime import datetime

class ModelVersion(Base):
    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True, index=True)
    version_id = Column(String(50), unique=True, index=True, nullable=False)
    architecture = Column(String(100), nullable=False)
    
    hyperparameters = Column(JSON)
    checkpoint_path = Column(String(255))
    metrics = Column(JSON)
    
    status = Column(String(50), default="CANDIDATE") # CANDIDATE, PRODUCTION, ARCHIVED
    created_at = Column(DateTime, default=datetime.utcnow)

class Experience(Base):
    """
    The core schema for storing Reinforcement Learning Experiences.
    This acts as the memory bank for the AI to perform offline Experience Replay.
    """
    __tablename__ = "experiences"
    extend_existing = True

    id = Column(Integer, primary_key=True, index=True)
    experience_id = Column(String(64), unique=True, index=True)
    timestamp = Column(BigInteger, index=True)
    symbol = Column(String(20), index=True)
    
    # Complex state vectors stored as JSON to allow flexible feature evolution
    market_state = Column(JSON)
    derivatives_state = Column(JSON, nullable=True)
    news_state = Column(JSON, nullable=True)
    macro_state = Column(JSON, nullable=True)
    crypto_state = Column(JSON, nullable=True)
    sentiment_state = Column(JSON, nullable=True)
    account_state = Column(JSON, nullable=True)
    portfolio_state = Column(JSON, nullable=True)
    
    # Action and Confidence
    action = Column(String(20)) # OPEN_LONG, CLOSE, HOLD, etc.
    action_probability = Column(Float, nullable=True)
    confidence = Column(Float, nullable=True)
    
    # Position tracking
    position_before = Column(Float, nullable=True)
    position_after = Column(Float, nullable=True)
    entry_price = Column(Float, nullable=True)
    exit_price = Column(Float, nullable=True)
    leverage = Column(Integer, nullable=True)
    margin = Column(Float, nullable=True)
    
    # Rewards and Costs
    reward = Column(Float, nullable=True)
    fees = Column(Float, nullable=True)
    funding_cost = Column(Float, nullable=True)
    slippage = Column(Float, nullable=True)
    realized_pnl = Column(Float, nullable=True)
    unrealized_pnl = Column(Float, nullable=True)
    
    # Meta Data
    holding_duration = Column(Integer, nullable=True) # in seconds or ms
    market_regime = Column(String(50), nullable=True)
    event_context = Column(String(100), nullable=True)
    model_version = Column(String(50), index=True)
    latency = Column(Float, nullable=True) # execution latency
    next_state = Column(JSON, nullable=True)
    episode_id = Column(String(64), nullable=True, index=True)
    data_quality_flags = Column(String(200), nullable=True)
