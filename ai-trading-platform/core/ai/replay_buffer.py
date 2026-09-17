import random
import torch
import numpy as np
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from typing import List, Dict, Any, Tuple
from core.database.models.ai import Experience
from core.config.settings import settings

class ReplayBuffer:
    """
    Phase 3: SQL-backed Replay Buffer.
    Provides intelligent sampling (recent, historical, rare) from MySQL.
    Uses RAM as a cache, NOT the source of truth.
    """
    def __init__(self, capacity_cache: int = 10000):
        self.capacity = capacity_cache
        self.engine = create_engine(settings.database_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        # RAM Cache
        self.cache: List[Experience] = []
        
    def add_experience(self, exp_data: Dict[str, Any]):
        """Saves a single experience to MySQL and pushes to cache."""
        try:
            with self.SessionLocal() as session:
                exp = Experience(**exp_data)
                session.add(exp)
                session.commit()
                session.refresh(exp)
                
                self.cache.append(exp)
                if len(self.cache) > self.capacity:
                    self.cache.pop(0)
        except Exception as e:
            # logger.error(f"Failed to save experience: {e}")
            pass
            
    def load_cache_from_db(self, limit: int = 10000):
        """Loads recent experiences from DB into RAM to bootstrap the cache."""
        try:
            with self.SessionLocal() as session:
                recent = session.query(Experience).order_by(Experience.timestamp.desc()).limit(limit).all()
                self.cache = recent[::-1] # Reverse to chronological
        except Exception as e:
            pass

    def sample(self, batch_size: int, 
               recent_pct: float = 0.4, 
               historical_pct: float = 0.3, 
               rare_pct: float = 0.2, 
               random_pct: float = 0.1) -> List[Experience]:
        """
        Samples a batch using configured categories to prevent catastrophic forgetting.
        """
        if len(self.cache) < batch_size:
            # Fallback to random if cache is too small
            return random.sample(self.cache, len(self.cache))
            
        n_recent = int(batch_size * recent_pct)
        n_rare = int(batch_size * rare_pct)
        n_random = int(batch_size * random_pct)
        n_historical = batch_size - n_recent - n_rare - n_random
        
        batch = []
        
        # 1. Recent (tail of cache)
        if len(self.cache) > n_recent:
            batch.extend(self.cache[-n_recent:])
            
        # 2. Rare (high absolute reward/punishment)
        # Sort cache by abs(reward) descending
        rare_pool = sorted(self.cache, key=lambda x: abs(x.reward or 0), reverse=True)
        if len(rare_pool) > n_rare:
            batch.extend(rare_pool[:n_rare])
            
        # 3. Random
        batch.extend(random.sample(self.cache, min(n_random, len(self.cache))))
        
        # 4. Historical (random from entire DB)
        try:
            with self.SessionLocal() as session:
                # Naive random sampling via SQL ORDER BY RAND() is slow on large tables.
                # In production, we'd use indexed random sampling.
                historical = session.query(Experience).order_by(Experience.timestamp.asc()).limit(n_historical).all()
                batch.extend(historical)
        except Exception:
            # Fallback to cache
            batch.extend(random.sample(self.cache, min(n_historical, len(self.cache))))
            
        return batch

    def build_tensors(self, batch: List[Experience]) -> Tuple[torch.Tensor, ...]:
        """Converts a batch of Experience ORM objects into PyTorch tensors."""
        states, actions, rewards, next_states, dones = [], [], [], [], []
        
        for exp in batch:
            # Flat state concatenation
            state = []
            if exp.portfolio_state: state.extend(exp.portfolio_state)
            if exp.market_state: state.extend(exp.market_state)
            if exp.position_state: state.extend(exp.position_state)
            if exp.macro_state: state.extend(exp.macro_state)
            if exp.event_context: state.extend(exp.event_context)
            
            n_state = exp.next_state if exp.next_state else state # fallback
            
            # Simple discrete/continuous action mock parse
            act = [0.0, exp.confidence or 0.0, exp.approved_size or 0.0]
            if exp.action_type == "OPEN_LONG": act[0] = 0.5
            elif exp.action_type == "OPEN_SHORT": act[0] = -0.5
            
            states.append(state)
            actions.append(act)
            rewards.append([exp.reward or 0.0])
            next_states.append(n_state)
            dones.append([1.0 if exp.done else 0.0])
            
        # Pad sequences or truncate depending on exact dimensions (Assumes uniform here)
        try:
            s = torch.tensor(states, dtype=torch.float32)
            a = torch.tensor(actions, dtype=torch.float32)
            r = torch.tensor(rewards, dtype=torch.float32)
            s_ = torch.tensor(next_states, dtype=torch.float32)
            d = torch.tensor(dones, dtype=torch.float32)
            return s, a, r, s_, d
        except ValueError:
            # If dims don't match, return empty tensors (to be handled by caller)
            return torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0)
