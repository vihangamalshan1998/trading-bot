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

    def cleanup_old_data(self, days: int = 30):
        """Automatically deletes experiences older than X days to prevent database bloat."""
        try:
            import time
            cutoff_timestamp = int(time.time()) - (days * 24 * 60 * 60)
            
            with self.SessionLocal() as session:
                deleted_count = session.query(Experience).filter(Experience.timestamp < cutoff_timestamp).delete()
                if deleted_count > 0:
                    session.commit()
                    print(f"DATABASE CLEANUP: Permanently deleted {deleted_count} experiences older than {days} days.")
        except Exception as e:
            print(f"Cleanup Error: {e}")

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
            # Check if this experience stores a 2D Sliding Window Sequence
            if exp.market_state and isinstance(exp.market_state, list) and len(exp.market_state) > 0 and isinstance(exp.market_state[0], list):
                state = exp.market_state
                # Force pad to 120 steps just in case it's a legacy sequence (e.g. 12 steps)
                while len(state) < 120:
                    state.insert(0, state[0])
            else:
                # Flat state concatenation (Legacy 1D)
                state = []
                if exp.portfolio_state: state.extend(exp.portfolio_state)
                if exp.market_state: state.extend(exp.market_state)
                state.append(exp.position_before or 0.0)
                if exp.macro_state: state.extend(exp.macro_state)
                else: state.extend([0.0] * 13) # Time (2), Funding (1), Blanks (10)
                
                # FAKE SEQUENCE: Pad Legacy 1D state to 120 steps to prevent PyTorch ValueError!
                state = [state] * 120
            
            n_state = exp.next_state if getattr(exp, "next_state", None) else state # fallback
            
            # 4 outputs for Actor Head: [Action, Confidence, Target_Size, Price_Offset]
            target_size = exp.derivatives_state.get("target_size", 0.0) if isinstance(exp.derivatives_state, dict) else 0.0
            price_offset = exp.derivatives_state.get("price_offset", 0.0) if isinstance(exp.derivatives_state, dict) else 0.0
            
            # Revert [0, 1] stored values back to [-1, 1] for neural network tanh targets!
            raw_confidence = (exp.confidence * 2.0) - 1.0 if exp.confidence is not None else 0.0
            raw_target_size = (target_size * 2.0) - 1.0 if target_size > 0 else 0.0
            raw_price_offset = (price_offset * 2.0) - 1.0 if price_offset > 0 else 0.0
            
            act = [0.0, raw_confidence, raw_target_size, raw_price_offset]
            if exp.action == "OPEN_LONG": act[0] = 0.5
            elif exp.action == "OPEN_SHORT": act[0] = -0.5
            elif exp.action == "CLOSE_LONG": act[0] = -0.5
            elif exp.action == "CLOSE_SHORT": act[0] = 0.5
            
            states.append(state)
            actions.append(act)
            rewards.append([exp.reward or 0.0])
            next_states.append(n_state)
            # If next_state is missing, treat as terminal to prevent critic value explosion
            dones.append([1.0 if getattr(exp, "next_state", None) is None else 0.0])
            
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
