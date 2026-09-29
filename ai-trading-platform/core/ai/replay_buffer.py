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
    def __init__(self, capacity_cache: int = 1000):
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
               recent_pct: float = 0.3, 
               historical_pct: float = 0.2, 
               rare_pct: float = 0.4, # UPGRADED: 40% Priority Experience Replay
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
        if len(self.cache) > 0:
            recent_pool = self.cache[-n_recent*5:] # Pool of recent items
            batch.extend(random.sample(recent_pool, min(n_recent, len(recent_pool))))
            
        # 2. Priority Experience Replay (PER) - Rare (Massive mistakes or huge wins)
        try:
            with self.SessionLocal() as session:
                from sqlalchemy import func
                # Query a larger pool of rare experiences, then randomly sample from it
                rare_db = session.query(Experience).order_by(func.abs(Experience.reward).desc()).limit(n_rare * 10).all()
                if len(rare_db) > 0: 
                    batch.extend(random.sample(rare_db, min(n_rare, len(rare_db))))
                else:
                    raise Exception("Not enough rare in DB, falling back to RAM cache")
        except Exception:
            # Fallback to sorting the RAM cache if DB query fails
            rare_pool = sorted(self.cache, key=lambda x: abs(x.reward or 0), reverse=True)
            if len(rare_pool) > 0:
                batch.extend(random.sample(rare_pool[:n_rare*10], min(n_rare, len(rare_pool[:n_rare*10]))))
            
        # 3. Random
        batch.extend(random.sample(self.cache, min(n_random, len(self.cache))))
        
        # 4. Historical (random from entire DB)
        try:
            with self.SessionLocal() as session:
                # To avoid slow ORDER BY RAND(), we grab a chunk of historical and sample in memory
                import time
                # Grab a random offset based on total experiences (approximate)
                total_count = session.query(Experience).count()
                offset = random.randint(0, max(0, total_count - 1000))
                historical_pool = session.query(Experience).offset(offset).limit(1000).all()
                
                if historical_pool:
                    batch.extend(random.sample(historical_pool, min(n_historical, len(historical_pool))))
                else:
                    raise Exception("No historical pool")
        except Exception:
            # Fallback to cache
            batch.extend(random.sample(self.cache, min(n_historical, len(self.cache))))
            
        return batch

    def _extract_sequence(self, exp: Experience, seq_len: int) -> List[List[float]]:
        # If the market_state is already a 2D movie (list of lists)
        if exp.market_state and isinstance(exp.market_state, list) and len(exp.market_state) > 0 and isinstance(exp.market_state[0], list):
            seq = []
            for frame in exp.market_state:
                # In V3, the live bot already builds the perfect 200-dim vector and passes it in the frame!
                if len(frame) == 200:
                    seq.append(frame)
                else:
                    # Legacy fallback
                    state = []
                    if exp.portfolio_state: state.extend(exp.portfolio_state)
                    state.extend(frame)
                    state.append(exp.position_before or 0.0)
                    if exp.macro_state: state.extend(exp.macro_state)
                    else: state.extend([0.0] * 13)
                    while len(state) < 200: state.append(0.0)
                    seq.append(state)
            
            # Pad or truncate to seq_len
            while len(seq) < seq_len:
                seq.insert(0, seq[0] if len(seq) > 0 else [0.0]*200)
            return seq[-seq_len:]
        else:
            # Fallback for old 1D records
            if isinstance(exp.market_state, list) and len(exp.market_state) == 200:
                return [exp.market_state] * seq_len
            
            state = []
            if exp.portfolio_state: state.extend(exp.portfolio_state)
            if exp.market_state: state.extend(exp.market_state)
            state.append(exp.position_before or 0.0)
            if exp.macro_state: state.extend(exp.macro_state)
            else: state.extend([0.0] * 13)
            
            # Pad to 200, or slice down to exactly 200 to guarantee it NEVER crashes
            while len(state) < 200: state.append(0.0)
            return [state[:200]] * seq_len

    def build_tensors(self, batch: List[Experience], seq_len: int = 300) -> Tuple[torch.Tensor, ...]:
        """Converts batch into PyTorch tensors with LSTM sequences and Multi-Horizon returns."""
        states, actions, rewards, next_states, dones = [], [], [], [], []
        
        with self.SessionLocal() as session:
            for exp in batch:
                # 1. BUILD LSTM SEQUENCE FROM EMBEDDED MOVIE
                seq_states = self._extract_sequence(exp, seq_len)
                next_seq_states = seq_states # Simplification for next state since we don't have the +1 frame


                # 2. CALCULATE MULTI-HORIZON PREDICTIONS (Combine DB Grades + RAM Cache)
                # If the Teacher's Assistant has graded this old test in the DB, use it!
                # If it's a brand new test (in cache) and hasn't been graded yet, sum whatever future we have in RAM.
                if exp.reward_5m is not None:
                    r_5m = exp.reward_5m
                    r_1h = exp.reward_1h
                    r_4h = exp.reward_4h
                else:
                    try:
                        idx = self.cache.index(exp)
                    except ValueError:
                        idx = -1
                        
                    if idx != -1:
                        # It's in the cache, so we can see up to 16 minutes into the future!
                        cache_len = len(self.cache)
                        r_5m = sum((self.cache[i].reward or 0.0) for i in range(idx, min(idx + 300, cache_len)))
                        r_1h = sum((self.cache[i].reward or 0.0) for i in range(idx, min(idx + 3600, cache_len)))
                        r_4h = sum((self.cache[i].reward or 0.0) for i in range(idx, min(idx + 14400, cache_len)))
                    else:
                        # Isolated rare DB sample that somehow wasn't graded yet (Edge case)
                        r_5m = r_1h = r_4h = (exp.reward or 0.0)
            
            # Action Mapping
            target_size = exp.derivatives_state.get("target_size", 0.0) if isinstance(exp.derivatives_state, dict) else 0.0
            price_offset = exp.derivatives_state.get("price_offset", 0.0) if isinstance(exp.derivatives_state, dict) else 0.0
            raw_confidence = (exp.confidence * 2.0) - 1.0 if exp.confidence is not None else 0.0
            raw_target_size = (target_size * 2.0) - 1.0 if target_size > 0 else 0.0
            raw_price_offset = (price_offset * 2.0) - 1.0 if price_offset > 0 else 0.0
            
            act = [0.0, raw_confidence, raw_target_size, raw_price_offset]
            if exp.action == "OPEN_LONG": act[0] = 0.5
            elif exp.action == "OPEN_SHORT": act[0] = -0.5
            elif exp.action == "CLOSE_LONG": act[0] = -0.5
            elif exp.action == "CLOSE_SHORT": act[0] = 0.5
            
            states.append(seq_states)
            actions.append(act)
            rewards.append([r_5m, r_1h, r_4h]) # 3 output heads!
            next_states.append(next_seq_states)
            dones.append([1.0 if getattr(exp, "next_state", None) is None else 0.0])
            
        try:
            s = torch.tensor(states, dtype=torch.float32)
            a = torch.tensor(actions, dtype=torch.float32)
            r = torch.tensor(rewards, dtype=torch.float32)
            s_ = torch.tensor(next_states, dtype=torch.float32)
            d = torch.tensor(dones, dtype=torch.float32)
            return s, a, r, s_, d
        except ValueError:
            return torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0)
