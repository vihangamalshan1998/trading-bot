import random
import torch
import numpy as np
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from typing import List, Dict, Any, Tuple
from core.database.models.ai import Experience
from core.config.settings import settings
from core.ai.sumtree import SumTree

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
        
        # Phase 4: PER SumTree — replaces slow ORDER BY ABS(reward) DB query.
        # Rebuilt every time cache is refreshed. O(log n) priority sampling.
        self.sumtree = SumTree(capacity=max(capacity_cache, 1))
        
    def add_experience(self, exp_data: Dict[str, Any]):
        """Saves a single experience to MySQL. Only caches if capacity > 0."""
        try:
            with self.SessionLocal() as session:
                exp = Experience(**exp_data)
                session.add(exp)
                session.commit()
                
                # MEMORY FIX: Only call session.refresh() if we actually need the object in cache.
                # When capacity=0 (storage service), refresh() loads the full ~480KB state_vector
                # from MySQL into RAM just to immediately discard it — causing a 1.5GB leak over hours.
                if self.capacity > 0:
                    session.refresh(exp)
                    self.cache.append(exp)
                    if len(self.cache) > self.capacity:
                        self.cache.pop(0)
                        
            # Periodic GC to prevent Python memory fragmentation over long runs
            if not hasattr(self, '_write_count'):
                self._write_count = 0
            self._write_count += 1
            if self._write_count % 100 == 0:
                import gc
                gc.collect()
        except Exception as e:
            pass

            
    def load_cache_from_db(self, limit: int = 10000):
        """Loads recent experiences from DB into RAM. Rebuilds SumTree after load."""
        try:
            with self.SessionLocal() as session:
                recent = session.query(Experience).order_by(Experience.timestamp.desc()).limit(limit).all()
                self.cache = recent[::-1] # Reverse to chronological
            # Phase 4: Rebuild SumTree from new cache so priorities are fresh
            if self.capacity > 0 and len(self.cache) > 0:
                self.sumtree.rebuild_from_list(self.cache, alpha=0.6)
        except Exception as e:
            from core.logging.logger import logger
            import traceback
            logger.error(f"CRITICAL ERROR loading cache from DB: {e}\n{traceback.format_exc()}")

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
            
        # 2. Phase 4: PER SumTree — Priority Experience Replay
        # Replaces slow ORDER BY ABS(reward) DB query with O(log n) tree lookup.
        # Samples proportional to |reward|^0.6 — high-reward trades seen more often.
        if self.sumtree.n_entries >= n_rare:
            per_samples = self.sumtree.sample_batch(n_rare)
            batch.extend([exp for _, _, exp in per_samples if exp is not None])
        else:
            # Fallback: sort cache by |reward| if SumTree not ready yet
            rare_pool = sorted(self.cache, key=lambda x: abs(x.reward or 0), reverse=True)
            if rare_pool:
                batch.extend(random.sample(rare_pool[:max(n_rare * 10, len(rare_pool))], min(n_rare, len(rare_pool))))
            
        # 3. Random
        batch.extend(random.sample(self.cache, min(n_random, len(self.cache))))
        
        # 4. Historical (random from entire DB)
        # BUG FIX: Never query the DB during the active training loop. It parses massive JSONs and freezes the thread.
        # The cache is already populated with thousands of rows via load_cache_from_db.
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
        
        for exp in batch:  # BUG FIX: Removed erroneous 'with session' wrapper - action mapping must be INSIDE this loop
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
                    # It's in the cache. The bot ticks every ~5 seconds, so:
                    # 5 min  = 300s  / 5s per tick = 60 cache slots
                    # 1 hour = 3600s / 5s per tick = 720 cache slots  
                    # 4 hour = 14400s/ 5s per tick = 2880 cache slots
                    cache_len = len(self.cache)
                    r_5m = sum((self.cache[i].reward or 0.0) for i in range(idx, min(idx + 60, cache_len)))
                    r_1h = sum((self.cache[i].reward or 0.0) for i in range(idx, min(idx + 720, cache_len)))
                    r_4h = sum((self.cache[i].reward or 0.0) for i in range(idx, min(idx + 2880, cache_len)))
                else:
                    # Isolated rare DB sample that somehow wasn't graded yet (Edge case)
                    r_5m = r_1h = r_4h = (exp.reward or 0.0)

            # Action Mapping  (BUG FIX: this block is now INSIDE the for loop)
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
            rewards.append([r_5m, r_1h, r_4h])  # 3 output heads!
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
