import random
import torch
import numpy as np
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from typing import List, Dict, Any, Tuple
from core.database.models.ai import Experience, TradeOutcome, GRADED_FLAG
from core.risk import exit_policy
from core.config.settings import settings
from core.ai.sumtree import SumTree
from core.schemas.state_layout import LOG_VOLUME_SLOTS, SPREAD_SLOT, WHALE_IMBALANCE_SLOT, WHALE_VOLUME_SLOT

# Longest reward horizon (4h) + grading tolerance: gap kept between training rows and the holdout
HOLDOUT_EMBARGO_SEC = 14400 + 1800

class ReplayBuffer:
    """
    Phase 3: SQL-backed Replay Buffer.
    Provides intelligent sampling (recent, historical, rare) from MySQL.
    Uses RAM as a cache, NOT the source of truth.
    """
    def __init__(self, capacity_cache: int = 1000, holdout_frac: float = 0.15):
        self.capacity = capacity_cache
        self.holdout_frac = holdout_frac
        self.holdout_cache: List[Experience] = []
        self.holdout_start_ts = float("inf")
        # Only load records saved from this time on (epoch s); set once the brain uses the inputs that
        # older records store as 0 (state_layout.FULL_INPUTS_SINCE). 0 = all records.
        self.min_timestamp = 0.0
        self.engine = create_engine(settings.database_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        # RAM Cache
        self.cache: List[Experience] = []
        self.golden_cache: List[Experience] = []
        
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
                        
            # We let Python handle its own garbage collection natively
            if not hasattr(self, '_write_count'):
                self._write_count = 0
            self._write_count += 1
        except Exception as e:
            # Never silent: a failed insert is a lost training record (the storage service logged
            # "Stored experience" for rows that were never written). Re-raised so the caller sees it.
            from core.logging.logger import logger
            logger.error(f"Experience insert FAILED ({exp_data.get('symbol')}, id {exp_data.get('experience_id')}): {e}")
            raise

            
    def load_cache_from_db(self, limit: int = 10000):
        """Loads recent experiences from DB into RAM. Rebuilds SumTree after load."""
        from core.logging.logger import logger
        try:
            logger.info(f"📥 Loading up to {limit} experiences from MySQL into RAM cache...")
            # Enforce 90% Normal / 10% Golden rule directly in the pull
            normal_limit = int(limit * 0.90)
            golden_limit = int(limit * 0.10)
            
            recent_limit = int(normal_limit * 0.50)
            historical_limit = normal_limit - recent_limit
            
            with self.SessionLocal() as session:
                from sqlalchemy.sql import text

                # Only rows graded with TRUE market returns are trainable (see reward_calculator.GRADED_FLAG).
                # Ungraded / legacy-graded rows previously taught the critic to count HOLD emissions.
                graded = Experience.data_quality_flags == GRADED_FLAG
                if self.min_timestamp > 0:
                    graded = graded & (Experience.timestamp >= int(self.min_timestamp))

                # OPTIMIZATION FIX: Fetch half Recent and half Random Historical using indexed id.desc()
                recent_ids_query = session.query(Experience.id).filter(graded).order_by(Experience.id.desc()).limit(recent_limit).all()
                recent_ids = [r[0] for r in recent_ids_query]

                try:
                    from sqlalchemy import func
                    max_id = session.query(func.max(Experience.id)).scalar() or 1
                    min_id = 1
                    if self.min_timestamp > 0:
                        min_id = session.query(func.min(Experience.id)).filter(
                            Experience.timestamp >= int(self.min_timestamp)).scalar() or max_id
                    span = max(1, max_id - min_id + 1)
                    sample_pool = random.sample(range(min_id, max_id + 1), min(historical_limit * 3, span))
                    historical_ids_query = session.query(Experience.id).filter(Experience.id.in_(sample_pool), graded).limit(historical_limit).all()
                    historical_ids = [r[0] for r in historical_ids_query]
                except Exception:
                    historical_ids = []

                all_normal_ids = list(set(recent_ids + historical_ids))

                if all_normal_ids:
                    # INDEXED LOOKUP: Order by primary key id.desc() to avoid multi-gigabyte disk filesort!
                    recent = session.query(Experience).filter(Experience.id.in_(all_normal_ids)).order_by(Experience.id.desc()).all()
                else:
                    recent = []

                # Chronological train / holdout split: the newest slice of graded data is NEVER trained on.
                # Candidate models are judged on it, so promotion reflects out-of-sample performance.
                recent.sort(key=lambda e: e.timestamp or 0)
                n_holdout = int(len(recent) * self.holdout_frac) if len(recent) >= 100 else 0
                self.holdout_cache = recent[len(recent) - n_holdout:] if n_holdout else []
                self.cache = recent[:len(recent) - n_holdout] if n_holdout else recent
                self.holdout_start_ts = (self.holdout_cache[0].timestamp or 0) if self.holdout_cache else float("inf")
                if self.holdout_cache:
                    # Purge embargo: training rows whose 4h forward window overlaps the holdout would leak it.
                    embargo_start = self.holdout_start_ts - HOLDOUT_EMBARGO_SEC
                    self.cache = [e for e in self.cache if (e.timestamp or 0) < embargo_start]

                # Golden Batch Integration (The 10% Permanent Memory)
                try:
                    golden_ids_query = session.execute(text(
                        f"SELECT id FROM golden_experiences WHERE data_quality_flags = '{GRADED_FLAG}' "
                        f"AND timestamp < {int(min(self.holdout_start_ts, 2**62))} "
                        f"AND timestamp >= {int(self.min_timestamp)} ORDER BY RAND() LIMIT {golden_limit}"
                    )).fetchall()
                    golden_ids = [r[0] for r in golden_ids_query]
                    
                    if golden_ids:
                        format_strings = ','.join([':id_' + str(i) for i in range(len(golden_ids))])
                        query = text(f"SELECT * FROM golden_experiences WHERE id IN ({format_strings})")
                        params = {f"id_{i}": golden_ids[i] for i in range(len(golden_ids))}
                        self.golden_cache = session.query(Experience).from_statement(query).params(**params).all()
                    else:
                        self.golden_cache = []
                except Exception:
                    self.golden_cache = []
                    
                n_labelled = self._attach_outcomes(session)

            # Rebuild SumTree from new cache so priorities are fresh
            if self.capacity > 0 and len(self.cache) > 0:
                self.sumtree.rebuild_from_list(self.cache, alpha=0.6)
                
            logger.info(f"✅ Replay buffer cache loaded successfully ({len(self.cache)} train, {len(self.holdout_cache)} holdout, {len(self.golden_cache)} golden in RAM, {n_labelled} with real trade outcomes)!")
                
        except Exception as e:
            from core.logging.logger import logger
            import traceback
            logger.error(f"CRITICAL ERROR loading cache from DB: {e}\n{traceback.format_exc()}")

    def _attach_outcomes(self, session) -> int:
        """
        Attaches path-aware trade outcomes (TradeOutcome rows written by the reward calculator) to the
        cached experiences as `exp._trade_outcome = (long_pct, short_pct)`. Rows without one fall back
        to the blended market return in build_tensors().
        """
        groups = {"exp": self.cache + self.holdout_cache, "gold": list(self.golden_cache)}
        labelled = 0
        for src, exps in groups.items():
            ids = [e.id for e in exps]
            found = {}
            try:
                for i in range(0, len(ids), 1000):
                    chunk = ids[i:i + 1000]
                    for exp_id, lp, sp in session.query(TradeOutcome.exp_id, TradeOutcome.long_pct, TradeOutcome.short_pct).filter(
                            TradeOutcome.src == src, TradeOutcome.exp_id.in_(chunk)).all():
                        if lp is not None and sp is not None:
                            found[exp_id] = (float(lp), float(sp))
            except Exception:
                found = {}  # table not created yet
            for e in exps:
                e._trade_outcome = found.get(e.id)
                labelled += e._trade_outcome is not None
        return labelled

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
               rare_pct: float = 0.3, # 30% Priority Experience Replay
               random_pct: float = 0.1,
               golden_pct: float = 0.1) -> List[Experience]: # 10% Golden Batch
        """
        Samples a batch using configured categories to prevent catastrophic forgetting.
        """
        if len(self.cache) < batch_size:
            # Fallback to random if cache is too small
            return random.sample(self.cache, len(self.cache))
            
        n_recent = int(batch_size * recent_pct)
        n_rare = int(batch_size * rare_pct)
        n_golden = int(batch_size * golden_pct)
        n_random = int(batch_size * random_pct)
        n_historical = batch_size - n_recent - n_rare - n_golden - n_random
        
        batch = []
        
        # 1. Golden Batch (The 10% permanent anchor against Forgetting)
        if hasattr(self, 'golden_cache') and len(self.golden_cache) > 0:
            actual_golden = min(n_golden, len(self.golden_cache))
            batch.extend(random.sample(self.golden_cache, actual_golden))
            # If we don't have enough golden, shift the remainder to random
            n_random += (n_golden - actual_golden)
        else:
            n_random += n_golden
        
        # 2. Recent (tail of cache)
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
        
        # 5. Historical (random from entire DB)
        # BUG FIX: Never query the DB during the active training loop. It parses massive JSONs and freezes the thread.
        # The cache is already populated with thousands of rows via load_cache_from_db.
        batch.extend(random.sample(self.cache, min(n_historical, len(self.cache))))
        
        # 6. Deduplication (Safety Net)
        # Ensure no overlapping experiences are accidentally fed
        unique_batch = []
        seen_ids = set()
        for exp in batch:
            if exp.id not in seen_ids:
                seen_ids.add(exp.id)
                unique_batch.append(exp)
                
        # 7. Fill missing slots (if deduplication removed anything)
        missing = batch_size - len(unique_batch)
        if missing > 0 and len(self.cache) > 0:
            fallback_pool = [e for e in self.cache if e.id not in seen_ids]
            if len(fallback_pool) >= missing:
                unique_batch.extend(random.sample(fallback_pool, missing))
            else:
                unique_batch.extend(fallback_pool) # Add whatever is left
            
        return unique_batch

    def count_graded_since(self, ts: float) -> int:
        """Number of trainable (graded) records saved at or after `ts` (epoch seconds)."""
        try:
            with self.SessionLocal() as session:
                return int(session.query(Experience.id).filter(
                    Experience.data_quality_flags == GRADED_FLAG, Experience.timestamp >= int(ts)).count())
        except Exception:
            return 0

    def sample_holdout(self, n: int) -> List[Experience]:
        """Random sample from the chronological holdout (never used for training)."""
        return random.sample(self.holdout_cache, min(n, len(self.holdout_cache)))

    @staticmethod
    def _normalize_frame(frame: List[float]) -> List[float]:
        """
        Scale raw dollar amounts to normalized ratios (base capital $5,000).
        Prevents large cash numbers ($5,000 balance, $81,000 BTC price) from blowing up
        LayerNorm variance and wiping out market indicators.
        """
        if not frame or len(frame) < 200:
            return frame
        f = list(frame)
        base_capital = 5000.0
        
        # Check if portfolio block is unnormalized (raw dollar amounts > 5.0)
        if f[0] > 5.0:
            f[0] = max(0.0, min(5.0, f[0] / base_capital))
            f[1] = max(0.0, min(5.0, f[1] / base_capital))
            f[2] = max(0.0, min(1.0, f[2] / base_capital))
        else:
            f[0] = max(0.0, min(5.0, f[0]))
            f[1] = max(0.0, min(5.0, f[1]))
            f[2] = max(0.0, min(1.0, f[2]))
            
        # Slot 4: exposure
        if f[4] > 10.0:
            f[4] = max(0.0, min(10.0, f[4] / base_capital))
        else:
            f[4] = max(0.0, min(10.0, f[4]))
            
        # Slot 10: position quantity vs signed direction
        # In new code: pos_signed_dir is in [-1.0, 1.0]. In legacy code: raw coins (e.g. -409 SOL)
        if abs(f[10]) > 1.0:
            f[10] = 1.0 if f[10] > 0 else -1.0
            
        # Slot 11: entry_price distance (if abs > 1.0, it was raw asset price like $81,000 BTC)
        if abs(f[11]) > 1.0:
            f[11] = 0.0
            
        # Market Features volume & orderbook quantity slots:
        # Slot 30 (vol_1m), 31 (buy_vol_1m), 32 (sell_vol_1m), 39 (vol_24h), 41 (bid_qty), 42 (ask_qty), 101 (liq_1m), 102 (liq_15m)
        # The live bot (_build_state_vector) ALREADY log-scales these before storing the frame, so only
        # rescale values that are still raw (> 5.0). Re-scaling already-scaled values shrank them ~10x
        # in training while inference saw full-size values (train/serve skew).
        for vol_idx in LOG_VOLUME_SLOTS:
            val = float(f[vol_idx])
            if abs(val) > 5.0:
                f[vol_idx] = float(np.sign(val) * np.log1p(abs(val)) / 10.0)

        # Slot 40: spread_bps (live bot already stores bps / 100; only rescale raw legacy values)
        # Whale flow: legacy records stored raw dollar pressure; map onto the live normalized definitions
        if abs(float(f[WHALE_IMBALANCE_SLOT])) > 1.0:
            f[WHALE_IMBALANCE_SLOT] = 0.0
        if float(f[WHALE_VOLUME_SLOT]) > 1.0:
            f[WHALE_VOLUME_SLOT] = float(min(1.0, max(0.0, np.log1p(float(f[WHALE_VOLUME_SLOT])) / 16.0)))

        if float(f[SPREAD_SLOT]) > 5.0:
            f[SPREAD_SLOT] = float(max(0.0, min(5.0, float(f[SPREAD_SLOT]) / 100.0)))
        
        # Universal clamp across all slots to protect LayerNorm from extreme outliers
        for idx in range(len(f)):
            f[idx] = float(max(-5.0, min(5.0, f[idx])))
            
        return f

    def _extract_sequence(self, exp: Experience, seq_len: int) -> List[List[float]]:
        # If the market_state is already a 2D movie (list of lists)
        if exp.market_state and isinstance(exp.market_state, list) and len(exp.market_state) > 0 and isinstance(exp.market_state[0], list):
            seq = []
            for frame in exp.market_state:
                # In V3, the live bot already builds the perfect 200-dim vector and passes it in the frame!
                if len(frame) == 200:
                    seq.append(self._normalize_frame(frame))
                else:
                    # Legacy fallback
                    state = []
                    if exp.portfolio_state:
                        if isinstance(exp.portfolio_state, list):
                            state.extend([x / 5000.0 if i < 3 and x > 10.0 else x for i, x in enumerate(exp.portfolio_state)])
                        else:
                            state.extend([1.0, 1.0, 1.0])
                    state.extend(frame)
                    state.append(exp.position_before or 0.0)
                    if exp.macro_state: state.extend(exp.macro_state)
                    else: state.extend([0.0] * 13)
                    while len(state) < 200: state.append(0.0)
                    seq.append(self._normalize_frame(state[:200]))
            
            # Pad or truncate to seq_len
            while len(seq) < seq_len:
                seq.insert(0, seq[0] if len(seq) > 0 else [0.0]*200)
            return seq[-seq_len:]
        else:
            # Fallback for old 1D records
            if isinstance(exp.market_state, list) and len(exp.market_state) == 200:
                norm_state = self._normalize_frame(exp.market_state)
                return [norm_state] * seq_len
            
            state = []
            if exp.portfolio_state:
                if isinstance(exp.portfolio_state, list):
                    state.extend([x / 5000.0 if i < 3 and x > 10.0 else x for i, x in enumerate(exp.portfolio_state)])
                else:
                    state.extend([1.0, 1.0, 1.0])
            if exp.market_state: state.extend(exp.market_state)
            state.append(exp.position_before or 0.0)
            if exp.macro_state: state.extend(exp.macro_state)
            else: state.extend([0.0] * 13)
            
            # Pad to 200, or slice down to exactly 200 to guarantee it NEVER crashes
            while len(state) < 200: state.append(0.0)
            norm_state = self._normalize_frame(state[:200])
            return [norm_state] * seq_len

    def build_tensors(self, batch: List[Experience], seq_len: int = 300) -> Tuple[torch.Tensor, ...]:
        """
        Converts a batch into PyTorch tensors with LSTM sequences and multi-horizon market returns.

        rewards[:, h] = TRUE forward market return of the symbol over 5m / 1h / 4h (% of price,
        direction-free). Because the market move does not depend on our (small) order, the outcome
        of ANY action can be scored from it: pnl(position) = position * return - costs. The trainer
        relies on this to evaluate freshly sampled actions instead of the logged ones.

        actions[:, 0] = logged direction (+0.5 long / -0.5 short / 0 hold) for diagnostics only.
        Rows without true market returns are dropped (never guessed).
        """
        states, actions, rewards, next_states, dones = [], [], [], [], []
        self.last_symbols = []  # symbol of each returned row (same order), used for per-coin holdout stats
        
        for exp in batch:
            if exp.reward_5m is None or exp.reward_1h is None or exp.reward_4h is None:
                continue
            if getattr(exp, "data_quality_flags", None) != GRADED_FLAG:
                continue

            seq_states = self._extract_sequence(exp, seq_len)

            act = [0.0, 0.0, 0.0, 0.0]
            if exp.action in ("OPEN_LONG", "CLOSE_SHORT"): act[0] = 0.5
            elif exp.action in ("OPEN_SHORT", "CLOSE_LONG"): act[0] = -0.5
            
            # Clamp returns to realistic percentage boundaries to protect the Critic from data glitches
            r_5m_clean = float(np.clip(exp.reward_5m, -5.0, 5.0))
            r_1h_clean = float(np.clip(exp.reward_1h, -15.0, 15.0))
            r_4h_clean = float(np.clip(exp.reward_4h, -25.0, 25.0))
            
            # Columns 3 / 4: realized gross % move of a LONG / SHORT opened here under the live exit
            # policy (stop, TP1, trailing runner, 4h limit). Fallback when no path label exists yet:
            # the blended market return clipped to the exit envelope (symmetric).
            outcome = getattr(exp, "_trade_outcome", None)
            if outcome is not None:
                long_pct, short_pct = outcome
            else:
                w = exit_policy.HORIZON_WEIGHTS
                blended = w[0] * r_5m_clean + w[1] * r_1h_clean + w[2] * r_4h_clean
                long_pct = float(np.clip(blended, -exit_policy.MAX_STOP * 100.0, exit_policy.MAX_STOP * exit_policy.HARD_TP_RR * 100.0))
                short_pct = float(np.clip(-blended, -exit_policy.MAX_STOP * 100.0, exit_policy.MAX_STOP * exit_policy.HARD_TP_RR * 100.0))

            states.append(seq_states)
            actions.append(act)
            rewards.append([r_5m_clean, r_1h_clean, r_4h_clean, long_pct, short_pct])  # 3 critic heads + 2 trade outcomes
            next_states.append(seq_states)
            self.last_symbols.append(getattr(exp, "symbol", "") or "")
            # Each sample is an independent one-step decision (contextual bandit): always terminal.
            dones.append([1.0])
            
        if not states:
            return torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0)
        try:
            s = torch.tensor(states, dtype=torch.float32)
            a = torch.tensor(actions, dtype=torch.float32)
            r = torch.tensor(rewards, dtype=torch.float32)
            d = torch.tensor(dones, dtype=torch.float32)
            # next_states are identical to states (no +1 frame is stored); share the tensor, don't copy 2x RAM.
            return s, a, r, s, d
        except ValueError:
            return torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0)
