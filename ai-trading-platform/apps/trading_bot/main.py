import asyncio
import json
import uuid
import torch
import numpy as np
import time
import os
import random
import collections
import datetime
import msgpack  # Phase 1: Binary compression for experience messages
from typing import Dict, Any

from core.db.redis import redis_manager
from core.ai.registry import ModelRegistry
from core.risk import exit_policy
from core.schemas.state_layout import MARKET_LOG_VOLUME_FEATURES, MARKET_SPREAD_FEATURE, MARKET_DEPRECATED_FEATURES, CROSS_SECTION_DIM, GLOBAL_KEYS, ADDED_2026_10_SLOTS
from apps.market_collector.global_context import GLOBAL_CONTEXT_KEY
from apps.research.model import SingleSymbolActorCritic
from core.risk.risk_manager import RiskManager
from core.exchange.binance_client import BinanceFuturesAdapter
from core.exchange.symbol_registry import registry
from core.logging.logger import logger, set_log_file
from core.ai.memory import EventMemoryBuffer
from core.schemas.state_schema import MarketState, PortfolioState, PositionState, MacroState, OrderRequest
from core.config.settings import settings
from core.ai.institutional_alpha import InstitutionalAlphaEngine
from core.risk.edge_estimator import EdgeEstimator
from core.risk.macro_crash_shield import macro_crash_shield
from core.ai.liquidation_cascade_hunter import liquidation_hunter, LiquidationSignal
from core.risk.beta_hedger import beta_hedger
from core.telemetry.telegram_service import telegram_notifier

ENTRY_GUARD_KEY = "bot:entry_guard"  # hourly entry cap + per-coin cooldowns, kept across restarts
STATE_HISTORY_KEY = "bot:state_history"  # the brain's input history (last 300 frames per coin), kept across restarts
STATE_HISTORY_SAVE_SECONDS = 60.0
STATE_HISTORY_MAX_AGE_SECONDS = 300.0  # older saved history is not restored: the gap would distort the brain's view

class ProductionTradingBot:
    """
    Phase 1: Safe Execution Engine with Canonical Schemas.
    Live execution engine for the fully upgraded RL model.
    """
    def __init__(self, symbols: list):
        self.symbols = symbols
        # Dynamic Symbol Validation: Ensure non-empty and all symbols are valid USDT perpetual contracts
        if not self.symbols or not all(isinstance(s, str) and s.endswith("USDT") for s in self.symbols):
            raise ValueError(f"Invalid symbol universe specification: {self.symbols}. All symbols must be valid USDT perpetuals.")
        self.num_symbols = len(self.symbols)
            
        self.redis = redis_manager
        self.risk_manager = RiskManager()
        self.registry = ModelRegistry()
        self.binance = BinanceFuturesAdapter()
        
        # Phase 1: Complete Safety Gates Configured in settings
        self.dry_run = settings.dry_run
        self.trading_enabled = settings.trading_enabled
        
        if not self.dry_run and not settings.allow_live_trading and not settings.allow_testnet:
            logger.error("SAFETY GATE: Live/Testnet trading disabled by settings.")
            self.dry_run = True
            
        if settings.emergency_stop:
            logger.critical("SAFETY GATE: EMERGENCY STOP ACTIVE")
            self.trading_enabled = False
            self.dry_run = True
            
        logger.info(f"Trading Mode: {'DRY_RUN' if self.dry_run else 'LIVE_OR_TESTNET'} | Enabled: {self.trading_enabled}")
        
        # 4. Strict Model Checkpoint Loading (graceful if no checkpoint yet)
        self.has_valid_model = False
        self.actor_signal_offset = 0.50
        self.trade_policy = {}  # holdout-derived per-coin filter + confidence threshold / calibration (version.json)
        self.model = SingleSymbolActorCritic(input_dim=200)
        
        # VPS CPU FIX: Restrict PyTorch inference to 1 core so the trading bot doesn't hog the host CPU
        torch.set_num_threads(1)
        
        try:
            # This calls the strictly validated loader that checks architecture and active status
            self.model = self.registry.load_model(self.model)
            self.model.eval()
            self.has_valid_model = True
            self._refresh_actor_signal_offset()
            logger.info(f"Loaded rigorously validated SingleSymbolActorCritic for {self.num_symbols} symbols.")
        except Exception as e:
            logger.warning(f"No trained model checkpoint found yet ({e}). Operating in DATA-COLLECTION ONLY mode. Real order execution is strictly DISABLED until a validated checkpoint is loaded.")
            self.model.eval()
            self.has_valid_model = False # Priority 3: Strictly forbid untrained models from executing orders
        
        self.running = False
        self.emergency_stop = False  # BUG FIX: was missing, caused AttributeError in monitor_kill_switch
        self.event_memory = EventMemoryBuffer()
        
        # Tracking live states using Canonical Schemas
        # SIMULATION ONLY: Safe zero account state
        self.portfolio_state = PortfolioState(
            wallet_balance=0.0, equity=0.0, used_margin=0.0,
            free_margin=0.0, total_unrealized_pnl=0.0, total_exposure=0.0,
            positions={sym: PositionState(symbol=sym) for sym in self.symbols}
        )
        self.macro_state = MacroState(timestamp=time.time())
        self.market_states: Dict[str, MarketState] = {}
        self.whale_states: Dict[str, dict] = {}
        self.statarb_states: Dict[str, dict] = {}
        self.global_context: Dict[str, float] = {}  # Coinbase premium / DVOL (Redis market:global_context)
        
        # Phase 12: LSTM Sliding Window Sequence (300 steps = 5 minutes)
        self.sequence_length = 300
        self.state_history = collections.defaultdict(lambda: collections.deque(maxlen=self.sequence_length))
        
        # LSTM True Memory State Tracking
        self.hidden_states = {sym: None for sym in self.symbols}
        
        # Limit Order Tracking
        self.active_orders = {} # symbol -> {'order_id': str, 'timestamp': float}
        
        # Self-Awareness Tracking
        self.session_trades = 0
        self.session_wins = 0
        self.session_win_rate = 0.0
        
        self.max_equity = 0.0
        self.initial_balance = None
        self.last_win_time = time.time()
        self.equity_history = collections.deque(maxlen=60) # Last 60 ticks (5 mins)
        self.last_close_time = {sym: 0.0 for sym in self.symbols} # Cooldown tracking per symbol
        self.recent_entry_times = collections.deque()  # new-position timestamps for the hourly entry cap
        self._saved_entry_guard = None  # last entry-guard state written to Redis (survives restarts)
        self._entry_guard_loaded = False
        self._last_history_save = 0.0
        self._warmup_logged = set()
        self.position_exit_levels = {}   # sym -> ExitLevels frozen at entry (the trainer scores trades with entry-time levels)
        self.disaster_stops = {}         # sym -> exchange-side backup stop {"algo_id", "trigger", "side", "entry_px", "qty"}
        self._disaster_stop_retry = {}   # sym -> earliest time a failed stop placement is retried
        self._restored_pos_state = {}    # sym -> per-position state from Redis, applied once Binance confirms the same position
        self.position_entry_times = {sym: 0.0 for sym in self.symbols} # Minimum holding tracking per symbol
        self.peak_pnl_pct = {sym: 0.0 for sym in self.symbols} # Peak leveraged PnL tracking for trailing profit locks
        self.pnl_history = {sym: collections.deque(maxlen=30) for sym in self.symbols} # (timestamp, pnl_pct_leveraged) for velocity shock detection
        self.smoothed_action_history = {sym: collections.deque(maxlen=3) for sym in self.symbols} # Institutional signal smoothing (3-tick window)
        self.smoothed_confidence_history = {sym: collections.deque(maxlen=3) for sym in self.symbols}
        self.last_hold_emission_time = {sym: 0.0 for sym in self.symbols} # VPS DISK SHIELD: Throttles HOLD experience emissions to once per 5 mins
        
        # Institutional Scale-Out & Free-Roll Runner Tracking
        self.partial_tp_triggered = {sym: False for sym in self.symbols}
        
        # Institutional Circuit Breaker (5% Daily Loss Kill Switch) & Profit Ratchet
        self.daily_start_equity = 0.0
        self.daily_peak_equity = 0.0
        self.daily_equity_reset_time = time.time()
        self.circuit_breaker_active_until = 0.0
        self.profit_ratchet_active = False
        
        # Institutional Hedge Fund Safeguards (Penalty Box, HTF Regime, Whale Gate, Capital De-escalation)
        self.consecutive_losses = {sym: 0 for sym in self.symbols}
        self.loss_lockout_until = {sym: {"LONG": 0.0, "SHORT": 0.0} for sym in self.symbols}
        self.htf_regimes = {sym: {"trend": "NEUTRAL", "adx": 0.0, "ema20": 0.0, "ema50": 0.0, "last_updated": 0.0} for sym in self.symbols}
        self.alpha_engine = InstitutionalAlphaEngine(symbols=self.symbols)
        self.edge_estimator = EdgeEstimator(min_net_edge_pct=0.08, default_horizon_seconds=4 * 3600.0)
        self.dynamic_active_symbols = set(self.symbols)
        self.last_universe_check_time = 0.0
        self._symbol_tasks: Dict[str, list] = {}
        
        # Institutional Upgrades State
        self.global_liq_state = {}
        self.pause_until = 0.0

        # Passive Airbag Safeguards State (Zero effect on normal trading frequency)
        self.recent_losses_history: collections.deque = collections.deque(maxlen=30)
        self.last_market_tick_time = time.time()
        self.last_margin_trim_time = 0.0

    def _entries_last_hour(self) -> int:
        cutoff = time.time() - 3600.0
        while self.recent_entry_times and self.recent_entry_times[0] < cutoff:
            self.recent_entry_times.popleft()
        return len(self.recent_entry_times)

    async def _load_entry_guard(self):
        """Restores the hourly entry count and per-coin cooldowns, so a restart does not open a burst of trades."""
        if self._entry_guard_loaded or not self.redis.redis:
            return
        try:
            raw = await self.redis.redis.get(ENTRY_GUARD_KEY)
            self._entry_guard_loaded = True
            if not raw:
                return
            data = json.loads(raw)
            cutoff = time.time() - 3600.0
            known = set(self.recent_entry_times)
            restored = sorted(float(t) for t in data.get("entries", []) if float(t) >= cutoff and float(t) not in known)
            self.recent_entry_times = collections.deque(sorted(list(self.recent_entry_times) + restored))
            for sym, t in (data.get("last_close") or {}).items():
                self.last_close_time[sym] = max(self.last_close_time.get(sym, 0.0), float(t))
            self._restore_risk_state(data.get("risk") or {})
            logger.info(f"Restored entry guard: {len(self.recent_entry_times)} entries in the last hour, "
                        f"{len(data.get('last_close') or {})} coin cooldowns.")
        except Exception as e:
            logger.warning(f"Could not restore entry guard: {e}")

    async def _save_entry_guard(self):
        await self._load_entry_guard()   # no-op once done; never overwrite stored state that was not read yet
        if not self._entry_guard_loaded:
            return
        state = {
            "entries": list(self.recent_entry_times),
            "last_close": {s: t for s, t in self.last_close_time.items() if t > time.time() - 7200.0},
            "risk": self._safe_risk_state(),
        }
        if state == self._saved_entry_guard:
            return
        try:
            await self.redis.redis.set(ENTRY_GUARD_KEY, json.dumps(state), ex=3 * 86400)
            self._saved_entry_guard = state
        except Exception as e:
            logger.warning(f"Could not save entry guard: {e}")

    def _exit_levels_for(self, sym: str, market) -> "exit_policy.ExitLevels":
        """Exit levels frozen when the position opened (volatility at entry), like the trainer's replay of the trade."""
        lv = self.position_exit_levels.get(sym)
        if lv is None:
            lv = exit_policy.exit_levels(float(getattr(market, "volatility", 0.0) or 0.0))
            self.position_exit_levels[sym] = lv
        return lv

    def _close_record(self, base: dict, entry_px: float, parts) -> dict:
        """
        Trade-history row for a close made of several fills. parts: [(qty, price, fee_rate), ...].
        PnL uses the real fill prices (maker part at its limit price, market part at Binance's average fill).
        """
        rec = dict(base)
        parts = [(float(q), float(px), float(r)) for q, px, r in parts if q and q > 0 and px and px > 0]
        qty = sum(q for q, _, _ in parts)
        if qty <= 0 or entry_px <= 0:
            return rec
        exit_px = sum(q * px for q, px, _ in parts) / qty
        is_long = "LONG" in str(rec.get("side", ""))
        lev = float(rec.get("leverage") or settings.max_leverage)
        gross = (exit_px - entry_px) * qty * (1 if is_long else -1)
        fees = entry_px * qty * settings.maker_fee_pct + sum(q * px * r for q, px, r in parts)
        margin = entry_px * qty / lev if lev else 0.0
        rec.update({
            "timestamp": time.time(),
            "quantity": qty,
            "price": exit_px,
            "entry_price": entry_px,
            "gross_pnl": gross,
            "fees_paid": fees,
            "realized_pnl": gross - fees,
            "roi_pct": (gross - fees) / margin * 100.0 if margin > 0 else 0.0,
            "notional": exit_px * qty,
        })
        return rec

    def _risk_state(self) -> dict:
        """Safety state that must survive restarts (daily loss / drawdown stops, penalty box, open-trade tracking)."""
        rm = self.risk_manager
        pos_state = {}
        for sym, pos in self.portfolio_state.positions.items():
            if abs(pos.quantity) > 1e-9 and pos.entry_price > 0:
                lv = self.position_exit_levels.get(sym)
                pos_state[sym] = {
                    "entry_px": float(pos.entry_price),
                    "entry_time": float(self.position_entry_times.get(sym) or 0.0),
                    "peak": float(self.peak_pnl_pct.get(sym, 0.0)),
                    "tp1": bool(self.partial_tp_triggered.get(sym, False)),
                    "levels": [lv.stop, lv.tp1, lv.hard_tp, lv.breakeven, lv.trail_gap] if lv else None,
                }
        return {
            "daily_start_equity": self.daily_start_equity,
            "daily_peak_equity": self.daily_peak_equity,
            "daily_equity_reset_time": self.daily_equity_reset_time,
            "circuit_breaker_active_until": self.circuit_breaker_active_until,
            "profit_ratchet_active": self.profit_ratchet_active,
            "max_equity": self.max_equity,
            "rm_daily_high": rm.daily_high_equity,
            "rm_global_high": rm.global_high_equity,
            "rm_last_day_reset": rm.last_day_reset,
            "consecutive_losses": {s: n for s, n in self.consecutive_losses.items() if n},
            "lockouts": {s: d for s, d in self.loss_lockout_until.items() if max(list(d.values()) or [0.0]) > time.time()},
            "positions": pos_state,
        }

    def _safe_risk_state(self) -> dict:
        # Called every inference tick: a failure here must never reach the loop (it halts the bot on any error)
        try:
            return self._risk_state()
        except Exception as e:
            logger.warning(f"Could not build risk state for saving: {e}")
            return {}

    def _restore_risk_state(self, risk: dict):
        if not risk:
            return
        rm = self.risk_manager
        self.daily_start_equity = float(risk.get("daily_start_equity") or 0.0)
        self.daily_peak_equity = float(risk.get("daily_peak_equity") or 0.0)
        self.daily_equity_reset_time = float(risk.get("daily_equity_reset_time") or time.time())
        self.circuit_breaker_active_until = float(risk.get("circuit_breaker_active_until") or 0.0)
        self.profit_ratchet_active = bool(risk.get("profit_ratchet_active", False))
        self.max_equity = max(self.max_equity, float(risk.get("max_equity") or 0.0))
        rm.daily_high_equity = max(rm.daily_high_equity, float(risk.get("rm_daily_high") or 0.0))
        rm.global_high_equity = max(rm.global_high_equity, float(risk.get("rm_global_high") or 0.0))
        rm.last_day_reset = float(risk.get("rm_last_day_reset") or rm.last_day_reset)
        for sym, n in (risk.get("consecutive_losses") or {}).items():
            self.consecutive_losses[sym] = int(n)
        for sym, d in (risk.get("lockouts") or {}).items():
            self.loss_lockout_until[sym] = {"LONG": float(d.get("LONG", 0.0)), "SHORT": float(d.get("SHORT", 0.0))}
        self._restored_pos_state = dict(risk.get("positions") or {})
        logger.info(f"Restored risk state: daily start ${self.daily_start_equity:.2f}, circuit breaker "
                    f"{'ON' if self.circuit_breaker_active_until > time.time() else 'off'}, "
                    f"{len(risk.get('lockouts') or {})} penalty-box coins, {len(self._restored_pos_state)} open-trade states.")

    def _apply_restored_position(self, sym: str, entry_px: float):
        """Called by the Binance sync for an open position: reuse the saved trade state only if it is the same trade."""
        r = self._restored_pos_state.pop(sym, None)
        if not r:
            return
        saved_px = float(r.get("entry_px") or 0.0)
        if saved_px <= 0 or abs(entry_px - saved_px) / saved_px > 1e-5:
            logger.info(f"[{sym}] Saved trade state ignored (entry {saved_px} != live {entry_px}).")
            return
        if r.get("entry_time"):
            self.position_entry_times[sym] = float(r["entry_time"])
            self.portfolio_state.positions[sym].entry_time = float(r["entry_time"])
        self.peak_pnl_pct[sym] = float(r.get("peak") or 0.0)
        self.partial_tp_triggered[sym] = bool(r.get("tp1", False))
        if r.get("levels"):
            self.position_exit_levels[sym] = exit_policy.ExitLevels(*[float(x) for x in r["levels"]])
        logger.info(f"[{sym}] Restored trade state: held {(time.time() - float(r.get('entry_time') or time.time())) / 60:.0f} min, "
                    f"50% profit taken: {bool(r.get('tp1'))}, best move so far {float(r.get('peak') or 0.0) * 100:+.2f}% (leveraged).")

    def _stops_allowed(self) -> bool:
        return (settings.disaster_stop_enabled and self.trading_enabled and not self.dry_run
                and not self.emergency_stop and (settings.allow_testnet or settings.allow_live_trading))

    async def _load_disaster_stops(self):
        """Adopts backup stops already on Binance (they survive bot restarts and order cleanups)."""
        if not self._stops_allowed():
            return
        try:
            for o in await self.binance.get_open_algo_orders():
                sym = o.get("symbol")
                if str(o.get("orderType") or o.get("type")) != "STOP_MARKET" or not o.get("closePosition"):
                    continue
                if sym in self.disaster_stops:
                    await self.binance.cancel_algo_order(o.get("algoId"))  # duplicate
                    continue
                self.disaster_stops[sym] = {"algo_id": o.get("algoId"), "trigger": float(o.get("triggerPrice") or 0.0),
                                            "side": str(o.get("side", "")).upper(), "entry_px": None, "qty": 0.0}
            if self.disaster_stops:
                logger.info(f"Found exchange backup stops for: {sorted(self.disaster_stops)}")
        except Exception as e:
            logger.warning(f"Could not list exchange backup stops: {e}")

    async def _ensure_disaster_stops(self, live_positions):
        """
        Every open position gets a Binance-side STOP_MARKET (closePosition) at disaster_stop_mult x its own stop.
        The bot's own stop fires first; this one only matters when the bot, VPS or price feed is down.
        """
        if not self._stops_allowed():
            return
        now_t = time.time()
        open_syms = {}
        for p in live_positions:
            amt = float(p.get("positionAmt", 0) or 0.0)
            if amt != 0 and p.get("symbol") in self.portfolio_state.positions:
                open_syms[p.get("symbol")] = (amt, float(p.get("entryPrice", 0) or 0.0))

        # Positions that disappeared: was it our backup stop that closed them?
        for sym in [x for x in self.disaster_stops if x not in open_syms]:
            info = self.disaster_stops.pop(sym)
            if now_t - self.last_close_time.get(sym, 0.0) < 120.0:
                continue  # the bot closed it
            try:
                st = await self.binance._request("GET", "/fapi/v1/algoOrder", signed=True, params={"algoId": info["algo_id"]})
                status = str(st.get("algoStatus", "")).upper()
            except Exception:
                status = ""
            if "TRIGGER" in status or "FINISH" in status:
                await self._record_disaster_stop_fill(sym, info)

        for sym, (amt, entry_px) in open_syms.items():
            if entry_px <= 0 or now_t < self._disaster_stop_retry.get(sym, 0.0):
                continue
            market = self.market_states.get(sym)
            lv = self._exit_levels_for(sym, market)
            dist = lv.stop * settings.disaster_stop_mult
            is_long = amt > 0
            cfg = registry.get_symbol(sym)
            if not cfg:
                continue
            trigger = float(cfg.format_price(entry_px * (1 - dist) if is_long else entry_px * (1 + dist)))
            side = "SELL" if is_long else "BUY"
            cur = self.disaster_stops.get(sym)
            if cur and cur["side"] == side and cur["trigger"] > 0 and abs(cur["trigger"] - trigger) / trigger < 0.002:
                cur.update({"entry_px": entry_px, "qty": abs(amt)})
                continue
            try:
                if cur:
                    try:
                        await self.binance.cancel_algo_order(cur["algo_id"])
                    except Exception:
                        pass
                    self.disaster_stops.pop(sym, None)
                res = await self.binance.place_close_stop(sym, side, cfg.format_price(trigger), f"ai_dstop_{uuid.uuid4().hex[:10]}")
                self.disaster_stops[sym] = {"algo_id": res.get("algoId"), "trigger": trigger, "side": side,
                                            "entry_px": entry_px, "qty": abs(amt)}
                logger.info(f"[{sym}] 🛡️ Exchange backup stop set at {trigger} ({dist * 100:.2f}% from entry {entry_px}).")
            except Exception as e:
                self._disaster_stop_retry[sym] = now_t + 60.0
                logger.warning(f"[{sym}] Could not set exchange backup stop: {e}")

    async def _record_disaster_stop_fill(self, sym: str, info: dict):
        is_long = info.get("side") == "SELL"
        entry_px = float(info.get("entry_px") or 0.0)
        qty = float(info.get("qty") or 0.0)
        gross, fees = None, None
        try:
            income = await self.binance._request("GET", "/fapi/v1/income", signed=True, params={"symbol": sym, "limit": 20})
            recent = [i for i in (income or []) if time.time() * 1000 - float(i.get("time", 0)) < 600000]
            if any(i.get("incomeType") == "REALIZED_PNL" for i in recent):
                gross = sum(float(i.get("income", 0)) for i in recent if i.get("incomeType") == "REALIZED_PNL")
                fees = sum(abs(float(i.get("income", 0))) for i in recent if i.get("incomeType") == "COMMISSION")
        except Exception as e:
            logger.warning(f"[{sym}] Income lookup for backup stop failed: {e}")
        self.last_close_time[sym] = time.time()
        closed_dir = "LONG" if is_long else "SHORT"
        self.loss_lockout_until.setdefault(sym, {"LONG": 0.0, "SHORT": 0.0})[closed_dir] = time.time() + 3600.0
        logger.critical(f"[{sym}] 🛡️ EXCHANGE BACKUP STOP FIRED: the bot did not close this {closed_dir} in time "
                        f"(trigger {info.get('trigger')}, entry {entry_px}). {closed_dir} locked for 60 min.")
        if self.redis.redis and qty > 0 and entry_px > 0:
            exit_px = float(info.get("trigger") or entry_px)
            margin = entry_px * qty / settings.max_leverage
            if gross is None:
                gross = (exit_px - entry_px) * qty * (1 if is_long else -1)
                fees = (entry_px * settings.maker_fee_pct + exit_px * settings.taker_fee_pct) * qty
            rec = {"timestamp": time.time(), "symbol": sym, "side": "CLOSE_LONG" if is_long else "CLOSE_SHORT",
                   "quantity": qty, "price": exit_px, "confidence": 0.0, "order_id": info.get("algo_id"),
                   "entry_price": entry_px, "gross_pnl": gross, "fees_paid": fees, "realized_pnl": gross - fees,
                   "roi_pct": (gross - fees) / margin * 100.0 if margin > 0 else 0.0, "leverage": settings.max_leverage,
                   "notional": exit_px * qty, "order_type": "EXCHANGE STOP", "exit_reason": "EXCHANGE BACKUP STOP"}
            await self.redis.redis.lpush("dashboard:trade_history", json.dumps(rec))
            await self.redis.redis.ltrim("dashboard:trade_history", 0, 99)

    def _history_warm(self, sym: str) -> bool:
        """
        True once the brain has a full window of real history for this coin. After a restart without a fresh
        saved history, the window is padded with copies of the current frame (the market looks frozen for
        25 minutes), so the AI must not open or close trades, or write training records, until it refills.
        """
        hist = getattr(self, "state_history", None)
        if hist is None:
            return True
        return len(hist[sym]) >= self.sequence_length

    def _log_warmup(self, sym: str, what: str):
        if sym not in self._warmup_logged:
            self._warmup_logged.add(sym)
            n = len(self.state_history[sym])
            logger.info(f"[{sym}] WARM-UP: brain history {n}/{self.sequence_length} frames after restart. "
                        f"No AI {what} until it is full (~{(self.sequence_length - n) * 5 / 60:.0f} min); stops and take-profits stay active.")

    async def _save_state_history(self):
        if time.time() - self._last_history_save < STATE_HISTORY_SAVE_SECONDS:
            return
        self._last_history_save = time.time()
        try:
            rb = self.redis.binary_redis
            if rb is None:
                return
            hist = {sym: np.asarray(list(h), dtype=np.float32).tobytes() for sym, h in self.state_history.items() if len(h)}
            payload = {"ts": time.time(), "dim": 200, "hist": hist}
            await rb.set(STATE_HISTORY_KEY, msgpack.packb(payload, use_bin_type=True), ex=3600)
        except Exception as e:
            logger.warning(f"Could not save brain history: {e}")

    async def _load_state_history(self):
        try:
            rb = self.redis.binary_redis
            raw = await rb.get(STATE_HISTORY_KEY) if rb is not None else None
            if not raw:
                logger.info("No saved brain history: AI decisions wait for a full window (~25 min warm-up).")
                return
            data = msgpack.unpackb(raw, raw=False)
            age = time.time() - float(data.get("ts", 0.0))
            if age > STATE_HISTORY_MAX_AGE_SECONDS:
                logger.info(f"Saved brain history is {age / 60:.0f} min old (max {STATE_HISTORY_MAX_AGE_SECONDS / 60:.0f}): "
                            f"not restored, AI decisions wait for a full window (~25 min warm-up).")
                return
            dim = int(data.get("dim", 200))
            restored = 0
            for sym, buf in (data.get("hist") or {}).items():
                frames = np.frombuffer(buf, dtype=np.float32).reshape(-1, dim)
                self.state_history[sym].extend(frames.astype(np.float64).tolist())
                restored += 1
            logger.info(f"Restored brain history for {restored} coins (saved {age:.0f}s ago).")
        except Exception as e:
            logger.warning(f"Could not restore brain history: {e}")

    async def _record_airbag_trim(self, sym: str, pos, qty: float, order_res, mid_price: float):
        """Adds an Airbag 4 margin trim to the dashboard trade history, so its profit/loss is not missing from the stats."""
        try:
            if not self.redis.redis or qty <= 0:
                return
            res = order_res if isinstance(order_res, dict) else {}
            exit_px = float(res.get("avgPrice") or 0.0) or float(mid_price or 0.0)
            entry_px = float(getattr(pos, "entry_price", 0.0) or 0.0) or exit_px
            is_long = pos.quantity > 0
            lev = settings.max_leverage
            gross = (exit_px - entry_px) * qty * (1 if is_long else -1)
            fees = entry_px * qty * settings.maker_fee_pct + exit_px * qty * settings.taker_fee_pct
            margin = entry_px * qty / lev if lev else 0.0
            record = {
                "timestamp": time.time(),
                "symbol": sym,
                "side": "PARTIAL_CLOSE_LONG" if is_long else "PARTIAL_CLOSE_SHORT",
                "quantity": qty,
                "price": exit_px,
                "confidence": 0.0,
                "order_id": res.get("orderId", "airbag4"),
                "entry_price": entry_px,
                "gross_pnl": gross,
                "fees_paid": fees,
                "realized_pnl": gross - fees,
                "roi_pct": (gross - fees) / margin * 100.0 if margin > 0 else 0.0,
                "leverage": int(lev),
                "notional": exit_px * qty,
                "order_type": "MARKET",
                "mid_price": float(mid_price or 0.0),
                "exit_reason": "AIRBAG MARGIN TRIM",
            }
            entry_t = self.position_entry_times.get(sym) or float(getattr(pos, "entry_time", 0.0) or 0.0)
            if entry_t:
                record["hold_seconds"] = max(0.0, time.time() - entry_t)
            await self.redis.redis.lpush("dashboard:trade_history", json.dumps(record))
            await self.redis.redis.ltrim("dashboard:trade_history", 0, 99)
        except Exception as e:
            logger.warning(f"[{sym}] Could not record airbag trim: {e}")

    async def _top_up_partial_entry(self, sym: str, side: str, intended_qty: float, filled_qty: float,
                                    limit_price: float, pred_move_pct: float) -> bool:
        """
        After a maker entry timed out, sends the unfilled rest as a MARKET order (taker fee, +0.03%) when the
        signal is strong enough to pay for it and the price has not already run away. Returns True if sent.
        """
        if intended_qty <= 0 or filled_qty >= intended_qty * settings.entry_topup_below_fill_frac:
            return False
        if abs(pred_move_pct) < settings.entry_topup_min_move_pct:
            return False
        market = self.market_states.get(sym)
        mid = float(getattr(market, "mid_price", 0.0) or 0.0)
        if mid <= 0 or limit_price <= 0:
            return False
        is_long = "LONG" in side
        adverse_pct = ((mid - limit_price) if is_long else (limit_price - mid)) / limit_price * 100.0
        if adverse_pct > settings.entry_topup_max_adverse_pct:
            logger.info(f"[{sym}] No entry top-up: price moved {adverse_pct:.3f}% past the limit (max {settings.entry_topup_max_adverse_pct}%).")
            return False
        sym_config = registry.get_symbol(sym)
        rest = float(sym_config.format_quantity(intended_qty - filled_qty)) if sym_config else 0.0
        if rest <= 0 or rest * mid < 25.0:
            return False
        logger.info(f"[{sym}] ENTRY TOP-UP: {filled_qty} of {intended_qty} filled as maker; sending the rest ({rest}) at market "
                    f"(predicted move {pred_move_pct:+.2f}% >= {settings.entry_topup_min_move_pct}%).")
        await self.binance.create_order(sym, "BUY" if is_long else "SELL", rest,
                                        f"ai_topup_{uuid.uuid4().hex[:10]}", order_type="MARKET")
        return True

    def _refresh_actor_signal_offset(self):
        """
        Legacy champions (trained on summed shaped rewards) learned a strong long bias, which was
        cancelled by hand with `raw_action - 0.50`. Champions approved by the v3 holdout evaluator are
        trained on true market returns and centered at 0 (|a| <= 0.2 = flat), so the offset must be 0;
        otherwise a neutral model would be read as a -0.5 SHORT on every tick.
        """
        evaluator = self.registry.get_evaluator_version()
        self.actor_signal_offset = 0.0 if evaluator.startswith("v3") else 0.50
        logger.info(f"Actor signal offset = {self.actor_signal_offset:.2f} (champion evaluator: {evaluator or 'legacy'})")
        self.trade_policy = self.registry.get_trade_policy() if evaluator.startswith("v3") else {}
        if self.trade_policy:
            logger.info(
                f"Model trade policy: min confidence {self.trade_policy.get('min_confidence')}, "
                f"blocked coins {self.trade_policy.get('blocked_symbols') or []}, "
                f"holdout PnL/trade {self.trade_policy.get('filtered_pnl_per_trade')}"
            )

    def _mask_unseen_inputs(self, x: torch.Tensor) -> torch.Tensor:
        """
        Legacy champions (no fitted feature statistics) were trained when the October 2026 inputs were
        always 0; feed them 0 there so new data cannot shift their behaviour. Fitted models mask unseen
        slots themselves (feat_mask). Stored experience frames always keep the real values.
        """
        if bool(getattr(self.model, "feat_fitted", False)):
            return x
        x = x.clone()
        x[..., list(ADDED_2026_10_SLOTS)] = 0.0
        return x

    def _emit_market_snapshot(self, sym: str, seq, action_logits) -> None:
        """
        One training record per coin every settings.experience_snapshot_seconds, whatever the bot decides.
        The trainer replays both LONG and SHORT from every record (path-aware outcomes), so it needs market
        coverage, not decisions. Records used to be written only when the decision logic reached a HOLD
        branch; the October 2026 entry gates skip past those, which cut the flow to ~200 records a day.
        Values are rounded to 6 significant digits: the JSON state column is the VPS disk cost (~0.77 MB raw).
        Shares the throttle with the older HOLD emitters, so a coin never gets two records in one window.
        """
        if not self._history_warm(sym):
            return
        now_t = time.time()
        if now_t - self.last_hold_emission_time.get(sym, 0.0) < settings.experience_snapshot_seconds:
            return
        self.last_hold_emission_time[sym] = now_t
        try:
            pos = self.portfolio_state.positions.get(sym)
            qty = float(getattr(pos, "quantity", 0.0) or 0.0)
            lev = int(getattr(pos, "leverage", 0) or settings.max_leverage)
            mid = float(getattr(self.market_states.get(sym), "mid_price", 0.0) or 0.0)
            conf = float(np.clip((float(action_logits[1]) + 1.0) / 2.0, 0.05, 0.99))
            entry_t = float(getattr(pos, "entry_time", 0.0) or 0.0)
            exp_data = {
                "experience_id": str(uuid.uuid4()),
                "timestamp": int(now_t),
                "symbol": sym,
                "market_state": [[float(f"{float(v):.6g}") for v in frame] for frame in seq],
                "macro_state": None,
                "portfolio_state": {
                    "free_margin": float(self.portfolio_state.free_margin),
                    "total_unrealized_pnl": float(self.portfolio_state.total_unrealized_pnl),
                },
                "derivatives_state": {
                    "target_size": float(np.clip((float(action_logits[2]) + 1.0) / 2.0, 0.0, 1.0)),
                    "price_offset": float(np.clip((float(action_logits[3]) + 1.0) / 2.0, 0.0, 1.0)),
                },
                "position_before": qty,
                "position_after": qty,
                "entry_price": float(getattr(pos, "entry_price", 0.0) or 0.0),
                "exit_price": None,
                "leverage": lev,
                "margin": abs(qty) * mid / lev if qty and lev else 0.0,
                "fees": 0.0,
                "holding_duration": int(now_t - entry_t) if qty and entry_t > 0 else 0,
                "action": "HOLD",
                "action_probability": conf,
                "confidence": conf,
                "model_version": self.registry.get_current_version(),
                "reward": 0.0,  # replaced by the reward grader's true market return
                "realized_pnl": 0.0,
                "unrealized_pnl": float(getattr(pos, "unrealized_pnl", 0.0) or 0.0),
            }
            asyncio.create_task(redis_manager.publish_binary("experience:completed", msgpack.packb(exp_data, use_bin_type=True)))
        except Exception as e:
            logger.warning(f"[{sym}] Market snapshot record skipped: {e}")

    def _calibrated_confidence(self, raw_conf: float) -> float:
        """Honest confidence: the win rate the champion's raw confidence actually achieved on unseen data."""
        cal = (self.trade_policy or {}).get("calibration") or {}
        xs, ys = cal.get("raw") or [], cal.get("win_rate") or []
        if len(xs) >= 2 and len(xs) == len(ys):
            return float(np.interp(raw_conf, xs, ys))
        return float(raw_conf)

    def is_valid_contract(self, sym: str) -> bool:
        """
        Validates symbol against actual Binance USDT-M exchange metadata.
        Strict verification:
        1. String structure (non-empty string, ends with USDT, min length).
        2. Configured exclusion check (respects settings.excluded_trading_symbols).
        3. Exchange metadata verification (PERPETUAL contract, quoteAsset == USDT, status ENABLED/TRADING).
        Fails closed (returns False) if validation cannot be confirmed.
        """
        if not isinstance(sym, str) or not sym.endswith("USDT") or len(sym) < 6:
            return False

        # Respect configured exclusions (e.g. SUIUSDT excluded from new entries)
        excluded = getattr(settings, "excluded_trading_symbols", [])
        if sym in excluded:
            logger.info(f"[{sym}] CONTRACT VALIDATION: Symbol is in excluded_trading_symbols.")
            return False

        # Enforce strict Binance Futures exchange metadata verification
        if not registry._symbols:
            logger.warning(f"[{sym}] CONTRACT VALIDATION: Binance Futures metadata registry is empty. Failing closed.")
            return False

        cfg = registry.get_symbol(sym)
        if not cfg:
            logger.warning(f"[{sym}] CONTRACT VALIDATION: Symbol not found in Binance Futures exchange metadata.")
            return False
        if cfg.status != "ENABLED" or not cfg.trading_enabled:
            logger.warning(f"[{sym}] CONTRACT VALIDATION: Contract status '{cfg.status}' is not tradable.")
            return False
        if cfg.quote_asset != "USDT":
            logger.warning(f"[{sym}] CONTRACT VALIDATION: Quote asset '{cfg.quote_asset}' != 'USDT'.")
            return False
        if cfg.market_type != "FUTURES":
            logger.warning(f"[{sym}] CONTRACT VALIDATION: Market type '{cfg.market_type}' != 'FUTURES'.")
            return False
        return True

    async def refresh_contract_metadata_if_needed(self):
        """Safely refreshes exchange contract metadata from Binance if empty or requested."""
        try:
            if not registry._symbols:
                logger.info("Refreshing Binance Futures contract metadata...")
                await registry.initialize_from_exchange(self.binance)
        except Exception as e:
            logger.error(f"Failed to refresh Binance contract metadata: {e}")

    def register_symbol(self, sym: str) -> bool:
        """
        Safely registers a dynamic symbol:
        Initializes portfolio tracking, history queues, deques, and stream subscriptions.
        """
        if not self.is_valid_contract(sym):
            logger.warning(f"Rejecting invalid dynamic symbol: {sym}")
            return False

        if sym not in self.symbols:
            self.symbols.append(sym)
            self.num_symbols = len(self.symbols)

        if sym not in self.portfolio_state.positions:
            self.portfolio_state.positions[sym] = PositionState(symbol=sym)

        if sym not in self.hidden_states:
            self.hidden_states[sym] = None
        if sym not in self.last_close_time:
            self.last_close_time[sym] = 0.0
        if sym not in self.position_entry_times:
            self.position_entry_times[sym] = 0.0
        if sym not in self.peak_pnl_pct:
            self.peak_pnl_pct[sym] = 0.0
        if sym not in self.pnl_history:
            self.pnl_history[sym] = collections.deque(maxlen=30)
        if sym not in self.smoothed_action_history:
            self.smoothed_action_history[sym] = collections.deque(maxlen=3)
        if sym not in self.smoothed_confidence_history:
            self.smoothed_confidence_history[sym] = collections.deque(maxlen=3)
        if sym not in self.last_hold_emission_time:
            self.last_hold_emission_time[sym] = 0.0
        if sym not in self.partial_tp_triggered:
            self.partial_tp_triggered[sym] = False
        if sym not in self.consecutive_losses:
            self.consecutive_losses[sym] = 0
        if sym not in self.loss_lockout_until:
            self.loss_lockout_until[sym] = {"LONG": 0.0, "SHORT": 0.0}
        if sym not in self.htf_regimes:
            self.htf_regimes[sym] = {"trend": "NEUTRAL", "adx": 0.0, "ema20": 0.0, "ema50": 0.0, "last_updated": 0.0}

        # Spawn listeners if bot is actively running
        if self.running and sym not in self._symbol_tasks:
            t_m = asyncio.create_task(self.listen_market(sym))
            t_w = asyncio.create_task(self.listen_whale(sym))
            t_s = asyncio.create_task(self.listen_statarb(sym))
            self._symbol_tasks[sym] = [t_m, t_w, t_s]
            logger.info(f"🌐 DYNAMIC UNIVERSE: Successfully registered and subscribed listeners for {sym}")

        return True

    async def listen_macro(self):
        await self.redis.connect()
        redis_conn = self.redis.redis
        pubsub = redis_conn.pubsub()
        await pubsub.subscribe("macro:state:global")
        
        async for message in pubsub.listen():
            if message["type"] == "message":
                try:
                    payload = json.loads(message["data"])
                    self.macro_state.sentiment_score = payload.get("sentiment_score", 0.0)
                    self.macro_state.volatility_expectation = payload.get("volatility_expectation", 0.5)
                    self.macro_state.regime = payload.get("regime", 0.0)
                    self.macro_state.sp500_momentum = payload.get("sp500_momentum", 0.0)
                    self.macro_state.dxy_momentum = payload.get("dxy_momentum", 0.0)
                    self.macro_state.vix_momentum = payload.get("vix_momentum", 0.0)
                    self.macro_state.gold_momentum = payload.get("gold_momentum", 0.0)
                    self.macro_state.treasury_yield_momentum = payload.get("treasury_yield_momentum", 0.0)
                    self.macro_state.ndx_momentum = payload.get("ndx_momentum", 0.0)
                    self.macro_state.defi_tvl_momentum = payload.get("defi_tvl_momentum", 0.0)
                    self.macro_state.fear_greed_index = payload.get("fear_greed_index", 0.5)
                    self.macro_state.timestamp = time.time()
                except Exception as e:
                    logger.error(f"Error parsing macro: {e}")
                    
    async def listen_market(self, symbol: str):
        await self.redis.connect()
        redis_conn = self.redis.redis
        pubsub = redis_conn.pubsub()
        await pubsub.subscribe(f"market:state:{symbol}")
        
        async for message in pubsub.listen():
            if message["type"] == "message":
                try:
                    payload = json.loads(message["data"])
                    if float(payload.get("mid_price", 0.0)) <= 0.0:
                        continue # Wait until market collector has real price data
                    # Parse into Canonical MarketState
                    m_state = MarketState(**payload)
                    self.market_states[symbol] = m_state
                    self.last_market_tick_time = time.time()
                    
                    # Institutional Upgrade 1: BTC Flash-Crash Shield update
                    if symbol == "BTCUSDT":
                        crash_now, drop_pct = macro_crash_shield.update_btc_price(m_state.mid_price, m_state.timestamp)
                        if crash_now:
                            peak_px = max([p for t, p in macro_crash_shield.btc_price_history])
                            asyncio.create_task(
                                telegram_notifier.broadcast_alert(
                                    "FLASH_CRASH",
                                    "",
                                    {
                                        "pct_drop": drop_pct,
                                        "peak_price": peak_px,
                                        "current_price": m_state.mid_price,
                                        "lockout_minutes": int(settings.macro_flash_crash_lockout_seconds / 60)
                                    }
                                )
                            )
                except Exception as e:
                    logger.error(f"Error parsing market state for {symbol}: {e}")
                    
    async def listen_whale(self, symbol: str):
        await self.redis.connect()
        redis_conn = self.redis.redis
        pubsub = redis_conn.pubsub()
        await pubsub.subscribe(f"whale:state:{symbol}")
        
        async for message in pubsub.listen():
            if message["type"] == "message":
                try:
                    payload = json.loads(message["data"])
                    self.whale_states[symbol] = payload
                except Exception as e:
                    logger.error(f"Error parsing whale state for {symbol}: {e}")

    async def listen_statarb(self, symbol: str):
        await self.redis.connect()
        redis_conn = self.redis.redis
        pubsub = redis_conn.pubsub()
        await pubsub.subscribe(f"statarb:state:{symbol}")
        
        async for message in pubsub.listen():
            if message["type"] == "message":
                try:
                    payload = json.loads(message["data"])
                    self.statarb_states[symbol] = payload
                except Exception as e:
                    logger.error(f"Error parsing statarb state for {symbol}: {e}")

    def _calc_ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """Calculates Exponential Moving Average across a 1D price series."""
        if len(data) == 0: return np.array([])
        alpha = 2.0 / (period + 1)
        ema = np.zeros_like(data)
        ema[0] = data[0]
        for i in range(1, len(data)):
            ema[i] = data[i] * alpha + ema[i-1] * (1 - alpha)
        return ema

    def _calc_adx(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> float:
        """Calculates Average Directional Index (ADX) measuring trend strength."""
        if len(closes) < period + 2: return 0.0
        try:
            tr = np.maximum(highs[1:] - lows[1:], np.maximum(abs(highs[1:] - closes[:-1]), abs(lows[1:] - closes[:-1])))
            plus_dm = np.where((highs[1:] - highs[:-1] > lows[:-1] - lows[1:]) & (highs[1:] - highs[:-1] > 0), highs[1:] - highs[:-1], 0.0)
            minus_dm = np.where((lows[:-1] - lows[1:] > highs[1:] - highs[:-1]) & (lows[:-1] - lows[1:] > 0), lows[:-1] - lows[1:], 0.0)
            atr_series = self._calc_ema(tr, period)
            atr = atr_series[-1] if len(atr_series) > 0 else 0.0
            if atr <= 0: return 0.0
            p_dm_series = self._calc_ema(plus_dm, period)
            m_dm_series = self._calc_ema(minus_dm, period)
            plus_di = (p_dm_series[-1] / atr) * 100.0 if len(p_dm_series) > 0 else 0.0
            minus_di = (m_dm_series[-1] / atr) * 100.0 if len(m_dm_series) > 0 else 0.0
            di_sum = plus_di + minus_di
            if di_sum <= 0: return 0.0
            dx = (abs(plus_di - minus_di) / di_sum) * 100.0
            return float(dx)
        except Exception:
            return 0.0

    async def update_htf_regimes_loop(self):
        """
        Periodically analyzes Higher-Timeframe (1H) trend, EMA stack, and ADX for all symbols.
        Prevents taking counter-trend positions during violent momentum expansions (Hedge Fund Rule).
        """
        while self.running:
            try:
                for sym in self.symbols:
                    try:
                        klines = await self.binance.get_klines(sym, interval="1h", limit=50)
                        if klines and len(klines) >= 20:
                            closes = np.array([float(k[4]) for k in klines])
                            highs = np.array([float(k[2]) for k in klines])
                            lows = np.array([float(k[3]) for k in klines])
                            cur_px = closes[-1]
                            
                            ema20 = float(self._calc_ema(closes, 20)[-1])
                            ema50 = float(self._calc_ema(closes, min(50, len(closes)))[-1])
                            adx_val = float(self._calc_adx(highs, lows, closes, period=14))
                            
                            # Bullish Expansion: Price > EMA20 > EMA50 with authentic trend momentum (ADX >= 20.0)
                            if cur_px > ema20 and ema20 > ema50 and adx_val >= 20.0:
                                trend = "STRONG_BULL"
                            # Bearish Breakdown: Price < EMA20 < EMA50 with authentic trend momentum (ADX >= 20.0)
                            elif cur_px < ema20 and ema20 < ema50 and adx_val >= 20.0:
                                trend = "STRONG_BEAR"
                            elif cur_px > ema20 and ema20 > ema50:
                                trend = "WEAK_BULL"
                            elif cur_px < ema20 and ema20 < ema50:
                                trend = "WEAK_BEAR"
                            else:
                                trend = "RANGING"
                                
                            self.htf_regimes[sym] = {
                                "trend": trend,
                                "adx": adx_val,
                                "ema20": ema20,
                                "ema50": ema50,
                                "last_updated": time.time()
                            }
                    except Exception as sym_err:
                        logger.debug(f"[{sym}] HTF update skipped: {sym_err}")
                    await asyncio.sleep(0.4)
            except Exception as e:
                logger.warning(f"Error in HTF regime loop: {e}")
            await asyncio.sleep(180.0) # Refresh every 3 minutes
                    
    async def poll_global_context_loop(self):
        """Refreshes market-wide signals published by the market collector (apps/market_collector/global_context.py)."""
        while self.running:
            try:
                if self.redis.redis:
                    raw = await self.redis.redis.get(GLOBAL_CONTEXT_KEY)
                    if raw:
                        self.global_context = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
            except Exception as e:
                logger.debug(f"Global context refresh failed: {e}")
            await asyncio.sleep(30.0)

    def _cross_section_features(self, symbol: str) -> list:
        """
        Ranks this coin against every coin with fresh data at the same moment (state slots 150-159):
        return relative to the basket average (5m / 15m / 30m / 24h), percentile rank in [-1, 1]
        (15m / 30m / 24h, funding) and the basket's own 15m / 30m move. Lets the model learn
        "will this coin beat the others", which survives market-wide crashes better than direction alone.
        """
        now = time.time()
        fresh = {s: m for s, m in self.market_states.items()
                 if m is not None and len(m.features) == 90 and now - float(getattr(m, "timestamp", 0.0) or 0.0) < 120.0}
        if symbol not in fresh or len(fresh) < 4:
            return [0.0] * CROSS_SECTION_DIM

        def col(i):
            return {s: float(m.features[i]) for s, m in fresh.items()}

        def rel(d):
            return d[symbol] - sum(d.values()) / len(d)

        def rank(d):
            v = d[symbol]
            below = sum(1 for x in d.values() if x < v)
            equal = sum(1 for x in d.values() if x == v)
            return 2.0 * (below + (equal - 1) / 2.0) / (len(d) - 1) - 1.0

        r5, r15, r30, r24, fund = col(3), col(4), col(5), col(6), col(80)
        return [rel(r5), rel(r15), rel(r30), rel(r24),
                rank(r15), rank(r30), rank(r24), rank(fund),
                sum(r15.values()) / len(r15), sum(r30.values()) / len(r30)]

    def _global_features(self) -> list:
        g = self.global_context or {}
        if time.time() - float(g.get("timestamp", 0.0) or 0.0) > 600.0:
            g = {}
        vals = [float(g.get(k, 0.0) or 0.0) for k in GLOBAL_KEYS]
        return vals + [0.0] * (10 - len(vals))

    def _build_state_vector(self, symbol: str) -> tuple[torch.Tensor, list]:
        obs = []
        
        # 1. Portfolio & Risk (10 dims)
        equity = self.portfolio_state.equity
        free_margin = self.portfolio_state.free_margin
        margin_utilization = (equity - free_margin) / equity if equity > 0 else 0.0
        
        # Advanced Risk: Portfolio Heat
        active_positions = sum(1 for p in self.portfolio_state.positions.values() if p.quantity != 0)
        portfolio_heat = active_positions / self.num_symbols if self.num_symbols > 0 else 0.0
        
        # Meta-Learning & StatArb Internal Metrics
        if equity > self.max_equity:
            self.max_equity = equity
        account_drawdown = (self.max_equity - equity) / self.max_equity if self.max_equity > 0 else 0.0
        
        self.equity_history.append(equity)
        import numpy as np
        portfolio_variance = float(np.var(self.equity_history) / equity) if equity > 0 and len(self.equity_history) > 1 else 0.0
        time_since_win_hrs = (time.time() - self.last_win_time) / 3600.0 if self.last_win_time > 0 else 0.0
        
        # Scale raw dollar amounts to normalized ratios (adaptive to account's starting capital)
        # Prevents large cash numbers from blowing up LayerNorm variance regardless of wallet size ($500, $5k, $50k)
        if getattr(self, 'initial_balance', None) is None or self.initial_balance <= 0:
            if self.portfolio_state.wallet_balance > 0:
                self.initial_balance = float(self.portfolio_state.wallet_balance)
            else:
                self.initial_balance = 5000.00
                
        base_capital = self.initial_balance if self.initial_balance > 0 else 5000.00
        norm_wallet = float(np.clip(self.portfolio_state.wallet_balance / base_capital, 0.0, 5.0))
        norm_equity = float(np.clip(equity / base_capital, 0.0, 5.0))
        norm_free_margin = float(np.clip(free_margin / equity if equity > 0 else 1.0, 0.0, 1.0))
        norm_exposure = float(np.clip(self.portfolio_state.total_exposure / equity if equity > 0 else 0.0, 0.0, 10.0))
        norm_time_since_win = float(np.clip(time_since_win_hrs / 24.0, 0.0, 2.0))
        
        obs.extend([
            norm_wallet, 
            norm_equity,
            norm_free_margin,
            margin_utilization,
            norm_exposure,
            float(portfolio_heat),
            float(self.session_win_rate),
            float(account_drawdown), 
            float(portfolio_variance), 
            norm_time_since_win
        ])
        
        # 2. Position Awareness (10 dims)
        pos = self.portfolio_state.positions[symbol]
        market = self.market_states.get(symbol)
        
        time_held_hours = 0.0
        entry_dist_vwap = 0.0
        current_pnl_pct = 0.0
        dist_to_liq = 0.0
        
        if pos.quantity != 0 and pos.entry_price > 0 and market:
            raw_pnl = (market.mid_price - pos.entry_price) / pos.entry_price
            current_pnl_pct = raw_pnl if pos.quantity > 0 else -raw_pnl
            
            # High & Low Water Mark tracking
            if current_pnl_pct > pos.max_unrealized_pnl:
                pos.max_unrealized_pnl = current_pnl_pct
            if current_pnl_pct < pos.max_drawdown_pnl:
                pos.max_drawdown_pnl = current_pnl_pct
                
            if pos.liquidation_price > 0:
                dist_to_liq = abs(market.mid_price - pos.liquidation_price) / market.mid_price
                
            time_held_hours = (time.time() - pos.entry_time) / 3600.0 if pos.entry_time > 0 else 0.0
            if market.vwap > 0:
                entry_dist_vwap = (pos.entry_price - market.vwap) / market.vwap
        else:
            # Reset High Water Mark when flat
            pos.max_unrealized_pnl = 0.0
            pos.max_drawdown_pnl = 0.0
            pos.entry_time = 0.0
                
        # Normalized entry price distance ratio rather than raw $81,000 BTC dollar price
        entry_dist = (pos.entry_price - market.mid_price) / market.mid_price if market and market.mid_price > 0 and pos.entry_price > 0 else 0.0
        pos_signed_dir = 1.0 if pos.quantity > 0 else (-1.0 if pos.quantity < 0 else 0.0)
        
        obs.extend([
            float(pos_signed_dir),
            float(np.clip(entry_dist, -1.0, 1.0)),
            float(pos.leverage) / 50.0, # normalized
            current_pnl_pct,
            dist_to_liq,
            # Fully utilizing the 10-slot Position Block
            float(pos.max_unrealized_pnl), 
            float(pos.max_drawdown_pnl),
            float(np.clip(time_held_hours / 24.0, 0.0, 2.0)),
            float(np.clip(entry_dist_vwap, -1.0, 1.0)),
            float(np.clip(pos.accumulated_funding, -1.0, 1.0))
        ])
        
        # 3. Market Features (90 dims) from engine.py
        if market and len(market.features) == 90:
            cleaned_features = []
            for idx, val in enumerate(market.features):
                # Per-slot transforms come from core/schemas/state_layout.py (single source of truth)
                if idx in MARKET_DEPRECATED_FEATURES:
                    cleaned_features.append(0.0)
                elif idx in MARKET_LOG_VOLUME_FEATURES:
                    v_abs = abs(float(val))
                    scaled_v = np.sign(float(val)) * (np.log1p(v_abs) / 10.0)
                    cleaned_features.append(float(np.clip(scaled_v, -5.0, 5.0)))
                elif idx == MARKET_SPREAD_FEATURE: # spread_bps (scale so 100 bps = 1.0)
                    cleaned_features.append(float(np.clip(float(val) / 100.0, 0.0, 5.0)))
                else:
                    cleaned_features.append(float(np.clip(float(val), -5.0, 5.0)))
            obs.extend(cleaned_features)
        else:
            obs.extend([0.0] * 90)
            
        # 4. Macro & Cross-Asset (20 dims)
        current_hour = time.localtime().tm_hour
        current_min = time.localtime().tm_min
        minute_of_day = current_hour * 60 + current_min
        time_sin = np.sin(2 * np.pi * minute_of_day / 1440.0)
        time_cos = np.cos(2 * np.pi * minute_of_day / 1440.0)
        
        day_w = time.localtime().tm_wday
        day_sin = np.sin(2 * np.pi * day_w / 7.0)
        day_cos = np.cos(2 * np.pi * day_w / 7.0)
        
        btc_market = self.market_states.get("BTCUSDT")
        btc_mom_1m = btc_market.features[2] if btc_market and len(btc_market.features) == 90 else 0.0
        btc_mom_15m = btc_market.features[4] if btc_market and len(btc_market.features) == 90 else 0.0
        
        # Cross-Coin Beta (Relative Strength to BTC)
        sym_mom_1m = market.features[2] if market and len(market.features) == 90 else 0.0
        cross_coin_beta = sym_mom_1m / btc_mom_1m if abs(btc_mom_1m) > 1e-6 else 0.0
        cross_coin_beta = float(np.clip(cross_coin_beta, -5.0, 5.0))
        
        macro_features = [
            float(time_sin), float(time_cos), float(day_sin), float(day_cos), float(cross_coin_beta), 0.0,
            float(self.macro_state.sentiment_score),
            float(self.macro_state.volatility_expectation),
            float(self.macro_state.regime),
            btc_mom_1m, btc_mom_15m, 
            float(self.macro_state.sp500_momentum), 
            float(self.macro_state.dxy_momentum), 
            float(self.macro_state.fear_greed_index),
            float(self.macro_state.vix_momentum),
            float(self.macro_state.gold_momentum),
            float(self.macro_state.treasury_yield_momentum),
            float(self.macro_state.ndx_momentum),
            float(self.macro_state.defi_tvl_momentum), 
            0.0
        ]
        obs.extend(macro_features)
        
        # 5. Future-Proofing Canvas: Whale & StatArb Injection (70 dims)
        whale = self.whale_states.get(symbol, {})
        statarb = self.statarb_states.get(symbol, {})
        
        # Institutional Whale Order Flow Normalization:
        # Prevent injecting raw 8-figure dollar values (e.g. $124M) into the LSTM neural network!
        w_buy_raw = float(whale.get("whale_buy_pressure", 0.0))
        w_sell_raw = float(whale.get("whale_sell_pressure", 0.0))
        total_whale_vol = w_buy_raw + w_sell_raw
        whale_imbalance = (w_buy_raw - w_sell_raw) / (total_whale_vol + 1e-5) if total_whale_vol > 0 else 0.0
        whale_vol_norm = float(np.clip(np.log1p(total_whale_vol) / 16.0, 0.0, 1.0))
        
        obs.extend([
            # Whale Tracking (Slots 130 - 139)
            float(whale.get("buy_wall_distance", 0.0)),
            float(whale.get("sell_wall_distance", 0.0)),
            float(whale_imbalance), # Normalized [-1.0, 1.0] Order Flow Imbalance
            float(whale_vol_norm),  # Normalized [0.0, 1.0] Log Volume
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            
            # StatArb Tracking (Slots 140 - 149)
            float(statarb.get("okx_premium", 0.0)),
            float(statarb.get("bybit_premium", 0.0)),
            float(statarb.get("okx_momentum_lead", 0.0)),
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ])
        
        # 6. Cross-sectional ranking (150-159) and market-wide signals (160-169), core/schemas/state_layout.py
        obs.extend(self._cross_section_features(symbol))
        obs.extend(self._global_features())

        # Reserved canvas for future connections (170-199)
        obs.extend([0.0] * 30)
        
        if len(obs) != 200:
            raise ValueError(f"State vector dimension mismatch. Expected 200, got {len(obs)}")
        
        # SAFETY NET (matches trainer's universal clamp in normalize_state_tensor):
        # A single unbounded input (e.g. raw $49B DeFi TVL) dominates LayerNorm(200) and makes
        # every symbol produce identical outputs. Replace NaN/Inf and clamp every slot to [-5, 5].
        obs_t = torch.tensor(obs, dtype=torch.float32)
        obs_t = torch.nan_to_num(obs_t, nan=0.0, posinf=5.0, neginf=-5.0).clamp_(-5.0, 5.0)
        return obs_t.unsqueeze(0), macro_features
                
    async def sync_dashboard(self):
        """Periodically fetches live positions from Binance and publishes to dashboard."""
        while self.running:
            try:
                if self.trading_enabled and not self.dry_run:
                    # Update live equity
                    usdt = await self.binance.get_account_details()
                    self.portfolio_state.wallet_balance = float(usdt.get("balance", 0.0))
                    self.portfolio_state.equity = float(usdt.get("crossWalletBalance", self.portfolio_state.wallet_balance)) + float(usdt.get("crossUnPnl", 0.0))
                    self.portfolio_state.free_margin = float(usdt.get("availableBalance", self.portfolio_state.wallet_balance))
                    if (getattr(self, 'initial_balance', None) is None or self.initial_balance <= 0) and self.portfolio_state.wallet_balance > 0:
                        self.initial_balance = self.portfolio_state.wallet_balance
                    
                    # Passive Airbag 4: Maintenance Margin Cushion (Anti-Exchange Liquidation Auto-Trim)
                    now_t = time.time()
                    if (self.portfolio_state.equity > 0 and 
                        (self.portfolio_state.free_margin / self.portfolio_state.equity) < 0.15 and 
                        (now_t - self.last_margin_trim_time) > 60.0 and
                        not self.emergency_stop and not self.dry_run and 
                        (settings.allow_testnet or settings.allow_live_trading)):
                        
                        active_pos_list = [
                            (s, p) for s, p in self.portfolio_state.positions.items() 
                            if abs(p.quantity) > 0.0001
                        ]
                        if active_pos_list:
                            def _calc_underwater_roi(item):
                                sym_name, pos_obj = item
                                m = self.market_states.get(sym_name)
                                px = m.mid_price if m else pos_obj.entry_price
                                if pos_obj.entry_price > 0:
                                    return ((px - pos_obj.entry_price) / pos_obj.entry_price) * (1 if pos_obj.quantity > 0 else -1)
                                return 0.0

                            worst_sym, worst_pos = min(active_pos_list, key=_calc_underwater_roi)
                            m_st = self.market_states.get(worst_sym)
                            sym_cfg = registry.get_symbol(worst_sym)
                            if m_st and sym_cfg:
                                trim_qty = abs(worst_pos.quantity) * 0.25
                                trim_qty_str = sym_cfg.format_quantity(trim_qty)
                                if float(trim_qty_str) > 0 and (float(trim_qty_str) * m_st.mid_price) >= sym_cfg.min_notional:
                                    trim_side = "SELL" if worst_pos.quantity > 0 else "BUY"
                                    free_pct = (self.portfolio_state.free_margin / self.portfolio_state.equity) * 100
                                    logger.critical(
                                        f"🚨 AIRBAG 4 ACTIVATED: CRITICAL MARGIN RATIO ({free_pct:.1f}% free margin < 15%)! "
                                        f"Executing auto-deleveraging trim: {trim_side} {trim_qty_str} {worst_sym} to restore margin cushion before Binance liquidation."
                                    )
                                    try:
                                        trim_res = await self.binance.create_order(
                                            symbol=worst_sym,
                                            side=trim_side,
                                            quantity=float(trim_qty_str),
                                            client_order_id=f"airbag4_{uuid.uuid4().hex[:8]}",
                                            order_type="MARKET",
                                            reduce_only=True
                                        )
                                        self.last_margin_trim_time = now_t
                                        signed_trim = float(trim_qty_str) if trim_side == "BUY" else -float(trim_qty_str)
                                        await self._record_airbag_trim(worst_sym, worst_pos, float(trim_qty_str), trim_res, m_st.mid_price)
                                        worst_pos.quantity += signed_trim
                                        asyncio.create_task(
                                            telegram_notifier.broadcast_alert(
                                                "MARGIN_CUSHION_TRIM",
                                                worst_sym,
                                                {
                                                    "free_margin_ratio": free_pct,
                                                    "trimmed_symbol": worst_sym,
                                                    "trimmed_qty": trim_qty_str
                                                }
                                            )
                                        )
                                    except Exception as trim_err:
                                        logger.error(f"Failed to execute Airbag 4 margin trim: {trim_err}")
                    
                    # Update live positions
                    live_positions = await self.binance.get_positions()
                    formatted_positions = []
                    # Update live positions without zeroing race conditions
                    active_synced_symbols = set()
                    for p in live_positions:
                        sym = p.get("symbol")
                        amt = float(p.get("positionAmt", 0))
                        
                        # Sync back to internal portfolio state
                        if sym in self.portfolio_state.positions:
                            active_synced_symbols.add(sym)
                            self.portfolio_state.positions[sym].quantity = amt
                            self.portfolio_state.positions[sym].leverage = int(p.get("leverage", 10))
                            if amt != 0:
                                self.portfolio_state.positions[sym].entry_price = float(p.get("entryPrice", 0))
                                if sym in self._restored_pos_state:
                                    self._apply_restored_position(sym, float(p.get("entryPrice", 0)))
                                # MEMORY RESILIENCE: If position is open but entry time was not tracked (e.g. after PM2 restart),
                                # recover the exact entry timestamp from Binance updateTime so the 600s breathing room stays active!
                                if self.position_entry_times.get(sym, 0.0) == 0.0:
                                    pos_update_ms = float(p.get("updateTime", 0.0))
                                    if pos_update_ms > 0:
                                        self.position_entry_times[sym] = pos_update_ms / 1000.0
                                        self.portfolio_state.positions[sym].entry_time = pos_update_ms / 1000.0
                                        logger.info(f"[{sym}] 🔄 RECOVERED ENTRY TIME from Binance: {time.time() - (pos_update_ms / 1000.0):.1f}s ago")
                                    else:
                                        self.position_entry_times[sym] = time.time()
                                        self.portfolio_state.positions[sym].entry_time = time.time()
                            else:
                                self.peak_pnl_pct[sym] = 0.0
                                self.pnl_history[sym].clear()
                                self.position_entry_times[sym] = 0.0
                                self.portfolio_state.positions[sym].entry_price = 0.0
                                self.partial_tp_triggered[sym] = False
                                self.position_exit_levels.pop(sym, None)
                                
                        # CRITICAL: Only send active positions to the dashboard (once per active position)
                        if abs(amt) > 0:
                            pnl = float(p.get("unRealizedProfit", 0))
                            formatted_positions.append({
                                "symbol": sym,
                                "side": "LONG" if amt > 0 else "SHORT",
                                "quantity": abs(amt),
                                "pnl": pnl,
                                "entryPrice": float(p.get("entryPrice", 0)),
                                "markPrice": float(p.get("markPrice", 0)),
                                "leverage": int(p.get("leverage", 1)),
                                "liquidationPrice": float(p.get("liquidationPrice", 0)),
                                "marginType": p.get("marginType", "cross"),
                                "entry_time": float(self.position_entry_times.get(sym) or 0.0),
                            })

                            # Auto-reconcile missing OPEN trade in history (ONLY for positions recovered after PM2 restart)
                            if self.redis.redis:
                                try:
                                    recent_close = (time.time() - self.last_close_time.get(sym, 0)) < 60.0
                                    inflight_order = sym in self.active_orders
                                    entry_px = float(p.get("entryPrice", 0))
                                    notional = abs(amt) * entry_px
                                    
                                    # Never auto-reconcile if a close just happened, order is in-flight, or it's sub-$10 micro-dust
                                    if not recent_close and not inflight_order and entry_px > 0 and notional >= 10.0:
                                        raw_history = await self.redis.redis.lrange("dashboard:trade_history", 0, 49)
                                        # The CURRENT position has an OPEN row only if this symbol's newest row is an OPEN
                                        # (or a partial close of it). An OPEN from an older, already-closed trade does not count.
                                        has_open_record = False
                                        for item in raw_history:
                                            try:
                                                th = json.loads(item)
                                                if th.get("symbol") == sym:
                                                    th_side = th.get("side", "")
                                                    has_open_record = th_side.startswith("OPEN") or th_side.startswith("PARTIAL_")
                                                    break
                                            except Exception:
                                                pass
                                        
                                        # Only reconcile if position has been open for > 15s (proves it's a persistent recovered position, not a transient order)
                                        entry_t = self.position_entry_times.get(sym, time.time())
                                        if not has_open_record and (time.time() - entry_t) > 15.0:
                                            open_record = {
                                                "timestamp": entry_t,
                                                "symbol": sym,
                                                "side": "OPEN_LONG" if amt > 0 else "OPEN_SHORT",
                                                "quantity": abs(amt),
                                                "price": entry_px,
                                                "confidence": 0.85,
                                                "order_id": f"synced_{int(entry_t)}",
                                                "entry_price": entry_px,
                                                "gross_pnl": 0.0,
                                                "fees_paid": 0.0,
                                                "realized_pnl": 0.0,
                                                "roi_pct": 0.0
                                            }
                                            await self.redis.redis.lpush("dashboard:trade_history", json.dumps(open_record))
                                            await self.redis.redis.ltrim("dashboard:trade_history", 0, 99)
                                            logger.info(f"[{sym}] 🔄 Auto-reconciled recovered OPEN trade into Dashboard History ({open_record['side']} {abs(amt)} at ${entry_px})")
                                except Exception as hist_err:
                                    logger.warning(f"[{sym}] History auto-reconcile check failed: {hist_err}")

                    # Zero out any symbols absent from Binance response
                    for s in self.symbols:
                        if s in self.portfolio_state.positions and s not in active_synced_symbols:
                            self.portfolio_state.positions[s].quantity = 0.0
                            self.portfolio_state.positions[s].entry_price = 0.0
                            self.partial_tp_triggered[s] = False
                            self.position_exit_levels.pop(s, None)

                    await self._ensure_disaster_stops(live_positions)
                    
                    if self.redis.redis:
                        payload = {
                            "equity": self.portfolio_state.equity,
                            "positions": formatted_positions
                        }
                        await self.redis.redis.set("dashboard:portfolio", json.dumps(payload))
                        
            except Exception as e:
                logger.error(f"Dashboard sync failed: {e}")
            await asyncio.sleep(5.0)

    async def inference_loop(self):
        logger.info("Starting production inference loop...")
        while self.running:
            try:
                # 0. Fast Authoritative Redis Kill Switch & Emergency Stop Check
                ks_active = False
                try:
                    if self.redis.redis:
                        ks = await self.redis.redis.get("system:kill_switch:active")
                        val = (ks.decode("utf-8") if isinstance(ks, bytes) else str(ks or "")).strip().lower()
                        if val in ["true", "1", "yes"]:
                            ks_active = True
                except Exception as ks_err:
                    logger.error(f"Failed to check Redis kill switch in inference loop: {ks_err}")

                if ks_active or self.emergency_stop or settings.emergency_stop:
                    if not self.emergency_stop or not settings.emergency_stop:
                        logger.critical("🚨 EMERGENCY KILL SWITCH DETECTED IN INFERENCE LOOP: Halting trading immediately!")
                        self.emergency_stop = True
                        self.trading_enabled = False
                        settings.emergency_stop = True
                        await self.panic_liquidate()
                    await asyncio.sleep(3.0)
                    continue

                # 0. Hot Reload Model (a failed load keeps the current brain and retries next tick: an error
                #    here used to halt the whole bot via FAIL CLOSED)
                if self.registry.check_for_updates():
                    logger.info("New model version detected. Hot reloading...")
                    try:
                        self.model = self.registry.load_model(self.model)
                        self.model.eval()
                        self.has_valid_model = True
                        self._refresh_actor_signal_offset()
                        logger.info("Hot reload complete.")
                    except Exception as reload_err:
                        logger.error(f"Hot reload failed, keeping the current brain: {reload_err}")
                # Trade policy (min confidence, calibration, signal offset) lives in version.json: re-read it
                # whenever that file changes, not only when the weights do.
                try:
                    v_mtime = os.path.getmtime(os.path.join(self.registry.model_dir, "version.json"))
                except OSError:
                    v_mtime = 0.0
                if v_mtime and v_mtime != getattr(self, "_version_mtime", 0.0):
                    if getattr(self, "_version_mtime", 0.0):
                        self._refresh_actor_signal_offset()
                    self._version_mtime = v_mtime

                now_t = time.time()
                # 0.1 Dynamic Universe Synchronization: Integrate hot coins from UniverseScreener
                if self.redis.redis and (now_t - getattr(self, "_last_universe_check", 0.0)) > 60.0:
                    self._last_universe_check = now_t
                    try:
                        raw_u = await self.redis.redis.get("system:active_symbols")
                        if raw_u:
                            u_list = json.loads(raw_u.decode("utf-8") if isinstance(raw_u, bytes) else raw_u)
                            if isinstance(u_list, list) and len(u_list) > 0:
                                self.dynamic_active_symbols = set(u_list)
                                for s in u_list:
                                    self.register_symbol(s)
                    except Exception as u_err:
                        logger.debug(f"Dynamic universe check: {u_err}")

                # 24-Hour Rolling Circuit Breaker Tracking (5% Daily Loss Kill Switch) & Profit Ratchet
                current_eq = float(self.portfolio_state.equity)
                if (now_t - self.daily_equity_reset_time) >= 86400.0 or self.daily_start_equity <= 0.0:
                    if current_eq > 0:
                        self.daily_start_equity = current_eq
                        self.daily_peak_equity = current_eq
                        self.daily_equity_reset_time = now_t
                        self.profit_ratchet_active = False
                        logger.info(f"📅 24-Hour Equity Benchmark Reset: Starting Equity = ${self.daily_start_equity:.2f}")

                if self.daily_start_equity > 0 and current_eq > 0:
                    # Update rolling daily high-water peak equity
                    if current_eq > self.daily_peak_equity:
                        self.daily_peak_equity = current_eq

                    # 1. 5% Daily Loss Circuit Breaker Kill Switch
                    daily_dd = (self.daily_start_equity - current_eq) / self.daily_start_equity
                    if daily_dd >= 0.05 and now_t > self.circuit_breaker_active_until:
                        self.circuit_breaker_active_until = now_t + (6 * 3600.0) # 6 hours standby
                        logger.critical(f"🚨 5% DAILY LOSS CIRCUIT BREAKER TRIPPED! Daily drawdown is {daily_dd*100:.2f}% >= 5.00%. Pausing new entries for 6 hours!")

                    # 2. Daily High-Water Profit Ratchet (Session Profit Lock)
                    # If daily peak profit reached >= +4.0% (e.g. +$180 on $4,500), protect accumulated daily profits!
                    # If current profit retraces by >= 50% from daily peak profit, arm Capital Preservation Mode.
                    peak_profit = self.daily_peak_equity - self.daily_start_equity
                    current_profit = current_eq - self.daily_start_equity
                    if peak_profit >= (0.040 * self.daily_start_equity):
                        if current_profit < (0.50 * peak_profit) and not self.profit_ratchet_active:
                            self.profit_ratchet_active = True
                            logger.warning(
                                f"🛡️ DAILY PROFIT RATCHET ARMED (Capital Preservation Mode): Daily peak profit was +${peak_profit:.2f} "
                                f"({(peak_profit/self.daily_start_equity)*100:.2f}%), but current profit retraced to +${current_profit:.2f}. "
                                f"Halving base margin to 5.0% and elevating confidence hurdle to 74% to permanently preserve winning day!"
                            )
                        elif current_profit >= (0.75 * peak_profit) and self.profit_ratchet_active:
                            self.profit_ratchet_active = False
                            logger.info(f"🛡️ Daily Profit Ratchet disarmed: Equity recovered to +${current_profit:.2f} (>= 75% of peak). Normal sizing restored.")

                # Institutional Liquidity Clock & Session Volatility Regimes
                utc_now = datetime.datetime.now(datetime.timezone.utc)
                is_weekend = (utc_now.weekday() >= 5) # Saturday (5) or Sunday (6)
                utc_hour = utc_now.hour + (utc_now.minute / 60.0)
                
                if is_weekend:
                    session_name = "WEEKEND_LOW_LIQUIDITY"
                    session_conf_hurdle = settings.conf_hurdle_weekend
                    session_max_spread = 0.00045 # Tight 0.045% limit in low liquidity
                    session_critic_hurdle = 0.30
                elif 13.5 <= utc_hour < 21.0: # US New York Peak Trend & Volatility (13:30 - 21:00 UTC)
                    session_name = "US_NEW_YORK_PEAK"
                    session_conf_hurdle = settings.conf_hurdle_ny_peak
                    session_max_spread = 0.00060 # Standard 0.060%
                    session_critic_hurdle = 0.22
                elif 8.0 <= utc_hour < 13.5: # London Morning & Pre-NY (08:00 - 13:30 UTC)
                    session_name = "LONDON_TREND"
                    session_conf_hurdle = settings.conf_hurdle_london
                    session_max_spread = 0.00060
                    session_critic_hurdle = 0.22
                else: # Asia Off-Peak (00:00 - 08:00 UTC) Mean-Reversion Chop
                    session_name = "ASIA_OFF_PEAK_CHOP"
                    session_conf_hurdle = settings.conf_hurdle_asia
                    session_max_spread = 0.00050 # 0.050%
                    session_critic_hurdle = 0.30

                # Phase 2: Batched Inference Pre-Pass
                # Collect ALL valid symbol state vectors and run ONE model call instead of 6.
                # SAFETY SHIELD: Symbols with wrong shape are silently skipped (fall back to individual inference).
                precomputed_actions = {}  # {sym: action_logits_np_array}
                precomputed_critics = {}  # {sym: critic_pred_np_array}
                precomputed_seqs = {}     # {sym: (state_vector_tensor, seq_list)} — cache to avoid double build
                
                batch_tensors = []
                batch_syms = []
                
                for sym in self.symbols:
                    if sym not in self.market_states:
                        continue
                    if sym in self.active_orders:
                        continue  # Handled inside main loop
                    try:
                        sv, mf = self._build_state_vector(sym)
                        sv_list = sv.squeeze(0).tolist()
                        # Build seq using current history + new frame (peek-ahead)
                        seq = list(self.state_history[sym]) + [sv_list]
                        while len(seq) < self.sequence_length:
                            seq.insert(0, seq[0])
                        seq = seq[-self.sequence_length:]
                        
                        st = torch.tensor(seq, dtype=torch.float32)
                        # SAFETY SHIELD: Only batch if shape is exactly right
                        if st.shape == (self.sequence_length, 200):
                            batch_tensors.append(st)
                            batch_syms.append(sym)
                            precomputed_seqs[sym] = (sv, seq)
                        else:
                            logger.warning(f"[{sym}] Skipping batch inference: shape {st.shape}")
                    except Exception as e:
                        logger.warning(f"[{sym}] Batch pre-pass failed: {e}")
                
                if batch_tensors:
                    try:
                        final_batch = torch.stack(batch_tensors)  # [N, seq_len, 200]
                        final_batch = torch.nan_to_num(final_batch, nan=0.0, posinf=1.0, neginf=-1.0)
                        
                        with torch.no_grad():
                            batch_logits, batch_critic = self.model(self._mask_unseen_inputs(final_batch), hidden_state=None, return_hidden=False)
                        for i, sym in enumerate(batch_syms):
                            precomputed_actions[sym] = batch_logits[i].numpy()
                            precomputed_critics[sym] = batch_critic[i].numpy()
                    except Exception as e:
                        logger.warning(f"Batch inference failed, falling back to individual: {e}")
                        precomputed_actions = {}  # Force fallback
                        precomputed_critics = {}
                
                # Dynamic Opportunity-Ranked Priority Queue:
                # 1. Existing positions & active limit orders are evaluated FIRST to guarantee instant SL/TP/trailing execution.
                # 2. Flat candidate symbols are sorted in descending order of Opportunity Score (Confidence × Expected Return),
                #    ensuring high-momentum breakouts on altcoins (e.g. NEAR, AVAX, LINK) get first access to open portfolio slots.
                # Phase 5: Dynamic Symbol Universe Sync from UniverseScreener
                now_u = time.time()
                if (now_u - self.last_universe_check_time) > 300.0 and self.redis.redis:
                    self.last_universe_check_time = now_u
                    try:
                        raw_u = await self.redis.redis.get("system:active_symbols")
                        if raw_u:
                            parsed_u = json.loads(raw_u)
                            if isinstance(parsed_u, list) and len(parsed_u) > 0:
                                valid_dyn = [s for s in parsed_u if s in self.symbols]
                                # Always retain any symbol with an active position or open order
                                for s, p in self.portfolio_state.positions.items():
                                    if abs(p.quantity) > 1e-5 and s not in valid_dyn:
                                        valid_dyn.append(s)
                                for s in self.active_orders:
                                    if s not in valid_dyn:
                                        valid_dyn.append(s)
                                if valid_dyn:
                                    self.dynamic_active_symbols = valid_dyn
                    except Exception as e:
                        logger.debug(f"Dynamic universe sync error: {e}")

                active_symbols = []
                flat_candidates = []
                # Book balance for ranking: with slots and the hourly entry cap scarce, a new trade that evens
                # out longs vs shorts is preferred (a balanced book survives market-wide moves). Never blocks.
                n_longs = sum(1 for p in self.portfolio_state.positions.values() if p.quantity > 1e-5)
                n_shorts = sum(1 for p in self.portfolio_state.positions.values() if p.quantity < -1e-5)
                rank_conf_floor = float((self.trade_policy or {}).get("min_confidence") or 0.70)
                
                for sym in self.symbols:
                    if sym not in self.market_states:
                        continue
                    pos = self.portfolio_state.positions.get(sym)
                    has_active_pos = pos is not None and abs(pos.quantity) > 1e-5
                    has_pending_order = sym in self.active_orders
                    
                    # Skip excluded coins (e.g. SUI with thin testnet orderbook) unless an active trade is open
                    if sym in getattr(settings, 'excluded_trading_symbols', []) and not has_active_pos and not has_pending_order:
                        continue
                    
                    if has_active_pos or has_pending_order:
                        active_symbols.append(sym)
                    else:
                        act_logits = precomputed_actions.get(sym)
                        crit_vals = precomputed_critics.get(sym)
                        score = 0.0
                        if act_logits is not None and len(act_logits) >= 2 and crit_vals is not None and len(crit_vals) >= 1:
                            raw_act = float(act_logits[0])
                            raw_conf = float(np.clip((act_logits[1] + 1.0) / 2.0, 0.05, 0.99))
                            exp_move = abs(exit_policy.expected_move_pct(crit_vals))
                            if abs(raw_act - self.actor_signal_offset) > 0.20 and raw_conf >= rank_conf_floor:
                                # Honest (holdout-calibrated) win probability x expected move
                                score = self._calibrated_confidence(raw_conf) * exp_move
                                is_long = (raw_act - self.actor_signal_offset) > 0
                                imbalance = n_longs - n_shorts
                                if (is_long and imbalance < 0) or (not is_long and imbalance > 0):
                                    score *= 1.25
                                elif (is_long and imbalance >= 2) or (not is_long and imbalance <= -2):
                                    score *= 0.75
                        flat_candidates.append((score, sym))
                
                # Sort flat symbols descending by opportunity score (highest score first)
                flat_candidates.sort(key=lambda x: x[0], reverse=True)
                sorted_eval_symbols = active_symbols + [sym for _, sym in flat_candidates]
                
                entries_this_cycle = 0  # new positions sent in this 5s cycle (capped by settings.max_entries_per_cycle)
                for sym in sorted_eval_symbols:

                        
                    # Check for pending Limit Order
                    # Check for pending Limit Order
                    if sym in self.active_orders:
                        # 1. ALWAYS VERIFY IF FILLED FIRST (Eliminates race conditions where a filled order is mistaken for a timeout!)
                        order_id_to_check = str(self.active_orders[sym]['order_id'])
                        is_still_open = True
                        try:
                            open_orders = await self.binance.get_open_orders(sym)
                            is_still_open = any(str(o.get('orderId')) == order_id_to_check or o.get('clientOrderId') == order_id_to_check for o in open_orders)
                        except Exception as e:
                            logger.warning(f"[{sym}] Error checking open orders: {e}")
                            
                        if not is_still_open:
                            logger.info(f"[{sym}] LIMIT ORDER FILLED/NO LONGER OPEN! Removing from tracker.")
                            
                            # Publish to dashboard history ONLY WHEN FILLED!
                            if 'trade_record' in self.active_orders[sym] and self.redis.redis:
                                self.active_orders[sym]['trade_record']['timestamp'] = time.time()
                                await self.redis.redis.lpush("dashboard:trade_history", json.dumps(self.active_orders[sym]['trade_record']))
                                await self.redis.redis.ltrim("dashboard:trade_history", 0, 99)
                                
                                # Send Telegram alert on confirmed LIMIT close fill
                                rec_side = self.active_orders[sym]['trade_record'].get('side', '')
                                if "CLOSE" in rec_side:
                                    tr = self.active_orders[sym]['trade_record']
                                    holding_mins = (time.time() - self.position_entry_times.get(sym, time.time())) / 60.0
                                    asyncio.create_task(
                                        telegram_notifier.broadcast_alert(
                                            "TRADE_EXIT",
                                            "",
                                            {
                                                "symbol": sym,
                                                "side": tr.get("side", ""),
                                                "price": tr.get("price", 0.0),
                                                "entry_price": tr.get("entry_price", 0.0),
                                                "realized_pnl": tr.get("realized_pnl", 0.0),
                                                "roi_pct": tr.get("roi_pct", 0.0),
                                                "fees": tr.get("fees_paid", 0.0),
                                                "reason": "Maker Limit Exit (Filled)",
                                                "holding_minutes": max(0.1, holding_mins)
                                            }
                                        )
                                    )
                                
                            # Publish experience to Redis ONLY WHEN CONFIRMED FILLED!
                            if 'experience_payload' in self.active_orders[sym]:
                                await redis_manager.publish_binary(
                                    "experience:completed",
                                    self.active_orders[sym]['experience_payload'],
                                )
                                
                            del self.active_orders[sym]
                            continue

                        # 2. ONLY IF STILL OPEN: Check for Timeout!
                        time_elapsed = time.time() - self.active_orders[sym].get('timestamp', time.time())
                        is_close_order = 'trade_record' in self.active_orders[sym] and "CLOSE" in self.active_orders[sym]['trade_record'].get('side', '')
                        timeout_limit = 15.0 if is_close_order else 30.0
                        
                        if time_elapsed > timeout_limit:
                            logger.info(f"[{sym}] POST-ONLY MAKER TIMEOUT (> {timeout_limit:.0f}s, is_close={is_close_order}). Cancelling unfilled order...")
                            try:
                                await self.binance.cancel_order(sym, self.active_orders[sym]['order_id'])
                            except Exception as e:
                                logger.warning(f"[{sym}] Failed to cancel (maybe already filled/cancelled): {e}")
                            
                            side = ""
                            qty = 0.0
                            if 'trade_record' in self.active_orders[sym]:
                                side = self.active_orders[sym]['trade_record']['side']
                                qty = float(self.active_orders[sym]['trade_record']['quantity'])
                                price = float(self.active_orders[sym]['trade_record']['price'])
                                margin_used = (qty * price) / settings.max_leverage
                                
                                # If an unfilled MAKER CLOSE timed out, fall back immediately to MARKET CLOSE to secure the exit!
                                # Sizes come from Binance: whatever the maker order already filled is not sold twice, and a
                                # 50% take-profit keeps its runner (entry time, best move and exit levels stay intact).
                                if "CLOSE" in side and qty > 0 and (self.trading_enabled and not self.dry_run):
                                    logger.info(f"[{sym}] ⏰ MAKER EXIT TIMEOUT. Executing immediate MARKET CLOSE fallback to secure exit!")
                                    close_binance_side = "BUY" if "SHORT" in side else "SELL"
                                    is_partial = side.startswith("PARTIAL_")
                                    try:
                                        live = await self.binance.get_positions()
                                        lp = next((x for x in live if x.get('symbol') == sym), None)
                                        live_abs = abs(float(lp.get('positionAmt', 0.0))) if lp else 0.0
                                        before_abs = float(self.active_orders[sym].get('pos_qty_before') or 0.0)
                                        maker_filled = min(qty, max(0.0, before_abs - live_abs)) if before_abs > 0 else 0.0
                                        rest = max(0.0, qty - maker_filled) if is_partial else live_abs
                                        sym_config = registry.get_symbol(sym)
                                        fmt_qty = float(sym_config.format_quantity(min(rest, live_abs))) if rest > 0 else 0.0
                                        fb_order = {}
                                        if fmt_qty > 0:
                                            fb_order = await self.binance.create_order(
                                                sym, close_binance_side, fmt_qty,
                                                f"ai_mkt_fb_{uuid.uuid4().hex[:8]}",
                                                order_type="MARKET", reduce_only=True
                                            )
                                        self.last_close_time[sym] = time.time()
                                        remaining_abs = max(0.0, live_abs - fmt_qty)
                                        if is_partial and remaining_abs > 0:
                                            self.portfolio_state.positions[sym].quantity = remaining_abs if "LONG" in side else -remaining_abs
                                            self.partial_tp_triggered[sym] = True
                                        else:
                                            self.portfolio_state.positions[sym].quantity = 0.0
                                            self.portfolio_state.positions[sym].entry_price = 0.0
                                            self.position_entry_times[sym] = 0.0
                                            self.position_exit_levels.pop(sym, None)

                                        # Record the close with the real fill prices
                                        if self.redis.redis and 'trade_record' in self.active_orders[sym] and (fmt_qty + maker_filled) > 0:
                                            base_rec = self.active_orders[sym]['trade_record']
                                            mkt_px = float(fb_order.get('avgPrice') or 0.0) if isinstance(fb_order, dict) else 0.0
                                            if mkt_px <= 0:
                                                m_fb = self.market_states.get(sym)
                                                mkt_px = float(m_fb.mid_price) if m_fb else price
                                            fb_rec = self._close_record(base_rec, float(base_rec.get('entry_price') or 0.0), [
                                                (maker_filled, price, settings.maker_fee_pct),
                                                (fmt_qty, mkt_px, settings.taker_fee_pct),
                                            ])
                                            fb_rec['order_id'] = fb_order.get('orderId', base_rec.get('order_id')) if isinstance(fb_order, dict) else base_rec.get('order_id')
                                            fb_rec['order_type'] = "MAKER + MARKET FALLBACK" if maker_filled > 0 else "MARKET (maker timeout)"
                                            await self.redis.redis.lpush("dashboard:trade_history", json.dumps(fb_rec))
                                            await self.redis.redis.ltrim("dashboard:trade_history", 0, 99)
                                    except Exception as fb_err:
                                        logger.warning(f"[{sym}] Fallback market exit error: {fb_err}")
                                else:
                                    # Revert optimistic portfolio update for OPEN order since it timed out and didn't fill
                                    if "OPEN_LONG" in side:
                                        self.portfolio_state.positions[sym].quantity -= qty
                                    elif "OPEN_SHORT" in side:
                                        self.portfolio_state.positions[sym].quantity += qty
                                        
                                    if abs(self.portfolio_state.positions[sym].quantity) < 1e-5:
                                        self.portfolio_state.positions[sym].quantity = 0.0
                                        self.portfolio_state.positions[sym].entry_price = 0.0
                                        self.position_entry_times[sym] = 0.0
                                        
                                    self.portfolio_state.free_margin += margin_used
                                    
                                    # Institutional Upgrade: Broadcast Order Timeout Alert to Telegram
                                    asyncio.create_task(
                                        telegram_notifier.broadcast_alert(
                                            "ORDER_TIMEOUT",
                                            "",
                                            {
                                                "symbol": sym,
                                                "timeout_seconds": timeout_limit,
                                                "reason": f"Post-only maker order at ${price:,.4f} was not filled within {timeout_limit:.0f}s. Cancelled to protect capital."
                                            }
                                        )
                                    )

                            # Partial-Fill Micro-Stub Protection:
                            # If an OPEN order timed out and only a tiny micro-stub (< 25% of intended size) filled,
                            # market-close the dust immediately to eliminate asymmetric underfunded risk!
                            if "OPEN" in side and qty > 0 and (self.trading_enabled and not self.dry_run):
                                try:
                                    active_positions = await self.binance.get_positions()
                                    matching = [p for p in active_positions if p.get('symbol') == sym]
                                    actual_pos = matching[0] if matching else None
                                    actual_qty = abs(float(actual_pos.get('positionAmt', 0.0))) if actual_pos else 0.0
                                    topped_up = False
                                    try:
                                        tr_open = self.active_orders[sym].get('trade_record', {})
                                        topped_up = await self._top_up_partial_entry(
                                            sym, side, qty, actual_qty, price, float(tr_open.get('predicted_move_pct') or 0.0))
                                    except Exception as tu_err:
                                        logger.warning(f"[{sym}] Entry top-up failed: {tu_err}")
                                    if topped_up:
                                        active_positions = await self.binance.get_positions()
                                        matching = [p for p in active_positions if p.get('symbol') == sym]
                                        actual_pos = matching[0] if matching else None
                                        actual_qty = abs(float(actual_pos.get('positionAmt', 0.0))) if actual_pos else 0.0
                                        self.active_orders[sym].get('trade_record', {})['order_type'] = "MAKER + MARKET TOP-UP"

                                    if actual_qty > 0:
                                        mkt_fb = self.market_states.get(sym)
                                        entry_px_val = float(actual_pos.get('entryPrice', 0.0)) or (mkt_fb.mid_price if mkt_fb else 0.0)
                                        notional_filled = actual_qty * entry_px_val
                                        if notional_filled >= 25.0:
                                            logger.info(f"[{sym}] Substantial partial fill confirmed (${notional_filled:.2f} notional >= $25.00). Letting position ride cleanly to preserve fees!")
                                            self.portfolio_state.positions[sym].quantity = actual_qty if "LONG" in side else -actual_qty
                                            self.portfolio_state.positions[sym].entry_price = entry_px_val
                                            self.position_entry_times[sym] = time.time()
                                            # The position is real: record the OPEN (dashboard + training experience).
                                            # Previously only the CLOSE was ever recorded for partially filled entries.
                                            if self.redis.redis and 'trade_record' in self.active_orders[sym]:
                                                pf_rec = dict(self.active_orders[sym]['trade_record'])
                                                pf_rec.update({"timestamp": time.time(), "quantity": actual_qty, "price": entry_px_val, "entry_price": entry_px_val})
                                                await self.redis.redis.lpush("dashboard:trade_history", json.dumps(pf_rec))
                                                await self.redis.redis.ltrim("dashboard:trade_history", 0, 99)
                                            if 'experience_payload' in self.active_orders[sym]:
                                                await redis_manager.publish_binary("experience:completed", self.active_orders[sym]['experience_payload'])
                                        else:
                                            logger.warning(f"[{sym}] 🧹 MICRO-STUB DETECTED: Tiny fill (${notional_filled:.2f} < $25.00 min notional). Market-closing dust stub immediately to eliminate asymmetric risk!")
                                            close_side = "SELL" if "LONG" in side else "BUY"
                                            sym_config = registry.get_symbol(sym)
                                            stub_qty_str = sym_config.format_quantity(actual_qty)
                                            if float(stub_qty_str) > 0:
                                                await self.binance.create_order(
                                                    symbol=sym,
                                                    side=close_side,
                                                    quantity=float(stub_qty_str),
                                                    client_order_id=f"ai_stub_{uuid.uuid4().hex[:10]}",
                                                    order_type="MARKET",
                                                    reduce_only=True
                                                )
                                            self.last_close_time[sym] = time.time() + 900.0  # 15m cooldown to prevent fee churn
                                            self.portfolio_state.positions[sym].quantity = 0.0
                                            self.portfolio_state.positions[sym].entry_price = 0.0
                                            self.position_entry_times[sym] = 0.0
                                            
                                            # Transparent Telemetry: Record in dashboard history so it never vanishes without a trace!
                                            if self.redis.redis:
                                                stub_record = {
                                                    "timestamp": time.time(),
                                                    "symbol": sym,
                                                    "side": f"DUST_CLEAN_{side}",
                                                    "quantity": float(stub_qty_str),
                                                    "price": entry_px_val,
                                                    "confidence": 0.0,
                                                    "order_id": f"ai_stub_{sym}",
                                                    "entry_price": entry_px_val,
                                                    "gross_pnl": 0.0,
                                                    "fees_paid": 0.0,
                                                    "realized_pnl": 0.0,
                                                    "roi_pct": 0.0
                                                }
                                                await self.redis.redis.lpush("dashboard:trade_history", json.dumps(stub_record))
                                                await self.redis.redis.ltrim("dashboard:trade_history", 0, 99)
                                except Exception as e:
                                    logger.warning(f"[{sym}] Could not verify micro-stub on Binance: {e}")

                            del self.active_orders[sym]
                            continue
                        else:
                            logger.debug(f"[{sym}] Limit order still pending. Skipping tick.")
                            continue
                        
                    # Phase 2: Use precomputed batched result if available
                    if sym in precomputed_actions and sym in precomputed_seqs:
                        # Use results from the batch pre-pass (no model call needed!)
                        state_vector, seq = precomputed_seqs[sym]
                        self.state_history[sym].append(state_vector.squeeze(0).tolist())
                        action_logits = precomputed_actions[sym]
                    else:
                        # Fallback: individual inference (for symbols skipped in pre-pass)
                        state_vector, macro_features = self._build_state_vector(sym)
                        self.state_history[sym].append(state_vector.squeeze(0).tolist())
                        seq = list(self.state_history[sym])
                        while len(seq) < self.sequence_length:
                            seq.insert(0, seq[0] if len(seq) > 0 else state_vector.squeeze(0).tolist())
                        state_tensor = torch.tensor([seq], dtype=torch.float32)
                        state_tensor = torch.nan_to_num(state_tensor, nan=0.0, posinf=1.0, neginf=-1.0)
                        with torch.no_grad():
                            action_logits_t, expected_return = self.model(self._mask_unseen_inputs(state_tensor), hidden_state=None, return_hidden=False)
                            action_logits = action_logits_t[0].numpy()
                            precomputed_critics[sym] = expected_return[0].numpy()

                    # Training record for this coin (every 10 min), before any trading gate can skip it
                    self._emit_market_snapshot(sym, seq, action_logits)

                    market = self.market_states[sym]
                    self.alpha_engine.update_tick(sym, market)
                    
                    # Update portfolio state with current price for RiskManager correlated exposure checks
                    if sym in self.portfolio_state.positions:
                        self.portfolio_state.positions[sym].current_price = market.mid_price
                    # Calculate Real-Time PNL Percentage, Adaptive Trailing Stop, Sudden Shock Detector, and Hard Stop Loss
                    hard_stop_triggered = False
                    hard_take_profit_triggered = False
                    sudden_shock_triggered = False
                    trailing_stop_triggered = False
                    stale_trade_triggered = False
                    partial_tp_triggered = False
                    freeroll_stop_triggered = False
                    pnl_pct_leveraged = 0.0
                    breadth_mult = 1.0
                    cvd_mult = 1.0
                    funding_mult = 1.0
                    dom_mult = 1.0
                    sq_mult = 1.0

                    if sym in self.portfolio_state.positions:
                        pos_check = self.portfolio_state.positions[sym]
                        if pos_check.quantity != 0 and pos_check.entry_price > 0:
                            now_t = time.time()
                            # PNL % = (Current - Entry) / Entry  (* -1 if short)
                            pnl_pct = (market.mid_price - pos_check.entry_price) / pos_check.entry_price
                            if pos_check.quantity < 0:
                                pnl_pct = -pnl_pct
                            
                            # Leverage amplifies PNL %
                            pnl_pct_leveraged = pnl_pct * pos_check.leverage
                            
                            # Record PnL time-series history for sudden shock / velocity reversal detection
                            if sym not in self.pnl_history:
                                self.pnl_history[sym] = collections.deque(maxlen=30)
                            self.pnl_history[sym].append((now_t, pnl_pct_leveraged))
                            
                            # Track Peak PnL for Dynamic Trailing
                            prev_peak = self.peak_pnl_pct.get(sym, 0.0)
                            if pnl_pct_leveraged > prev_peak:
                                self.peak_pnl_pct[sym] = pnl_pct_leveraged
                                prev_peak = pnl_pct_leveraged

                            is_in_freeroll = self.partial_tp_triggered.get(sym, False)

                            # Shared exit policy (core/risk/exit_policy.py), in PRICE moves so it is independent
                            # of leverage and identical to what the trainer scores. Stop <= 0.75% (never wider
                            # than before), first take-profit >= 1.5x the stop, fee-covering breakeven, and the
                            # runner trails one stop-distance behind its peak only after TP1 is banked.
                            levels = self._exit_levels_for(sym, market)
                            lev = float(pos_check.leverage) if pos_check.leverage else 1.0
                            peak_move = prev_peak / lev
                            entry_time = self.position_entry_times.get(sym, 0.0)
                            held_seconds = (now_t - entry_time) if entry_time > 0 else 0.0
                            decision = exit_policy.evaluate_exit(pnl_pct, peak_move, is_in_freeroll, held_seconds, levels)

                            if decision.reason == "HARD STOP":
                                logger.warning(f"[{sym}] 🛑 HARD STOP LOSS TRIGGERED: Price move {pnl_pct*100:+.2f}% <= -{levels.stop*100:.2f}% stop. Overriding AI to protect capital.")
                                hard_stop_triggered = True
                            elif decision.reason == "HARD TAKE-PROFIT":
                                logger.info(f"[{sym}] 🎯 HARD TAKE-PROFIT TRIGGERED: Price move {pnl_pct*100:+.2f}% >= +{levels.hard_tp*100:.2f}% (3x stop).")
                                hard_take_profit_triggered = True
                            elif decision.reason == "PARTIAL TP 50%":
                                logger.info(f"[{sym}] 💰 50% SCALE-OUT (TP1): Price move {pnl_pct*100:+.2f}% >= +{levels.tp1*100:.2f}% ({levels.tp1/levels.stop:.1f}x the {levels.stop*100:.2f}% stop). Banking half, runner protected at fee-covering breakeven.")
                                partial_tp_triggered = True
                            elif decision.reason == "FREE-ROLL BREAKEVEN":
                                logger.info(f"[{sym}] 🔒 FREE-ROLL BREAKEVEN HIT: Runner back to {pnl_pct*100:+.2f}% (<= +{levels.breakeven*100:.2f}% which covers round-trip fees). Closing remainder.")
                                freeroll_stop_triggered = True
                            elif decision.reason == "TRAILING STOP":
                                logger.info(f"[{sym}] 🔒 RUNNER TRAIL: Peak {peak_move*100:+.2f}%, now {pnl_pct*100:+.2f}% (trail gap {levels.trail_gap*100:.2f}%). Securing gain!")
                                trailing_stop_triggered = True
                            elif decision.reason == "STALE PRUNE":
                                logger.info(f"[{sym}] ♻️ STALE TRADE PRUNER: Held {held_seconds/3600.0:.1f}h with no progress ({pnl_pct*100:+.2f}%). Pruning to recycle capital!")
                                stale_trade_triggered = True
                            elif decision.reason == "MAX HOLD":
                                logger.info(f"[{sym}] ⏱️ MAX HOLD: Held {held_seconds/3600.0:.1f}h (limit {exit_policy.MAX_HOLD_SECONDS/3600:.0f}h, same as training). Closing at {pnl_pct*100:+.2f}%.")
                                stale_trade_triggered = True
                        else:
                            self.peak_pnl_pct[sym] = 0.0
                            if sym in self.pnl_history:
                                self.pnl_history[sym].clear()

                    exit_override = hard_stop_triggered or hard_take_profit_triggered or sudden_shock_triggered or trailing_stop_triggered or stale_trade_triggered or freeroll_stop_triggered or partial_tp_triggered
                    if exit_override:
                        # Force an immediate MARKET CLOSE
                        action_val = -1.0 if pos_check.quantity > 0 else 1.0
                        confidence = 1.0
                        target_size = 0.0
                        price_offset = 0.0 # Force immediate execution (0 offset)
                        self.smoothed_action_history[sym].clear()
                        self.smoothed_confidence_history[sym].clear()
                    else:
                        # Pure Deterministic Policy (Learned AI Weights - Zero Random Exploration Noise)
                        raw_action = float(action_logits[0])
                        # Unscale confidence from [-1.0, 1.0] (tanh policy output) to [0.05, 0.99]
                        raw_conf = float(np.clip((action_logits[1] + 1.0) / 2.0, 0.05, 0.99))
                        
                        # 1. Institutional Temporal Window Smoothing (3-tick moving average)
                        # Eliminates high-frequency neural jitter and noisy 1-tick oscillations
                        self.smoothed_action_history[sym].append(raw_action)
                        self.smoothed_confidence_history[sym].append(raw_conf)
                        raw_action_val = float(np.mean(self.smoothed_action_history[sym]))
                        confidence = float(np.mean(self.smoothed_confidence_history[sym]))
                        
                        critic_vals = precomputed_critics.get(sym)
                        pred_ret_5m = float(critic_vals[0]) if critic_vals is not None and len(critic_vals) > 0 else 0.0
                        pred_ret_1h = float(critic_vals[1]) if critic_vals is not None and len(critic_vals) > 1 else pred_ret_5m
                        pred_ret_4h = float(critic_vals[2]) if critic_vals is not None and len(critic_vals) > 2 else pred_ret_1h

                        # Actor-Critic Directional Fusion:
                        # 1. The Critic head forecasts real 5m / 1h / 4h price moves; their blend (exit_policy
                        #    HORIZON_WEIGHTS) is the expected move of a trade held under the 1-4h exit policy.
                        # 2. When that expected move is decisive (|move| >= 0.15%), it drives the thesis.
                        # 3. Otherwise the Actor policy modulates the state.
                        exp_move = exit_policy.expected_move_pct(critic_vals)
                        critic_signal = float(np.clip(exp_move / 1.5, -0.80, 0.80))
                        actor_signal = float(np.clip(raw_action_val - self.actor_signal_offset, -0.50, 0.50))
                        
                        if exp_move <= -0.15:
                            action_val = float(np.clip(critic_signal + min(0.0, actor_signal), -1.0, -0.25))
                        elif exp_move >= 0.15:
                            action_val = float(np.clip(critic_signal + max(0.0, actor_signal), 0.25, 1.0))
                        else:
                            action_val = float(np.clip(actor_signal + critic_signal, -1.0, 1.0))

                        # Dynamic Conviction-Scaled Position Sizing (Unlocks dynamic sizes beyond 25% floor):
                        # 1. Base model requested preference [0.0 to 1.0]
                        model_requested_size = (action_logits[2] + 1.0) / 2.0

                        # 2. Dynamic Conviction Multiplier:
                        # High confidence (> 0.65) and strong directional return (> 0.08%) scale up position allocation
                        conviction_mult = 1.0
                        if confidence > 0.65:
                            conviction_mult += (confidence - 0.65) * 1.5
                        
                        dir_expected_ret = -exp_move if action_val < -0.1 else exp_move
                        if dir_expected_ret > 0.08:
                            conviction_mult += min(0.50, dir_expected_ret * 2.0)

                        # 3. Dynamic target size bounded between 0.25 (minimum safety floor) and 1.0 (max budget)
                        effective_size = max(0.25, model_requested_size) * conviction_mult
                        target_size = float(np.clip(effective_size, 0.25, 1.0))
                        price_offset = float(np.clip((action_logits[3] + 1.0) / 2.0, 0.0, 1.0))
                    
                    pos = self.portfolio_state.positions[sym]
                    lead_lag_mult = 1.00
                    liq_mult = 1.00
                    
                    # Institutional Upgrade 4: Telegram Pause Guard
                    if not exit_override and pos.quantity == 0 and time.time() < self.pause_until:
                        rem_p = (self.pause_until - time.time()) / 60.0
                        logger.info(f"[{sym}] SKIPPING OPEN: Trading temporarily paused via Telegram ({rem_p:.1f}m remaining).")
                        continue

                    # (Removed: the liquidation "sniper" used to overwrite the AI action with a forced 0.85 LONG.
                    #  Liquidation volume is a model input (slots 101/102); the AI decides, the gates only protect.)

                    # 2. Institutional Hysteresis Deadband
                    # Entries from flat require |action_val| > 0.20
                    # Reversals/Exits against an established position require passing a wider barrier (|action_val| > 0.35)
                    # This prevents 1-tick micro-wobbles from closing positions early into chop!
                    if pos.quantity > 0:
                        is_closing = (action_val < -0.35) or exit_override
                    elif pos.quantity < 0:
                        is_closing = (action_val > 0.35) or exit_override
                    else:
                        is_closing = False
                    if is_closing and not exit_override and not self._history_warm(sym):
                        self._log_warmup(sym, "close")
                        is_closing = False

                    # Predicted side tracking for telemetry
                    if action_val < -0.2: predicted_side = "SHORT"
                    elif action_val > 0.2: predicted_side = "LONG"
                    else: predicted_side = "HOLD"
                    
                    if self.redis.redis:
                        ai_state = {
                            "predicted_side": predicted_side,
                            "confidence": float(confidence),
                            "confidence_calibrated": self._calibrated_confidence(float(confidence)),
                            "target_size": float(target_size),
                            "critic_return": float(pred_ret_5m),
                            "critic_1h": float(pred_ret_1h),
                            "critic_4h": float(pred_ret_4h),
                            "timestamp": time.time(),
                        }
                        await self.redis.redis.set(f"ai:state:{sym}", json.dumps(ai_state))

                    # GUARD 0.4: Dynamic Universe Check (Allow all curated universe symbols, skip explicitly excluded ones)
                    # Existing open positions and pending orders are NEVER abandoned and continue full risk/exit management!
                    if not is_closing and pos.quantity == 0 and sym in getattr(settings, "excluded_trading_symbols", []):
                        logger.info(f"[{sym}] SKIPPING OPEN: Symbol is in excluded_trading_symbols.")
                        continue

                    # GUARD 0.5: Circuit Breaker Check (5% Daily Drawdown Kill Switch)
                    if not is_closing and time.time() < self.circuit_breaker_active_until:
                        time_left_cb = (self.circuit_breaker_active_until - time.time()) / 3600.0
                        logger.warning(f"[{sym}] 🛑 CIRCUIT BREAKER ACTIVE: Halting new entries ({time_left_cb:.1f}h remaining). Preserving capital!")
                        continue

                    # GUARD 1: Per-coin cooldown after ANY close (no fast-pass). Re-opening a coin minutes after
                    # closing it paid a full round-trip fee for no new information (fees were 5x the trading loss).
                    if not is_closing and pos.quantity == 0:
                        time_since_close = time.time() - self.last_close_time.get(sym, 0.0)
                        if time_since_close < settings.entry_cooldown_seconds:
                            logger.debug(f"[{sym}] SKIPPING OPEN: cooldown after last close ({max(0.0, time_since_close):.0f}s / {settings.entry_cooldown_seconds:.0f}s).")
                            continue

                    # GUARD 1.2: Net Portfolio Beta & Correlation Exposure Governor
                    # Caps simultaneous unidirectional open positions to max 5 Longs or 5 Shorts
                    if not is_closing and pos.quantity == 0:
                        open_longs = [s for s, p in self.portfolio_state.positions.items() if p.quantity > 1e-5]
                        open_shorts = [s for s, p in self.portfolio_state.positions.items() if p.quantity < -1e-5]
                        
                        long_margin = sum((p.quantity * self.market_states[s].mid_price / p.leverage)
                                          for s, p in self.portfolio_state.positions.items()
                                          if p.quantity > 1e-5 and s in self.market_states and self.market_states[s].mid_price > 0)
                        short_margin = sum((abs(p.quantity) * self.market_states[s].mid_price / p.leverage)
                                           for s, p in self.portfolio_state.positions.items()
                                           if p.quantity < -1e-5 and s in self.market_states and self.market_states[s].mid_price > 0)
                        
                        net_exposure_ceiling = max(800.0, self.portfolio_state.equity * 0.50) # 50% of equity (min $800 safety buffer)
                        
                        if action_val > 0.20:
                            if len(open_longs) >= 5:
                                logger.info(f"[{sym}] 🛑 PORTFOLIO BETA GOVERNOR: Maximum concurrent Longs reached ({len(open_longs)}/5: {open_longs}). Blocking new Long to prevent correlated drawdown!")
                                continue
                            if (long_margin - short_margin) >= net_exposure_ceiling:
                                logger.info(f"[{sym}] 🛑 PORTFOLIO BETA GOVERNOR: Net Long margin exposure (${long_margin - short_margin:.2f} >= ${net_exposure_ceiling:.2f} limit). Blocking Long!")
                                continue
                        elif action_val < -0.20:
                            if len(open_shorts) >= 5:
                                logger.info(f"[{sym}] 🛑 PORTFOLIO BETA GOVERNOR: Maximum concurrent Shorts reached ({len(open_shorts)}/5: {open_shorts}). Blocking new Short to prevent correlated drawdown!")
                                continue
                            if (short_margin - long_margin) >= net_exposure_ceiling:
                                logger.info(f"[{sym}] 🛑 PORTFOLIO BETA GOVERNOR: Net Short margin exposure (${short_margin - long_margin:.2f} >= ${net_exposure_ceiling:.2f} limit). Blocking Short!")
                                continue

                    # GUARD 1.3: Institutional Penalty Box & Stop-Loss Lockout (Citadel / Jane Street Model)
                    # Blocks re-entering a direction that just suffered a stop-loss (Tier 1: 60m lockout, Tier 2: 4h quarantine)
                    if not is_closing and pos.quantity == 0:
                        intended_dir = "LONG" if action_val > 0.20 else ("SHORT" if action_val < -0.20 else None)
                        if intended_dir:
                            lockout_expiry = self.loss_lockout_until.get(sym, {}).get(intended_dir, 0.0)
                            if time.time() < lockout_expiry:
                                rem_mins = (lockout_expiry - time.time()) / 60.0
                                logger.info(f"[{sym}] 🛑 PENALTY BOX ACTIVE: {intended_dir} locked out for {rem_mins:.1f} more mins after recent stop-loss. Refusing to fight trend!")
                                continue

                    # GUARD 1.5: Spread Firewall (Protect against illiquid book slippage bleed)
                    if not is_closing and market and market.mid_price > 0:
                        spread_pct = market.spread / market.mid_price
                        if spread_pct > session_max_spread:
                            logger.warning(f"[{sym}] 🛑 SPREAD FIREWALL ({session_name}): Spread {spread_pct*100:.3f}% > {session_max_spread*100:.3f}% limit. Skipping entry to protect from slippage fee bleed!")
                            continue

                    # ENTRY GATES SIMPLIFIED (2026-10): the HTF/ADX regime gate, whale gates, the 12 alpha-engine
                    # "pillars", macro-trend / anti-countertrend / over-extension / lead-lag / OFI / statarb / funding /
                    # liquidation-waterfall gates were removed. Every one of those signals is already a model input,
                    # and hand rules vetoing the AI meant (a) it never learned from the trades it wanted and (b) results
                    # could not be attributed. Kept: kill switch / circuit breaker / pause, exposure caps, cooldowns and
                    # rate limit, spread firewall, AI confidence, expected-move and net-edge cost checks, size floors.

                    # GUARD 2: Minimum Hold Duration (900s / 15 mins with Massive Reversal Escape Door)
                    if is_closing and not exit_override:
                        # 1. Early exit if in solid impulse profit (>= +4.0% leveraged to cover fees)
                        is_profit_impulse = pnl_pct_leveraged >= 0.04
                        
                        # 2. Smart Trap & Massive Reversal Escape Door:
                        # If AI sees a massive multi-timeframe reversal, exit immediately as soon as possible!
                        # Triggers on:
                        # a) Critic multi-timeframe adverse breakdown (5m >= 0.40% AND 1h >= 0.60% adverse, or severe 5m shock >= 0.75% adverse)
                        # b) High-Conviction Actor thesis reversal (|action_val| >= 0.55 opposing position with confidence >= 0.75)
                        critic_vals = precomputed_critics.get(sym)
                        pred_ret_5m = float(critic_vals[0]) if critic_vals is not None and len(critic_vals) > 0 else 0.0
                        pred_ret_1h = float(critic_vals[1]) if critic_vals is not None and len(critic_vals) > 1 else pred_ret_5m
                        
                        critic_reversal = (pos.quantity > 0 and ((pred_ret_5m <= -0.40 and pred_ret_1h <= -0.60) or pred_ret_5m <= -0.75) and confidence >= 0.70) or \
                                          (pos.quantity < 0 and ((pred_ret_5m >= 0.40 and pred_ret_1h >= 0.60) or pred_ret_5m >= 0.75) and confidence >= 0.70)
                        
                        actor_reversal = (pos.quantity > 0 and action_val <= -0.55 and confidence >= 0.75) or \
                                         (pos.quantity < 0 and action_val >= 0.55 and confidence >= 0.75)
                        
                        is_massive_reversal = critic_reversal or actor_reversal
                        is_protective_loss = pnl_pct_leveraged <= -0.12  # Allows surviving wicks up to -12% leveraged (-0.60% price move)
                        
                        entry_t = self.position_entry_times.get(sym, 0.0)
                        time_held = (time.time() - entry_t) if entry_t > 0 else 999.0
                        
                        if is_massive_reversal:
                            trigger_src = "Critic adverse breakdown" if critic_reversal else "Actor high-conviction thesis flip"
                            logger.warning(f"[{sym}] 🚨 MASSIVE REVERSAL DETECTED ({trigger_src}): Exiting market as soon as possible (PnL: {pnl_pct_leveraged*100:+.2f}%, Held: {time_held:.1f}s)!")
                        elif time_held < 900.0 and not (is_profit_impulse or is_protective_loss):
                            logger.info(f"[{sym}] ⏳ MICRO-SWING BREATHING: Minimum hold (900s / 15m) not met yet ({time_held:.1f}s, PnL: {pnl_pct_leveraged*100:+.2f}%). Giving swing room to run!")
                            continue

                    # GUARD 3: Confidence Filter. A champion approved on the holdout ships the threshold above
                    # which its unseen-data trades made money after costs; legacy champions use session hurdles.
                    policy_conf = (self.trade_policy or {}).get("min_confidence")
                    if policy_conf is not None:
                        effective_conf_hurdle = float(policy_conf) + (0.05 if self.profit_ratchet_active else 0.0)
                    else:
                        effective_conf_hurdle = max(0.74 if self.profit_ratchet_active else session_conf_hurdle, session_conf_hurdle)
                    if not is_closing and confidence < effective_conf_hurdle: 
                        if policy_conf is not None:
                            reason = f"model holdout threshold, honest win rate {self._calibrated_confidence(confidence):.0%}"
                        else:
                            reason = "Profit Ratchet Preservation" if self.profit_ratchet_active and effective_conf_hurdle > session_conf_hurdle else session_name
                        logger.info(f"[{sym}] SKIPPING OPEN (Low Confidence: {confidence:.2f} < {effective_conf_hurdle:.2f} in {reason})")
                        
                        # Emit authentic live HOLD experience into DB when flat (throttled to once per 5 mins per coin to protect VPS disk)
                        # Populates LINK, ADA, AVAX, NEAR, LTC with genuine 200-dim market state sequences
                        now_hold_t = time.time()
                        if pos.quantity == 0 and self._history_warm(sym) and (now_hold_t - self.last_hold_emission_time.get(sym, 0.0)) >= settings.experience_snapshot_seconds:
                            self.last_hold_emission_time[sym] = now_hold_t
                            active_version = self.registry.get_current_version()
                            hold_exp_data = {
                                "experience_id": str(uuid.uuid4()),
                                "timestamp": int(time.time()),
                                "symbol": sym,
                                "market_state": seq, 
                                "macro_state": None, 
                                "portfolio_state": {
                                    "free_margin": float(self.portfolio_state.free_margin),
                                    "total_unrealized_pnl": float(self.portfolio_state.total_unrealized_pnl)
                                },
                                "derivatives_state": {
                                    "target_size": float(target_size),
                                    "price_offset": float(price_offset)
                                },
                                "position_before": 0.0,
                                "position_after": 0.0,
                                "entry_price": 0.0,
                                "exit_price": None,
                                "leverage": int(pos.leverage) if hasattr(pos, 'leverage') else settings.max_leverage,
                                "margin": 0.0,
                                "fees": 0.0,
                                "holding_duration": 0,
                                "action": "HOLD",
                                "action_probability": float(confidence),
                                "confidence": float(confidence),
                                "model_version": active_version,
                                "reward": 0.05, 
                                "realized_pnl": 0.0,
                                "unrealized_pnl": 0.0
                            }
                            asyncio.create_task(redis_manager.publish_binary("experience:completed", msgpack.packb(hold_exp_data, use_bin_type=True)))
                        continue 

                    if is_closing and not exit_override:
                        # PILLAR 2: Dynamic Profit Harvesting & Multi-Horizon Critic-Assisted Reversal Exit
                        critic_vals = precomputed_critics.get(sym)
                        pred_ret_5m = float(critic_vals[0]) if critic_vals is not None and len(critic_vals) > 0 else 0.0
                        pred_ret_1h = float(critic_vals[1]) if critic_vals is not None and len(critic_vals) > 1 else pred_ret_5m
                        pred_ret_4h = float(critic_vals[2]) if critic_vals is not None and len(critic_vals) > 2 else pred_ret_1h
                        
                        has_reversal_critic = (pos.quantity > 0 and (pred_ret_5m <= -0.15 or pred_ret_1h <= -0.25)) or \
                                              (pos.quantity < 0 and (pred_ret_5m >= 0.15 or pred_ret_1h >= 0.25))
                        
                        # Thresholds follow the shared exit policy (price moves): an AI close of a winner must be
                        # worth at least one stop-distance, otherwise wins stay smaller than losses.
                        cur_move = pnl_pct_leveraged / max(1.0, float(pos.leverage or 1))
                        close_levels = self._exit_levels_for(sym, market)

                        # 1. Critic-Assisted Exit: in profit beyond 2x costs and the Critic forecasts an adverse reversal
                        if cur_move >= 2.0 * exit_policy.ROUND_TRIP_COST and has_reversal_critic:
                            logger.info(f"[{sym}] 🎯 CRITIC-ASSISTED PROFIT EXIT: Move {cur_move*100:+.2f}% and Critic forecasts adverse reversal (5m: {pred_ret_5m:+.2f}%, 1h: {pred_ret_1h:+.2f}%). Securing profit!")
                        elif 0.0 <= cur_move < close_levels.stop:
                            # 2. Small profit: let it run toward TP1 instead of banking a win smaller than a loss
                            logger.info(f"[{sym}] SKIPPING PREMATURE PROFIT CLOSE: Move {cur_move*100:+.2f}% < {close_levels.stop*100:.2f}% (one stop-distance). Letting trend ride!")
                            continue
                        elif cur_move >= close_levels.stop:
                            # 3. Solid Profit (>= one stop-distance): Allow smooth profit taking (0.40 hurdle)
                            close_hurdle = 0.40
                            if confidence < close_hurdle:
                                logger.info(f"[{sym}] SKIPPING PROFIT CLOSE (Low Confidence: {confidence:.2f} < {close_hurdle:.2f}, PnL: +{pnl_pct_leveraged*100:.2f}%)")
                                continue
                        else:
                            # 4. Negative PnL (< 0.0% leveraged):
                            # Crypto wicks are normal (noise buffer allows breathing room up to -15% hard stop).
                            # Cut early ONLY if Critic confirms a REAL adverse trend breakdown (5m >= 0.50% or 1h >= 0.70% adverse move)
                            has_adverse_critic = (pos.quantity > 0 and (pred_ret_5m <= -0.50 or pred_ret_1h <= -0.70)) or \
                                                 (pos.quantity < 0 and (pred_ret_5m >= 0.50 or pred_ret_1h >= 0.70))
                            
                            if cur_move > -close_levels.stop and not has_adverse_critic:
                                logger.info(f"[{sym}] SKIPPING WICK EXIT: PnL is {pnl_pct_leveraged*100:+.2f}% (surviving wick noise buffer up to -15%, Critic return 5m: {pred_ret_5m:+.2f}%, 1h: {pred_ret_1h:+.2f}% is not severe). Giving swing room to run!")
                                continue
                            
                            # Deep loss cut: require >= 0.65 conviction to cut before hard stop
                            if confidence < 0.65:
                                logger.info(f"[{sym}] SKIPPING LOSS CLOSE (Conviction {confidence:.2f} < 0.65, PnL: {pnl_pct_leveraged*100:+.2f}%)")
                                continue

                    # GUARD 4: Critic Multi-Horizon Swing Return Check (5m, 1h, 4h)
                    critic_vals = precomputed_critics.get(sym)
                    if not is_closing and critic_vals is not None and len(critic_vals) >= 1:
                        pred_ret_5m = float(critic_vals[0])
                        pred_ret_1h = float(critic_vals[1]) if len(critic_vals) > 1 else pred_ret_5m
                        pred_ret_4h = float(critic_vals[2]) if len(critic_vals) > 2 else pred_ret_1h
                        
                        min_hurdle = max(session_critic_hurdle, settings.min_open_expected_return_pct,
                                     settings.min_edge_cost_multiple * exit_policy.ROUND_TRIP_COST * 100.0)  # >= 2x fees
                        
                        # 1. Expected-Move Entry Hurdle: the blended 1-4h forecast (the horizon the exits are built
                        #    for) must clear costs with margin. A 5m forecast almost never exceeds this hurdle.
                        exp_move = exit_policy.expected_move_pct(critic_vals)
                        if action_val > 0.2 and exp_move < min_hurdle:
                            logger.info(f"[{sym}] SKIPPING OPEN_LONG: Expected move {exp_move:+.2f}% < {min_hurdle:.2f}% hurdle.")
                            continue
                        elif action_val < -0.2 and exp_move > -min_hurdle:
                            logger.info(f"[{sym}] SKIPPING OPEN_SHORT: Expected move {exp_move:+.2f}% > -{min_hurdle:.2f}% hurdle.")
                            continue
                            
                        # 2. 1-Hour & 4-Hour Multi-Horizon Swing Alignment (Strict Anti-Suicide Rule)
                        # Strictly forbids entering counter-trend positions into strong macro swing momentum
                        if action_val > 0.2:
                            if pred_ret_1h < -0.20:
                                logger.info(f"[{sym}] 🛑 SKIPPING OPEN_LONG: 1h Critic ({pred_ret_1h:+.2f}%) opposes Long swing!")
                                continue
                            if pred_ret_4h < -0.30 and pred_ret_1h <= 0.00:
                                logger.info(f"[{sym}] 🛑 SKIPPING OPEN_LONG: 4h Critic ({pred_ret_4h:+.2f}%) is heavily bearish and 1h Critic ({pred_ret_1h:+.2f}%) has not confirmed reversal!")
                                continue
                        elif action_val < -0.2:
                            if pred_ret_1h > 0.20:
                                logger.info(f"[{sym}] 🛑 SKIPPING OPEN_SHORT: 1h Critic ({pred_ret_1h:+.2f}%) opposes Short swing!")
                                continue
                            if pred_ret_4h > 0.30 and pred_ret_1h >= 0.00:
                                logger.info(f"[{sym}] 🛑 SKIPPING OPEN_SHORT: 4h Critic ({pred_ret_4h:+.2f}%) is heavily bullish and 1h Critic ({pred_ret_1h:+.2f}%) has not confirmed reversal!")
                                continue

                    # GUARD 4.5: Entry rate limit (safety net against overtrading / runaway loops)
                    if not is_closing and pos.quantity == 0 and abs(action_val) > 0.20:
                        if self._entries_last_hour() >= settings.max_entries_per_hour:
                            logger.info(f"[{sym}] SKIPPING OPEN: hourly entry cap reached ({settings.max_entries_per_hour}/h).")
                            continue
                        if entries_this_cycle >= settings.max_entries_per_cycle:
                            logger.info(f"[{sym}] SKIPPING OPEN: {settings.max_entries_per_cycle} entries already sent this cycle; re-checking next cycle.")
                            continue

                    # GUARD 5: Institutional Dynamic Volatility & Conviction Margin Sizing (Kelly / ATR Scaling)
                    # Dynamically scales between 20.0% and 40.0% based on conviction, horizon alignment, and coin volatility.
                    # Scaled to 20.0% base margin in Profit Ratchet Capital Preservation Mode.
                    base_margin_pct = 0.20 if self.profit_ratchet_active else 0.35
                    
                    # 1. Conviction Multiplier (0.80x to 1.30x scaled to session hurdle)
                    conviction_mult = float(np.clip(0.80 + (confidence - effective_conf_hurdle) * 1.5, 0.80, 1.30))
                    
                    # 2. Multi-Horizon Critic Alignment Multiplier (0.85x to 1.15x)
                    critic_vals = precomputed_critics.get(sym)
                    if critic_vals is not None and len(critic_vals) >= 2:
                        c5m = float(critic_vals[0])
                        c1h = float(critic_vals[1])
                        # If 5m and 1h agree with trade direction, scale up; if divergent, scale down
                        if (action_val > 0.2 and c5m > 0 and c1h > 0) or (action_val < -0.2 and c5m < 0 and c1h < 0):
                            critic_mult = 1.15
                        elif (action_val > 0.2 and c1h < 0) or (action_val < -0.2 and c1h > 0):
                            critic_mult = 0.85
                        else:
                            critic_mult = 1.00
                    else:
                        critic_mult = 1.00
                        
                    # 3. Asset Beta / Volatility Dampener
                    # Large caps (BTC, ETH) can handle higher margin; high-beta altcoins get dampener to prevent big fee bleed
                    if sym in ["BTCUSDT", "ETHUSDT"]:
                        beta_dampener = 1.10
                    elif sym in ["SOLUSDT", "BNBUSDT", "LTCUSDT"]:
                        beta_dampener = 1.00
                    elif sym in ["1000PEPEUSDT", "WIFUSDT", "DOGEUSDT"]:
                        beta_dampener = 0.80 # High-beta meme coins: conservative margin dampener to stay extra safe
                    else: # ADA, XRP, NEAR, AVAX, LINK, SUI, RENDER, APT, TAO, INJ (wild altcoins)
                        beta_dampener = 0.85
                        
                    # 4. Volatility Scaling from Market Observation
                    vol_scale = 1.0
                    if market and hasattr(market, 'volatility') and market.volatility > 0.015:
                        vol_scale = float(np.clip(0.015 / market.volatility, 0.70, 1.00))
                        
                    # 5. Whale Flow Alignment Multiplier (Asset-Tiered)
                    whale_mult = 1.0
                    whale_data = self.whale_states.get(sym)
                    is_major = sym in ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
                    whale_boost_hurdle = 250_000.0 if is_major else 120_000.0
                    if whale_data and (time.time() - float(whale_data.get("timestamp", 0.0))) < 60.0:
                        w_b = float(whale_data.get("whale_buy_pressure", 0.0))
                        w_s = float(whale_data.get("whale_sell_pressure", 0.0))
                        if action_val > 0.20 and w_b >= whale_boost_hurdle and w_b > (1.5 * w_s):
                            whale_mult = 1.05
                        elif action_val < -0.20 and w_s >= whale_boost_hurdle and w_s > (1.5 * w_b):
                            whale_mult = 1.05
                            
                    # 6. Institutional Alpha Multipliers
                    lead_lag_mult = 1.0
                    liq_mult = 1.0
                    alpha_health_mult = float(np.clip(breadth_mult * cvd_mult * funding_mult * dom_mult * sq_mult, 0.65, 1.35))
                    
                    if is_major:
                        # Major Coins (BTC, ETH, SOL, BNB): 25% - 40% margin
                        min_margin_bound = 0.15 if self.profit_ratchet_active else settings.major_min_margin_pct
                        max_margin_bound = 0.25 if self.profit_ratchet_active else settings.major_max_margin_pct
                    else:
                        # Altcoins (ADA, XRP, LINK, AVAX, NEAR, LTC, TAO, etc.): 20% - 35% margin
                        min_margin_bound = 0.12 if self.profit_ratchet_active else settings.alt_min_margin_pct
                        max_margin_bound = 0.22 if self.profit_ratchet_active else settings.alt_max_margin_pct
                    dynamic_margin_pct = float(np.clip(base_margin_pct * conviction_mult * critic_mult * beta_dampener * vol_scale * whale_mult * lead_lag_mult * liq_mult * alpha_health_mult, min_margin_bound, max_margin_bound))
                    
                    # System 4: Dynamic High-Water Equity Auto-Compounder (Asymmetric Kelly Scaling)
                    # Tracks peak session equity (HWM). When at new peaks, accelerates compounding (+5%).
                    # When in drawdown (D), dampens risk (1.0 - 2.5*D) to protect principal from compounding downward.
                    current_eq = float(self.portfolio_state.equity)
                    if current_eq > self.max_equity:
                        self.max_equity = current_eq

                    hwm_equity = self.max_equity
                    if hwm_equity > 0:
                        drawdown_from_hwm = max(0.0, (hwm_equity - current_eq) / hwm_equity)
                        if drawdown_from_hwm < 0.01:
                            # At or near all-time high: Expansion Regime (compounding accelerator)
                            kelly_compound_mult = 1.05
                        else:
                            # In drawdown: Asymmetrical capital preservation dampener
                            kelly_compound_mult = float(np.clip(1.0 - (2.5 * drawdown_from_hwm), 0.70, 1.00))
                    else:
                        kelly_compound_mult = 1.00

                    # System 4.5: Institutional Capital De-Escalation (Inverse Sizing on Staggered Losses)
                    # When an asset struggles, de-escalate capital allocation to prevent compounding drawdown:
                    # 0 previous losses: 100% normal size (~$600 margin)
                    # 1 previous loss: 50% margin cut (~$300 margin)
                    # 2+ previous losses: 25% margin cut (~$150 margin)
                    sym_losses = self.consecutive_losses.get(sym, 0)
                    if sym_losses == 1:
                        de_escalation_mult = 0.50
                        logger.info(f"[{sym}] 📉 CAPITAL DE-ESCALATION: 1 previous loss on symbol. Scaling margin down to 50% ($300 target).")
                    elif sym_losses >= 2:
                        de_escalation_mult = 0.25
                        logger.info(f"[{sym}] 📉 CAPITAL DE-ESCALATION: {sym_losses} previous losses on symbol. Scaling margin down to 25% ($150 target).")
                    else:
                        de_escalation_mult = 1.00

                    max_margin_budget = max(0.0, current_eq * dynamic_margin_pct * kelly_compound_mult * de_escalation_mult)
                    # Margin reserve: new entries never use the last entry_margin_reserve_pct of equity, so fills, fees
                    # and adverse moves cannot push free margin under Airbag 4's 15% line (forced market trims).
                    entry_margin_room = self.portfolio_state.free_margin - settings.entry_margin_reserve_pct * current_eq
                    margin_allocated = min(max(0.0, entry_margin_room), max_margin_budget * max(0.70, target_size))
                    notional_requested = margin_allocated * pos.leverage
                    
                    # Side determination respecting Hysteresis Deadband (Strict One-Trade-One-Entry: No Add-ons)
                    side = "HOLD"
                    if pos.quantity > 0:
                        if is_closing:
                            side = "CLOSE_LONG"
                        # When already in a Long, hold the position. Never generate OPEN_LONG to add contracts!
                    elif pos.quantity < 0:
                        if is_closing:
                            side = "CLOSE_SHORT"
                        # When already in a Short, hold the position. Never generate OPEN_SHORT to add contracts!
                    else:
                        if sym in getattr(settings, 'excluded_trading_symbols', []):
                            side = "HOLD"
                        elif sym in ((self.trade_policy or {}).get("blocked_symbols") or []) and abs(action_val) > 0.20:
                            # Per-coin filter: this champion lost money on this coin in its unseen-data test.
                            # Still HOLD (not skip) so the coin keeps producing training records.
                            side = "HOLD"
                            logger.info(f"[{sym}] SKIPPING OPEN: coin blocked by model per-coin filter (lost money on holdout).")
                        elif abs(action_val) > 0.20 and not self._history_warm(sym):
                            self._log_warmup(sym, "entry")
                            side = "HOLD"
                        elif action_val < -0.20:
                            side = "OPEN_SHORT"
                        elif action_val > 0.20:
                            side = "OPEN_LONG"
                    
                    if side == "HOLD":
                        logger.info(f"[{sym}] HOLDING (Action Val: {action_val:.2f})")
                        
                        # High-Quality HOLD Experiences (throttled to once per 5 mins per coin to protect VPS disk)
                        now_hold_t = time.time()
                        if self._history_warm(sym) and (now_hold_t - self.last_hold_emission_time.get(sym, 0.0)) >= settings.experience_snapshot_seconds:
                            self.last_hold_emission_time[sym] = now_hold_t
                            hold_dur = 0
                            if pos.entry_price > 0 and getattr(pos, 'entry_time', 0.0) > 0:
                                hold_dur = int(time.time() - pos.entry_time)
                            
                            # Reward patience: If flat, +0.05 reward for preserving capital and avoiding fee bleed
                            unreal_pnl = float(pos.unrealized_pnl) if hasattr(pos, 'unrealized_pnl') else 0.0
                            hold_reward = 0.05 if pos.quantity == 0 else max(-1.0, min(1.0, unreal_pnl * 0.1))
                            active_version = self.registry.get_current_version()
                                
                            exp_data = {
                                "experience_id": str(uuid.uuid4()),
                                "timestamp": int(time.time()),
                                "symbol": sym,
                                "market_state": seq, 
                                "macro_state": None, 
                                "portfolio_state": {
                                    "free_margin": float(self.portfolio_state.free_margin),
                                    "total_unrealized_pnl": float(self.portfolio_state.total_unrealized_pnl)
                                },
                                "derivatives_state": {
                                    "target_size": float(target_size),
                                    "price_offset": float(price_offset)
                                },
                                "position_before": float(pos.quantity),
                                "position_after": float(pos.quantity),
                                "entry_price": float(pos.entry_price),
                                "exit_price": None,
                                "leverage": int(pos.leverage) if hasattr(pos, 'leverage') else settings.max_leverage,
                                "margin": (float(pos.quantity) * market.mid_price / pos.leverage) if pos.leverage and pos.quantity else 0.0,
                                "fees": 0.0,
                                "holding_duration": hold_dur,
                                "action": "HOLD",
                                "action_probability": float(confidence),
                                "confidence": float(confidence),
                                "model_version": active_version,
                                "reward": float(hold_reward), 
                                "realized_pnl": 0.0,
                                "unrealized_pnl": float(pos.unrealized_pnl) if hasattr(pos, 'unrealized_pnl') else 0.0
                            }
                            asyncio.create_task(redis_manager.publish_binary("experience:completed", msgpack.packb(exp_data, use_bin_type=True)))
                    
                    if side != "HOLD":
                        active_version = self.registry.get_current_version()
                        target_qty = notional_requested / market.mid_price # Use EXPLICIT mid_price, no feature[6] hack
                        
                        # MICRO-SWING CONSOLIDATED ENTRY: Strict One-Trade-One-Entry Rule
                        if "OPEN" in side:
                            if abs(pos.quantity) > 0:
                                # STRICT ONE-TRADE-ONE-ENTRY POLICY: Zero multi-open add-ons permitted.
                                # Averaging into an existing position dilutes entries, corrupts hold timers,
                                # and has historically caused repeated fee bleed. Let active trades ride cleanly!
                                logger.info(f"[{sym}] 🔒 ONE-TRADE-ONE-ENTRY: Position already active ({abs(pos.quantity):.4f}). Strictly forbidding add-ons!")
                                continue
                            
                            # Clean One-Shot Entry (Dynamically scaled to account size)
                            # Minimum viable size is at least 60% of target budget (min $25 notional for small live accounts)
                            min_viable_notional = max(25.0, current_eq * min_margin_bound * pos.leverage * 0.60)
                            if notional_requested < min_viable_notional:
                                logger.info(f"[{sym}] Skipping initial entry: notional ${notional_requested:.2f} < ${min_viable_notional:.2f} minimum viable size. Waiting for full margin to ensure clean one-shot entry!")
                                continue
                            qty_raw = target_qty
                                
                            # Filter micro-dust orders under minimum symbol notional
                            dust_floor = max(10.0, min(50.0, current_eq * 0.05 * pos.leverage))
                            if (qty_raw * market.mid_price) < dust_floor:
                                logger.info(f"[{sym}] Skipping micro-order: notional ${qty_raw * market.mid_price:.2f} < ${dust_floor:.2f} minimum.")
                                continue
                                
                            # Phase 2: Pre-Trade Net-Edge Estimator (Ensures positive expectancy after fees, spread, and slippage)
                            edge_move_pred = exit_policy.expected_move_pct(precomputed_critics[sym]) if (sym in precomputed_critics and len(precomputed_critics[sym]) > 0) else (0.25 if "LONG" in side else -0.25)
                            active_regime = "NEUTRAL"
                            if sym in self.htf_regimes:
                                active_regime = self.htf_regimes[sym].get("trend", "NEUTRAL")
                            edge_result = self.edge_estimator.estimate(
                                symbol=sym,
                                side=side,
                                predicted_move_pct=edge_move_pred,
                                confidence=float(confidence),
                                market=market,
                                is_maker=True,
                                order_size_notional=qty_raw * market.mid_price,
                                regime=active_regime
                            )
                            if not edge_result.approved:
                                logger.info(f"[{sym}] 🛑 NET-EDGE ESTIMATOR: Skipping {side}. {edge_result.reason_code} | Est Costs: {edge_result.estimated_costs_pct:.3f}%")
                                continue
                        else:
                            # For CLOSE actions, we close the entire existing quantity, UNLESS it's a 50% partial TP!
                            if partial_tp_triggered:
                                sym_config = registry.get_symbol(sym)
                                partial_qty = abs(pos.quantity) * 0.50
                                if (partial_qty * market.mid_price) >= sym_config.min_notional:
                                    qty_raw = partial_qty
                                else:
                                    logger.info(f"[{sym}] 50% partial quantity notional (${partial_qty * market.mid_price:.2f}) < Min Notional (${sym_config.min_notional:.2f}). Letting full position ride!")
                                    partial_tp_triggered = False
                                    qty_raw = 0.0
                            else:
                                qty_raw = abs(pos.quantity)
                            
                        if qty_raw <= 0.0001:
                            logger.info(f"[{sym}] Target quantity reached. No additional {side} needed.")
                            continue
                        
                        request = OrderRequest(
                            symbol=sym, action_type=side, confidence=float(confidence),
                            requested_quantity=float(qty_raw), target_position=float(target_size),
                            model_version=active_version, timestamp=time.time()
                        )
                        
                        # Evaluate Risk via Schema
                        decision = self.risk_manager.evaluate(request, self.portfolio_state, market, self.macro_state)
                        
                        if decision.approved and decision.adjusted_quantity > 0:
                            sym_config = registry.get_symbol(sym)
                            qty_str = sym_config.format_quantity(decision.adjusted_quantity)
                            
                            if float(qty_str) <= 0:
                                logger.info(f"[{sym}] Formatted quantity is zero. Skipping execution.")
                                continue
                                
                            notional_val = float(qty_str) * market.mid_price
                            if notional_val < sym_config.min_notional and "OPEN" in side:
                                logger.info(f"[{sym}] Notional {notional_val:.2f} < Min Notional {sym_config.min_notional}. Skipping.")
                                continue
                                
                            logger.info(f"[{sym}] EXECUTING: {side} {qty_str} (Confidence: {confidence:.2f})")
                            
                            if (self.trading_enabled and 
                                not self.emergency_stop and
                                not settings.emergency_stop and
                                not self.dry_run and 
                                (settings.allow_testnet or settings.allow_live_trading) and 
                                self.has_valid_model and 
                                decision.approved):
                                
                                
                                # Phase 9: Execution and Experience Broadcasting
                                try:
                                    # Set leverage before executing
                                    try:
                                        await self.binance.set_leverage(sym, settings.max_leverage)
                                    except Exception as e:
                                        logger.warning(f"[{sym}] Could not set leverage (might already be set): {e}")

                                    # Calculate LIMIT order price using offset
                                    # Price Offset [0, 1] mapped to spread (e.g. 0 to 10 ticks away)
                                    tick_size = sym_config.tick_size
                                    offset_amount = (price_offset * 10) * tick_size
                                    
                                    binance_side = "BUY" if side in ["OPEN_LONG", "CLOSE_SHORT"] else "SELL"
                                    
                                    # System 2: Volume-Weighted Micro-Price (P_micro) Execution Pegger (Jane Street Model)
                                    # Calculates queue pressure: P_micro = I * P_ask + (1 - I) * P_bid
                                    # Ensures Post-Only Maker queue priority and protects against adverse selection fills.
                                    ob_skew = float(market.order_book_imbalance) if hasattr(market, 'order_book_imbalance') and market.order_book_imbalance is not None else 0.0
                                    imb_weight = float(np.clip((ob_skew + 1.0) / 2.0, 0.0, 1.0))
                                    
                                    bid_px = market.bid if market.bid > 0 else (market.mid_price - tick_size)
                                    ask_px = market.ask if market.ask > 0 else (market.mid_price + tick_size)
                                    p_micro = (imb_weight * ask_px) + ((1.0 - imb_weight) * bid_px)
                                    
                                    if binance_side == "BUY":
                                        # If sellers dominate and press down (p_micro < mid): peg 1 tick deeper to avoid toxic fill!
                                        if p_micro < (market.mid_price - (0.5 * tick_size)):
                                            limit_price = bid_px - tick_size
                                            logger.debug(f"[{sym}] 🏛️ P_MICRO PEG: Sellers pressing (P_micro ${p_micro:.4f} < Mid ${market.mid_price:.4f}). Shifting bid 1 tick deeper.")
                                        else:
                                            limit_price = bid_px
                                    else:
                                        # If buyers dominate and push up (p_micro > mid): peg 1 tick higher to harvest the wick!
                                        if p_micro > (market.mid_price + (0.5 * tick_size)):
                                            limit_price = ask_px + tick_size
                                            logger.debug(f"[{sym}] 🏛️ P_MICRO PEG: Buyers pushing (P_micro ${p_micro:.4f} > Mid ${market.mid_price:.4f}). Shifting ask 1 tick higher.")
                                        else:
                                            limit_price = ask_px
                                        
                                    limit_price_str = sym_config.format_price(limit_price)
                                    
                                    # Pre-capture entry price BEFORE sending the order to avoid race conditions with market fill
                                    entry_px = self.portfolio_state.positions[sym].entry_price if sym in self.portfolio_state.positions else 0.0
                                    pos_leverage = settings.max_leverage  # set on Binance just above, before every order
                                    
                                    # Always fetch authoritative Binance live position entry price before sending close order
                                    if "CLOSE" in side and self.trading_enabled and not self.dry_run:
                                        try:
                                            live_pos_check = await self.binance.get_positions()
                                            for lp in live_pos_check:
                                                if lp.get("symbol") == sym and float(lp.get("positionAmt", 0)) != 0:
                                                    binance_ep = float(lp.get("entryPrice", 0.0))
                                                    if binance_ep > 0:
                                                        entry_px = binance_ep
                                                        self.portfolio_state.positions[sym].entry_price = binance_ep
                                                        break
                                        except Exception as e:
                                            logger.warning(f"[{sym}] Could not pre-fetch entry price from Binance: {e}")

                                    # Execute on Binance (Hybrid Maker/Market Engine)
                                    client_order_id = f"ai_bot_{uuid.uuid4().hex[:10]}"
                                    is_closing = "CLOSE" in side
                                    
                                    # Emergency exits must ALWAYS execute via immediate MARKET order for 100% execution certainty
                                    is_emergency_exit = (
                                        hard_stop_triggered or 
                                        sudden_shock_triggered or 
                                        (is_closing and pnl_pct_leveraged <= -0.10)
                                    )
                                    
                                    # Planned profitable exits (Take-Profit, Trailing Stop, Free-Roll, or profitable AI Close >= +1.5%)
                                    # are routed through Post-Only Maker GTX to capture the ultra-low 0.02% Maker fee (saves ~43% on fees!)
                                    is_maker_exit = is_closing and not is_emergency_exit and (
                                        pnl_pct_leveraged >= 0.015 or 
                                        partial_tp_triggered or 
                                        hard_take_profit_triggered or 
                                        trailing_stop_triggered
                                    )
                                    
                                    is_market_exit = is_emergency_exit or (is_closing and not is_maker_exit) or (exit_override and not is_maker_exit)
                                    
                                    if is_market_exit:
                                        exit_reason = "HARD STOP" if hard_stop_triggered else ("HARD TAKE-PROFIT" if hard_take_profit_triggered else ("SUDDEN SHOCK" if sudden_shock_triggered else ("TRAILING STOP" if trailing_stop_triggered else ("STALE PRUNE" if stale_trade_triggered else ("FREE-ROLL BREAKEVEN" if freeroll_stop_triggered else ("PARTIAL TP 50%" if partial_tp_triggered else "AI CLOSE"))))))
                                        logger.info(f"[{sym}] MARKET INTENT ({exit_reason}): {binance_side} {qty_str}")
                                        try:
                                              await self.binance.cancel_all_orders(sym)
                                        except Exception as e:
                                              logger.warning(f"[{sym}] Failed to cancel lingering orders: {e}")
                                        order_res = await self.binance.create_order(sym, binance_side, float(qty_str), client_order_id, order_type="MARKET", reduce_only=True)
                                        self.active_orders.pop(sym, None)
                                        self.last_close_time[sym] = time.time()
                                        if not partial_tp_triggered:
                                            self.peak_pnl_pct[sym] = 0.0
                                            if sym in self.pnl_history:
                                                self.pnl_history[sym].clear()
                                    else:
                                        # Institutional Post-Only Maker Execution (GTX)
                                        # Forces Maker fill (0.02% fee) and strictly forbids taker fill (0.05% fee)
                                        exit_tag = " [PROFIT MAKER EXIT - 0.02% FEE]" if is_closing else ""
                                        logger.info(f"[{sym}] 🏛️ POST-ONLY MAKER INTENT (GTX){exit_tag}: {binance_side} at {limit_price_str} (reduce_only={is_closing})")
                                        try:
                                            order_res = await self.binance.create_order(sym, binance_side, float(qty_str), client_order_id, order_type="LIMIT", price=limit_price_str, time_in_force="GTX", reduce_only=is_closing)
                                        except Exception as e:
                                            # If post-only was rejected (-5022: would match and take liquidity)
                                            if "-5022" in str(e) or "Post Only" in str(e):
                                                if is_closing:
                                                    # For profit close, if limit crosses, execute immediate market close to capture the banked gain!
                                                    logger.info(f"[{sym}] Maker profit exit would cross book. Executing immediate MARKET exit to bank gain!")
                                                    order_res = await self.binance.create_order(sym, binance_side, float(qty_str), client_order_id, order_type="MARKET", reduce_only=True)
                                                    is_market_exit = True
                                                else:
                                                    logger.info(f"[{sym}] Post-Only GTX would cross book. Re-pegging 1 tick deeper...")
                                                    if binance_side == "BUY":
                                                        limit_price -= tick_size
                                                    else:
                                                        limit_price += tick_size
                                                    limit_price_str = sym_config.format_price(limit_price)
                                                    order_res = await self.binance.create_order(sym, binance_side, float(qty_str), client_order_id, order_type="LIMIT", price=limit_price_str, time_in_force="GTX", reduce_only=False)
                                            else:
                                                raise e
                                    
                                    actual_order_id = order_res.get('orderId') or client_order_id
                                    logger.info(f"[{sym}] ORDER SUCCESS: {actual_order_id}")
                                    
                                    avg_fill = float(order_res.get('avgPrice', 0.0))
                                    exec_qty = float(order_res.get('executedQty', 0.0))
                                    cum_quote = float(order_res.get('cumQuote', 0.0))
                                    if avg_fill <= 0.0 and exec_qty > 0.0 and cum_quote > 0.0:
                                        avg_fill = cum_quote / exec_qty
                                    order_exec_price = avg_fill if avg_fill > 0 else (float(limit_price_str) if not is_market_exit else market.mid_price)

                                    # Passive Airbag 3: Real-Time Execution Slippage Anomaly Guard
                                    if market.mid_price > 0 and avg_fill > 0:
                                        slippage_pct = abs(avg_fill - market.mid_price) / market.mid_price
                                        if slippage_pct > 0.0075:  # > 0.75% abnormal slippage (liquidity hole)
                                            quarantine_secs = 900.0  # 15 minutes
                                            self.loss_lockout_until[sym]["LONG"] = max(self.loss_lockout_until[sym].get("LONG", 0.0), time.time() + quarantine_secs)
                                            self.loss_lockout_until[sym]["SHORT"] = max(self.loss_lockout_until[sym].get("SHORT", 0.0), time.time() + quarantine_secs)
                                            logger.critical(
                                                f"[{sym}] ⚠️ AIRBAG 3 ACTIVATED: SLIPPAGE ANOMALY GUARD! "
                                                f"Fill Price ${avg_fill:,.4f} deviated {slippage_pct*100:.2f}% from Mid ${market.mid_price:,.4f} (> 0.75% threshold)! "
                                                f"Quarantining {sym} for 15 minutes to allow orderbook depth to normalize."
                                            )
                                            asyncio.create_task(
                                                telegram_notifier.broadcast_alert(
                                                    "SLIPPAGE_ANOMALY",
                                                    sym,
                                                    {
                                                        "symbol": sym,
                                                        "slippage_pct": slippage_pct * 100,
                                                        "mid_price": market.mid_price,
                                                        "fill_price": avg_fill,
                                                        "quarantine_minutes": 15
                                                    }
                                                )
                                            )

                                    # Prepare Trade History Record
                                    record_side = f"PARTIAL_{side}" if partial_tp_triggered else side
                                    trade_record = {
                                        "timestamp": time.time(),
                                        "symbol": sym,
                                        "side": record_side,
                                        "quantity": float(qty_str),
                                        "price": order_exec_price,
                                        "confidence": float(confidence),
                                        "order_id": actual_order_id,
                                        "entry_price": entry_px if ("CLOSE" in side and entry_px > 0) else order_exec_price,
                                        "gross_pnl": 0.0,
                                        "fees_paid": 0.0,
                                        "realized_pnl": 0.0,
                                        "roi_pct": 0.0
                                    }
                                    # Extra detail for the dashboard's trade history
                                    trade_record.update({
                                        "leverage": int(pos_leverage or 0),
                                        "notional": float(qty_str) * order_exec_price,
                                        "order_type": "MARKET" if is_market_exit else "MAKER LIMIT",
                                        "mid_price": float(market.mid_price or 0.0),
                                        "regime": (self.htf_regimes.get(sym) or {}).get("trend"),
                                    })
                                    if "OPEN" in side:
                                        trade_record.update({
                                            "confidence_calibrated": self._calibrated_confidence(float(confidence)),
                                            "predicted_move_pct": float(edge_move_pred),
                                            "est_cost_pct": float(edge_result.estimated_costs_pct),
                                        })
                                    else:
                                        trade_record["exit_reason"] = (
                                            "HARD STOP" if hard_stop_triggered else "HARD TAKE-PROFIT" if hard_take_profit_triggered
                                            else "SUDDEN SHOCK" if sudden_shock_triggered else "TRAILING STOP" if trailing_stop_triggered
                                            else "MAX HOLD / STALE" if stale_trade_triggered else "FREE-ROLL BREAKEVEN" if freeroll_stop_triggered
                                            else "PARTIAL TP 50%" if partial_tp_triggered else "AI CLOSE")
                                        entry_t = self.position_entry_times.get(sym) or getattr(self.portfolio_state.positions.get(sym), "entry_time", 0.0)
                                        if entry_t:
                                            trade_record["hold_seconds"] = max(0.0, time.time() - entry_t)

                                    # Add Analytics for Closures
                                    if "CLOSE" in side and entry_px > 0:
                                        close_px = order_exec_price
                                        qty_closed = float(qty_str)
                                        if side == "CLOSE_LONG":
                                             gross_pnl = (close_px - entry_px) * qty_closed
                                        else:
                                            gross_pnl = (entry_px - close_px) * qty_closed
                                            
                                        # Accurate trading fees: Maker for entry, Taker for market/hard stop or Maker for limit
                                        notional_cost_exit = qty_closed * close_px
                                        exit_fee_rate = settings.taker_fee_pct if is_market_exit else settings.maker_fee_pct
                                        dashboard_trade_fees = (entry_px * qty_closed * settings.maker_fee_pct) + (notional_cost_exit * exit_fee_rate)
                                        pnl = gross_pnl - dashboard_trade_fees
                                        
                                        # Record close timestamp for symbol cooldown (Guard 1)
                                        self.last_close_time[sym] = time.time()
                                        
                                        # Calculate Net ROI
                                        margin_used_for_trade = (entry_px * qty_closed) / pos_leverage
                                        roi = (pnl / margin_used_for_trade) * 100 if margin_used_for_trade > 0 else 0.0
                                            
                                        trade_record["entry_price"] = entry_px
                                        trade_record["gross_pnl"] = gross_pnl
                                        trade_record["fees_paid"] = dashboard_trade_fees
                                        trade_record["realized_pnl"] = pnl
                                        trade_record["roi_pct"] = roi
                                        
                                        # Update Self-Awareness Win Rate
                                        self.session_trades += 1
                                        if roi > 0:
                                            self.session_wins += 1
                                            self.last_win_time = time.time()
                                        self.session_win_rate = self.session_wins / self.session_trades
                                        
                                        # Exact Binance Income Ledger Audit (Matches exchange to the penny)
                                        if self.trading_enabled and not self.dry_run:
                                            try:
                                                await asyncio.sleep(0.5)
                                                recent_income = await self.binance._request(
                                                    "GET", "/fapi/v1/income", signed=True,
                                                    params={"symbol": sym, "limit": 10}
                                                )
                                                now_ms = time.time() * 1000
                                                real_pnl = 0.0
                                                real_comm = 0.0
                                                matched = False
                                                for inc in (recent_income or []):
                                                    if (now_ms - float(inc.get("time", 0))) < 15000:
                                                        if inc.get("incomeType") == "REALIZED_PNL":
                                                            real_pnl += float(inc.get("income", 0.0))
                                                            matched = True
                                                        elif inc.get("incomeType") == "COMMISSION":
                                                            real_comm += abs(float(inc.get("income", 0.0)))
                                                if matched:
                                                    gross_pnl = real_pnl
                                                    dashboard_trade_fees = real_comm
                                                    pnl = gross_pnl - dashboard_trade_fees
                                                    trade_record["gross_pnl"] = gross_pnl
                                                    trade_record["fees_paid"] = dashboard_trade_fees
                                                    trade_record["realized_pnl"] = pnl
                                                    margin_used_for_trade = (entry_px * qty_closed) / pos_leverage if pos_leverage else (entry_px * qty_closed / 10)
                                                    roi = (pnl / margin_used_for_trade) * 100 if margin_used_for_trade > 0 else 0.0
                                                    trade_record["roi_pct"] = roi
                                                    logger.info(f"[{sym}] 📜 BINANCE OFFICIAL AUDIT: Gross: ${gross_pnl:+.4f}, Fees: ${dashboard_trade_fees:.4f}, Net: ${pnl:+.4f} (ROI: {roi:+.2f}%)")
                                            except Exception as audit_err:
                                                logger.warning(f"[{sym}] Income audit skipped: {audit_err}")
                                        
                                    # Only log completed trade and alert Telegram immediately if this was an immediate MARKET order.
                                    # Pending MAKER LIMIT orders are logged ONLY when confirmed filled (or on market fallback timeout).
                                    if self.redis.redis and (exit_override or is_closing) and is_market_exit:
                                        await self.redis.redis.lpush("dashboard:trade_history", json.dumps(trade_record))
                                        await self.redis.redis.ltrim("dashboard:trade_history", 0, 99)
                                        
                                        # Institutional Upgrade 4: Telegram Exit Alert
                                        holding_mins = (time.time() - self.position_entry_times.get(sym, time.time())) / 60.0
                                        asyncio.create_task(
                                            telegram_notifier.broadcast_alert(
                                                "TRADE_EXIT",
                                                "",
                                                {
                                                    "symbol": sym,
                                                    "side": record_side,
                                                    "price": order_exec_price,
                                                    "entry_price": entry_px,
                                                    "realized_pnl": trade_record.get("realized_pnl", 0.0),
                                                    "roi_pct": trade_record.get("roi_pct", 0.0),
                                                    "fees": trade_record.get("fees_paid", 0.0),
                                                    "reason": exit_reason if 'exit_reason' in locals() else "AI Close",
                                                    "holding_minutes": max(0.1, holding_mins)
                                                }
                                            )
                                        )
                                    elif "OPEN" in side:
                                        # Institutional Upgrade 4: Telegram Entry Alert
                                        asyncio.create_task(
                                            telegram_notifier.broadcast_alert(
                                                "TRADE_ENTRY",
                                                "",
                                                {
                                                    "symbol": sym,
                                                    "side": side,
                                                    "price": order_exec_price,
                                                    "quantity": float(qty_str),
                                                    "leverage": settings.max_leverage,
                                                    "confidence": float(confidence),
                                                    "reason": "Institutional Alpha Signal",
                                                    "stop_loss": order_exec_price * (0.9925 if "LONG" in side else 1.0075),
                                                    "take_profit": order_exec_price * (1.0175 if "LONG" in side else 0.9825)
                                                }
                                            )
                                        )
                                        
                                    # Only track pending LIMIT orders (market closes fill immediately)
                                    if not is_market_exit:
                                        self.active_orders[sym] = {'order_id': actual_order_id, 'timestamp': time.time(), 'trade_record': trade_record,
                                                                   'pos_qty_before': abs(self.portfolio_state.positions[sym].quantity)}
                                    
                                    # Update local position state immediately to prevent over-buying before the next sync
                                    notional_cost = float(qty_str) * order_exec_price
                                    margin_used = notional_cost / settings.max_leverage

                                    if "OPEN_LONG" in side:
                                        if self.portfolio_state.positions[sym].quantity == 0.0:
                                            self.recent_entry_times.append(time.time())
                                            entries_this_cycle += 1
                                            self.position_exit_levels[sym] = exit_policy.exit_levels(float(getattr(market, "volatility", 0.0) or 0.0))
                                            self.portfolio_state.positions[sym].entry_time = time.time()
                                            self.position_entry_times[sym] = time.time()
                                            self.portfolio_state.positions[sym].entry_price = order_exec_price
                                        else:
                                            old_q = abs(self.portfolio_state.positions[sym].quantity)
                                            new_q = float(qty_str)
                                            old_ep = self.portfolio_state.positions[sym].entry_price
                                            self.portfolio_state.positions[sym].entry_price = ((old_ep * old_q) + (order_exec_price * new_q)) / (old_q + new_q)
                                        self.portfolio_state.positions[sym].quantity += float(qty_str)
                                        self.portfolio_state.free_margin -= margin_used
                                    elif "CLOSE_LONG" in side:
                                        self.portfolio_state.positions[sym].quantity -= float(qty_str)
                                        self.portfolio_state.free_margin += margin_used
                                        if abs(self.portfolio_state.positions[sym].quantity) < 1e-5:
                                            self.portfolio_state.positions[sym].quantity = 0.0
                                            self.portfolio_state.positions[sym].entry_price = 0.0
                                            self.position_entry_times[sym] = 0.0
                                            self.peak_pnl_pct[sym] = 0.0
                                            self.partial_tp_triggered[sym] = False
                                            self.position_exit_levels.pop(sym, None)
                                            if sym in self.pnl_history:
                                                self.pnl_history[sym].clear()
                                    elif "OPEN_SHORT" in side:
                                        if self.portfolio_state.positions[sym].quantity == 0.0:
                                            self.recent_entry_times.append(time.time())
                                            entries_this_cycle += 1
                                            self.position_exit_levels[sym] = exit_policy.exit_levels(float(getattr(market, "volatility", 0.0) or 0.0))
                                            self.portfolio_state.positions[sym].entry_time = time.time()
                                            self.position_entry_times[sym] = time.time()
                                            self.portfolio_state.positions[sym].entry_price = order_exec_price
                                        else:
                                            old_q = abs(self.portfolio_state.positions[sym].quantity)
                                            new_q = float(qty_str)
                                            old_ep = self.portfolio_state.positions[sym].entry_price
                                            self.portfolio_state.positions[sym].entry_price = ((old_ep * old_q) + (order_exec_price * new_q)) / (old_q + new_q)
                                        self.portfolio_state.positions[sym].quantity -= float(qty_str)
                                        self.portfolio_state.free_margin -= margin_used
                                    elif "CLOSE_SHORT" in side:
                                        self.portfolio_state.positions[sym].quantity += float(qty_str)
                                        self.portfolio_state.free_margin += margin_used
                                        if abs(self.portfolio_state.positions[sym].quantity) < 1e-5:
                                            self.portfolio_state.positions[sym].quantity = 0.0
                                            self.portfolio_state.positions[sym].entry_price = 0.0
                                            self.position_entry_times[sym] = 0.0
                                            self.peak_pnl_pct[sym] = 0.0
                                            self.partial_tp_triggered[sym] = False
                                            self.position_exit_levels.pop(sym, None)
                                            if sym in self.pnl_history:
                                                self.pnl_history[sym].clear()
                                                
                                    if partial_tp_triggered:
                                        self.partial_tp_triggered[sym] = True
                                        logger.info(f"[{sym}] 🚀 FREE-ROLL RUNNER ARMED: 50% banked at market. Remaining {abs(self.portfolio_state.positions[sym].quantity):.4f} is now a 100% Risk-Free Free-Roll!")
                                    
                                    # Calculate immediate RL Reward
                                    imm_reward = 0.0
                                    imm_pnl = 0.0
                                    exit_px = None
                                    trade_fees = 0.0
                                    hold_dur = None
                                    
                                    if "CLOSE" in side and entry_px > 0:
                                        exit_px = float(order_exec_price)
                                        trade_fees = dashboard_trade_fees
                                        # pnl is already net of fees from above calculation
                                        
                                        # --- INSTITUTIONAL PENALTY BOX & CONSECUTIVE LOSS TRACKER ---
                                        closed_dir = "LONG" if "CLOSE_LONG" in side else "SHORT"
                                        if pnl < -1.0: # Real loss (exceeding minor fee noise)
                                            now_t = time.time()
                                            self.recent_losses_history.append((now_t, sym, pnl))
                                            
                                            # Passive Airbag 1: Portfolio-Wide Loss Velocity Breaker
                                            recent_losses = [entry for entry in self.recent_losses_history if (now_t - entry[0]) <= 120.0]
                                            if len(recent_losses) >= 3:
                                                cooldown_secs = 900.0  # 15 minutes
                                                self.pause_until = max(self.pause_until, now_t + cooldown_secs)
                                                sym_list = ", ".join(set(e[1] for e in recent_losses))
                                                logger.critical(
                                                    f"🚨 AIRBAG 1 ACTIVATED: LOSS VELOCITY BREAKER! "
                                                    f"{len(recent_losses)} stop-losses in {int(now_t - recent_losses[0][0])}s ({sym_list})! "
                                                    f"Freezing all new trade entries for {int(cooldown_secs/60)} minutes to protect portfolio capital."
                                                )
                                                if self.redis.redis:
                                                    asyncio.create_task(self.redis.redis.set("system:pause_until", str(self.pause_until)))
                                                asyncio.create_task(
                                                    telegram_notifier.broadcast_alert(
                                                        "LOSS_VELOCITY_BREAKER",
                                                        "",
                                                        {
                                                            "loss_count": len(recent_losses),
                                                            "window_seconds": 120,
                                                            "symbols": sym_list,
                                                            "cooldown_minutes": int(cooldown_secs / 60)
                                                        }
                                                    )
                                                )

                                            self.consecutive_losses[sym] = self.consecutive_losses.get(sym, 0) + 1
                                            loss_cnt = self.consecutive_losses[sym]
                                            
                                            if loss_cnt == 1:
                                                # Tier 1 Penalty Box: Lock out the SAME direction for 60 minutes (3,600s)
                                                self.loss_lockout_until[sym][closed_dir] = time.time() + 3600.0
                                                logger.warning(f"[{sym}] 🥊 PENALTY BOX (Tier 1): Stop-loss hit (${pnl:+.2f}). Direction '{closed_dir}' locked out for 60 mins to prevent squeeze fight!")
                                            elif loss_cnt >= 2:
                                                # Tier 2 Penalty Box: Lock out the ENTIRE symbol for 4 hours (14,400s)
                                                self.loss_lockout_until[sym]["LONG"] = time.time() + 14400.0
                                                self.loss_lockout_until[sym]["SHORT"] = time.time() + 14400.0
                                                logger.warning(f"[{sym}] 🥊 PENALTY BOX HARD QUARANTINE (Tier 2): {loss_cnt} consecutive losses! Symbol locked out for 4 hours!")
                                        elif pnl > 2.0: # Clean winning trade
                                            if self.consecutive_losses.get(sym, 0) > 0:
                                                logger.info(f"[{sym}] 🏆 WINNING EXIT (${pnl:+.2f}): Resetting consecutive loss counter and clearing Penalty Box!")
                                            self.consecutive_losses[sym] = 0
                                            self.loss_lockout_until[sym] = {"LONG": 0.0, "SHORT": 0.0}
                                        
                                        entry_t = getattr(self.portfolio_state.positions[sym], 'entry_time', 0.0)
                                        if entry_t > 0:
                                            hold_dur = int(time.time() - entry_t)
                                        
                                        # --- ACCOUNT-AGNOSTIC REWARD UPGRADE ---
                                        # Use ROI (Percentage) instead of absolute PnL so the AI 
                                        # does not get confused when moving from $100k Testnet to $500 Live.
                                        #
                                        # FIX: Confidence-Weighted Reward — teaches the AI that
                                        # high confidence on a losing trade = EXTRA pain,
                                        # high confidence on a winning trade = EXTRA reward.
                                        # This creates a gradient signal for calibrating confidence honestly.
                                        if roi < 0:
                                            # Confidence-Amplified Loss Penalty:
                                            # At confidence=0.0 → 1.0x base penalty
                                            # At confidence=0.5 → 1.75x penalty  
                                            # At confidence=1.0 → 2.5x penalty (overconfident = extra pain)
                                            confidence_penalty = 1.0 + (confidence * 1.5)
                                            imm_reward = roi * confidence_penalty
                                        elif roi > 0 and roi < 5.0:
                                            # Time Decay Proxy: Punish tiny "lazy" wins (< 5% ROI)
                                            # Forces the AI to look for real momentum instead of micro-scalping
                                            imm_reward = -abs(roi * 0.5) 
                                        else:
                                            # Confidence-Amplified Sniper Bonus:
                                            # High confidence on a big win = rewarded for accurate conviction
                                            confidence_bonus = 1.0 + (confidence * 0.5)
                                            imm_reward = roi * confidence_bonus
                                            
                                        imm_pnl = pnl
                                    elif "OPEN" in side:
                                        # Maker fee penalty (Standardized to ROI percentage to avoid scale-attachment)
                                        imm_reward = -(settings.maker_fee_pct * 100.0) * pos_leverage
                                        
                                    pos_after = float(self.portfolio_state.positions[sym].quantity)
                                    
                                    # Record Experience
                                    exp_data = {
                                        "experience_id": str(uuid.uuid4()),
                                        "timestamp": int(time.time()),
                                        "symbol": sym,
                                        "market_state": seq, # Save full 2D sequence [300, 108] so the AI has the full movie for this specific trade!
                                        "macro_state": None, 
                                        "portfolio_state": {
                                            "free_margin": float(self.portfolio_state.free_margin),
                                            "total_unrealized_pnl": float(self.portfolio_state.total_unrealized_pnl)
                                        },
                                        "derivatives_state": {
                                            "target_size": float(target_size),
                                            "price_offset": float(price_offset)
                                        },
                                        "position_before": float(pos.quantity),
                                        "position_after": pos_after,
                                        "entry_price": float(entry_px if ("CLOSE" in side and entry_px > 0) else order_exec_price),
                                        "exit_price": exit_px,
                                        "leverage": int(pos_leverage),
                                        "margin": float(margin_used),
                                        "fees": float(trade_fees) if trade_fees else 0.0,
                                        "holding_duration": hold_dur,
                                        "action": side,
                                        "action_probability": float(confidence),
                                        "confidence": float(confidence),
                                        "model_version": active_version,
                                        "reward": float(imm_reward),
                                        "realized_pnl": float(imm_pnl),
                                        "unrealized_pnl": 0.0
                                    }
                                    experience_payload = msgpack.packb(
                                        exp_data,
                                        use_bin_type=True,
                                    )
                                    if hard_stop_triggered:
                                        # Market order executed immediately
                                        await redis_manager.publish_binary(
                                            "experience:completed",
                                            experience_payload,
                                        )
                                    elif sym in self.active_orders:
                                        # Limit order: attach to active_orders to publish ONLY when filled!
                                        self.active_orders[sym]['experience_payload'] = experience_payload
                                except Exception as e:
                                    logger.error(f"[{sym}] ORDER/EXPERIENCE FAILED: {e}")
                                    if "-4118" in str(e) or "ReduceOnly Order Failed" in str(e):
                                        logger.warning(f"[{sym}] Ghost limit order detected. Clearing order book to unstuck AI.")
                                        try:
                                            await self.binance.cancel_all_orders(sym)
                                            # Clean up local active orders tracking to match reality
                                            if sym in self.active_orders:
                                                del self.active_orders[sym]
                                        except Exception as cancel_e:
                                            logger.error(f"[{sym}] Failed to clear ghost orders: {cancel_e}")
                                    
                            else:
                                logger.info(f"[{sym}] NO ORDER (Blocked by final safety gate): {side} {qty_str}")
                        else:
                            logger.warning(f"[{sym}] Risk Manager REJECTED: {decision.reason}")
                            
            except Exception as e:
                logger.critical(f"FAIL CLOSED: Critical error in inference loop: {e}")
                self.running = False # Halt the bot immediately
                raise RuntimeError(f"FAIL CLOSED: Inference loop error: {e}")

            await self._save_entry_guard()
            await self._save_state_history()
            await asyncio.sleep(5.0)
            
    async def monitor_kill_switch(self):
        while self.running:
            try:
                if self.redis.redis:
                    ks = await self.redis.redis.get("system:kill_switch:active")
                    if not ks:
                        ks = await self.redis.redis.get("kill_switch")
                    val = (ks.decode("utf-8") if isinstance(ks, bytes) else str(ks or "")).strip().lower()
                    if val in ["true", "1", "yes"]:
                        if not self.emergency_stop:
                            logger.critical("🚨 EMERGENCY KILL SWITCH ACTIVATED VIA REDIS! 🚨")
                            self.emergency_stop = True
                            self.trading_enabled = False
                            settings.emergency_stop = True
                            await self.panic_liquidate()
                    else:
                        if self.emergency_stop:
                            logger.info("🟢 EMERGENCY KILL SWITCH DISARMED VIA REDIS. Normal trading restored.")
                            self.emergency_stop = False
                            self.trading_enabled = settings.trading_enabled
                            settings.emergency_stop = False
                            
                    # Monitor Telegram temporary pause timer
                    pause_val = await self.redis.redis.get("system:pause_until")
                    if pause_val:
                        p_ts = float(pause_val.decode("utf-8") if isinstance(pause_val, bytes) else pause_val)
                        self.pause_until = p_ts
                    else:
                        self.pause_until = 0.0
            except Exception as e:
                logger.error(f"Error in monitor_kill_switch: {e}")
            await asyncio.sleep(2.0)

    async def monitor_heartbeat_watchdog(self):
        """Passive Airbag 2: Monitors tick freshness. If WebSockets stall while resting limit orders exist, cancel them."""
        logger.info("Starting WebSocket Heartbeat & Resting Order Watchdog...")
        while self.running:
            try:
                now_t = time.time()
                stall_duration = now_t - self.last_market_tick_time
                if stall_duration > 5.0 and self.active_orders:
                    order_count = len(self.active_orders)
                    logger.warning(
                        f"🚨 AIRBAG 2 ACTIVATED: WEBSOCKET HEARTBEAT STALL ({stall_duration:.1f}s > 5.0s)! "
                        f"Cancelling {order_count} resting limit orders to avoid stale orderbook fills."
                    )
                    for sym, order_info in list(self.active_orders.items()):
                        try:
                            if (self.trading_enabled and not self.dry_run and 
                                (settings.allow_testnet or settings.allow_live_trading)):
                                await self.binance.cancel_order(sym, order_info.get("order_id"))
                                logger.info(f"[{sym}] Safe cancelled resting order {order_info.get('order_id')} due to WebSocket stall.")
                        except Exception as cancel_e:
                            logger.debug(f"[{sym}] Watchdog cancel error: {cancel_e}")
                        self.active_orders.pop(sym, None)
                    
                    asyncio.create_task(
                        telegram_notifier.broadcast_alert(
                            "HEARTBEAT_STALL",
                            "",
                            {
                                "stall_duration": stall_duration,
                                "cancelled_count": order_count
                            }
                        )
                    )
            except Exception as e:
                logger.error(f"Error in monitor_heartbeat_watchdog: {e}")
            await asyncio.sleep(2.0)

    async def panic_liquidate(self):
        """Immediately closes all open positions and cancels open orders."""
        logger.critical("Initiating PANIC LIQUIDATION for all open positions!")
        try:
            live_positions = await self.binance.get_positions()
            for p in live_positions:
                sym = p.get("symbol")
                amt = float(p.get("positionAmt", 0))
                if abs(amt) > 0.0001:
                    logger.critical(f"Panic liquidating {sym} {amt}")
                    side = "SELL" if amt > 0 else "BUY"
                    
                    if not self.dry_run:
                        # Cancel any open orders for this symbol first
                        try:
                            await self.binance.cancel_all_orders(sym)
                        except Exception as e:
                            logger.warning(f"Failed to cancel orders for {sym}: {e}")
                            
                        # Send market order to close
                        client_order_id = f"panic_{uuid.uuid4().hex[:10]}"
                        try:
                            await self.binance.create_order(
                                symbol=sym,
                                side=side,
                                quantity=abs(amt),
                                client_order_id=client_order_id,
                                order_type="MARKET",
                                reduce_only=True
                            )
                            logger.critical(f"Panic Market {side} for {sym} executed.")
                        except Exception as order_e:
                            logger.error(f"Failed to execute panic liquidation for {sym}: {order_e}")
        except Exception as e:
            logger.error(f"Error during panic liquidation: {e}")

    async def listen_liquidation_cascade(self):
        """Institutional Upgrade 2: Subscribes to multi-exchange liquidation streams."""
        await self.redis.connect()
        if not self.redis.redis:
            return
        pubsub = self.redis.redis.pubsub()
        await pubsub.subscribe("liquidation:cascade:global")
        for sym in self.symbols:
            await pubsub.subscribe(f"liquidation:cascade:{sym}")
            
        logger.info("Subscribed to multi-exchange liquidation cascade channels.")
        async for message in pubsub.listen():
            if not self.running:
                break
            if message["type"] == "message":
                try:
                    ch = message["channel"]
                    channel = ch.decode("utf-8") if isinstance(ch, bytes) else str(ch)
                    raw_data = message["data"]
                    data_str = raw_data.decode("utf-8") if isinstance(raw_data, bytes) else str(raw_data)
                    data = json.loads(data_str)
                    
                    if channel == "liquidation:cascade:global":
                        self.global_liq_state = data
                    elif channel.startswith("liquidation:cascade:"):
                        sym = channel.split(":")[-1]
                        liquidation_hunter.update_liquidation_state(sym, data)
                except Exception as e:
                    logger.debug(f"Error parsing liquidation stream: {e}")

    async def beta_hedging_loop(self):
        """Market-Beta Hedger has been removed/disabled. Trading runs in pure directional alpha mode."""
        logger.info("Market-Beta Hedger is completely disabled.")
        return

    async def listen_macro_shield_loop(self):
        """Institutional Upgrade 1: Monitors upcoming economic releases and logs/alerts before high-impact events."""
        logger.info("Starting Macro & Economic Event Shield monitor loop...")
        last_alerted_event = None
        while self.running:
            try:
                next_event = macro_crash_shield.get_next_economic_event()
                if next_event:
                    mins_until = next_event.get("minutes_until", 999.0)
                    if 0 <= mins_until <= 15.0 and last_alerted_event != next_event["name"]:
                        last_alerted_event = next_event["name"]
                        logger.warning(
                            f"🏛️ ECONOMIC EVENT SHIELD: {next_event['name']} in {mins_until:.1f} mins! "
                            f"Pausing new order executions to avoid spread blowout."
                        )
                        asyncio.create_task(
                            telegram_notifier.broadcast_alert(
                                "ECONOMIC_SHIELD",
                                "",
                                {
                                    "name": next_event["name"],
                                    "status": "IMMINENT RELEASE (Execution Paused)",
                                    "time_desc": f"Releasing in {mins_until:.1f} mins"
                                }
                            )
                        )
            except Exception as e:
                logger.error(f"Error in listen_macro_shield_loop: {e}")
            await asyncio.sleep(30.0)

    async def start(self):
        self.running = True
        await registry.initialize_from_exchange(self.binance)
        
        # Cancel all open orders on startup to ensure a clean slate and free up margin
        try:
            for sym in self.symbols:
                await self.binance.cancel_all_orders(sym)
            logger.info("Cleared all pre-existing open orders on startup.")
        except Exception as e:
            logger.warning(f"Failed to clear open orders on startup: {e}")
        
        # Fetch actual account balance to allow RiskManager to approve trades
        try:
            usdt = await self.binance.get_account_details()
            if not usdt and self.dry_run:
                usdt = {"balance": 1000.0, "crossWalletBalance": 1000.0, "crossUnPnl": 0.0, "availableBalance": 1000.0}
            self.portfolio_state.wallet_balance = float(usdt.get("balance", 0.0))
            self.portfolio_state.equity = float(usdt.get("crossWalletBalance", self.portfolio_state.wallet_balance)) + float(usdt.get("crossUnPnl", 0.0))
            self.portfolio_state.free_margin = float(usdt.get("availableBalance", self.portfolio_state.wallet_balance))
            balance = self.portfolio_state.wallet_balance
            logger.info(f"Initialized Portfolio with Balance: {balance} USDT")
        except Exception as e:
            logger.error(f"Failed to fetch account balance: {e}")
            if self.dry_run:
                self.portfolio_state.wallet_balance = 1000.0
                self.portfolio_state.equity = 1000.0
                self.portfolio_state.free_margin = 1000.0
                logger.info("Using mock 1000 USDT balance for Dry Run.")

        # Redis must be up before the startup checks below (it was otherwise only connected later by the listener tasks)
        try:
            await self.redis.connect()
        except Exception as e:
            logger.error(f"Redis connect at startup failed: {e}")

        # Check if kill switch is active at startup
        try:
            if self.redis.redis:
                ks = await self.redis.redis.get("system:kill_switch:active")
                val = (ks.decode("utf-8") if isinstance(ks, bytes) else str(ks or "")).strip().lower()
                if val in ["true", "1", "yes"]:
                    logger.critical("🚨 STARTUP BLOCKED: Kill switch is currently ACTIVE in Redis! Halting trading immediately.")
                    self.emergency_stop = True
                    self.trading_enabled = False
                    settings.emergency_stop = True
                    await self.panic_liquidate()
        except Exception as ks_err:
            logger.error(f"Failed to check initial kill switch: {ks_err}")

        await self._load_entry_guard()
        await self._load_state_history()
        await self._load_disaster_stops()

        tasks = [
            asyncio.create_task(self.listen_macro()),
            asyncio.create_task(self.inference_loop()),
            asyncio.create_task(self.sync_dashboard()),
            asyncio.create_task(self.monitor_kill_switch()),
            asyncio.create_task(self.update_htf_regimes_loop()),
            asyncio.create_task(self.poll_global_context_loop()),
            asyncio.create_task(self.listen_liquidation_cascade()),
            asyncio.create_task(self.listen_macro_shield_loop()),
            asyncio.create_task(self.monitor_heartbeat_watchdog())
        ]
        for sym in list(self.symbols):
            t_m = asyncio.create_task(self.listen_market(sym))
            t_w = asyncio.create_task(self.listen_whale(sym))
            t_s = asyncio.create_task(self.listen_statarb(sym))
            self._symbol_tasks[sym] = [t_m, t_w, t_s]
            tasks.extend([t_m, t_w, t_s])
            
        try:
            await asyncio.gather(*tasks)
        finally:
            await self.binance.close()
        
    def stop(self):
        self.running = False

if __name__ == "__main__":
    set_log_file("logs/trading_bot.log")
    from core.config.settings import settings
    bot = ProductionTradingBot(symbols=settings.symbol_universe)
    try:
        asyncio.run(bot.start())
    except KeyboardInterrupt:
        bot.stop()
    except Exception as e:
        logger.critical(f"FATAL ERROR ON STARTUP: {e}", exc_info=True)
