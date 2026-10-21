import asyncio
import json
import time
from types import SimpleNamespace

import pytest

import apps.trading_bot.main as bot_main
from core.exchange.symbol_registry import SymbolConfig
from core.risk import exit_policy
from core.risk.risk_manager import RiskManager
from core.schemas.state_schema import PortfolioState, PositionState


def _bot():
    b = bot_main.ProductionTradingBot.__new__(bot_main.ProductionTradingBot)
    b.risk_manager = RiskManager()
    b.portfolio_state = PortfolioState(wallet_balance=1000.0, equity=1000.0, used_margin=0.0, free_margin=1000.0,
                                       total_unrealized_pnl=0.0, total_exposure=0.0,
                                       positions={"XRPUSDT": PositionState(symbol="XRPUSDT"),
                                                  "ETHUSDT": PositionState(symbol="ETHUSDT")})
    b.position_entry_times = {"XRPUSDT": 0.0, "ETHUSDT": 0.0}
    b.peak_pnl_pct = {"XRPUSDT": 0.0, "ETHUSDT": 0.0}
    b.partial_tp_triggered = {"XRPUSDT": False, "ETHUSDT": False}
    b.position_exit_levels = {}
    b.consecutive_losses = {"XRPUSDT": 0, "ETHUSDT": 0}
    b.loss_lockout_until = {"XRPUSDT": {"LONG": 0.0, "SHORT": 0.0}, "ETHUSDT": {"LONG": 0.0, "SHORT": 0.0}}
    b.daily_start_equity = 0.0
    b.daily_peak_equity = 0.0
    b.daily_equity_reset_time = time.time()
    b.circuit_breaker_active_until = 0.0
    b.profit_ratchet_active = False
    b.max_equity = 0.0
    b._restored_pos_state = {}
    b.last_close_time = {"XRPUSDT": 0.0, "ETHUSDT": 0.0}
    b.market_states = {"XRPUSDT": SimpleNamespace(mid_price=1.38, volatility=0.009)}
    b.disaster_stops = {}
    b._disaster_stop_retry = {}
    b.trading_enabled = True
    b.dry_run = False
    b.emergency_stop = False
    b.redis = SimpleNamespace(redis=None)
    return b


def test_close_record_uses_real_fills():
    b = _bot()
    base = {"side": "CLOSE_LONG", "leverage": 20, "symbol": "XRPUSDT"}
    rec = b._close_record(base, 1.3676, [(2000.0, 1.3820, 0.0002), (5580.2, 1.3810, 0.0005)])
    exit_px = (2000 * 1.3820 + 5580.2 * 1.3810) / 7580.2
    assert rec["price"] == pytest.approx(exit_px)
    assert rec["gross_pnl"] == pytest.approx((exit_px - 1.3676) * 7580.2)
    fees = 1.3676 * 7580.2 * 0.0002 + 2000 * 1.3820 * 0.0002 + 5580.2 * 1.3810 * 0.0005
    assert rec["realized_pnl"] == pytest.approx(rec["gross_pnl"] - fees)


def test_risk_state_survives_restart_and_only_restores_the_same_trade():
    before = _bot()
    before.daily_start_equity = 4400.0
    before.circuit_breaker_active_until = time.time() + 3000
    before.consecutive_losses["ETHUSDT"] = 1
    before.loss_lockout_until["ETHUSDT"] = {"LONG": 0.0, "SHORT": time.time() + 1800}
    pos = before.portfolio_state.positions["XRPUSDT"]
    pos.quantity, pos.entry_price = 7580.2, 1.3676
    before.position_entry_times["XRPUSDT"] = time.time() - 1500
    before.peak_pnl_pct["XRPUSDT"] = 0.21
    before.partial_tp_triggered["XRPUSDT"] = True
    before.position_exit_levels["XRPUSDT"] = exit_policy.exit_levels(0.009)
    state = json.loads(json.dumps(before._risk_state()))

    after = _bot()
    after._restore_risk_state(state)
    assert after.daily_start_equity == 4400.0
    assert after.circuit_breaker_active_until > time.time()
    assert after.consecutive_losses["ETHUSDT"] == 1 and after.loss_lockout_until["ETHUSDT"]["SHORT"] > time.time()
    after._apply_restored_position("XRPUSDT", 1.3676)            # Binance confirms the same trade
    assert after.partial_tp_triggered["XRPUSDT"] is True
    assert after.peak_pnl_pct["XRPUSDT"] == pytest.approx(0.21)
    assert after.position_exit_levels["XRPUSDT"] == exit_policy.exit_levels(0.009)

    other = _bot()
    other._restore_risk_state(state)
    other._apply_restored_position("XRPUSDT", 1.4000)            # a different trade: ignore saved state
    assert other.partial_tp_triggered["XRPUSDT"] is False


def test_exit_levels_are_frozen_at_entry():
    b = _bot()
    first = b._exit_levels_for("XRPUSDT", SimpleNamespace(volatility=0.004))
    later = b._exit_levels_for("XRPUSDT", SimpleNamespace(volatility=0.009))
    assert later == first


class FakeExchange:
    def __init__(self):
        self.placed, self.cancelled = [], []
        self.status = "NEW"

    async def place_close_stop(self, sym, side, trigger, cid):
        self.placed.append((sym, side, trigger))
        return {"algoId": len(self.placed)}

    async def cancel_algo_order(self, algo_id):
        self.cancelled.append(algo_id)

    async def _request(self, method, path, signed=False, params=None):
        if path == "/fapi/v1/algoOrder":
            return {"algoStatus": self.status}
        return []


def test_backup_stop_is_placed_once_and_detects_a_fill(monkeypatch):
    cfg = SymbolConfig("XRPUSDT")
    cfg.price_precision, cfg.tick_size = 4, 0.0001
    monkeypatch.setattr(bot_main.registry, "get_symbol", lambda s: cfg)
    monkeypatch.setattr(bot_main.settings, "allow_testnet", True)
    b = _bot()
    ex = FakeExchange()
    b.binance = ex
    live = [{"symbol": "XRPUSDT", "positionAmt": "7580.2", "entryPrice": "1.3676"}]

    async def run():
        await b._ensure_disaster_stops(live)
        await b._ensure_disaster_stops(live)                    # second sync: nothing new
        ex.status = "TRIGGERED"
        await b._ensure_disaster_stops([])                      # position gone, bot did not close it
    asyncio.run(run())

    stop = exit_policy.exit_levels(0.009).stop * bot_main.settings.disaster_stop_mult
    assert len(ex.placed) == 1
    sym, side, trigger = ex.placed[0]
    assert side == "SELL" and float(trigger) == pytest.approx(1.3676 * (1 - stop), abs=2e-4)
    assert b.disaster_stops == {}
    assert b.loss_lockout_until["XRPUSDT"]["LONG"] > time.time()  # stop-out counted like a stop-loss
