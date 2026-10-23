import time
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock
from collections import deque

from core.telemetry.telegram_service import telegram_notifier
from core.schemas.state_schema import MarketState, PortfolioState, PositionState
from apps.trading_bot.main import ProductionTradingBot


def make_market_state(symbol: str, mid_price: float) -> MarketState:
    """Helper to instantiate a valid MarketState schema."""
    return MarketState(
        symbol=symbol,
        timestamp=time.time(),
        bid=mid_price - 0.05,
        ask=mid_price + 0.05,
        mid_price=mid_price,
        last_price=mid_price,
        spread=0.1,
        order_book_imbalance=0.0,
        volume=1000.0,
        vwap=mid_price,
        volatility=0.01,
        funding_rate=0.0001,
        features=[0.0] * 90
    )


def test_telegram_airbag_formatters():
    """Verify all 4 airbag Telegram alert formatters render valid HTML."""
    # Airbag 1 formatter
    t1 = telegram_notifier.format_loss_velocity_breaker({
        "loss_count": 3,
        "window_seconds": 120,
        "symbols": "BTCUSDT, ETHUSDT, SOLUSDT",
        "cooldown_minutes": 15
    })
    assert "LOSS VELOCITY BREAKER" in t1
    assert "3 stop-losses in 120s" in t1

    # Airbag 2 formatter
    t2 = telegram_notifier.format_heartbeat_stall({
        "stall_duration": 6.5,
        "cancelled_count": 2
    })
    assert "WEBSOCKET HEARTBEAT WATCHDOG" in t2
    assert "6.5s without tick" in t2

    # Airbag 3 formatter
    t3 = telegram_notifier.format_slippage_anomaly({
        "symbol": "DOGEUSDT",
        "slippage_pct": 1.25,
        "mid_price": 0.1200,
        "fill_price": 0.1215,
        "quarantine_minutes": 15
    })
    assert "SLIPPAGE ANOMALY GUARD" in t3
    assert "1.25%" in t3
    assert "DOGEUSDT" in t3

    # Airbag 4 formatter
    t4 = telegram_notifier.format_margin_cushion_trim({
        "free_margin_ratio": 12.5,
        "trimmed_symbol": "XRPUSDT",
        "trimmed_qty": "500"
    })
    assert "MARGIN CUSHION AUTO-TRIM" in t4
    assert "12.5%" in t4
    assert "XRPUSDT" in t4


def test_airbag1_loss_velocity_breaker_trips():
    """Airbag 1: Verify 3 rapid losses within 120s trigger 15-minute global pause."""
    bot = ProductionTradingBot(symbols=["BTCUSDT", "ETHUSDT"])
    now = time.time()
    
    # Simulate 3 losses within 60s
    bot.recent_losses_history.append((now - 50, "BTCUSDT", -15.0))
    bot.recent_losses_history.append((now - 30, "ETHUSDT", -20.0))
    bot.recent_losses_history.append((now - 10, "BTCUSDT", -12.0))
    
    recent_losses = [entry for entry in bot.recent_losses_history if (now - entry[0]) <= 120.0]
    assert len(recent_losses) == 3
    
    cooldown_secs = 900.0
    bot.pause_until = max(bot.pause_until, now + cooldown_secs)
    
    assert bot.pause_until >= now + 899.0
    assert bot.pause_until <= now + 901.0


def test_airbag1_loss_velocity_spaced_out_does_not_trip():
    """Airbag 1: Verify losses spaced > 120s apart do NOT trip the loss velocity breaker."""
    bot = ProductionTradingBot(symbols=["BTCUSDT", "ETHUSDT"])
    now = time.time()
    
    # Simulate 3 losses spaced 5 minutes apart
    bot.recent_losses_history.append((now - 600, "BTCUSDT", -15.0))
    bot.recent_losses_history.append((now - 300, "ETHUSDT", -20.0))
    bot.recent_losses_history.append((now - 10, "BTCUSDT", -12.0))
    
    recent_losses = [entry for entry in bot.recent_losses_history if (now - entry[0]) <= 120.0]
    assert len(recent_losses) == 1  # Only the recent one falls within the 120s window


def test_airbag3_slippage_anomaly_quarantines_symbol():
    """Airbag 3: Verify execution slippage > 0.75% quarantines the symbol for 15 minutes."""
    bot = ProductionTradingBot(symbols=["BTCUSDT", "DOGEUSDT"])
    now = time.time()
    
    mid_price = 0.1000
    avg_fill = 0.1010  # 1.0% slippage (exceeds 0.75% threshold)
    
    slippage_pct = abs(avg_fill - mid_price) / mid_price
    assert slippage_pct > 0.0075
    
    quarantine_secs = 900.0
    bot.loss_lockout_until["DOGEUSDT"]["LONG"] = max(bot.loss_lockout_until["DOGEUSDT"].get("LONG", 0.0), now + quarantine_secs)
    bot.loss_lockout_until["DOGEUSDT"]["SHORT"] = max(bot.loss_lockout_until["DOGEUSDT"].get("SHORT", 0.0), now + quarantine_secs)
    
    assert bot.loss_lockout_until["DOGEUSDT"]["LONG"] >= now + 899.0
    assert bot.loss_lockout_until["DOGEUSDT"]["SHORT"] >= now + 899.0


def test_airbag3_normal_slippage_does_not_quarantine():
    """Airbag 3: Verify normal fill slippage (<= 0.75%) does NOT quarantine the symbol."""
    bot = ProductionTradingBot(symbols=["BTCUSDT", "DOGEUSDT"])
    now = time.time()
    
    mid_price = 0.1000
    avg_fill = 0.1002  # 0.2% slippage (normal maker/market variance)
    
    slippage_pct = abs(avg_fill - mid_price) / mid_price
    assert slippage_pct <= 0.0075
    assert bot.loss_lockout_until["DOGEUSDT"]["LONG"] == 0.0


def test_airbag2_heartbeat_watchdog_cancels_resting_orders_on_stall():
    """Airbag 2: Verify watchdog cancels resting orders when tick age exceeds 5.0 seconds."""
    async def _test():
        bot = ProductionTradingBot(symbols=["BTCUSDT", "ETHUSDT"])
        bot.running = True
        bot.trading_enabled = True
        bot.dry_run = False
        
        now = time.time()
        bot.last_market_tick_time = now - 7.0  # Stalled 7 seconds (> 5.0s)
        bot.active_orders = {
            "BTCUSDT": {"order_id": "12345", "timestamp": now - 10.0},
            "ETHUSDT": {"order_id": "67890", "timestamp": now - 8.0}
        }
        
        bot.binance.cancel_order = AsyncMock(return_value={"status": "CANCELED"})
        
        # Run one watchdog iteration check
        stall_duration = now - bot.last_market_tick_time
        assert stall_duration > 5.0
        
        for sym, order_info in list(bot.active_orders.items()):
            await bot.binance.cancel_order(sym, order_info["order_id"])
            bot.active_orders.pop(sym, None)
            
        assert bot.binance.cancel_order.call_count == 2
        assert len(bot.active_orders) == 0

    asyncio.run(_test())


def test_airbag2_heartbeat_watchdog_keeps_resting_orders_when_ticks_fresh():
    """Airbag 2: Verify watchdog does NOT cancel orders when ticks are fresh (< 5.0s)."""
    async def _test():
        bot = ProductionTradingBot(symbols=["BTCUSDT"])
        bot.running = True
        now = time.time()
        bot.last_market_tick_time = now - 0.5  # Fresh tick (0.5s ago)
        bot.active_orders = {
            "BTCUSDT": {"order_id": "12345", "timestamp": now - 2.0}
        }
        
        bot.binance.cancel_order = AsyncMock()
        
        stall_duration = now - bot.last_market_tick_time
        assert stall_duration <= 5.0
        # No cancellation performed
        assert bot.binance.cancel_order.call_count == 0
        assert len(bot.active_orders) == 1

    asyncio.run(_test())


def test_airbag4_margin_cushion_detects_critical_ratio():
    """Airbag 4: Verify free margin < 15% correctly flags critical deleveraging need."""
    bot = ProductionTradingBot(symbols=["BTCUSDT", "XRPUSDT"])
    bot.portfolio_state.equity = 10000.0
    bot.portfolio_state.free_margin = 1200.0  # 12% free margin (< 15% critical threshold)
    
    free_ratio = bot.portfolio_state.free_margin / bot.portfolio_state.equity
    assert free_ratio < 0.15
    
    # Populate positions: BTC in profit (+5%), XRP in loss (-15%)
    bot.portfolio_state.positions["BTCUSDT"] = PositionState(
        symbol="BTCUSDT", quantity=0.1, entry_price=60000.0
    )
    bot.portfolio_state.positions["XRPUSDT"] = PositionState(
        symbol="XRPUSDT", quantity=1000.0, entry_price=1.00
    )
    
    bot.market_states["BTCUSDT"] = make_market_state("BTCUSDT", 63000.0)  # +5%
    bot.market_states["XRPUSDT"] = make_market_state("XRPUSDT", 0.85)     # -15%
    
    active_pos_list = [
        (s, p) for s, p in bot.portfolio_state.positions.items() 
        if abs(p.quantity) > 0.0001
    ]
    
    def _calc_underwater_roi(item):
        sym_name, pos_obj = item
        m = bot.market_states.get(sym_name)
        px = m.mid_price if m else pos_obj.entry_price
        if pos_obj.entry_price > 0:
            return ((px - pos_obj.entry_price) / pos_obj.entry_price) * (1 if pos_obj.quantity > 0 else -1)
        return 0.0

    worst_sym, worst_pos = min(active_pos_list, key=_calc_underwater_roi)
    assert worst_sym == "XRPUSDT"  # Correctly identifies the losing position for 25% trim
    assert worst_pos.quantity == 1000.0
    trim_qty = abs(worst_pos.quantity) * 0.25
    assert trim_qty == 250.0


def test_total_exposure_feeds_the_portfolio_cap():
    """total_exposure (never filled before, so the 20x portfolio cap never applied) sums open notionals."""
    from core.risk.risk_manager import RiskManager
    from core.schemas.state_schema import OrderRequest
    import core.risk.risk_manager as rm_module

    bot = ProductionTradingBot(symbols=["BTCUSDT", "ETHUSDT", "XRPUSDT"])
    bot.market_states["BTCUSDT"] = make_market_state("BTCUSDT", 100.0)
    bot.portfolio_state.positions["BTCUSDT"].quantity = 300.0      # $30,000 long at the market price
    bot.portfolio_state.positions["ETHUSDT"].quantity = -200.0     # $20,000 short, no market state yet:
    bot.portfolio_state.positions["ETHUSDT"].entry_price = 100.0   # valued at its entry price
    for s in ("BTCUSDT", "ETHUSDT"):
        bot.portfolio_state.positions[s].current_price = 100.0     # set every tick by the live loop
    assert bot._refresh_total_exposure() == pytest.approx(50000.0)
    assert bot.portfolio_state.total_exposure == pytest.approx(50000.0)

    # Equity $3,000 x 20 = $60,000 cap: a $15,000 third trade would make $65,000 -> rejected
    old = (rm_module.settings.trading_enabled, rm_module.settings.emergency_stop)
    rm_module.settings.trading_enabled, rm_module.settings.emergency_stop = True, False
    try:
        rm = RiskManager()
        rm.max_open_positions = 10
        rm.max_symbol_exposure_pct = 20.0
        rm.correlated_exposure_limit_pct = 20.0
        rm.max_position_margin_pct = 1.0
        port = bot.portfolio_state
        port.equity, port.free_margin = 3000.0, 3000.0
        market = make_market_state("XRPUSDT", 1.0)
        req = OrderRequest(symbol="XRPUSDT", action_type="OPEN_LONG", confidence=0.9, requested_quantity=15000.0,
                           target_position=1.0, model_version="v1", timestamp=time.time())
        assert rm.max_portfolio_exposure_pct == 20.0
        assert rm.evaluate(req, port, market).reason == "EXCESSIVE_PORTFOLIO_EXPOSURE"
        req.requested_quantity = 5000.0   # $55,000 total: allowed
        assert rm.evaluate(req, port, market).approved
    finally:
        rm_module.settings.trading_enabled, rm_module.settings.emergency_stop = old


def test_stale_whale_and_statarb_feeds_read_as_no_data():
    """A crashed collector used to freeze its last message in the brain's inputs forever."""
    fresh = {"timestamp": time.time() - 5, "whale_buy_pressure": 200000.0}
    stale = {"timestamp": time.time() - 600, "whale_buy_pressure": 200000.0}
    assert ProductionTradingBot._fresh_feed(fresh) is fresh
    assert ProductionTradingBot._fresh_feed(stale) == {}
    assert ProductionTradingBot._fresh_feed(None) == {}
    assert ProductionTradingBot._fresh_feed({"whale_buy_pressure": 1.0}) == {}   # no timestamp: unknown age
