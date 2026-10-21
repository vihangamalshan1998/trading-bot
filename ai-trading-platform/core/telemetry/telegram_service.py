import aiohttp
import json
import time
from typing import Dict, Any, Optional
from core.config.settings import settings
from core.logging.logger import logger
from core.db.redis import redis_manager

class TelegramNotifier:
    """
    Asynchronous Telegram notification broadcaster.
    Publishes rich events to Redis and sends direct Telegram Bot API messages.
    """

    def __init__(self):
        self.bot_token = settings.telegram_bot_token
        self.chat_id = settings.telegram_chat_id
        self.enabled = settings.telegram_enabled and bool(self.bot_token) and bool(self.chat_id)
        self.session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10))
        return self.session

    async def send_message(self, text: str, reply_markup: Optional[Dict[str, Any]] = None, parse_mode: str = "HTML") -> bool:
        """Sends a message directly to the configured Telegram chat."""
        if not self.enabled:
            logger.debug(f"[TELEGRAM DRY-RUN]: {text[:100]}...")
            return False
            
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": parse_mode,
            "disable_web_page_preview": True
        }
        if reply_markup:
            payload["reply_markup"] = reply_markup
            
        try:
            session = await self._get_session()
            async with session.post(url, json=payload) as resp:
                if resp.status == 200:
                    return True
                err = await resp.text()
                if "can't parse entities" in err:
                    # A bare "<" in an alert (e.g. "< 15%") breaks HTML mode: deliver it as plain text instead
                    payload.pop("parse_mode", None)
                    async with session.post(url, json=payload) as retry:
                        if retry.status == 200:
                            return True
                        err = await retry.text()
                logger.error(f"Telegram API error {resp.status}: {err}")
                return False
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
            return False

    async def broadcast_alert(self, alert_type: str, message: str, payload: Optional[Dict[str, Any]] = None):
        """
        Broadcasts alert to Redis channel 'telegram:alerts' so the Telegram Bot service
        can dispatch it with interactive buttons.
        """
        event = {
            "type": alert_type,
            "message": message,
            "payload": payload or {},
            "timestamp": time.time()
        }
        try:
            if redis_manager.redis:
                await redis_manager.redis.publish("telegram:alerts", json.dumps(event))
            else:
                # Direct fallback
                await self.send_message(message)
        except Exception as e:
            logger.error(f"Failed to broadcast Telegram alert: {e}")

    # =========================================================================
    # Rich Alert Formatters
    # =========================================================================
    @staticmethod
    def format_trade_entry(trade: Dict[str, Any]) -> str:
        sym = trade.get("symbol", "UNKNOWN")
        side = trade.get("side", "BUY")
        px = trade.get("price", 0.0)
        qty = trade.get("quantity", 0.0)
        leverage = trade.get("leverage", 20)
        notional = float(px) * float(qty)
        conf = trade.get("confidence", 0.0)
        reason = trade.get("reason", "AI Institutional Alpha Signal")
        sl = trade.get("stop_loss", 0.0)
        tp = trade.get("take_profit", 0.0)
        
        emoji = "🟢" if "LONG" in side or side == "BUY" else "🔴"
        return (
            f"<b>{emoji} TRADE ENTRY (POST-ONLY MAKER): {sym}</b>\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"• <b>Action:</b> <code>{side}</code>\n"
            f"• <b>Entry Price:</b> <code>${px:,.4f}</code>\n"
            f"• <b>Quantity:</b> <code>{qty}</code> (${notional:,.2f})\n"
            f"• <b>Leverage:</b> <code>{leverage}x</code>\n"
            f"• <b>AI Conviction:</b> <code>{conf*100:.1f}%</code>\n"
            f"• <b>Catalyst:</b> <i>{reason}</i>\n"
            f"• <b>Stop-Loss:</b> <code>${sl:,.4f}</code>\n"
            f"• <b>Take-Profit:</b> <code>${tp:,.4f}</code>\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"🕒 <i>{time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}</i>"
        )

    @staticmethod
    def format_trade_exit(trade: Dict[str, Any]) -> str:
        sym = trade.get("symbol", "UNKNOWN")
        side = trade.get("side", "CLOSE")
        pnl = trade.get("realized_pnl", 0.0)
        roi = trade.get("roi_pct", 0.0)
        exit_px = trade.get("price", 0.0)
        reason = trade.get("reason", "Take-Profit / Trailing Stop")
        duration = trade.get("holding_minutes", 0.0)
        fees = trade.get("fees", 0.0)
        
        win = pnl >= 0
        emoji = "💰" if win else "🛑"
        pnl_str = f"+${pnl:,.2f}" if win else f"-${abs(pnl):,.2f}"
        roi_str = f"+{roi:.2f}%" if win else f"{roi:.2f}%"
        
        return (
            f"<b>{emoji} TRADE CLOSED: {sym}</b>\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"• <b>Exit Reason:</b> <code>{reason}</code>\n"
            f"• <b>Net Realized PnL:</b> <b>{pnl_str} ({roi_str})</b>\n"
            f"• <b>Exit Price:</b> <code>${exit_px:,.4f}</code>\n"
            f"• <b>Round-Trip Fees:</b> <code>${fees:,.2f}</code>\n"
            f"• <b>Hold Time:</b> <code>{duration:.1f} mins</code>\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"🕒 <i>{time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}</i>"
        )

    @staticmethod
    def format_flash_crash(crash_data: Dict[str, Any]) -> str:
        pct = crash_data.get("pct_drop", 0.0)
        peak = crash_data.get("peak_price", 0.0)
        cur = crash_data.get("current_price", 0.0)
        lockout = crash_data.get("lockout_minutes", 30)
        
        return (
            f"<b>🚨 MACRO FLASH-CRASH SHIELD ACTIVATED</b>\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"• <b>Bitcoin Crash:</b> <b>{pct*100:.2f}% drop</b> in < 5 mins!\n"
            f"• <b>BTC Peak:</b> <code>${peak:,.2f}</code> ➔ <b>Now:</b> <code>${cur:,.2f}</code>\n"
            f"• <b>Altcoin Defense:</b> <b>ALL New Altcoin Longs PAUSED for {lockout} mins</b>\n"
            f"• <b>Trailing Stops:</b> Ratcheted to 90% floor to bank open profits\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"🛡️ <i>Capital preservation protocol engaged.</i>"
        )

    @staticmethod
    def format_economic_shield(econ_data: Dict[str, Any]) -> str:
        event_name = econ_data.get("name", "High-Impact Economic Release")
        status = econ_data.get("status", "Active")
        time_desc = econ_data.get("time_desc", "")
        
        return (
            f"<b>🏛️ ECONOMIC EVENT SHIELD: {status}</b>\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"• <b>Event:</b> <b>{event_name}</b>\n"
            f"• <b>Release Window:</b> {time_desc}\n"
            f"• <b>Safety Action:</b> New order execution paused (±15m buffer)\n"
            f"• <b>Purpose:</b> Avoid 5x-10x spread blowout & wick stop-outs\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"🕒 <i>{time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}</i>"
        )

    @staticmethod
    def format_liquidation_cascade(liq_data: Dict[str, Any]) -> str:
        vol = liq_data.get("cascade_usd", 0.0)
        sym = liq_data.get("symbol", "GLOBAL")
        wick_px = liq_data.get("bottom_wick_price", 0.0)
        confidence = liq_data.get("confidence", 0.90)
        
        return (
            f"<b>🎯 LIQUIDATION CASCADE HUNTER FIRED</b>\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"• <b>Cascade Magnitude:</b> <b>${vol:,.0f} liquidated in < 3m</b>\n"
            f"• <b>Symbol:</b> <code>{sym}</code>\n"
            f"• <b>Exhaustion Wick Bottom:</b> <code>${wick_px:,.4f}</code>\n"
            f"• <b>Conviction:</b> <code>{confidence*100:.1f}%</code> (>80% win-rate historical)\n"
            f"• <b>Signal:</b> <b>Bottom-Wick Mean Reversion Long Sniped!</b>\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"🕒 <i>{time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}</i>"
        )

    @staticmethod
    def format_loss_velocity_breaker(data: Dict[str, Any]) -> str:
        cnt = data.get("loss_count", 3)
        window = data.get("window_seconds", 120)
        symbols = data.get("symbols", "PORTFOLIO")
        mins = data.get("cooldown_minutes", 15)
        return (
            f"<b>🚨 AIRBAG ACTIVATED: LOSS VELOCITY BREAKER</b>\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"• <b>Trigger:</b> <b>{cnt} stop-losses in {window}s</b> ({symbols})\n"
            f"• <b>Action:</b> <b>All new trade entries FROZEN for {mins} minutes</b>\n"
            f"• <b>Open Positions:</b> Active trailing stops remain monitored\n"
            f"• <b>Purpose:</b> Halt knife-catching during rapid market liquidation waterfall\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"🛡️ <i>Capital preservation protocol engaged.</i>"
        )

    @staticmethod
    def format_slippage_anomaly(data: Dict[str, Any]) -> str:
        sym = data.get("symbol", "UNKNOWN")
        slip = data.get("slippage_pct", 0.0)
        mid_px = data.get("mid_price", 0.0)
        fill_px = data.get("fill_price", 0.0)
        mins = data.get("quarantine_minutes", 15)
        return (
            f"<b>⚠️ AIRBAG ACTIVATED: SLIPPAGE ANOMALY GUARD</b>\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"• <b>Symbol:</b> <code>{sym}</code>\n"
            f"• <b>Slippage:</b> <b>{slip:.2f}%</b> (> 0.75% threshold!)\n"
            f"• <b>Mid Price:</b> <code>${mid_px:,.4f}</code> ➔ <b>Fill:</b> <code>${fill_px:,.4f}</code>\n"
            f"• <b>Action:</b> <b>{sym} quarantined for {mins} minutes</b>\n"
            f"• <b>Purpose:</b> Avoid repeatedly filling into thin/vacuous orderbooks\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"🕒 <i>{time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}</i>"
        )

    @staticmethod
    def format_margin_cushion_trim(data: Dict[str, Any]) -> str:
        sym = data.get("trimmed_symbol", "UNKNOWN")
        qty = data.get("trimmed_qty", 0.0)
        margin_pct = data.get("free_margin_ratio", 0.0)
        return (
            f"<b>🛡️ AIRBAG ACTIVATED: MARGIN CUSHION AUTO-TRIM</b>\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"• <b>Free Margin Ratio:</b> <b>{margin_pct:.1f}%</b> (< 15.0% critical threshold!)\n"
            f"• <b>Deleveraging Trim:</b> Market-closed 25% of {sym} (Qty: {qty})\n"
            f"• <b>Purpose:</b> Restored margin buffer to prevent Binance auto-liquidation\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"🕒 <i>{time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}</i>"
        )

    @staticmethod
    def format_heartbeat_stall(data: Dict[str, Any]) -> str:
        dur = data.get("stall_duration", 0.0)
        cancelled = data.get("cancelled_count", 0)
        return (
            f"<b>🔌 AIRBAG ACTIVATED: WEBSOCKET HEARTBEAT WATCHDOG</b>\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"• <b>Tick Stall Duration:</b> <b>{dur:.1f}s without tick</b> (> 5.0s threshold)\n"
            f"• <b>Safety Action:</b> Cancelled {cancelled} resting limit orders on Binance\n"
            f"• <b>Purpose:</b> Prevent stale orderbook fills during exchange connection gap\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"🕒 <i>{time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}</i>"
        )

    @staticmethod
    def format_beta_hedge(data: Dict[str, Any]) -> str:
        action = data.get("action_type", "REBALANCE_HEDGE")
        side = data.get("side", "SELL")
        sym = data.get("symbol", "BTCUSDT")
        qty = data.get("quantity", 0.0)
        px = data.get("price", 0.0)
        notional = data.get("notional_usd", px * qty)
        alt_delta = data.get("net_altcoin_delta_usd", 0.0)
        target_hedge = data.get("target_btc_hedge_usd", 0.0)
        
        side_desc = "SHORT (SELL)" if side == "SELL" else "BUY (COVER / LONG)"
        return (
            f"<b>🛡️ MARKET-BETA DELTA HEDGE EXECUTED</b>\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"• <b>Hedge Action:</b> <code>{action}</code>\n"
            f"• <b>Instrument:</b> <code>{sym}</code> ({side_desc})\n"
            f"• <b>Quantity:</b> <code>{qty} {sym.replace('USDT', '')}</code> (${notional:,.2f})\n"
            f"• <b>BTC Execution Price:</b> <code>${px:,.2f}</code>\n"
            f"• <b>Net Altcoin Beta Delta:</b> <code>${alt_delta:+,.2f} USD</code>\n"
            f"• <b>Target Protected Hedge:</b> <code>${target_hedge:+,.2f} USD</code>\n"
            f"• <b>Portfolio Status:</b> <b>🟢 DELTA-NEUTRAL (Protected)</b>\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"💡 <i>Broad market risk neutralized. Altcoin alpha isolated!</i>\n"
            f"🕒 <i>{time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}</i>"
        )

    @staticmethod
    def format_order_timeout(data: Dict[str, Any]) -> str:
        sym = data.get("symbol", "UNKNOWN")
        timeout = data.get("timeout_seconds", 30.0)
        reason = data.get("reason", "Order timed out unfilled. Cancelled to protect capital.")
        return (
            f"<b>⏱️ LIMIT ORDER TIMED OUT: {sym}</b>\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"• <b>Action:</b> <b>Unfilled Order Cancelled</b>\n"
            f"• <b>Timeout:</b> <code>{timeout:.0f} seconds</code>\n"
            f"• <b>Protection:</b> <i>{reason}</i>\n"
            f"• <b>Portfolio:</b> <b>No position opened. Margin restored.</b>\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"🕒 <i>{time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}</i>"
        )

# Global Singleton Instance
telegram_notifier = TelegramNotifier()

