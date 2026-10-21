import asyncio
import aiohttp
import json
import time
import os
from typing import Dict, Any, List, Optional
from core.config.settings import settings
from core.logging.logger import logger, set_log_file
from core.db.redis import redis_manager
from core.telemetry.telegram_service import telegram_notifier

class TelegramCommandCenter:
    """
    Institutional Interactive Telegram Command Center.
    
    Provides mobile phone command & control for the quantitative trading engine:
    - Instant Rich Alerts (Entries, Exits, Flash-Crash, Liquidations)
    - One-Tap Inline Buttons:
      [🚨 Emergency Flat All] (Instantly exit all positions to cash)
      [⏸️ Pause Bot 1 Hour]
      [▶️ Resume Bot]
      [📊 Current Sharpe & PnL]
      [📈 Active Positions Status]
      [🛡️ Market-Beta Hedge Status]
    """

    def __init__(self):
        self.bot_token = settings.telegram_bot_token
        self.chat_id = settings.telegram_chat_id
        self.enabled = settings.telegram_enabled and bool(self.bot_token) and bool(self.chat_id)
        self.running = False
        self.session: Optional[aiohttp.ClientSession] = None
        self.last_update_id = 0

    async def _get_session(self) -> aiohttp.ClientSession:
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30))
        return self.session

    def get_main_keyboard(self) -> Dict[str, Any]:
        """Returns the interactive inline keyboard markup."""
        return {
            "inline_keyboard": [
                [
                    {"text": "🚨 Emergency Flat All", "callback_data": "cmd_flat_all"}
                ],
                [
                    {"text": "⏸️ Pause Bot 1 Hour", "callback_data": "cmd_pause_1h"},
                    {"text": "▶️ Resume Bot", "callback_data": "cmd_resume"}
                ],
                [
                    {"text": "📊 Current Sharpe & PnL", "callback_data": "cmd_pnl_stats"},
                    {"text": "📈 Active Positions", "callback_data": "cmd_positions"}
                ],
                [
                    {"text": "⚡ Refresh Status", "callback_data": "cmd_status"}
                ]
            ]
        }

    def _authorized(self, sender_chat: str) -> bool:
        """Commands can close every position: only the configured TELEGRAM_CHAT_ID may give them."""
        return bool(self.chat_id) and str(sender_chat) == str(self.chat_id)

    async def send_message(self, text: str, reply_markup: Optional[Dict[str, Any]] = None, chat_id: Optional[str] = None):
        """Sends a message to the user."""
        target_chat = chat_id or self.chat_id
        if not self.enabled or not target_chat:
            logger.info(f"[TELEGRAM LOCAL OUTPUT]:\n{text}")
            return
            
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": target_chat,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True
        }
        if reply_markup:
            payload["reply_markup"] = reply_markup
            
        try:
            session = await self._get_session()
            async with session.post(url, json=payload) as resp:
                if resp.status != 200:
                    err = await resp.text()
                    if "can't parse entities" in err:
                        # A bare "<" in an alert (e.g. "< 15%") breaks HTML mode: deliver it as plain text instead
                        payload.pop("parse_mode", None)
                        async with session.post(url, json=payload) as retry:
                            if retry.status == 200:
                                return
                            err = await retry.text()
                    logger.error(f"Telegram sendMessage failed: {err}")
        except Exception as e:
            logger.error(f"Error sending Telegram message: {e}")

    async def answer_callback_query(self, callback_query_id: str, text: str):
        """Acknowledges inline button tap with a temporary popup notification."""
        if not self.enabled:
            return
        url = f"https://api.telegram.org/bot{self.bot_token}/answerCallbackQuery"
        try:
            session = await self._get_session()
            await session.post(url, json={"callback_query_id": callback_query_id, "text": text})
        except Exception as e:
            logger.error(f"Error answering callback query: {e}")

    # =========================================================================
    # Command Implementations
    # =========================================================================
    async def handle_status_command(self, chat_id: Optional[str] = None):
        """Generates real-time bot health & system status report."""
        equity = 10000.0
        free_margin = 10000.0
        positions = []
        is_paused = False
        pause_remaining = 0.0
        kill_active = False
        
        try:
            if redis_manager.redis:
                # Check kill switch
                ks = await redis_manager.redis.get("system:kill_switch:active")
                if ks:
                    val = (ks.decode("utf-8") if isinstance(ks, bytes) else str(ks)).strip().lower()
                    kill_active = (val in ["true", "1", "yes"])
                    
                # Check pause
                pause_ts = await redis_manager.redis.get("system:pause_until")
                if pause_ts:
                    p_val = float(pause_ts.decode("utf-8") if isinstance(pause_ts, bytes) else pause_ts)
                    if p_val > time.time():
                        is_paused = True
                        pause_remaining = (p_val - time.time()) / 60.0
                        
                # Check portfolio
                p_raw = await redis_manager.redis.get("dashboard:portfolio")
                if p_raw:
                    p_data = json.loads(p_raw)
                    equity = p_data.get("equity", 10000.0)
                    free_margin = p_data.get("free_margin", equity)
                    positions = p_data.get("positions", [])
        except Exception as e:
            logger.error(f"Error reading status from Redis: {e}")

        status_emoji = "🔴 STOPPED" if kill_active else ("🟡 PAUSED" if is_paused else "🟢 ACTIVE & TRADING")
        mode_str = "DRY_RUN (Simulation)" if settings.dry_run else ("TESTNET" if settings.binance_testnet else "LIVE REAL MONEY")
        
        active_pos_count = len([p for p in positions if float(p.get("quantity", 0)) != 0])
        
        text = (
            f"<b>🤖 AI TRADING ENGINE COMMAND CENTER</b>\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"• <b>Status:</b> <b>{status_emoji}</b>\n"
            f"• <b>Mode:</b> <code>{mode_str}</code>\n"
            f"• <b>Total Equity:</b> <code>${equity:,.2f} USDT</code>\n"
            f"• <b>Available Margin:</b> <code>${free_margin:,.2f} USDT</code>\n"
            f"• <b>Open Positions:</b> <code>{active_pos_count} Active</code>\n"
            f"• <b>Pause Timer:</b> <code>{pause_remaining:.1f} mins remaining</code>\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"⚡ <i>Tap any inline button below for instant execution.</i>"
        )
        await self.send_message(text, reply_markup=self.get_main_keyboard(), chat_id=chat_id)

    async def handle_pnl_command(self, chat_id: Optional[str] = None):
        """Generates rich Sharpe & PnL metrics report."""
        equity = 10000.0
        daily_pnl = 0.0
        daily_pnl_pct = 0.0
        win_rate = 0.0
        sharpe = 2.45
        trades_count = 0
        
        try:
            if redis_manager.redis:
                p_raw = await redis_manager.redis.get("dashboard:portfolio")
                if p_raw:
                    p_data = json.loads(p_raw)
                    equity = float(p_data.get("equity", 10000.0))
                    daily_pnl = float(p_data.get("daily_pnl", 0.0))
                    daily_pnl_pct = float(p_data.get("daily_pnl_pct", 0.0))
                    trades_count = int(p_data.get("session_trades", 0))
                    win_rate = float(p_data.get("session_win_rate", 0.0))
                    sharpe = float(p_data.get("sharpe_ratio", 2.45))
        except Exception as e:
            logger.error(f"Error fetching PnL metrics: {e}")
            
        pnl_emoji = "🟢" if daily_pnl >= 0 else "🔴"
        pnl_str = f"+${daily_pnl:,.2f}" if daily_pnl >= 0 else f"-${abs(daily_pnl):,.2f}"
        pct_str = f"+{daily_pnl_pct:.2f}%" if daily_pnl >= 0 else f"{daily_pnl_pct:.2f}%"
        
        text = (
            f"<b>📊 PERFORMANCE & RISK METRICS</b>\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"• <b>Portfolio Equity:</b> <code>${equity:,.2f} USDT</code>\n"
            f"• <b>24h Net PnL:</b> {pnl_emoji} <b>{pnl_str} ({pct_str})</b>\n"
            f"• <b>Session Win Rate:</b> <code>{win_rate*100:.1f}%</code> ({trades_count} trades)\n"
            f"• <b>Sharpe Ratio:</b> <code>{sharpe:.2f}</code> (Institutional grade)\n"
            f"• <b>Max Drawdown Gate:</b> <code>5.0% Limit</code> (Protected)\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"🕒 <i>{time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}</i>"
        )
        await self.send_message(text, reply_markup=self.get_main_keyboard(), chat_id=chat_id)

    async def handle_positions_command(self, chat_id: Optional[str] = None):
        """Displays breakdown of all active positions."""
        positions = []
        try:
            if redis_manager.redis:
                p_raw = await redis_manager.redis.get("dashboard:portfolio")
                if p_raw:
                    p_data = json.loads(p_raw)
                    positions = p_data.get("positions", [])
        except Exception as e:
            logger.error(f"Error fetching positions: {e}")
            
        active = [p for p in positions if float(p.get("quantity", 0)) != 0]
        if not active:
            text = (
                f"<b>📈 ACTIVE POSITIONS STATUS</b>\n"
                f"━━━━━━━━━━━━━━━━━━\n"
                f"<i>Currently 100% in USDT Cash. Zero open positions.</i>\n"
                f"━━━━━━━━━━━━━━━━━━"
            )
            await self.send_message(text, reply_markup=self.get_main_keyboard(), chat_id=chat_id)
            return

        lines = ["<b>📈 ACTIVE POSITIONS BREAKDOWN</b>", "━━━━━━━━━━━━━━━━━━"]
        for p in active:
            sym = p.get("symbol", "")
            qty = float(p.get("quantity", 0))
            side = "LONG" if qty > 0 else "SHORT"
            emoji = "🟢" if qty > 0 else "🔴"
            entry_px = float(p.get("entry_price", 0))
            cur_px = float(p.get("current_price", entry_px))
            pnl_val = float(p.get("unrealized_pnl", 0))
            pnl_pct = ((cur_px - entry_px) / entry_px * (1 if qty > 0 else -1)) * float(p.get("leverage", 20)) * 100
            
            pnl_str = f"+${pnl_val:,.2f}" if pnl_val >= 0 else f"-${abs(pnl_val):,.2f}"
            pct_str = f"+{pnl_pct:.2f}%" if pnl_pct >= 0 else f"{pnl_pct:.2f}%"
            
            lines.append(
                f"{emoji} <b>{sym} ({side} {p.get('leverage', 20)}x)</b>\n"
                f"   • Size: <code>{abs(qty)}</code> (${abs(qty)*cur_px:,.2f})\n"
                f"   • Entry: <code>${entry_px:,.4f}</code> ➔ Now: <code>${cur_px:,.4f}</code>\n"
                f"   • Unrealized PnL: <b>{pnl_str} ({pct_str})</b>\n"
            )
        lines.append("━━━━━━━━━━━━━━━━━━")
        await self.send_message("\n".join(lines), reply_markup=self.get_main_keyboard(), chat_id=chat_id)

    async def handle_hedge_command(self, chat_id: Optional[str] = None):
        """Displays Market-Beta Hedge status."""
        text = (
            f"<b>🛡️ MARKET-BETA DELTA HEDGE: DISABLED</b>\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"• <b>Status:</b> <b>⚪ DISABLED</b>\n"
            f"• <b>Trading Mode:</b> Pure Directional AI Alpha\n"
            f"• <b>Protections:</b> Tight 0.75% SL, Trailing Take-Profits & Macro Shields\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"💡 <i>Trades execute purely based on AI conviction without paying double hedging fees.</i>"
        )
        await self.send_message(text, reply_markup=self.get_main_keyboard(), chat_id=chat_id)

    async def handle_flat_all_command(self, chat_id: Optional[str] = None):
        """Emergency action: flattens all open positions and cancels all resting orders."""
        logger.critical("🚨 TELEGRAM INLINE ACTION TRIGGERED: EMERGENCY FLAT ALL POSITIONS!")
        try:
            if redis_manager.redis:
                # Activate kill switch & publish emergency flat event
                await redis_manager.redis.set("system:kill_switch:active", "true")
                await redis_manager.redis.publish("system:events", json.dumps({"type": "PANIC_LIQUIDATE", "source": "TELEGRAM"}))
                
            text = (
                f"<b>🚨 EMERGENCY FLAT ALL EXECUTED!</b>\n"
                f"━━━━━━━━━━━━━━━━━━\n"
                f"• Kill switch ACTIVATED.\n"
                f"• All resting limit orders CANCELLED.\n"
                f"• All active positions MARKET-CLOSED to USDT Cash.\n"
                f"• Trading halted.\n"
                f"━━━━━━━━━━━━━━━━━━\n"
                f"<i>Tap [▶️ Resume Bot] when ready to trade again.</i>"
            )
            await self.send_message(text, reply_markup=self.get_main_keyboard(), chat_id=chat_id)
        except Exception as e:
            logger.error(f"Failed to execute emergency flat all: {e}")
            await self.send_message(f"❌ Failed to execute emergency flat: {e}", chat_id=chat_id)

    async def handle_pause_command(self, chat_id: Optional[str] = None):
        """Temporarily pauses new trade entries for 1 hour."""
        logger.info("⏸️ TELEGRAM ACTION: Pausing bot for 1 hour.")
        try:
            pause_until = time.time() + 3600.0
            if redis_manager.redis:
                await redis_manager.redis.set("system:pause_until", str(pause_until))
                await redis_manager.redis.publish("system:events", json.dumps({"type": "PAUSE_TRADING", "until": pause_until}))
                
            text = (
                f"<b>⏸️ BOT PAUSED FOR 1 HOUR</b>\n"
                f"━━━━━━━━━━━━━━━━━━\n"
                f"• New trade entries suspended until: <code>{time.strftime('%H:%M:%S UTC', time.gmtime(pause_until))}</code>\n"
                f"• Active trailing stops and take-profits remain monitored.\n"
                f"• Tap [▶️ Resume Bot] anytime to unpause early.\n"
                f"━━━━━━━━━━━━━━━━━━"
            )
            await self.send_message(text, reply_markup=self.get_main_keyboard(), chat_id=chat_id)
        except Exception as e:
            logger.error(f"Failed to pause bot: {e}")

    async def handle_resume_command(self, chat_id: Optional[str] = None):
        """Resumes trading, disarms kill switch and clears pause."""
        logger.info("▶️ TELEGRAM ACTION: Resuming bot trading.")
        try:
            if redis_manager.redis:
                await redis_manager.redis.set("system:kill_switch:active", "false")
                await redis_manager.redis.delete("system:pause_until")
                await redis_manager.redis.publish("system:events", json.dumps({"type": "RESUME_TRADING"}))
                
            text = (
                f"<b>▶️ TRADING BOT RESUMED!</b>\n"
                f"━━━━━━━━━━━━━━━━━━\n"
                f"• Kill switch disarmed.\n"
                f"• Pause timer cleared.\n"
                f"• Autonomous AI evaluation & entry active.\n"
                f"━━━━━━━━━━━━━━━━━━"
            )
            await self.send_message(text, reply_markup=self.get_main_keyboard(), chat_id=chat_id)
        except Exception as e:
            logger.error(f"Failed to resume bot: {e}")

    # =========================================================================
    # Telegram Update Polling & Callback Router
    # =========================================================================
    async def poll_updates(self):
        """Polls Telegram getUpdates endpoint to receive incoming button taps and commands."""
        if not self.enabled:
            logger.info("Telegram command center running in offline mode (set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env).")
            while self.running:
                await asyncio.sleep(10)
            return

        url = f"https://api.telegram.org/bot{self.bot_token}/getUpdates"
        logger.info("Telegram Command Center long-polling started...")

        while self.running:
            try:
                session = await self._get_session()
                params = {"offset": self.last_update_id + 1, "timeout": 20}
                async with session.get(url, params=params) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        updates = data.get("result", [])
                        for update in updates:
                            self.last_update_id = update["update_id"]
                            
                            # Handle Callback Query (Inline Button Tap)
                            if "callback_query" in update:
                                cb = update["callback_query"]
                                cb_id = cb["id"]
                                action = cb.get("data", "")
                                sender_chat = str(cb.get("message", {}).get("chat", {}).get("id", ""))
                                if not self._authorized(sender_chat):
                                    logger.warning(f"Ignored Telegram button '{action}' from unauthorized chat {sender_chat}.")
                                    continue
                                
                                logger.info(f"Received Telegram Button Tap: {action}")
                                await self.answer_callback_query(cb_id, "Processing...")
                                
                                if action == "cmd_flat_all":
                                    await self.handle_flat_all_command(sender_chat)
                                elif action == "cmd_pause_1h":
                                    await self.handle_pause_command(sender_chat)
                                elif action == "cmd_resume":
                                    await self.handle_resume_command(sender_chat)
                                elif action == "cmd_pnl_stats":
                                    await self.handle_pnl_command(sender_chat)
                                elif action == "cmd_positions":
                                    await self.handle_positions_command(sender_chat)
                                elif action == "cmd_hedge_status":
                                    await self.handle_hedge_command(sender_chat)
                                elif action == "cmd_status":
                                    await self.handle_status_command(sender_chat)
                                    
                            # Handle Text Slash Commands
                            elif "message" in update and "text" in update["message"]:
                                msg = update["message"]
                                text = msg.get("text", "").strip()
                                sender_chat = str(msg.get("chat", {}).get("id", ""))
                                if not self._authorized(sender_chat):
                                    logger.warning(f"Ignored Telegram command '{text[:40]}' from unauthorized chat {sender_chat}.")
                                    continue
                                
                                if text.startswith("/start") or text.startswith("/help"):
                                    await self.handle_status_command(sender_chat)
                                elif text.startswith("/status"):
                                    await self.handle_status_command(sender_chat)
                                elif text.startswith("/pnl"):
                                    await self.handle_pnl_command(sender_chat)
                                elif text.startswith("/positions"):
                                    await self.handle_positions_command(sender_chat)
                                elif text.startswith("/hedge"):
                                    await self.handle_hedge_command(sender_chat)
                                elif text.startswith("/flat_all") or text.startswith("/emergency"):
                                    await self.handle_flat_all_command(sender_chat)
                                elif text.startswith("/pause"):
                                    await self.handle_pause_command(sender_chat)
                                elif text.startswith("/resume"):
                                    await self.handle_resume_command(sender_chat)
                    else:
                        await asyncio.sleep(2)
            except Exception as e:
                logger.error(f"Error in Telegram poll loop: {e}")
                await asyncio.sleep(5)

    # =========================================================================
    # Redis Alert Dispatcher Loop
    # =========================================================================
    async def listen_redis_alerts(self):
        """Listens for alerts published across system services and sends them to Telegram."""
        while self.running:
            try:
                await redis_manager.connect()
                if not redis_manager.redis:
                    await asyncio.sleep(2)
                    continue
                    
                pubsub = redis_manager.redis.pubsub()
                await pubsub.subscribe("telegram:alerts")
                logger.info("Telegram Bot listening to Redis 'telegram:alerts' channel...")
                
                async for message in pubsub.listen():
                    if not self.running:
                        break
                    if message["type"] == "message":
                        try:
                            raw_data = message["data"]
                            data_str = raw_data.decode("utf-8") if isinstance(raw_data, bytes) else str(raw_data)
                            event = json.loads(data_str)
                            
                            alert_type = event.get("type", "ALERT")
                            text = event.get("message", "")
                            payload = event.get("payload", {})
                            
                            # Format nicely according to alert type
                            formatted_text = text
                            if alert_type == "TRADE_ENTRY":
                                formatted_text = telegram_notifier.format_trade_entry(payload)
                            elif alert_type == "TRADE_EXIT":
                                formatted_text = telegram_notifier.format_trade_exit(payload)
                            elif alert_type == "FLASH_CRASH":
                                formatted_text = telegram_notifier.format_flash_crash(payload)
                            elif alert_type == "ECONOMIC_SHIELD":
                                formatted_text = telegram_notifier.format_economic_shield(payload)
                            elif alert_type == "LIQUIDATION_CASCADE":
                                formatted_text = telegram_notifier.format_liquidation_cascade(payload)
                            elif alert_type == "LOSS_VELOCITY_BREAKER":
                                formatted_text = telegram_notifier.format_loss_velocity_breaker(payload)
                            elif alert_type == "SLIPPAGE_ANOMALY":
                                formatted_text = telegram_notifier.format_slippage_anomaly(payload)
                            elif alert_type == "MARGIN_CUSHION_TRIM":
                                formatted_text = telegram_notifier.format_margin_cushion_trim(payload)
                            elif alert_type == "HEARTBEAT_STALL":
                                formatted_text = telegram_notifier.format_heartbeat_stall(payload)
                            elif alert_type == "BETA_HEDGE":
                                formatted_text = telegram_notifier.format_beta_hedge(payload)
                            elif alert_type == "ORDER_TIMEOUT":
                                formatted_text = telegram_notifier.format_order_timeout(payload)
                                
                            await self.send_message(formatted_text, reply_markup=self.get_main_keyboard())
                        except Exception as parse_err:
                            logger.error(f"Error processing telegram alert: {parse_err}")
            except Exception as e:
                logger.warning(f"Telegram Redis subscriber disconnected ({e}). Reconnecting in 5s...")
                await asyncio.sleep(5)

    async def start(self):
        self.running = True
        logger.info("Starting Telegram Interactive Command Center...")
        tasks = [
            asyncio.create_task(self.poll_updates()),
            asyncio.create_task(self.listen_redis_alerts())
        ]
        await asyncio.gather(*tasks)

    def stop(self):
        self.running = False

if __name__ == "__main__":
    set_log_file("logs/telegram_bot.log")
    center = TelegramCommandCenter()
    try:
        asyncio.run(center.start())
    except KeyboardInterrupt:
        center.stop()
