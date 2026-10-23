import time
import collections
import json
import datetime
from typing import Tuple, Dict, Any, List, Optional
from core.config.settings import settings
from core.logging.logger import logger
from core.db.redis import redis_manager

class MacroCrashShield:
    """
    Institutional Macro & Flash-Crash Shield:
    
    1. BTC Sympathy Crash Guard:
       - Monitors BTC rolling 5-minute return.
       - If BTC drops >= 2.5% in a 5-minute candle/window:
         * Automatically pauses new Altcoin Longs for 30 minutes.
         * Tightens trailing stop retention floors on all active positions to protect against 8%-12% altcoin sympathy dumps.
       
    2. Economic Event Shield:
       - Tracks US CPI, FOMC Rate Decisions, FOMC Press Conferences, Non-Farm Payrolls (NFP), and Core PCE.
       - Spreads on crypto perpetuals widen 5x to 10x during major economic releases.
       - Automatically pauses execution 15 minutes BEFORE and 15 minutes AFTER high-impact releases.
    """

    # Institutional Economic Calendar: High-Impact US Releases (Timestamps in UTC)
    # Includes standard recurring schedules (CPI, FOMC, NFP, PCE)
    KNOWN_EVENTS = [
        # Format: (name, date_str, time_utc_str, impact)
        # 2026 Economic Releases
        ("US Non-Farm Payrolls (NFP)", "2026-10-02", "12:30", "CRITICAL"),
        ("US CPI Inflation", "2026-10-14", "12:30", "CRITICAL"),
        ("US FOMC Rate Decision", "2026-10-28", "18:00", "CRITICAL"),
        ("FOMC Press Conference (Powell)", "2026-10-28", "18:30", "CRITICAL"),
        ("US Non-Farm Payrolls (NFP)", "2026-11-06", "13:30", "CRITICAL"),
        ("US CPI Inflation", "2026-11-12", "13:30", "CRITICAL"),
        ("US Non-Farm Payrolls (NFP)", "2026-12-04", "13:30", "CRITICAL"),
        ("US FOMC Rate Decision", "2026-12-09", "19:00", "CRITICAL"),
        ("FOMC Press Conference (Powell)", "2026-12-09", "19:30", "CRITICAL"),
        ("US CPI Inflation", "2026-12-15", "13:30", "CRITICAL"),
        # 2027 Economic Releases
        ("US Non-Farm Payrolls (NFP)", "2027-01-08", "13:30", "CRITICAL"),
        ("US CPI Inflation", "2027-01-13", "13:30", "CRITICAL"),
        ("US FOMC Rate Decision", "2027-01-27", "19:00", "CRITICAL"),
        ("FOMC Press Conference (Powell)", "2027-01-27", "19:30", "CRITICAL"),
        ("US Non-Farm Payrolls (NFP)", "2027-02-05", "13:30", "CRITICAL"),
        ("US CPI Inflation", "2027-02-12", "13:30", "CRITICAL"),
        ("US FOMC Rate Decision", "2027-03-17", "18:00", "CRITICAL"),
        ("FOMC Press Conference (Powell)", "2027-03-17", "18:30", "CRITICAL"),
    ]

    def __init__(self):
        # BTC Price Rolling Window (5 minutes = 300 seconds)
        self.btc_price_history = collections.deque(maxlen=600)  # (timestamp, price)
        
        # Crash Guard State
        self.flash_crash_active = False
        self.altcoin_lockout_until = 0.0
        self.last_crash_drop_pct = 0.0
        self.last_crash_trigger_time = 0.0
        
        # Economic Shield State
        self.parsed_events: List[Dict[str, Any]] = []
        self._init_calendar()
        self.last_economic_alert_sent = 0.0

    def _init_calendar(self):
        """Parses calendar into UTC epoch timestamps."""
        self.parsed_events = []
        for name, date_str, time_str, impact in self.KNOWN_EVENTS:
            try:
                dt_str = f"{date_str} {time_str}"
                dt = datetime.datetime.strptime(dt_str, "%Y-%m-%d %H:%M").replace(tzinfo=datetime.timezone.utc)
                ts = dt.timestamp()
                self.parsed_events.append({
                    "name": name,
                    "timestamp": ts,
                    "date_str": date_str,
                    "time_str": time_str,
                    "impact": impact
                })
            except Exception as e:
                logger.error(f"Error parsing economic event {name}: {e}")

    def add_custom_event(self, name: str, timestamp: float, impact: str = "HIGH"):
        """Allows dynamic injection of newly scheduled central bank speeches or unexpected events."""
        self.parsed_events.append({
            "name": name,
            "timestamp": timestamp,
            "date_str": datetime.datetime.utcfromtimestamp(timestamp).strftime("%Y-%m-%d"),
            "time_str": datetime.datetime.utcfromtimestamp(timestamp).strftime("%H:%M UTC"),
            "impact": impact
        })

    def add_feed_events(self, events) -> int:
        """
        Adds releases from the live economic calendar ([[epoch, importance], ...] published by the market
        collector, apps/market_collector/global_context.py). The built-in KNOWN_EVENTS list is hand-written
        (it missed releases and ends in March 2027); one already listed within 30 minutes is not added twice.
        """
        added = 0
        for ts, importance in events or []:
            ts = float(ts)
            if any(abs(e["timestamp"] - ts) < 1800 for e in self.parsed_events):
                continue
            dt = datetime.datetime.fromtimestamp(ts, datetime.timezone.utc)
            self.parsed_events.append({
                "name": "US Fed decision (calendar feed)" if float(importance) >= 1.0 else "US CPI / jobs / PCE release (calendar feed)",
                "timestamp": ts,
                "date_str": dt.strftime("%Y-%m-%d"),
                "time_str": dt.strftime("%H:%M UTC"),
                "impact": "CRITICAL",
            })
            added += 1
        if added:
            cutoff = time.time() - 7 * 86400
            self.parsed_events = [e for e in self.parsed_events if e["timestamp"] >= cutoff]
        return added

    def update_btc_price(self, price: float, timestamp: Optional[float] = None) -> Tuple[bool, float]:
        """
        Ingests BTC price tick and evaluates 5-minute sympathy crash criteria.
        Returns: (crash_triggered_this_tick, max_drop_pct)
        """
        if price <= 0:
            return False, 0.0
            
        now = timestamp or time.time()
        self.btc_price_history.append((now, price))
        
        # Prune prices older than 5 minutes (300 seconds)
        while self.btc_price_history and (now - self.btc_price_history[0][0]) > 300.0:
            self.btc_price_history.popleft()
            
        if len(self.btc_price_history) < 10:
            return False, 0.0
            
        # Find peak price in the 5m rolling window
        prices = [p for t, p in self.btc_price_history]
        peak_5m = max(prices)
        
        if peak_5m <= 0:
            return False, 0.0
            
        pct_drop = (price - peak_5m) / peak_5m
        threshold = settings.macro_flash_crash_threshold_pct # default 0.025 (2.5%)
        
        crash_triggered_now = False
        if pct_drop <= -threshold:
            lockout_duration = settings.macro_flash_crash_lockout_seconds # default 1800s (30m)
            self.flash_crash_active = True
            self.altcoin_lockout_until = max(self.altcoin_lockout_until, now + lockout_duration)
            self.last_crash_drop_pct = pct_drop
            self.last_crash_trigger_time = now
            crash_triggered_now = True
            
            logger.critical(
                f"🚨 BTC FLASH-CRASH GUARD TRIGGERED: BTC dumped {pct_drop*100:.2f}% in 5m! "
                f"Peak: ${peak_5m:,.2f} -> Current: ${price:,.2f}. "
                f"Pausing Altcoin Longs & tightening trailing stops for {lockout_duration/60:.0f} mins!"
            )
            
        # Check if previous crash lockout has expired
        if self.flash_crash_active and now >= self.altcoin_lockout_until:
            self.flash_crash_active = False
            logger.info("✅ BTC Flash-Crash Guard expired. Normal altcoin trading resumed.")
            
        return crash_triggered_now, pct_drop

    def is_altcoin_long_allowed(self, symbol: str) -> Tuple[bool, str]:
        """
        Evaluates whether an Altcoin Long entry is permitted under Macro Crash Shield rules.
        BTC and ETH are bellwethers; all other symbols are considered Altcoins.
        """
        now = time.time()
        
        # Bellwethers can still be bought if their own models approve
        if symbol in ["BTCUSDT", "ETHUSDT"]:
            return True, "Bellwether symbol"
            
        if self.flash_crash_active and now < self.altcoin_lockout_until:
            rem_mins = (self.altcoin_lockout_until - now) / 60.0
            return (
                False,
                f"BTC_SYMPATHY_CRASH_GUARD: BTC dumped {self.last_crash_drop_pct*100:.2f}% in 5m. "
                f"Altcoin Longs locked out for {rem_mins:.1f} more mins."
            )
            
        return True, "Macro crash shield clear"

    def get_trailing_retention_override(self) -> Optional[float]:
        """
        When crash guard is active, ratchets the trailing stop retention floor
        (e.g. to 0.90 = retain 90% of peak profit) to bank gains before altcoins dump 8-12%.
        """
        now = time.time()
        if self.flash_crash_active and now < self.altcoin_lockout_until:
            return settings.macro_flash_crash_trailing_retention
        return None

    def is_economic_freeze_active(self, buffer_minutes: Optional[float] = None) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """
        Checks if current time is within +/- buffer_minutes of a high-impact US economic release
        (e.g. CPI, FOMC, NFP).
        """
        if not settings.economic_shield_enabled:
            return False, "Economic shield disabled", None
            
        now = time.time()
        buf_seconds = (buffer_minutes or settings.economic_shield_buffer_minutes) * 60.0
        
        for event in self.parsed_events:
            event_ts = event["timestamp"]
            diff = event_ts - now  # positive = upcoming, negative = just passed
            
            # If within 15 mins before or 15 mins after
            if -buf_seconds <= diff <= buf_seconds:
                if diff > 0:
                    status_desc = f"releasing in {diff/60.0:.1f} mins"
                else:
                    status_desc = f"released {abs(diff)/60.0:.1f} mins ago"
                    
                reason = (
                    f"ECONOMIC_EVENT_SHIELD: {event['name']} ({event['date_str']} {event['time_str']}) {status_desc}. "
                    f"Execution paused to avoid violent spread blowout."
                )
                return True, reason, event
                
        return False, "No imminent economic events", None

    def get_next_economic_event(self) -> Optional[Dict[str, Any]]:
        """Returns the next upcoming economic event in the calendar."""
        now = time.time()
        upcoming = [e for e in self.parsed_events if e["timestamp"] > now]
        if not upcoming:
            return None
        upcoming.sort(key=lambda x: x["timestamp"])
        next_event = dict(upcoming[0])
        next_event["seconds_until"] = next_event["timestamp"] - now
        next_event["minutes_until"] = (next_event["timestamp"] - now) / 60.0
        next_event["hours_until"] = (next_event["timestamp"] - now) / 3600.0
        return next_event

    def get_status(self) -> Dict[str, Any]:
        """Returns complete serializable status dictionary for Dashboard and Telegram."""
        now = time.time()
        is_econ, econ_reason, current_event = self.is_economic_freeze_active()
        next_event = self.get_next_economic_event()
        
        return {
            "flash_crash_active": self.flash_crash_active and now < self.altcoin_lockout_until,
            "lockout_remaining_seconds": max(0.0, self.altcoin_lockout_until - now),
            "last_crash_drop_pct": self.last_crash_drop_pct,
            "economic_shield_active": is_econ,
            "economic_shield_reason": econ_reason if is_econ else None,
            "current_economic_event": current_event,
            "next_economic_event": next_event
        }

# Global Singleton Instance
macro_crash_shield = MacroCrashShield()
