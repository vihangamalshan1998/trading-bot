#!/usr/bin/env python3
"""
scripts/auto_watchdog.py - Production Service Health Watchdog & Log Janitor
-------------------------------------------------------------------------
1. Actively monitors heartbeats of all trading bot microservices in Redis:
   - whale-tracker (whale:heartbeat)
   - statarb-collector (statarb:heartbeat)
   - market-collector (market:state:BTCUSDT)
   - trading-bot (ai:state:BTCUSDT)
2. Automatically issues `pm2 restart <service>` if a service freezes or stops ticking for > 90s.
   (Includes 120s anti-churn cooldown per service).
3. Log Janitor:
   - Scans `logs/` and PM2 log directories every 10 minutes.
   - Deletes stale archived log files (> 2 days old).
   - Truncates oversized active logs (> 15MB) to the last 2,000 lines.
   - Flushes PM2 logs if disk usage exceeds 50MB.
"""

import sys
import os
import time
import glob
import json
import asyncio
import subprocess
import argparse

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.db.redis import redis_manager
from core.logging.logger import logger, clean_old_logs

SERVICES_TO_MONITOR = {
    "whale-tracker": {
        "heartbeat_key": "whale:heartbeat",
        "max_silent_seconds": 90.0,
        "is_json": False,
    },
    "statarb-collector": {
        "heartbeat_key": "statarb:heartbeat",
        "max_silent_seconds": 90.0,
        "is_json": False,
    },
    "market-collector": {
        "heartbeat_key": "market:state:BTCUSDT",
        "max_silent_seconds": 60.0,
        "is_json": True,
    },
    "trading-bot": {
        "heartbeat_key": "ai:state:BTCUSDT",
        "max_silent_seconds": 60.0,
        "is_json": True,
    }
}

class AutoWatchdog:
    def __init__(self, check_interval: float = 30.0, log_clean_interval: float = 600.0):
        self.check_interval = check_interval
        self.log_clean_interval = log_clean_interval
        self.last_restart_time = {svc: 0.0 for svc in SERVICES_TO_MONITOR}
        self.restart_cooldown = 120.0  # Min 2 minutes between restarts per service
        self.running = True

    def restart_service_pm2(self, service_name: str, reason: str):
        now = time.time()
        time_since_restart = now - self.last_restart_time.get(service_name, 0.0)
        if time_since_restart < self.restart_cooldown:
            logger.warning(f"[WATCHDOG] Skipping restart for {service_name}: currently in cooldown ({time_since_restart:.1f}s / {self.restart_cooldown:.0f}s).")
            return

        self.last_restart_time[service_name] = now
        logger.critical(f"🚨 [WATCHDOG] RESTARTING {service_name}! Reason: {reason}")
        
        try:
            # Check if PM2 is available
            res = subprocess.run(f"pm2 restart {service_name}", shell=True, capture_output=True, text=True)
            if res.returncode == 0:
                logger.info(f"✅ [WATCHDOG] PM2 successfully restarted {service_name}.")
            else:
                logger.error(f"❌ [WATCHDOG] Failed to restart {service_name} via PM2: {res.stderr.strip()}")
        except Exception as e:
            logger.error(f"[WATCHDOG] Subprocess error executing PM2 restart for {service_name}: {e}")

    def clean_all_logs(self, max_size_mb: int = 15, max_age_days: int = 2):
        """Scans logs/ and PM2 log directories, purging old files and truncating large ones."""
        logger.info("[JANITOR] Running scheduled log cleanup...")
        
        # 1. Clean local project logs directory
        local_log_dir = os.path.abspath("logs")
        clean_old_logs(local_log_dir, max_size_mb=max_size_mb, max_age_days=max_age_days)
        
        # 2. Clean PM2 log directories (common Linux VPS paths)
        pm2_paths = ["/root/.pm2/logs", os.path.expanduser("~/.pm2/logs")]
        for pm2_dir in pm2_paths:
            if os.path.exists(pm2_dir):
                clean_old_logs(pm2_dir, max_size_mb=max_size_mb, max_age_days=max_age_days)
                
        # 3. Check total PM2 log size and run pm2 flush if excessive
        for pm2_dir in pm2_paths:
            if os.path.exists(pm2_dir):
                try:
                    total_pm2_bytes = sum(os.path.getsize(f) for f in glob.glob(os.path.join(pm2_dir, "*")) if os.path.isfile(f))
                    if total_pm2_bytes > (50 * 1024 * 1024):  # > 50MB
                        logger.warning(f"[JANITOR] Total PM2 logs ({total_pm2_bytes / 1024 / 1024:.1f}MB) > 50MB. Running 'pm2 flush'...")
                        subprocess.run("pm2 flush", shell=True, capture_output=True, text=True)
                except Exception as e:
                    logger.debug(f"[JANITOR] PM2 flush check skipped: {e}")

        logger.info("[JANITOR] Log cleanup complete.")

    async def check_heartbeats(self):
        """Checks Redis heartbeats for each service and triggers recovery if frozen."""
        now = time.time()
        for service_name, config in SERVICES_TO_MONITOR.items():
            key = config["heartbeat_key"]
            max_silent = config["max_silent_seconds"]
            is_json = config["is_json"]
            
            try:
                raw_val = await redis_manager.redis.get(key)
                if not raw_val:
                    # Key does not exist yet (service might never have started or crashed before writing)
                    # Allow 120s grace period from bot startup
                    continue
                
                last_time = 0.0
                if is_json:
                    try:
                        data = json.loads(raw_val)
                        last_time = float(data.get("timestamp", 0.0))
                    except Exception:
                        last_time = 0.0
                else:
                    try:
                        last_time = float(raw_val)
                    except Exception:
                        last_time = 0.0
                
                if last_time > 0:
                    silent_seconds = now - last_time
                    if silent_seconds > max_silent:
                        self.restart_service_pm2(
                            service_name, 
                            f"No heartbeat update on '{key}' for {silent_seconds:.1f}s (threshold: {max_silent:.0f}s)"
                        )
            except Exception as e:
                logger.error(f"[WATCHDOG] Error inspecting heartbeat for {service_name}: {e}")

    async def run(self):
        logger.info("🛡️ Starting Production Health Watchdog & Log Janitor Daemon...")
        await redis_manager.connect()
        
        # Initial cleanup on boot
        self.clean_all_logs()
        last_log_clean = time.time()
        
        while self.running:
            try:
                await self.check_heartbeats()
                
                # Check if it's time for periodic log janitor run
                if (time.time() - last_log_clean) >= self.log_clean_interval:
                    self.clean_all_logs()
                    last_log_clean = time.time()
                    
            except Exception as e:
                logger.error(f"[WATCHDOG] Unexpected loop error: {e}")
                
            await asyncio.sleep(self.check_interval)

def main():
    parser = argparse.ArgumentParser(description="Service Health Watchdog & Log Janitor")
    parser.add_argument("--clean-now", action="store_true", help="Clean all logs immediately and exit")
    parser.add_argument("--interval", type=float, default=30.0, help="Heartbeat check interval in seconds (default: 30)")
    args = parser.parse_args()

    watchdog = AutoWatchdog(check_interval=args.interval)

    if args.clean_now:
        print("🧹 Cleaning all log files immediately...")
        watchdog.clean_all_logs(max_size_mb=10, max_age_days=1)
        print("✅ Cleanup complete.")
        return

    try:
        asyncio.run(watchdog.run())
    except KeyboardInterrupt:
        print("\n[WATCHDOG] Stopped by user.")

if __name__ == "__main__":
    main()
