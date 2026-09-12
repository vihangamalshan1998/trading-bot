import subprocess
import sys
import time
from core.logging.logger import logger

def start_process(name, cmd):
    logger.info(f"Starting {name}...")
    return subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)

def main():
    logger.info("Initializing Autonomous AI Trading System...")
    
    processes = []
    
    try:
        # 1. Dashboard API
        api_proc = start_process("Dashboard API", [sys.executable, "-m", "uvicorn", "apps.dashboard_api.main:app", "--host", "0.0.0.0", "--port", "8000"])
        processes.append(("Dashboard API", api_proc))
        
        # 2. Market Data Collector
        collector_proc = start_process("Market Collector", [sys.executable, "-m", "apps.market_collector.market_data"])
        processes.append(("Market Collector", collector_proc))
        
        # 3. News & Macro Collector
        news_proc = start_process("News Collector", [sys.executable, "-m", "apps.market_collector.news_macro"])
        processes.append(("News Collector", news_proc))
        
        # 4. Experience Storage Worker
        storage_proc = start_process("Experience Storage", [sys.executable, "-m", "apps.experience.storage"])
        processes.append(("Experience Storage", storage_proc))
        
        # 5. Trading Bot
        bot_proc = start_process("Trading Bot", [sys.executable, "-m", "apps.trading_bot.main"])
        processes.append(("Trading Bot", bot_proc))
        
        # Monitor Loop
        while True:
            for name, proc in processes:
                if proc.poll() is not None:
                    logger.error(f"CRITICAL: {name} crashed with exit code {proc.returncode}!")
                    # In a real setup, we'd restart it here or let Systemd handle it
                    
            time.sleep(5)
            
    except KeyboardInterrupt:
        logger.info("Shutting down system...")
        for name, proc in processes:
            logger.info(f"Terminating {name}...")
            proc.terminate()
            proc.wait()
            
    logger.info("System shutdown complete.")

if __name__ == "__main__":
    main()
