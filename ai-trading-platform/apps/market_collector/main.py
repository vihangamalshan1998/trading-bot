import asyncio
import sys
import signal

from apps.market_collector.service import MarketCollectorService
from core.logging.logger import logger, set_log_file

async def main():
    set_log_file("logs/market_collector.log")
    service = MarketCollectorService()
    
    # Graceful shutdown handler
    def handle_sigint(sig, frame):
        logger.info("Received exit signal, stopping service...")
        asyncio.create_task(service.stop())

    signal.signal(signal.SIGINT, handle_sigint)
    signal.signal(signal.SIGTERM, handle_sigint)

    await service.start()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.error(f"Unhandled exception in Market Collector: {e}", exc_info=True)
        sys.exit(1)
