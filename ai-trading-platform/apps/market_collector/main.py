import asyncio
import sys
import signal

from apps.market_collector.service import MarketCollectorService
from core.logging.logger import logger

async def main():
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
        logger.error("Unhandled exception in Market Collector", extra={"error": str(e)})
        sys.exit(1)
