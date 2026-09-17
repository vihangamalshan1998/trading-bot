import asyncio
import sys
import signal

from apps.macro_collector.service import MacroCollectorService
from core.logging.logger import logger

async def main():
    service = MacroCollectorService()
    
    # Graceful shutdown handler
    def handle_sigint(sig, frame):
        logger.info("Received exit signal, stopping Macro Collector service...")
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
        logger.error(f"Unhandled exception in Macro Collector: {e}", exc_info=True)
        sys.exit(1)
