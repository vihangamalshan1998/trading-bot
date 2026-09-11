import asyncio
import os
import sys
import subprocess
from core.logging.logger import logger

class SystemOrchestrator:
    """
    Master orchestrator script that launches and monitors all microservices concurrently.
    """
    def __init__(self):
        self.processes = []
        
        # Define the services to run
        self.services = [
            {"name": "Universe Screener", "path": "apps/market_collector/universe.py"},
            {"name": "News & Macro Collector", "path": "apps/market_collector/news_macro.py"},
            {"name": "Market Data Collector", "path": "apps/market_collector/market_data.py"},
            {"name": "Online Trainer", "path": "apps/trading_bot/online_trainer.py"},
            {"name": "Production Trading Bot", "path": "apps/trading_bot/main.py"}
        ]

    async def run_service(self, service: dict):
        logger.info(f"Starting {service['name']}...")
        
        # Use subprocess to run each service in its own isolated environment
        # We assume the current virtualenv is active and PYTHONPATH="." is handled by the script runner
        env = os.environ.copy()
        env["PYTHONPATH"] = "."
        
        process = await asyncio.create_subprocess_exec(
            sys.executable, service["path"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env
        )
        
        self.processes.append(process)
        
        # Asynchronously read output
        async def read_stream(stream, is_stderr=False):
            while True:
                line = await stream.readline()
                if line:
                    decoded = line.decode('utf-8').strip()
                    if is_stderr:
                        logger.error(f"[{service['name']}] {decoded}")
                    else:
                        logger.info(f"[{service['name']}] {decoded}")
                else:
                    break

        await asyncio.gather(
            read_stream(process.stdout),
            read_stream(process.stderr, is_stderr=True)
        )
        
        await process.wait()
        logger.warning(f"Service {service['name']} exited with code {process.returncode}")

    async def start_all(self):
        logger.info("=== Starting AI Trading Platform ===")
        
        # Run all services concurrently
        tasks = [self.run_service(service) for service in self.services]
        await asyncio.gather(*tasks)

    def shutdown(self):
        logger.info("Shutting down all services...")
        for p in self.processes:
            try:
                p.terminate()
            except Exception:
                pass

if __name__ == "__main__":
    orchestrator = SystemOrchestrator()
    try:
        asyncio.run(orchestrator.start_all())
    except KeyboardInterrupt:
        orchestrator.shutdown()
        logger.info("System gracefully shut down.")
