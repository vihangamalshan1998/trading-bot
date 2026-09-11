import asyncio
from core.logging.logger import logger
from core.exchange.binance_client import BinanceFuturesClient
from core.exchange.symbol_registry import registry

async def test_binance_adapter():
    client = BinanceFuturesClient()
    
    logger.info("=== Testing BinanceFuturesClient ===")
    
    try:
        # 1. Fetch Exchange Info & Populate Registry
        logger.info("Testing get_exchange_info...")
        await registry.initialize_from_exchange(client)
        
        btc_config = registry.get_symbol("BTCUSDT")
        assert btc_config is not None, "Failed to load BTCUSDT config"
        logger.info(f"BTCUSDT Config -> Price Precision: {btc_config.price_precision}, Qty Precision: {btc_config.quantity_precision}")
        
        # Test formatting
        raw_qty = 1.23456789
        fmt_qty = btc_config.format_quantity(raw_qty)
        logger.info(f"Raw Qty: {raw_qty} -> Formatted: {fmt_qty}")
        assert len(fmt_qty.split(".")[1]) == btc_config.quantity_precision, "Formatting precision mismatch"
        
        # 2. Check Balance (Requires valid API keys in .env)
        logger.info("Testing get_account_balance...")
        balance = await client.get_account_balance()
        logger.info(f"Testnet USDT Balance: {balance}")
        
    except Exception as e:
        logger.error(f"Binance API test failed: {e}")
        logger.warning("Make sure you have valid Testnet API keys in your .env file!")
        
    finally:
        await client.close()

if __name__ == "__main__":
    asyncio.run(test_binance_adapter())
