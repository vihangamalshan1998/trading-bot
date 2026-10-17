import asyncio
import datetime
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))

from core.exchange.binance_client import BinanceFuturesAdapter

async def run():
    client = BinanceFuturesAdapter()
    
    # 1. Exact Server Time
    srv_time_res = await client._request("GET", "/fapi/v1/time")
    srv_ms = srv_time_res.get("serverTime")
    srv_dt = datetime.datetime.fromtimestamp(srv_ms / 1000.0, tz=datetime.timezone.utc)
    
    exact_24h_ago_ms = srv_ms - (24 * 3600 * 1000)
    exact_24h_ago_dt = datetime.datetime.fromtimestamp(exact_24h_ago_ms / 1000.0, tz=datetime.timezone.utc)
    
    # 2. Exact current balance
    acc = await client.get_account_details()
    current_wallet_exact = float(acc.get("balance", 0))
    
    # 3. Sum of all income events strictly within [exact_24h_ago_ms, srv_ms]
    incomes = await client._request("GET", "/fapi/v1/income", signed=True, params={"startTime": exact_24h_ago_ms, "limit": 1000})
    
    total_net_income_24h = 0.0
    for inc in incomes:
        amt = float(inc.get("income", 0))
        total_net_income_24h += amt
        
    exact_starting_balance = current_wallet_exact - total_net_income_24h
    
    print("==================================================")
    print("      EXACT 24-HOUR BALANCE AUDIT (8 DECIMALS)    ")
    print("==================================================")
    print(f"Current UTC Time:            {srv_dt.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"Exact 24 Hours Ago:          {exact_24h_ago_dt.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("--------------------------------------------------")
    print(f"CURRENT EXACT WALLET:        ${current_wallet_exact:.8f} USDT")
    print(f"TOTAL NET INCOME (24H):      ${total_net_income_24h:+.8f} USDT")
    print("--------------------------------------------------")
    print(f"EXACT VALUE 24 HOURS AGO:    ${exact_starting_balance:.8f} USDT")
    print(f"EXACT VALUE (ROUNDED 2 DEC): ${exact_starting_balance:.2f} USDT")
    print("==================================================")
    
    await client.close()

if __name__ == "__main__":
    asyncio.run(run())
