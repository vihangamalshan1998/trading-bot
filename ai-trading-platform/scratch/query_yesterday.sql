SELECT 
    symbol,
    COUNT(*) as trades,
    SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) as wins,
    SUM(CASE WHEN realized_pnl < 0 THEN 1 ELSE 0 END) as losses,
    ROUND(MAX(realized_pnl), 2) as max_win_usd,
    ROUND(SUM(realized_pnl), 2) as gross_pnl,
    ROUND(SUM(fees), 2) as fees,
    ROUND(SUM(realized_pnl - fees), 2) as net_pnl
FROM experiences
WHERE action != 'HOLD' 
  AND timestamp >= UNIX_TIMESTAMP('2026-10-13 00:00:00')
  AND timestamp < UNIX_TIMESTAMP('2026-10-14 00:00:00')
GROUP BY symbol
ORDER BY gross_pnl DESC;

SELECT 
    DATE_FORMAT(FROM_UNIXTIME(timestamp), '%H:00') as hour_slot,
    COUNT(*) as trades,
    SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) as wins,
    ROUND(SUM(realized_pnl), 2) as gross_pnl,
    ROUND(SUM(fees), 2) as fees,
    ROUND(SUM(realized_pnl - fees), 2) as net_pnl
FROM experiences
WHERE action != 'HOLD'
  AND timestamp >= UNIX_TIMESTAMP('2026-10-13 00:00:00')
  AND timestamp < UNIX_TIMESTAMP('2026-10-14 00:00:00')
GROUP BY hour_slot
ORDER BY hour_slot ASC;

SELECT 
    symbol, action, FROM_UNIXTIME(timestamp) as time_utc,
    ROUND(realized_pnl, 2) as gross, ROUND(fees, 2) as fees, ROUND(realized_pnl - fees, 2) as net
FROM experiences
WHERE action != 'HOLD' 
  AND timestamp >= UNIX_TIMESTAMP('2026-10-13 00:00:00')
  AND timestamp < UNIX_TIMESTAMP('2026-10-14 00:00:00')
  AND realized_pnl > 0
ORDER BY realized_pnl DESC
LIMIT 15;
