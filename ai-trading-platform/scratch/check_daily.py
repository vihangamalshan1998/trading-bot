import pymysql

try:
    connection = pymysql.connect(
        host='127.0.0.1',
        user='root',
        password='',
        database='ai_trading',
        cursorclass=pymysql.cursors.DictCursor
    )
    with connection.cursor() as cursor:
        query = '''
            SELECT 
                DATE(FROM_UNIXTIME(timestamp)) as date, 
                COUNT(*) as total_actions,
                SUM(reward) as total_reward,
                AVG(reward) as avg_reward,
                SUM(CASE WHEN reward > 0 THEN 1 ELSE 0 END) as winners,
                SUM(CASE WHEN reward <= 0 THEN 1 ELSE 0 END) as losers,
                SUM(realized_pnl) as net_pnl
            FROM experiences 
            WHERE FROM_UNIXTIME(timestamp) >= DATE_SUB(CURDATE(), INTERVAL 15 DAY)
            GROUP BY DATE(FROM_UNIXTIME(timestamp))
            ORDER BY date ASC;
        '''
        cursor.execute(query)
        results = cursor.fetchall()
        print('Daily Experiences Data (from sandbox history):')
        for r in results:
            pnl = r['net_pnl'] or 0
            reward = r['total_reward'] or 0
            avg = r['avg_reward'] or 0
            print(f"Date: {r['date']} | Actions: {r['total_actions']:>5} | Net PnL: {pnl:>8.2f} | Net Reward: {reward:>8.2f} | Avg Rwd: {avg:>6.2f} | W: {r['winners']:>4} | L: {r['losers']:>4}")
            
except Exception as e:
    print('Error:', e)
