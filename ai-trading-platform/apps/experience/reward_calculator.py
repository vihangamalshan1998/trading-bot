import time
import asyncio
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from core.database.models.ai import Experience
from core.config.settings import settings
from core.logging.logger import logger, set_log_file

class RewardCalculator:
    """
    The 'Teacher's Assistant'.
    Runs in the background. Finds old experiences that are missing their 
    Multi-Horizon macro rewards, calculates the exact future sum from the database,
    and writes the final grades back to the database in red pen.
    """
    def __init__(self):
        self.engine = create_engine(settings.database_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
    def process_batch(self, batch_size=100):
        try:
            current_time = int(time.time())
            # 4 hours = 14,400 seconds. Add a tiny buffer (15,000) to be safe.
            safe_cutoff = current_time - 15000
            
            with self.SessionLocal() as session:
                # Find records that are old enough, but haven't been graded yet
                ungraded = session.query(Experience).filter(
                    Experience.timestamp < safe_cutoff,
                    Experience.reward_4h == None
                ).order_by(Experience.timestamp.asc()).limit(batch_size).all()
                
                if not ungraded:
                    return 0 # Nothing to do
                    
                for exp in ungraded:
                    # 5-min, 1-hour, and 4-hour rewards
                    t_start = exp.timestamp
                    
                    # FETCH ALL FUTURE ROWS ONCE (Up to 4 hours)
                    future_rows = session.query(Experience.reward, Experience.timestamp).filter(
                        Experience.timestamp >= t_start,
                        Experience.timestamp <= t_start + 14400,
                        Experience.symbol == exp.symbol
                    ).order_by(Experience.timestamp.asc()).all()
                    
                    if not future_rows:
                        exp.reward_5m = exp.reward_1h = exp.reward_4h = (exp.reward or 0.0)
                        continue
                        
                    # CALCULATE IN FAST RAM INSTEAD OF 3 SQL QUERIES
                    r_5m_sum = 0.0
                    r_1h_sum = 0.0
                    r_4h_sum = 0.0
                    
                    for r, t in future_rows:
                        val = r or 0.0
                        if t <= t_start + 300: r_5m_sum += val
                        if t <= t_start + 3600: r_1h_sum += val
                        if t <= t_start + 14400: r_4h_sum += val
                        
                    exp.reward_5m = r_5m_sum
                    exp.reward_1h = r_1h_sum
                    exp.reward_4h = r_4h_sum
                    
                session.commit()
                return len(ungraded)
                
        except Exception as e:
            logger.error(f"Reward Calculator Error: {e}")
            return 0

async def main():
    set_log_file("logs/reward_calculator.log")
    logger.info("Starting Teacher's Assistant (Reward Calculator)...")
    calculator = RewardCalculator()
    
    while True:
        processed = calculator.process_batch()
        if processed > 0:
            logger.info(f"Graded {processed} historical experiences with Multi-Horizon rewards.")
            await asyncio.sleep(2.0) # Throttle to prevent 100% CPU usage while clearing backlogs
        else:
            await asyncio.sleep(60.0) # Sleep for a minute if caught up

if __name__ == "__main__":
    asyncio.run(main())
