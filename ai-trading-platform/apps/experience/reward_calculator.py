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
            # 4 hours = 14,400 seconds. Add a 1-hour buffer (18,000s total) to guarantee
            # all future data exists before we grade the experience.
            safe_cutoff = current_time - 18000
            
            with self.SessionLocal() as session:
                # Find records that are old enough, but haven't been graded yet
                ungraded = session.query(Experience).filter(
                    Experience.timestamp < safe_cutoff,
                    Experience.reward_4h == None
                ).order_by(Experience.timestamp.asc()).limit(batch_size).all()
                
                if not ungraded:
                    return 0 # Nothing to do
                    
                for exp in ungraded:
                    # Phase 1.B: Extract initial price and action direction
                    p_start = exp.entry_price
                    if not p_start and exp.market_state:
                        p_start = exp.market_state.get("mid_price") or exp.market_state.get("last_price")
                        
                    action_str = str(exp.action or "").upper()
                    if "LONG" in action_str:
                        direction = 1.0
                    elif "SHORT" in action_str:
                        direction = -1.0
                    else:
                        direction = 0.0
                        
                    round_trip_fee_pct = (settings.maker_fee_pct + settings.taker_fee_pct) * 100.0  # e.g. 0.07%
                    
                    # Track closest future price for 5m, 1h, 4h
                    p_5m = None
                    p_1h = None
                    p_4h = None
                    
                    for row in future_rows:
                        t = row.timestamp
                        f_price = row.entry_price
                        if not f_price and row.market_state:
                            f_price = row.market_state.get("mid_price") or row.market_state.get("last_price")
                            
                        if f_price:
                            if t <= t_start + 300:
                                p_5m = f_price
                            if t <= t_start + 3600:
                                p_1h = f_price
                            if t <= t_start + 14400:
                                p_4h = f_price
                                
                    def compute_horizon_outcome(p_future):
                        if not p_start or not p_future or p_start <= 0:
                            return 0.0
                        price_delta_pct = ((p_future - p_start) / p_start) * 100.0
                        if direction != 0.0:
                            gross = price_delta_pct * direction
                            net = gross - round_trip_fee_pct
                            return float(net)
                        else:
                            # HOLD: 0.0 baseline, no arbitrary reward bonus
                            return 0.0

                    exp.reward_5m = compute_horizon_outcome(p_5m)
                    exp.reward_1h = compute_horizon_outcome(p_1h)
                    exp.reward_4h = compute_horizon_outcome(p_4h)
                    
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
