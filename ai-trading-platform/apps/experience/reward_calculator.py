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
                    t_start = exp.timestamp
                    if not t_start:
                        continue

                    # Extract initial price and action direction
                    p_start = None
                    if exp.entry_price and exp.entry_price > 0:
                        p_start = float(exp.entry_price)
                    elif exp.market_state and isinstance(exp.market_state, dict):
                        p_val = exp.market_state.get("mid_price") or exp.market_state.get("last_price")
                        if p_val and float(p_val) > 0:
                            p_start = float(p_val)

                    if not p_start:
                        # Cannot grade without an initial reference price
                        exp.reward_5m = 0.0
                        exp.reward_1h = 0.0
                        exp.reward_4h = 0.0
                        continue

                    action_str = str(exp.action or "").upper()
                    if "LONG" in action_str:
                        direction = 1.0
                    elif "SHORT" in action_str:
                        direction = -1.0
                    else:
                        direction = 0.0

                    # Configurable round-trip trading cost assumption
                    round_trip_fee_pct = (settings.maker_fee_pct + settings.taker_fee_pct) * 100.0  # e.g. 0.07%

                    # Fetch future observations for this symbol up to 4.5 hours ahead
                    future_rows = session.query(
                        Experience.timestamp,
                        Experience.entry_price,
                        Experience.market_state,
                        Experience.fees
                    ).filter(
                        Experience.symbol == exp.symbol,
                        Experience.timestamp >= t_start,
                        Experience.timestamp <= t_start + 16200
                    ).order_by(Experience.timestamp.asc()).all()

                    if not future_rows:
                        continue

                    t_5m_target = t_start + 300
                    t_1h_target = t_start + 3600
                    t_4h_target = t_start + 14400

                    def extract_price(row):
                        if row.entry_price and float(row.entry_price) > 0:
                            return float(row.entry_price)
                        if row.market_state and isinstance(row.market_state, dict):
                            p = row.market_state.get("mid_price") or row.market_state.get("last_price")
                            if p and float(p) > 0:
                                return float(p)
                        return None

                    p_5m, p_1h, p_4h = None, None, None
                    best_diff_5m, best_diff_1h, best_diff_4h = float("inf"), float("inf"), float("inf")

                    for row in future_rows:
                        t = row.timestamp
                        f_price = extract_price(row)
                        if not f_price:
                            continue

                        # 5m horizon (closest at or after target, within 180s tolerance)
                        diff_5m = abs(t - t_5m_target)
                        if diff_5m < best_diff_5m and (t >= t_5m_target - 60):
                            best_diff_5m = diff_5m
                            p_5m = f_price

                        # 1h horizon (closest at or after target, within 600s tolerance)
                        diff_1h = abs(t - t_1h_target)
                        if diff_1h < best_diff_1h and (t >= t_1h_target - 150):
                            best_diff_1h = diff_1h
                            p_1h = f_price

                        # 4h horizon (closest at or after target, within 1200s tolerance)
                        diff_4h = abs(t - t_4h_target)
                        if diff_4h < best_diff_4h and (t >= t_4h_target - 300):
                            best_diff_4h = diff_4h
                            p_4h = f_price

                    def compute_outcome(p_future):
                        if not p_future or p_future <= 0:
                            return 0.0
                        delta_pct = ((p_future - p_start) / p_start) * 100.0
                        if direction != 0.0:
                            net = (delta_pct * direction) - round_trip_fee_pct
                            return float(net)
                        else:
                            return 0.0

                    if p_5m:
                        exp.reward_5m = compute_outcome(p_5m)
                    if p_1h:
                        exp.reward_1h = compute_outcome(p_1h)
                    if p_4h:
                        exp.reward_4h = compute_outcome(p_4h)
                    
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
