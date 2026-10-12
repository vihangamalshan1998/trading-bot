import sys
import os
sys.path.insert(0, os.path.abspath("."))
import time
from core.ai.replay_buffer import ReplayBuffer

t0 = time.time()
print("Testing ReplayBuffer.load_cache_from_db(limit=2000)...")
buffer = ReplayBuffer()
buffer.load_cache_from_db(limit=2000)
t1 = time.time()
print(f"DONE! Loaded {len(buffer.cache)} normal and {len(buffer.golden_cache)} golden records in {t1 - t0:.2f} seconds.")
