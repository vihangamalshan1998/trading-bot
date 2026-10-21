"""
Trainer heartbeat for the VPS dashboard.

The trainer runs on the laptop; MASTER_TRAINER's SSH tunnel maps the laptop's localhost:6379 to the
VPS Redis. Every ~30s the trainer writes a small status and the tail of logs/ai_trainer.log there
(`trainer:status`, `trainer:log_tail`, 10-minute expiry), so the dashboard can show whether training
is alive and read its log. Best effort: failures are silent and never affect training.
"""
import json
import os
import socket
import time
from typing import Optional

from core.config.settings import settings

STATUS_KEY = "trainer:status"
LOG_KEY = "trainer:log_tail"
TTL_SECONDS = 600
LOG_LINES = 400
LOG_PATH = "logs/ai_trainer.log"
_client = None
STARTED_AT = time.time()


def _tail(path: str, n: int) -> list:
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 400 * 1024))
            lines = f.read().decode("utf-8", errors="replace").splitlines()
        return lines[-n:]
    except Exception:
        return []


def publish(step: int, extra: Optional[dict] = None) -> bool:
    global _client
    try:
        import redis
        if _client is None:
            _client = redis.Redis.from_url(settings.redis_url, socket_connect_timeout=3, socket_timeout=3)
        status = {"timestamp": time.time(), "started_at": STARTED_AT, "host": socket.gethostname(), "step": int(step)}
        status.update(extra or {})
        _client.set(STATUS_KEY, json.dumps(status), ex=TTL_SECONDS)
        _client.set(LOG_KEY, json.dumps(_tail(LOG_PATH, LOG_LINES)), ex=TTL_SECONDS)
        return True
    except Exception:
        _client = None
        return False
