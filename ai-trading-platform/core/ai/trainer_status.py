"""
Trainer heartbeat for the VPS dashboard.

The trainer runs on the laptop; MASTER_TRAINER's SSH tunnel maps the laptop's localhost:6379 to the
VPS Redis. Every ~30s the trainer writes a small status and the tail of logs/ai_trainer.log there
(`trainer:status`, `trainer:log_tail`, 10-minute expiry), so the dashboard can show whether training
is alive, what it is doing (`phase`) and read its log. Best effort: failures are silent and never
affect training.
"""
import json
import os
import socket
import threading
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


_phase = {"phase": "starting", "step_fn": lambda: 0, "extra_fn": lambda: {}}
_thread = None


def set_phase(phase: str, step_fn=None, extra_fn=None) -> None:
    """What the trainer is doing now (shown on the dashboard), plus optional live step / stats getters."""
    _phase["phase"] = phase
    if step_fn is not None:
        _phase["step_fn"] = step_fn
    if extra_fn is not None:
        _phase["extra_fn"] = extra_fn


def start_heartbeat(interval: float = 30.0) -> None:
    """Publishes every `interval` seconds from launch on, so the dashboard sees the trainer while it is
    still loading data (that alone can take ~10 minutes over the tunnel)."""
    global _thread
    if _thread is not None:
        return

    def loop():
        while True:
            try:
                extra = {"phase": _phase["phase"]}
                extra.update(_phase["extra_fn"]() or {})
                publish(_phase["step_fn"](), extra)
            except Exception:
                pass
            time.sleep(interval)

    _thread = threading.Thread(target=loop, name="trainer-heartbeat", daemon=True)
    _thread.start()


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
