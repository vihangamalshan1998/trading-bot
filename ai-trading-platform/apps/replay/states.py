"""
Replayed 200-slot frames on disk, one compressed file per coin and UTC day.

A replayed training record is only an index row (coin, time, day, frame index) in the replay_records table;
its 300-frame sequence is cut from these files when the trainer loads it. Storing every record as a
300 x 200 JSON column cost ~0.35 s of disk time per record on the training laptop's hard disk and ~0.7 MB
each; a coin-day file holds all ~16,300 frames once.

Frames are stored rounded to 6 significant digits, like the live bot's stored records.
Light module (numpy only): the trainer imports it.
"""
import os
from datetime import date, timedelta
from typing import Dict, Optional

import numpy as np

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                         "data", "replay_cache")
STATES_DIR = os.path.join(CACHE_DIR, "states")
SEQ_LEN = 300
REPLAY_TABLE = "replay_records"   # one index row per replayed training record (apps/replay/run.py)
REPLAY_OUTCOME_SRC = "rpl"        # trade_outcomes.src of those rows
MAX_FRAME_GAP_SECONDS = 60.0  # a longer gap between frames breaks a sequence


def round_sig(values: np.ndarray, sig: int = 6) -> np.ndarray:
    """Rounds to `sig` significant digits (vectorized form of the bot's f"{v:.6g}" storage rounding)."""
    a = np.asarray(values, dtype=np.float64).copy()
    nz = np.isfinite(a) & (a != 0)
    mag = np.floor(np.log10(np.abs(a[nz])))
    scale = 10.0 ** (sig - 1 - mag)
    a[nz] = np.round(a[nz] * scale) / scale
    return a


def state_path(symbol: str, day: date) -> str:
    return os.path.join(STATES_DIR, symbol, f"{symbol}-{day.isoformat()}.npz")


def save_states(symbol: str, day: date, t: np.ndarray, state: np.ndarray, valid: np.ndarray) -> str:
    path = state_path(symbol, day)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, t=t, state=round_sig(state).astype(np.float32), valid=valid)
    os.replace(tmp, path)
    return path


def load_states(symbol: str, day: date) -> Optional[Dict[str, np.ndarray]]:
    path = state_path(symbol, day)
    if not os.path.exists(path):
        return None
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


class SequenceReader:
    """Cuts 300-frame sequences out of coin-day files, keeping recently used files in memory."""

    def __init__(self, max_files: int = 64):
        self.max_files = max_files
        self._files: Dict[tuple, Optional[Dict[str, np.ndarray]]] = {}

    def _get(self, symbol: str, day: date):
        key = (symbol, day)
        if key not in self._files:
            if len(self._files) >= self.max_files:
                self._files.pop(next(iter(self._files)))
            self._files[key] = load_states(symbol, day)
        return self._files[key]

    def sequence(self, symbol: str, day: date, frame_idx: int) -> Optional[np.ndarray]:
        """(300, 200) float32 frames ending at frame_idx of that day (continuing from the previous day)."""
        cur = self._get(symbol, day)
        if cur is None or frame_idx >= len(cur["t"]):
            return None
        if frame_idx + 1 >= SEQ_LEN:
            sl = slice(frame_idx + 1 - SEQ_LEN, frame_idx + 1)
            t, st, ok = cur["t"][sl], cur["state"][sl], cur["valid"][sl]
        else:
            prev = self._get(symbol, day - timedelta(days=1))
            if prev is None:
                return None
            need = SEQ_LEN - (frame_idx + 1)
            t = np.concatenate([prev["t"][-need:], cur["t"][:frame_idx + 1]])
            st = np.concatenate([prev["state"][-need:], cur["state"][:frame_idx + 1]])
            ok = np.concatenate([prev["valid"][-need:], cur["valid"][:frame_idx + 1]])
        if len(st) != SEQ_LEN or not ok.all() or np.max(np.diff(t)) > MAX_FRAME_GAP_SECONDS:
            return None
        return st
