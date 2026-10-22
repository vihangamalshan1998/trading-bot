"""
Heat guard for the replay's pass 1 on a laptop: fewer workers while the CPU runs hot.

Windows exposes thermal zones without admin rights (performance counter "Thermal Zone Information",
in Kelvin). The guard reads the hottest zone at most every CHECK_SECONDS; after HOT_READINGS readings in a
row at or above HOT_C it lowers the worker limit to HOT_WORKERS, and after COOL_READINGS readings at or
below COOL_C it goes back to the full count. No reading (other OS, counter missing): full count.
"""
import subprocess
import sys
import time
from typing import Optional

HOT_C = 90.0
COOL_C = 80.0
HOT_WORKERS = 2
HOT_READINGS = 2
COOL_READINGS = 3
CHECK_SECONDS = 20.0

_PS = ("$t = @(Get-CimInstance Win32_PerfFormattedData_Counters_ThermalZoneInformation "
       "-ErrorAction SilentlyContinue | ForEach-Object { $_.Temperature - 273.15 }); "
       "if ($t.Count) { ($t | Measure-Object -Maximum).Maximum }")


def cpu_temp_c() -> Optional[float]:
    if sys.platform != "win32":
        return None
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", _PS],
                             capture_output=True, text=True, timeout=15,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout.strip()
        return float(out) if out else None
    except Exception:
        return None


class ThermalGuard:
    def __init__(self, workers: int):
        self.full = workers
        self.hot_limit = max(1, min(HOT_WORKERS, workers))
        self.current = workers
        self._hot = self._cool = 0
        self._next_check = 0.0

    def limit(self) -> int:
        now = time.time()
        if now < self._next_check or self.full <= self.hot_limit:
            return self.current
        self._next_check = now + CHECK_SECONDS
        t = cpu_temp_c()
        if t is None:
            return self.current
        self._hot = self._hot + 1 if t >= HOT_C else 0
        self._cool = self._cool + 1 if t <= COOL_C else 0
        if self.current == self.full and self._hot >= HOT_READINGS:
            self.current = self.hot_limit
            print(f"heat guard: CPU {t:.0f} C -> {self.current} workers until it cools below {COOL_C:.0f} C", flush=True)
        elif self.current != self.full and self._cool >= COOL_READINGS:
            self.current = self.full
            print(f"heat guard: CPU {t:.0f} C -> back to {self.current} workers", flush=True)
        return self.current
