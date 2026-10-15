"""Tiny in-memory usage counters for the public deployment.

One server process, so a dict is enough (with many servers this would move to Redis).
Counts reset when the server restarts, which is fine for a demo.
"""
import threading
import time
from collections import defaultdict

DAY = 24 * 3600
HOUR = 3600


class Limiter:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._hits: dict[str, list[float]] = defaultdict(list)

    def _live(self, key: str, window: float) -> list[float]:
        cutoff = time.time() - window
        live = [t for t in self._hits[key] if t > cutoff]
        self._hits[key] = live
        return live

    def used(self, key: str, window: float) -> int:
        with self._lock:
            return len(self._live(key, window))

    def has_room(self, key: str, limit: int, window: float) -> bool:
        return self.used(key, window) < limit

    def hit(self, key: str) -> None:
        with self._lock:
            self._hits[key].append(time.time())

    def allow(self, key: str, limit: int, window: float) -> bool:
        """Check and count in one step."""
        with self._lock:
            live = self._live(key, window)
            if len(live) >= limit:
                return False
            live.append(time.time())
            return True

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


limiter = Limiter()
