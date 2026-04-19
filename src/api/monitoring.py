"""
ATLAS-X API – In-memory request monitoring.

Tracks: request counts, latencies, decision distribution, fraud rate.
Thread-safe via a single lock.  For production, swap the in-process store
for Redis (e.g. redis-py INCR / LPUSH) — the interface stays the same.
"""
import threading
import time
from collections import deque
from typing import Dict


class StatsTracker:
    """Singleton accumulating prediction metrics."""

    _instance = None
    _lock     = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    inst = super().__new__(cls)
                    inst._init()
                    cls._instance = inst
        return cls._instance

    def _init(self):
        self._mu             = threading.Lock()
        self.total           = 0
        self.fraud_count     = 0          # transactions predicted as fraud (BLOCK+FLAG)
        self.decisions: Dict[str, int] = {"APPROVE": 0, "FLAG": 0, "BLOCK": 0}
        self._latencies      = deque(maxlen=10_000)   # rolling window
        self._recent         = deque(maxlen=50)        # last 50 prediction summaries
        self.started_at      = time.time()

    # ── Write ─────────────────────────────────────────────────────────────────

    def record(self, *, decision: str, latency_ms: float,
               fraud_probability: float, extra: dict = None) -> None:
        with self._mu:
            self.total += 1
            self._latencies.append(latency_ms)
            if decision in self.decisions:
                self.decisions[decision] += 1
            if decision in ("FLAG", "BLOCK"):
                self.fraud_count += 1
            # store summary for /api/v1/recent
            entry = {
                "fraud_probability": round(fraud_probability, 4),
                "decision":          decision,
                "latency_ms":        round(latency_ms, 2),
            }
            if extra:
                entry.update(extra)
            self._recent.appendleft(entry)

    # ── Read ──────────────────────────────────────────────────────────────────

    def snapshot(self) -> dict:
        with self._mu:
            n   = self.total
            lat = list(self._latencies)

        import numpy as np
        avg_lat = float(np.mean(lat)) if lat else 0.0
        return {
            "total_predictions": n,
            "fraud_rate":        round(self.fraud_count / max(1, n), 4),
            "avg_latency_ms":    round(avg_lat, 2),
            "decisions":         dict(self.decisions),
        }

    def recent(self) -> list:
        with self._mu:
            return list(self._recent)

    def uptime(self) -> float:
        return time.time() - self.started_at
