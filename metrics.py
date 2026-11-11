from __future__ import annotations

import threading
from collections import defaultdict

_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10)


class Metrics:
    """Minimal Prometheus-style metrics (counters + latency histogram)."""

    def __init__(self):
        self._lock = threading.Lock()
        self.counters: dict[str, float] = defaultdict(float)
        self.buckets = [0] * len(_BUCKETS)
        self.lat_sum = 0.0
        self.lat_count = 0

    def inc(self, name: str, v: float = 1.0) -> None:
        with self._lock:
            self.counters[name] += v

    def observe(self, seconds: float) -> None:
        with self._lock:
            self.lat_sum += seconds
            self.lat_count += 1
            for i, b in enumerate(_BUCKETS):
                if seconds <= b:
                    self.buckets[i] += 1

    def render(self) -> str:
        with self._lock:
            lines = [f"rag_{k} {v}" for k, v in sorted(self.counters.items())]
            for b, c in zip(_BUCKETS, self.buckets):
                lines.append(f'rag_request_seconds_bucket{{le="{b}"}} {c}')
            lines += [f'rag_request_seconds_bucket{{le="+Inf"}} {self.lat_count}',
                      f"rag_request_seconds_sum {self.lat_sum:.6f}",
                      f"rag_request_seconds_count {self.lat_count}"]
        return "\n".join(lines) + "\n"
