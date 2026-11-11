from __future__ import annotations

import time
from collections import OrderedDict

import numpy as np


class SemanticCache:
    """LRU + TTL cache keyed by query embedding: paraphrases of a cached question are cache hits."""

    def __init__(self, threshold: float = 0.92, ttl_s: int = 3600, size: int = 256):
        self.threshold, self.ttl, self.size = threshold, ttl_s, size
        self._items: OrderedDict[int, tuple[np.ndarray, dict, float]] = OrderedDict()
        self._next = 0
        self.hits = self.misses = 0

    def get(self, qvec: np.ndarray) -> dict | None:
        now = time.time()
        for key in [k for k, (_, _, t) in self._items.items() if now - t > self.ttl]:
            del self._items[key]
        best_key, best = None, self.threshold
        for key, (v, _, _) in self._items.items():
            sim = float(v @ qvec)
            if sim >= best:
                best_key, best = key, sim
        if best_key is None:
            self.misses += 1
            return None
        self._items.move_to_end(best_key)
        self.hits += 1
        return self._items[best_key][1]

    def put(self, qvec: np.ndarray, value: dict) -> None:
        self._items[self._next] = (qvec, value, time.time())
        self._next += 1
        while len(self._items) > self.size:
            self._items.popitem(last=False)
