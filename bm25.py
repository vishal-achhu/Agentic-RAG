from __future__ import annotations

import math
from collections import Counter, defaultdict

import numpy as np


class BM25:
    """Okapi BM25 with an inverted index (no external dependencies)."""

    def __init__(self, k1: float = 1.4, b: float = 0.75):
        self.k1, self.b = k1, b
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.doc_len: np.ndarray = np.zeros(0)
        self.avgdl = 0.0
        self.n_docs = 0
        self._idf: dict[str, float] = {}

    def fit(self, docs: list[list[str]]) -> "BM25":
        self.postings = defaultdict(list)
        self.n_docs = len(docs)
        self.doc_len = np.array([len(d) for d in docs], dtype=np.float32)
        self.avgdl = float(self.doc_len.mean()) if self.n_docs else 0.0
        for i, doc in enumerate(docs):
            for term, tf in Counter(doc).items():
                self.postings[term].append((i, tf))
        self._idf = {t: math.log(1 + (self.n_docs - len(p) + 0.5) / (len(p) + 0.5))
                     for t, p in self.postings.items()}
        return self

    def idf(self, term: str) -> float:
        # unseen terms are treated as maximally informative (rare), not ignored
        return self._idf.get(term, math.log(1 + self.n_docs + 0.5))

    def scores(self, query_terms: list[str]) -> np.ndarray:
        s = np.zeros(self.n_docs, dtype=np.float32)
        for t in set(query_terms):
            idf = self._idf.get(t)
            if idf is None:
                continue
            for i, tf in self.postings[t]:
                denom = tf + self.k1 * (1 - self.b + self.b * self.doc_len[i] / (self.avgdl or 1))
                s[i] += idf * tf * (self.k1 + 1) / denom
        return s
