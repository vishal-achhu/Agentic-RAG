from __future__ import annotations

import numpy as np

from .bm25 import BM25
from .glossary import expand_query
from .text_utils import terms
from .models import Hit
from .vectorstore import VectorStore


class HybridRetriever:
    """BM25 (lexical) + dense (semantic) retrieval fused with Reciprocal Rank Fusion."""

    def __init__(self, store: VectorStore, embedder, settings):
        self.store, self.embedder, self.s = store, embedder, settings
        self.bm25 = BM25()
        self.chunk_terms: list[set[str]] = []

    def rebuild(self) -> None:
        docs = [terms(c.index_text) for c in self.store.chunks]
        self.chunk_terms = [set(d) for d in docs]
        self.bm25.fit(docs)

    @staticmethod
    def _top(scores: np.ndarray, n: int) -> list[int]:
        if not len(scores):
            return []
        order = np.argsort(-scores)[:n]
        return [int(i) for i in order if scores[i] > 1e-9]

    def search(self, query: str, k: int = 5, mode: str = "hybrid") -> list[Hit]:
        n = len(self.store)
        if n == 0:
            return []
        query = expand_query(query)
        pool = min(max(self.s.candidates, k), n)
        bm_scores = self.bm25.scores(terms(query))
        dn_scores = self.store.scores(self.embedder.embed([query])[0])
        bm_rank, dn_rank = self._top(bm_scores, pool), self._top(dn_scores, pool)

        if mode == "bm25":
            return [Hit(self.store.chunks[i], float(bm_scores[i]), i, "bm25", {"bm25": r + 1})
                    for r, i in enumerate(bm_rank[:k])]
        if mode == "dense":
            return [Hit(self.store.chunks[i], float(dn_scores[i]), i, "dense", {"dense": r + 1})
                    for r, i in enumerate(dn_rank[:k])]

        fused: dict[int, float] = {}
        ranks: dict[int, dict] = {}
        for name, lst in (("bm25", bm_rank), ("dense", dn_rank)):
            for r, i in enumerate(lst):
                fused[i] = fused.get(i, 0.0) + 1.0 / (self.s.rrf_k + r + 1)
                ranks.setdefault(i, {})[name] = r + 1
        top = sorted(fused, key=fused.get, reverse=True)[:k]
        return [Hit(self.store.chunks[i], fused[i], i, "hybrid", ranks[i]) for i in top]

    def mmr(self, hits: list[Hit], k: int, lam: float | None = None) -> list[Hit]:
        """Maximal Marginal Relevance: trade relevance against redundancy among selected passages."""
        if len(hits) <= k:
            return hits
        lam = self.s.mmr_lambda if lam is None else lam
        vecs = self.store.matrix[[h.idx for h in hits]]
        rel = np.array([h.rerank or h.score for h in hits], dtype=np.float32)
        rel = rel / (rel.max() or 1.0)
        sel = [int(rel.argmax())]
        rest = set(range(len(hits))) - set(sel)
        while len(sel) < k and rest:
            best = max(rest, key=lambda i: lam * rel[i] - (1 - lam) * max(float(vecs[i] @ vecs[j]) for j in sel))
            sel.append(best)
            rest.remove(best)
        return [hits[i] for i in sel]
