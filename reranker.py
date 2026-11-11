from __future__ import annotations

from .glossary import expand_query
from .text_utils import terms
from .models import Hit


class LexicalReranker:
    """Cheap cross-feature reranker: IDF-weighted query coverage + phrase (bigram) matches.

    `relevance` (pure coverage) doubles as the abstention signal: if even the best passage
    covers little of the question, the agent says "I don't know" instead of hallucinating.
    Swap for a cross-encoder (e.g. bge-reranker) behind the same `rerank` signature.
    """

    def __init__(self, retriever, w_lex: float = 0.7):
        self.retriever, self.w_lex = retriever, w_lex

    def _coverage(self, q_terms: list[str], c_terms: set[str]) -> float:
        if not q_terms:
            return 0.0
        total = got = 0.0
        for t in q_terms:
            w = self.retriever.bm25.idf(t)
            total += w
            if t in c_terms:
                got += w
            elif len(t) >= 5 and any(c.startswith(t[:5]) for c in c_terms):
                got += 0.7 * w                      # fuzzy stem match
        return got / total if total else 0.0

    def rerank(self, query: str, hits: list[Hit]) -> list[Hit]:
        if not hits:
            return hits
        q_terms = list(dict.fromkeys(terms(expand_query(query))))
        q_bi = set(zip(q_terms, q_terms[1:]))
        top_fused = max(h.score for h in hits) or 1.0
        for h in hits:
            c_list = terms(h.chunk.index_text)
            cov = self._coverage(q_terms, set(c_list))
            bi = len(q_bi & set(zip(c_list, c_list[1:]))) / len(q_bi) if q_bi else 0.0
            h.relevance = cov
            h.rerank = self.w_lex * (0.8 * cov + 0.2 * bi) + (1 - self.w_lex) * (h.score / top_fused)
        return sorted(hits, key=lambda h: h.rerank, reverse=True)
