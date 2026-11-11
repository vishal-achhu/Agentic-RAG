from __future__ import annotations

import time
from typing import Iterator

from .guardrails import Guardrails, grounding_score
from .llm import LLM
from .tools import fmt_number, safe_calculate
from .models import Action, AgentResult, AgentState, Hit, Step

ABSTAIN = "I couldn't find enough information in the knowledge base to answer that."


class Agent:
    """Plan -> act (search / calculate) -> observe loop with guardrails, caching and abstention."""

    def __init__(self, retriever, reranker, llm: LLM, embedder, guard: Guardrails, cache, metrics, settings):
        self.retriever, self.reranker, self.llm = retriever, reranker, llm
        self.embedder, self.guard, self.cache, self.metrics, self.s = embedder, guard, cache, metrics, settings

    # ------------------------------------------------------------------ tools
    def _search(self, query: str, state: AgentState) -> str:
        hits = self.retriever.search(query, k=self.s.candidates)
        hits, dropped = self.guard.filter_context(hits)
        if dropped:
            self.metrics.inc("context_injection_dropped_total", dropped)
        hits = self.reranker.rerank(query, hits)
        hits = self.retriever.mmr(hits, self.s.per_query_k)
        for h in hits:
            h.query = query
            h.queries = [query]
        state.evidence.extend(hits)
        if not hits:
            return "no passages found"
        top = hits[0]
        return (f"{len(hits)} passages; best: {top.chunk.title} > {top.chunk.section or 'intro'} "
                f"(relevance {top.relevance:.2f})")

    def _calculate(self, expr: str, state: AgentState) -> str:
        try:
            value = safe_calculate(expr)
        except ValueError as e:
            return f"error: {e}"
        result = f"{expr} = {fmt_number(value)}"
        state.tool_results.append(result)
        return result

    # ------------------------------------------------------------------ evidence selection
    def _select_evidence(self, state: AgentState) -> tuple[list[Hit], list[str]]:
        queries = [s.action.input for s in state.steps if s.action.tool == "search"]
        by_q: dict[str, list[Hit]] = {}
        for h in state.evidence:
            by_q.setdefault(h.query, []).append(h)
        answered, unanswered = [], []
        for q in queries:
            hs = by_q.get(q, [])
            (answered if hs and max(h.relevance for h in hs) >= self.s.min_relevance else unanswered).append(q)
        # round-robin across answerable sub-queries so every part of the question gets evidence;
        # a passage that serves several sub-queries is kept once but tagged with all of them
        by_id: dict[str, Hit] = {}
        picked: list[Hit] = []
        pools = {q: sorted(by_q[q], key=lambda h: h.rerank, reverse=True) for q in answered}
        while len(picked) < self.s.max_evidence and any(pools.values()):
            for q in answered:
                while pools[q]:
                    h = pools[q].pop(0)
                    if h.relevance < self.s.min_relevance * 0.6:
                        continue
                    if h.chunk.id in by_id:
                        if q not in by_id[h.chunk.id].queries:
                            by_id[h.chunk.id].queries.append(q)
                        continue
                    by_id[h.chunk.id] = h
                    picked.append(h)
                    break
        picked.sort(key=lambda h: answered.index(h.queries[0]))   # keep sub-query grouping stable
        return picked, unanswered

    # ------------------------------------------------------------------ main loop
    def run_stream(self, question: str) -> Iterator[dict]:
        t0 = time.perf_counter()
        self.metrics.inc("requests_total")

        def finish(res: AgentResult) -> dict:
            res.latency_ms = round((time.perf_counter() - t0) * 1000, 2)
            res.llm = self.llm.name
            self.metrics.observe(res.latency_ms / 1000)
            return {"type": "final", "result": res.to_dict()}

        g = self.guard.check_input(question)
        if g.blocked:
            self.metrics.inc("blocked_total")
            yield finish(AgentResult(question, "Request blocked by safety guardrails.", blocked=True,
                                     warnings=g.reasons))
            return
        q = g.sanitized
        warnings = list(g.reasons)

        qvec = self.embedder.embed([q])[0]
        if self.s.cache_enabled:
            cached = self.cache.get(qvec)
            if cached:
                self.metrics.inc("cache_hits_total")
                res = AgentResult(**{**cached, "question": question, "cached": True})
                yield finish(res)
                return

        state = AgentState(question=q)
        seen: set[tuple] = set()
        for i in range(1, self.s.max_steps + 1):
            action: Action = self.llm.next_action(state)
            if action.tool == "finish" or (action.tool, action.input) in seen:
                break
            seen.add((action.tool, action.input))
            ts = time.perf_counter()
            if action.tool == "search":
                obs = self._search(action.input, state)
            elif action.tool == "calculate":
                obs = self._calculate(action.input, state)
            else:
                obs = f"unknown tool {action.tool!r}"
            step = Step(i, action, obs, (time.perf_counter() - ts) * 1000)
            state.steps.append(step)
            yield {"type": "step", "step": step.to_dict()}

        evidence, unanswered = self._select_evidence(state)
        top_rel = max((h.relevance for h in evidence), default=0.0)

        if not evidence:
            self.metrics.inc("abstained_total")
            yield finish(AgentResult(question, ABSTAIN, steps=[s.to_dict() for s in state.steps],
                                     abstained=True, top_relevance=top_rel, warnings=warnings))
            return

        parts: list[str] = []
        for tok in self.llm.stream(q, evidence, state.tool_results):
            parts.append(tok)
            yield {"type": "token", "text": tok}
        answer = "".join(parts).strip()
        if unanswered:
            answer += " I couldn't find information about: " + "; ".join(unanswered) + "."
            warnings.append("partial_answer")
        answer, out_warn = self.guard.check_output(answer)
        warnings += out_warn

        grounding = grounding_score(answer, [h.chunk.text for h in evidence] + state.tool_results)
        if grounding < self.s.min_grounding:
            warnings.append("low_grounding")
            self.metrics.inc("low_grounding_total")

        citations = [{"n": n, "doc_id": h.chunk.doc_id, "chunk_id": h.chunk.id, "title": h.chunk.title,
                      "section": h.chunk.section, "snippet": h.chunk.text[:220],
                      "relevance": round(h.relevance, 3), "retrieval": h.ranks}
                     for n, h in enumerate(evidence, 1)]
        res = AgentResult(question, answer, citations, [s.to_dict() for s in state.steps],
                          round(grounding, 3), round(top_rel, 3), warnings=warnings)
        if self.s.cache_enabled and not unanswered and "low_grounding" not in warnings:
            self.cache.put(qvec, {k: v for k, v in res.to_dict().items() if k not in ("cached", "latency_ms", "llm")})
        yield finish(res)

    def run(self, question: str) -> AgentResult:
        final = None
        for ev in self.run_stream(question):
            if ev["type"] == "final":
                final = ev["result"]
        return AgentResult(**final)
