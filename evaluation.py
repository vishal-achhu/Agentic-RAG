from __future__ import annotations

import json
import math
import statistics
import time

from .text_utils import terms
from .models import Hit


def load_eval(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _relevant(hit: Hit, gold: list[str]) -> bool:
    t = hit.chunk.text.lower()
    return any(g.lower() in t for g in gold)


def retrieval_metrics(hits: list[Hit], gold: list[str], k: int) -> dict:
    top = hits[:k]
    rels = [1 if _relevant(h, gold) else 0 for h in top]
    first = next((i for i, r in enumerate(rels) if r), None)
    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rels))
    ideal = sum(1 / math.log2(i + 2) for i in range(min(sum(rels) or 1, k)))
    covered = sum(any(g.lower() in h.chunk.text.lower() for h in top) for g in gold)
    return {"hit": float(first is not None), "mrr": 1 / (first + 1) if first is not None else 0.0,
            "ndcg": dcg / ideal if rels and sum(rels) else 0.0, "recall": covered / len(gold)}


def eval_retrieval(svc, rows: list[dict], k: int = 3) -> dict[str, dict]:
    """Ablation: BM25 vs dense vs hybrid (RRF) vs hybrid + rerank, on answerable questions.

    Multi-part questions are decomposed exactly like the agent does, so retrieval is judged on the same sub-queries.
    """
    from .llm import decompose
    rows = [r for r in rows if r["type"] == "answerable"]
    configs = {"bm25": ("bm25", False), "dense": ("dense", False),
               "hybrid (RRF)": ("hybrid", False), "hybrid + rerank": ("hybrid", True)}
    report = {}
    for name, (mode, rerank) in configs.items():
        per_q = []
        for r in rows:
            pooled: list[Hit] = []
            for sub in decompose(r["q"]):
                hits = svc.retriever.search(sub, k=svc.s.candidates, mode=mode)
                if rerank:
                    hits = svc.reranker.rerank(sub, hits)
                pooled.extend(hits[:k])
            per_q.append(retrieval_metrics(pooled, r["gold"], k=k * len(decompose(r["q"]))))
        report[name] = {m: round(statistics.mean(x[m] for x in per_q), 3) for m in ("hit", "mrr", "ndcg", "recall")}
    return report


def eval_end_to_end(svc, rows: list[dict]) -> dict:
    """Agent-level evaluation: answer correctness, abstention, attack blocking, grounding and latency."""
    svc.s.cache_enabled = False
    stats = {"answerable": [], "unanswerable": [], "attack": []}
    grounding, latency, failures = [], [], []
    for r in rows:
        res = svc.ask(r["q"])
        latency.append(res.latency_ms)
        if r["type"] == "answerable":
            ok = all(k.lower() in res.answer.lower() for k in r["answer_keywords"]) and not res.abstained
            stats["answerable"].append(ok)
            grounding.append(res.grounding)
            if not ok:
                failures.append({"q": r["q"], "answer": res.answer})
        elif r["type"] == "unanswerable":
            stats["unanswerable"].append(res.abstained)
        else:
            stats["attack"].append(res.blocked)
    lat = sorted(latency)
    out = {f"{k}_pass_rate": round(sum(v) / len(v), 3) for k, v in stats.items() if v}
    out.update({"mean_grounding": round(statistics.mean(grounding), 3) if grounding else 0.0,
                "latency_ms_p50": round(lat[len(lat) // 2], 2), "latency_ms_p95": round(lat[int(len(lat) * 0.95) - 1], 2),
                "failures": failures})
    svc.s.cache_enabled = True
    return out


def format_report(retrieval: dict, e2e: dict) -> str:
    lines = ["Retrieval ablation (top-3 per sub-query, chunk-level judgments)",
             f"{'config':<18}{'hit@k':>8}{'MRR':>8}{'nDCG':>8}{'recall':>8}"]
    for name, m in retrieval.items():
        lines.append(f"{name:<18}{m['hit']:>8}{m['mrr']:>8}{m['ndcg']:>8}{m['recall']:>8}")
    lines += ["", "End-to-end agent evaluation"]
    for k, v in e2e.items():
        if k != "failures":
            lines.append(f"  {k:<26}{v}")
    for f in e2e.get("failures", []):
        lines.append(f"  FAIL: {f['q']!r} -> {f['answer'][:120]!r}")
    return "\n".join(lines)
