# Agentic RAG: hybrid retrieval, tool-using agent, guardrails & evaluation

A production-style Retrieval-Augmented Generation platform in Python. It runs **fully offline** (no API key, deterministic, CI-friendly)
and switches to **Claude** for planning and answering when `ANTHROPIC_API_KEY` is set.

```
question ─► input guardrails (injection block, PII redaction)
         ─► semantic cache ──hit──► answer
                 │miss
                 ▼
        ┌─────── agent loop (plan → act → observe, max N steps) ───────┐
        │  search(sub-query): BM25 ┐                                    │
        │                          ├─ RRF fusion ─► rerank ─► MMR       │
        │                 dense  ──┘      └─ drop poisoned passages     │
        │  calculate(expr): AST-sandboxed arithmetic                    │
        └───────────────────────────────────────────────────────────────┘
                 ▼
   evidence selection (round-robin per sub-query) ─► abstain if weak
                 ▼
   LLM synthesis with [n] citations (streamed) ─► grounding score ─► PII output filter
```

## Run it
```bash
pip install -r requirements.txt
python -m agentic_rag ingest                       # chunk + embed + index data/docs
python -m agentic_rag ask "What is the monthly cost for 12 seats on the Pro plan?" --stream
python -m agentic_rag eval                         # retrieval ablation + end-to-end metrics
python -m agentic_rag serve                        # FastAPI on :8000  (/query, /query/stream, /ingest, /metrics, /health)
python -m pytest -q                                # 32 tests, incl. a golden-set quality gate
export ANTHROPIC_API_KEY=...                       # switches planner + answerer to Claude (model: $ANTHROPIC_MODEL)
```

## What's inside
| Area | Implementation |
|---|---|
| Ingestion | Heading-aware chunking with sentence overlap; content-hash incremental re-indexing; persisted index |
| Retrieval | From-scratch BM25 + dense vectors, **Reciprocal Rank Fusion**, lexical cross-feature reranker, **MMR** diversification, domain glossary query expansion |
| Agent | Plan/act/observe loop; multi-part questions are decomposed into sub-queries; `search` + sandboxed `calculate` tools; loop/duplicate-action protection |
| Safety | Prompt-injection blocking, **indirect injection defence** (poisoned retrieved passages dropped), PII/card (Luhn) redaction, AST calculator (no `eval`) |
| Reliability | Abstains instead of hallucinating; faithfulness (grounding) score on every answer; planner failures degrade gracefully |
| Serving | FastAPI, SSE token streaming, API-key auth, input validation, Prometheus-style metrics, semantic LRU+TTL cache, Dockerfile |
| Evaluation | Chunk-level hit@k / MRR / nDCG / recall ablation (BM25 vs dense vs hybrid vs +rerank); end-to-end answer accuracy, abstention, attack-block rate, latency p50/p95; CI gate |
| LLM layer | `LLM` interface: Claude backend + deterministic offline backend. Claude path is tested against a scripted fake client |

## Measured results (`python -m agentic_rag eval`, offline backend)
Corpus: 7 documents / 26 chunks. Golden set: 18 answerable (8 multi-part), 2 out-of-scope, 2 attack prompts.

| Retrieval config | hit@3 | MRR | nDCG | recall |
|---|---|---|---|---|
| BM25 | 1.00 | 0.880 | 0.899 | 1.00 |
| Dense (hashed char n-grams) | 1.00 | 0.880 | 0.910 | 1.00 |
| **Hybrid (RRF)** | 1.00 | **0.907** | **0.920** | 1.00 |
| Hybrid + rerank | 1.00 | 0.898 | 0.905 | 1.00 |

End-to-end: answer accuracy **77.8%** (14/18, extractive offline answerer), abstention on out-of-scope **100%**,
attack blocking **100%**, mean grounding **1.0**, latency p50 ≈ 2 ms.

### Honest limitations
- The corpus and golden set are small, so retriever differences are small; treat them as a harness demonstration, not a benchmark.
  Fusion beating either retriever alone is the expected pattern, and the CI gate enforces it.
- The lexical reranker did **not** beat plain hybrid here. A cross-encoder (`bge-reranker`) is the natural upgrade behind the same interface.
- The 4 failing questions need semantic understanding ("databases" → PostgreSQL, "10 days" < "14 days"). The offline answerer is a
  keyword-extractive baseline; use Claude (and `sentence-transformers`, auto-detected) for real semantic quality, then re-run `eval`.
- Thresholds (`min_relevance`, glossary, stopwords) were tuned on this small set. Use a held-out set on real data.
- The Claude backend was verified with a stubbed client, not against the live API in this repo's CI.

## Resume bullets (adapt the numbers to your own runs)
- Built an **agentic RAG platform** in Python/FastAPI: hybrid BM25 + dense retrieval with Reciprocal Rank Fusion, reranking and MMR, plus a plan-act-observe agent that decomposes multi-part questions and uses search and calculator tools; pluggable Claude / offline LLM backends.
- Designed **safety and reliability layers**: prompt-injection blocking, indirect-injection filtering of retrieved passages, PII redaction, abstention on low-relevance queries, and a faithfulness score on every answer.
- Implemented an **evaluation harness** (hit@k, MRR, nDCG, end-to-end accuracy, abstention and attack-block rates) with a retrieval ablation and a CI quality gate; hybrid fusion improved MRR over BM25 and dense alone on the golden set.
- Shipped production features: SSE streaming, semantic cache (LRU+TTL), incremental idempotent indexing, API-key auth, Prometheus metrics, Docker, GitHub Actions, and 32 automated tests.

## Extending it
Swap `VectorStore` for pgvector/FAISS; add a cross-encoder reranker; add an LLM-as-judge for faithfulness on sampled traffic; add tools
(SQL, web search) by registering them in `Agent._execute`; add per-tenant filters via chunk `meta`.
