from __future__ import annotations

import glob
import hashlib
import os
import threading

from .agent import Agent
from .cache import SemanticCache
from .chunker import chunk_document
from .config import Settings
from .embeddings import get_embedder
from .guardrails import Guardrails
from .llm import get_llm
from .metrics import Metrics
from .reranker import LexicalReranker
from .retriever import HybridRetriever
from .vectorstore import VectorStore


class RAGService:
    """Facade wiring ingestion, retrieval, the agent, caching and metrics together."""

    def __init__(self, settings: Settings | None = None, llm=None, load: bool = True):
        self.s = settings or Settings()
        self.embedder = get_embedder(self.s)
        self.store = VectorStore(self.embedder.dim)
        self.retriever = HybridRetriever(self.store, self.embedder, self.s)
        self.reranker = LexicalReranker(self.retriever)
        self.llm = llm or get_llm(self.s)
        if hasattr(self.llm, "idf"):
            self.llm.idf = self.retriever.bm25.idf
        self.metrics = Metrics()
        self.cache = SemanticCache(self.s.cache_threshold, self.s.cache_ttl_s, self.s.cache_size)
        self.agent = Agent(self.retriever, self.reranker, self.llm, self.embedder, Guardrails(),
                           self.cache, self.metrics, self.s)
        self._lock = threading.RLock()
        if load and self.store.load(self.s.index_dir, self.embedder.name):
            self.retriever.rebuild()

    # ------------------------------------------------------------------ ingestion
    def ingest_texts(self, docs: list[dict], persist: bool = True) -> dict:
        """Idempotent, incremental ingestion: unchanged documents are skipped, changed ones replaced."""
        added = skipped = replaced = 0
        with self._lock:
            for d in docs:
                h = hashlib.sha256(d["text"].encode()).hexdigest()
                if self.store.doc_hashes.get(d["doc_id"]) == h:
                    skipped += 1
                    continue
                if self.store.remove_doc(d["doc_id"]):
                    replaced += 1
                chunks = chunk_document(d["doc_id"], d["text"], d.get("title"), self.s.chunk_size,
                                        self.s.chunk_overlap, d.get("meta"))
                if chunks:
                    self.store.add(chunks, self.embedder.embed([c.index_text for c in chunks]))
                self.store.doc_hashes[d["doc_id"]] = h
                added += 1
            if added:
                self.retriever.rebuild()
                self.cache._items.clear()          # cached answers may be stale after a corpus change
                if persist:
                    self.store.save(self.s.index_dir, self.embedder.name)
        return {"added": added, "replaced": replaced, "skipped": skipped, "chunks": len(self.store)}

    def ingest_dir(self, path: str, persist: bool = True) -> dict:
        docs = []
        for fp in sorted(glob.glob(os.path.join(path, "**", "*"), recursive=True)):
            if os.path.isfile(fp) and fp.endswith((".md", ".txt")):
                with open(fp, encoding="utf-8") as f:
                    docs.append({"doc_id": os.path.splitext(os.path.basename(fp))[0], "text": f.read()})
        return self.ingest_texts(docs, persist)

    # ------------------------------------------------------------------ querying
    def ask(self, question: str):
        return self.agent.run(question)

    def ask_stream(self, question: str):
        return self.agent.run_stream(question)
