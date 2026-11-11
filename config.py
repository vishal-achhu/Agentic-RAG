from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class Settings:
    index_dir: str = field(default_factory=lambda: os.getenv("RAG_INDEX_DIR", ".index"))
    # chunking
    chunk_size: int = 420
    chunk_overlap: int = 80
    # retrieval
    embedder: str = field(default_factory=lambda: os.getenv("RAG_EMBEDDER", "auto"))  # auto|hashing|st
    embed_dim: int = 1024
    candidates: int = 20       # per-retriever candidate pool before fusion
    rrf_k: int = 60            # reciprocal-rank-fusion constant
    per_query_k: int = 3       # passages kept per sub-query
    max_evidence: int = 6
    mmr_lambda: float = 0.8
    # agent
    max_steps: int = 6
    llm_provider: str = field(default_factory=lambda: os.getenv("RAG_LLM", "auto"))  # auto|anthropic|mock
    anthropic_model: str = field(default_factory=lambda: os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5"))
    # safety / quality gates
    min_relevance: float = 0.15   # below this the agent abstains (calibrated: off-topic ~0.0, real paraphrases >=0.2)
    min_grounding: float = 0.60   # below this the answer is flagged as weakly grounded
    # semantic cache
    cache_enabled: bool = True
    cache_threshold: float = 0.92
    cache_ttl_s: int = 3600
    cache_size: int = 256
