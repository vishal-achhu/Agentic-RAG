from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class Chunk:
    id: str
    doc_id: str
    title: str
    section: str
    text: str
    meta: dict = field(default_factory=dict)

    @property
    def index_text(self) -> str:
        """Text used for indexing: title and section give each chunk context."""
        return f"{self.title}. {self.section}. {self.text}"


@dataclass
class Hit:
    chunk: Chunk
    score: float                      # retrieval (fused) score
    idx: int = -1                     # row in the vector store
    source: str = "hybrid"
    ranks: dict = field(default_factory=dict)
    rerank: float = 0.0               # final score after reranking
    relevance: float = 0.0            # lexical coverage in [0, 1], used for abstention
    query: str = ""                   # sub-query that retrieved this hit
    queries: list = field(default_factory=list)  # every sub-query this passage serves (after de-duplication)


@dataclass
class Action:
    tool: str                         # search | calculate | finish
    input: str = ""
    thought: str = ""


@dataclass
class Step:
    index: int
    action: Action
    observation: str
    latency_ms: float

    def to_dict(self) -> dict:
        return {"index": self.index, "tool": self.action.tool, "input": self.action.input,
                "thought": self.action.thought, "observation": self.observation,
                "latency_ms": round(self.latency_ms, 2)}


@dataclass
class AgentState:
    question: str
    steps: list = field(default_factory=list)
    evidence: list = field(default_factory=list)
    tool_results: list = field(default_factory=list)
    scratch: dict = field(default_factory=dict)


@dataclass
class AgentResult:
    question: str
    answer: str
    citations: list = field(default_factory=list)
    steps: list = field(default_factory=list)
    grounding: float = 0.0
    top_relevance: float = 0.0
    abstained: bool = False
    blocked: bool = False
    cached: bool = False
    warnings: list = field(default_factory=list)
    latency_ms: float = 0.0
    llm: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
