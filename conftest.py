import pathlib
import pytest

from agentic_rag import RAGService, Settings

DOCS = str(pathlib.Path(__file__).resolve().parents[1] / "data" / "docs")
EVAL = str(pathlib.Path(__file__).resolve().parents[1] / "data" / "eval" / "golden.jsonl")


@pytest.fixture()
def svc(tmp_path):
    s = RAGService(Settings(index_dir=str(tmp_path / "idx"), embedder="hashing", llm_provider="mock"))
    s.ingest_dir(DOCS)
    return s
