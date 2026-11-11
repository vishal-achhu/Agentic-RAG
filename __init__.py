"""Agentic RAG: hybrid retrieval, tool-using agent, guardrails and evaluation."""
from .config import Settings
from .service import RAGService

__all__ = ["Settings", "RAGService"]
__version__ = "1.0.0"
