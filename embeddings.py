from __future__ import annotations

import zlib
from collections import Counter

import numpy as np

from .text_utils import STOPWORDS, tokenize


class HashingEmbedder:
    """Offline, dependency-free dense embedder (feature hashing of char n-grams + word bigrams).

    It is deliberately *different* from BM25: character n-grams are robust to typos and
    morphology ("pricing" ~ "price"), so fusing the two retrievers is genuinely useful.
    """
    name = "hashing"

    def __init__(self, dim: int = 1024):
        self.dim = dim

    def _features(self, text: str) -> Counter:
        toks = [t for t in tokenize(text) if t not in STOPWORDS]
        feats: Counter = Counter(toks)
        feats.update(f"{a}_{b}" for a, b in zip(toks, toks[1:]))
        for t in toks:
            w = f"<{t}>"
            for n in (3, 4, 5):
                feats.update(w[i:i + n] for i in range(len(w) - n + 1))
        return feats

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for f, c in self._features(text).items():
                h = zlib.crc32(f.encode())
                sign = 1.0 if (h >> 31) & 1 else -1.0
                out[row, h % self.dim] += sign * (1.0 + np.log(c))
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-9)


class SentenceTransformerEmbedder:
    """Optional higher-quality embedder: pip install sentence-transformers."""

    def __init__(self, model: str = "BAAI/bge-small-en-v1.5"):
        from sentence_transformers import SentenceTransformer  # lazy import
        self.model = SentenceTransformer(model)
        self.dim = self.model.get_sentence_embedding_dimension()
        self.name = f"st:{model}"

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts, normalize_embeddings=True), dtype=np.float32)


def get_embedder(settings):
    if settings.embedder in ("st", "auto"):
        try:
            return SentenceTransformerEmbedder()
        except Exception:
            if settings.embedder == "st":
                raise
    return HashingEmbedder(settings.embed_dim)
