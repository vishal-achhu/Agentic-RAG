from __future__ import annotations

import json
import os
from dataclasses import asdict

import numpy as np

from .models import Chunk


class VectorStore:
    """In-memory exact cosine-similarity store with on-disk persistence.

    Exact search is optimal up to ~100k chunks; swap in FAISS/pgvector behind the same API beyond that.
    """

    def __init__(self, dim: int):
        self.dim = dim
        self.matrix = np.zeros((0, dim), dtype=np.float32)
        self.chunks: list[Chunk] = []
        self.doc_hashes: dict[str, str] = {}

    def __len__(self) -> int:
        return len(self.chunks)

    def add(self, chunks: list[Chunk], vecs: np.ndarray) -> None:
        assert len(chunks) == len(vecs)
        self.chunks.extend(chunks)
        self.matrix = np.vstack([self.matrix, vecs.astype(np.float32)])

    def remove_doc(self, doc_id: str) -> int:
        keep = [i for i, c in enumerate(self.chunks) if c.doc_id != doc_id]
        removed = len(self.chunks) - len(keep)
        if removed:
            self.chunks = [self.chunks[i] for i in keep]
            self.matrix = self.matrix[keep] if keep else np.zeros((0, self.dim), dtype=np.float32)
        self.doc_hashes.pop(doc_id, None)
        return removed

    def scores(self, qvec: np.ndarray) -> np.ndarray:
        return self.matrix @ qvec if len(self) else np.zeros(0, dtype=np.float32)

    def save(self, directory: str, embedder_name: str) -> None:
        os.makedirs(directory, exist_ok=True)
        np.save(os.path.join(directory, "vectors.npy"), self.matrix)
        with open(os.path.join(directory, "chunks.json"), "w") as f:
            json.dump({"embedder": embedder_name, "dim": self.dim, "doc_hashes": self.doc_hashes,
                       "chunks": [asdict(c) for c in self.chunks]}, f)

    def load(self, directory: str, embedder_name: str) -> bool:
        p_meta, p_vec = os.path.join(directory, "chunks.json"), os.path.join(directory, "vectors.npy")
        if not (os.path.exists(p_meta) and os.path.exists(p_vec)):
            return False
        with open(p_meta) as f:
            data = json.load(f)
        if data.get("embedder") != embedder_name or data.get("dim") != self.dim:
            return False  # index was built with a different embedder: must re-ingest
        self.matrix = np.load(p_vec)
        self.chunks = [Chunk(**c) for c in data["chunks"]]
        self.doc_hashes = data.get("doc_hashes", {})
        return True
