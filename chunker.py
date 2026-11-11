from __future__ import annotations

import re

from .text_utils import split_sentences
from .models import Chunk


def _pack(sentences: list[str], size: int, overlap: int) -> list[str]:
    """Greedy sentence packing with sentence-level overlap between neighbours."""
    chunks: list[str] = []
    cur: list[str] = []
    n = 0
    for s in sentences:
        if cur and n + len(s) + 1 > size:
            chunks.append(" ".join(cur))
            keep: list[str] = []
            m = 0
            for prev in reversed(cur):
                if m + len(prev) > overlap:
                    break
                keep.insert(0, prev)
                m += len(prev) + 1
            cur, n = keep, m
        cur.append(s)
        n += len(s) + 1
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def chunk_document(doc_id: str, text: str, title: str | None = None, size: int = 420,
                   overlap: int = 80, meta: dict | None = None) -> list[Chunk]:
    """Structure-aware chunking: markdown headings define sections, chunks never cross them."""
    title_given = title is not None
    title = title or doc_id
    sections: list[tuple[str, list[str]]] = []
    heading, cur = "", []

    for block in re.split(r"\n\s*\n", text.strip()):
        block = block.strip()
        if not block:
            continue
        if block.startswith("#"):
            first, _, rest = block.partition("\n")
            level = len(first) - len(first.lstrip("#"))
            head = first.lstrip("#").strip()
            if cur:
                sections.append((heading, cur))
                cur = []
            if level == 1 and not title_given:
                title, heading = head, ""
            else:
                heading = head
            if rest.strip():
                cur.extend(split_sentences(rest))
        else:
            cur.extend(split_sentences(block))
    if cur:
        sections.append((heading, cur))

    chunks: list[Chunk] = []
    for heading, sents in sections:
        for piece in _pack(sents, size, overlap):
            chunks.append(Chunk(id=f"{doc_id}#{len(chunks)}", doc_id=doc_id, title=title,
                                section=heading, text=piece, meta=dict(meta or {})))
    return chunks
