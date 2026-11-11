from __future__ import annotations

import re

_TOKEN = re.compile(r"[a-z0-9]+")
_SENT = re.compile(r"(?<=[.!?])\s+|\n+")

STOPWORDS = frozenset("""
a an and are as at be but by can could do does did for from has have had how i if in into is it its
me my of on or our so than that the their them then there these they this to us was we were what
when where which who whom why will with would you your s t m re ve ll d about after also any been
being both just not only other over should some such very
long much many often take takes get gets happen happens tell know please need needs want
""".split())

_SUFFIXES = (("ies", "y"), ("ing", ""), ("ion", ""), ("ment", ""), ("ed", ""), ("es", ""), ("s", ""),
             ("ly", ""), ("e", ""))


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def _stem_once(tok: str) -> str:
    for suf, rep in _SUFFIXES:
        if tok.endswith(suf) and len(tok) - len(suf) >= 3:
            if suf == "s" and tok.endswith("ss"):
                return tok
            return tok[: -len(suf)] + rep
    return tok


def stem(tok: str) -> str:
    """Tiny iterative suffix stripper (regions -> region -> reg): consistent, dependency-free."""
    if len(tok) <= 3 or tok.isdigit():
        return tok
    for _ in range(2):
        nxt = _stem_once(tok)
        if nxt == tok:
            break
        tok = nxt
    return tok


def terms(text: str) -> list[str]:
    return [stem(t) for t in tokenize(text) if t not in STOPWORDS]


def split_sentences(text: str) -> list[str]:
    out = []
    for part in _SENT.split(text):
        part = re.sub(r"^\s*(?:[-*]|\d+[.)])\s+", "", part).strip()
        if part:
            out.append(part)
    return out
