from __future__ import annotations

import json
import os
import re

# Domain glossary: bridges acronyms / colloquial phrasing to the vocabulary used in the documents.
# Override with a JSON file via RAG_GLOSSARY=/path/glossary.json
DEFAULT_GLOSSARY = {
    "sso": "single sign-on",
    "mfa": "multi-factor authentication",
    "2fa": "multi-factor authentication",
    "money back": "refund",
    "cost": "price",
    "tier": "plan",
    "data location": "data residency region",
}


def _load() -> dict[str, str]:
    path = os.getenv("RAG_GLOSSARY")
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return {k.lower(): v for k, v in json.load(f).items()}
    return DEFAULT_GLOSSARY


GLOSSARY = _load()


def expand_query(query: str) -> str:
    """Append glossary aliases for matched phrases. Idempotent: never adds an alias twice."""
    low = query.lower()
    extra = [alias for key, alias in GLOSSARY.items()
             if re.search(rf"(?<!\w){re.escape(key)}(?!\w)", low) and alias.lower() not in low]
    return query + (" " + " ".join(extra) if extra else "")
