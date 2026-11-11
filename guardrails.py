from __future__ import annotations

import re
from dataclasses import dataclass, field

from .text_utils import split_sentences, terms
from .models import Hit

_INJECTION = [re.compile(p, re.I) for p in (
    r"ignore (all |any |the )?(previous|prior|above|earlier) (instructions|prompts?|rules)",
    r"disregard (all |any |the )?(previous|prior|above|system) ",
    r"(reveal|show|print|repeat|leak) (me )?(your |the )?(system|hidden|initial) (prompt|instructions|message)",
    r"you are now (in )?(dan|developer mode|jailbroken|an? unrestricted)",
    r"(developer|god|jailbreak) mode",
    r"act as (if you have )?no (rules|restrictions|guidelines)",
    r"override (your )?(safety|guardrails|instructions)",
)]
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_PHONE = re.compile(r"(?<!\w)(?:\+?\d{1,3}[\s-]?)?(?:\(?\d{3,4}\)?[\s-]?)\d{3}[\s-]?\d{3,4}(?!\w)")
_CARD = re.compile(r"\b(?:\d[ -]?){13,19}\b")
_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")


def _luhn(digits: str) -> bool:
    total, alt = 0, False
    for d in reversed(digits):
        n = int(d)
        if alt:
            n = n * 2 - 9 if n * 2 > 9 else n * 2
        total += n
        alt = not alt
    return total % 10 == 0


@dataclass
class GuardResult:
    blocked: bool = False
    reasons: list = field(default_factory=list)
    sanitized: str = ""


def redact_cards_ssn(text: str) -> tuple[str, int]:
    count = 0

    def card(m):
        nonlocal count
        digits = re.sub(r"\D", "", m.group())
        if 13 <= len(digits) <= 19 and _luhn(digits):
            count += 1
            return "[CARD]"
        return m.group()

    def ssn(_):
        nonlocal count
        count += 1
        return "[SSN]"

    return _SSN.sub(ssn, _CARD.sub(card, text)), count


def grounding_score(answer: str, evidence_texts: list[str], threshold: float = 0.7) -> float:
    """Fraction of answer sentences whose content words are covered by the retrieved evidence.

    A fast, deterministic faithfulness proxy (no LLM call). In production, pair it with an
    LLM-as-judge on a sampled fraction of traffic.
    """
    ev = set(terms(" ".join(evidence_texts)))
    clean = re.sub(r"\[\d+\]", "", answer)
    sents = [s for s in split_sentences(clean) if len(terms(s)) >= 3]
    if not sents:
        return 1.0
    ok = 0
    for s in sents:
        ts = terms(s)
        ok += (sum(t in ev for t in ts) / len(ts)) >= threshold
    return ok / len(sents)


class Guardrails:
    def check_input(self, text: str) -> GuardResult:
        reasons = [f"prompt_injection:{p.pattern[:30]}" for p in _INJECTION if p.search(text)]
        if reasons:
            return GuardResult(True, reasons, text)
        clean = _EMAIL.sub("[EMAIL]", text)
        clean, _ = redact_cards_ssn(clean)
        clean = _PHONE.sub(lambda m: "[PHONE]" if len(re.sub(r"\D", "", m.group())) >= 10 else m.group(), clean)
        return GuardResult(False, ["pii_redacted"] if clean != text else [], clean)

    def filter_context(self, hits: list[Hit]) -> tuple[list[Hit], int]:
        """Indirect prompt-injection defence: drop retrieved passages that try to instruct the model."""
        safe = [h for h in hits if not any(p.search(h.chunk.text) for p in _INJECTION)]
        return safe, len(hits) - len(safe)

    def check_output(self, text: str) -> tuple[str, list[str]]:
        clean, n = redact_cards_ssn(text)
        return clean, (["output_pii_redacted"] if n else [])
