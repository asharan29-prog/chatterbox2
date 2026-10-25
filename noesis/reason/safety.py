"""Defense in depth against prompt injection carried by ingested text.

The primary defense is structural (untrusted text only ever appears inside ``<document>``
wrappers under the safety system prompt). This module adds an output-side filter: sentences
that look like instructions aimed at an AI are never promoted into hypotheses, questions or
answers, and are reported in the trace instead.
"""

from __future__ import annotations

import re

from .. import nlp

__all__ = ["looks_like_injection", "strip_injection"]

_PATTERNS = [
    r"\bignore (all |any )?(the )?(previous|prior|above|earlier) (instructions|prompts?|rules)\b",
    r"\b(note|message|instructions?) to (any|all|the) (ai|assistant|model|llm)s?\b",
    r"\b(as an ai|you are now|act as|new instructions|system prompt|developer mode)\b",
    r"\bmark (every|all|each) claims?\b.*\b(believed|true|confidence)\b",
    r"\b(set|give) (the |its |every )?confidence (to|of) 1(\.0+)?\b",
    r"\bdisregard (the |your )?(rules|instructions|guidelines)\b",
]
_RE = re.compile("|".join(_PATTERNS), re.I)


def looks_like_injection(text: str) -> bool:
    return bool(_RE.search(text or ""))


def strip_injection(text: str) -> tuple[str, list[str]]:
    """Remove injection-like sentences. Returns ``(clean_text, removed_sentences)``."""
    if not looks_like_injection(text):
        return text, []
    kept, removed = [], []
    for sent, _s, _e in nlp.split_sentences(text) or [(text, 0, len(text))]:
        (removed if looks_like_injection(sent) else kept).append(sent)
    return " ".join(kept).strip(), removed
