"""Pure scoring functions for the Noesis benchmark (no store / model access here).

Everything in this module is deliberately simple and transparent so a reader can check how a
number was produced. Answer correctness uses per-question keyword groups (:data:`GOLD_KEYS`)
plus, for yes/no questions, the polarity of the answer's first on-topic sentence. This is a
heuristic: it can be fooled by paraphrase and by answers that hedge both ways. Use the optional
LLM judge (``--llm-judge``) with a real model for a second opinion.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from typing import Any

__all__ = [
    "GOLD_KEYS",
    "normalize",
    "tokens",
    "split_sentences",
    "split_clauses",
    "strip_hedge",
    "is_negated",
    "keyword_groups_hit",
    "score_answer",
    "token_overlap",
    "brier",
    "ece",
    "quote_in_chunk",
    "mean",
]

# Per-question gold keys for examples/corpus/questions.json.
#   require:  list of keyword groups; every group needs at least one hit (substring match on
#             normalized text, so "dehydrat" matches "dehydration").
#   polarity: "yes" / "no" for yes/no questions -- checked on the first sentence that hits the
#             first require group.
GOLD_KEYS: dict[str, dict[str, Any]] = {
    "q01": {"polarity": "yes", "require": [["mortality", "death", "die"]]},
    "q02": {"polarity": "no", "require": [["pressure"], ["lasting", "long", "persist", "habitual", "regular", "tolerance"]]},
    "q03": {"polarity": "yes", "require": [["pressure"], ["rarely", "briefly", "short", "hour", "temporar", "mmhg"]]},
    "q04": {"require": [["lower"], ["diabetes"]]},
    "q05": {"polarity": "no", "require": [["dehydrat", "fluid", "diuretic"]]},
    "q06": {"polarity": "no", "require": [["heart"]]},
    "q07": {"require": [["cafestol"]]},
    "q08": {"require": [["200"]]},
    "q09": {"require": [["5 hour", "five hour", "5-hour"]]},
    "q10": {"require": [["sleep"], ["40", "reduc", "less", "shorter", "fewer", "lost"]]},
    "q11": {"polarity": "yes", "require": [["decaf"], ["diabetes", "similar"]]},
    "q12": {"polarity": "no", "require": [["prove", "caus", "observational", "associat"]]},
    "q13": {"require": [["400"]]},
    "q14": {"polarity": "no", "require": [["atrial", "fibrillation"]]},
}

_NUM_WORDS = {
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7",
    "eight": "8", "nine": "9", "ten": "10", "forty": "40", "two hundred": "200", "four hundred": "400",
}
_NEG = re.compile(
    r"\b(not|never|cannot|neither|nor|insufficient|unproven|\w+n't)\b"
    r"|\bno (association|evidence|link|effect|lasting|increase|significant|proof|clear|good|higher|lower)\b"
    r"|\b(not enough|lack of|no longer)\b|^\s*no\b",
    re.I,
)
_STOP = set(
    "a an the of to in on for and or but is are was were be been being by with as at from that this these those "
    "it its does do did can could should would will may might has have had than then so such about into per "
    "which who whom what why how when where there their they them his her he she you your we our i".split()
)


def normalize(text: str) -> str:
    t = " ".join((text or "").lower().replace("’", "'").split())
    for w, d in sorted(_NUM_WORDS.items(), key=lambda kv: -len(kv[0])):
        t = re.sub(rf"\b{w}\b", d, t)
    return t


def _stem(w: str) -> str:
    for suf in ("ations", "ation", "ings", "ing", "edly", "ed", "ies", "es", "ly", "s"):
        if len(w) > len(suf) + 2 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def tokens(text: str) -> list[str]:
    return [_stem(w) for w in re.findall(r"[a-z0-9]+", normalize(text)) if w not in _STOP]


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(#])", (text or "").strip())
    return [p.strip() for p in parts if p.strip()]


def split_clauses(text: str) -> list[str]:
    """Sentences further split on contrastive joins (", but", "; ", ", whereas", ", although")."""
    out: list[str] = []
    for sent in split_sentences(text):
        out += [c.strip() for c in re.split(r",?\s+\bbut\b\s+|;\s+|,\s+(?:whereas|although|while|however)\s+", sent) if c.strip()]
    return out


def strip_hedge(answer: str) -> str:
    """Remove Noesis' 'Insufficient evidence ... Tentatively:' wrapper, keeping the tentative content."""
    a = answer or ""
    m = re.search(r"tentatively:\s*", a, re.I)
    if m and a.lower().startswith("insufficient evidence"):
        return a[m.end():]
    if a.lower().startswith("insufficient evidence"):
        return ""
    return a


def is_negated(sentence: str) -> bool:
    return bool(_NEG.search(sentence or ""))


def keyword_groups_hit(text: str, groups: Sequence[Sequence[str]]) -> list[bool]:
    t = normalize(text)
    return [any(normalize(k) in t for k in g) for g in groups]


def _lead_polarity(text: str) -> str | None:
    m = re.match(r"\s*(yes|no)\b", text or "", re.I)
    return m.group(1).lower() if m else None


def score_answer(qid: str, answer: str, insufficient: bool) -> dict[str, Any]:
    """Keyword/polarity correctness for one corpus question.

    Returns ``correct`` (strict: an abstention is never correct for an answerable question) and
    ``correct_if_committed`` (the same check applied to the tentative content of a hedged answer).
    """
    spec = GOLD_KEYS.get(qid)
    if spec is None:
        return {"correct": False, "correct_if_committed": False, "detail": "no gold keys"}
    content = strip_hedge(answer)
    groups = spec["require"]
    hits = keyword_groups_hit(content, groups)
    ok = bool(content) and all(hits)
    detail: dict[str, Any] = {"keyword_hits": hits}
    pol = spec.get("polarity")
    if ok and pol:
        lead = _lead_polarity(content)
        if lead:
            got = lead
        else:
            # polarity of the first clause that hits every keyword group (else the first group)
            clauses = split_clauses(content)
            full = [c for c in clauses if all(keyword_groups_hit(c, groups))]
            topical = full or [c for c in clauses if keyword_groups_hit(c, groups[:1])[0]]
            first = topical[0] if topical else content
            got = "no" if is_negated(first) else "yes"
            detail["polarity_sentence"] = first[:200]
        detail["polarity"] = got
        ok = got == pol
    return {"correct": ok and not insufficient, "correct_if_committed": ok, "detail": detail}


def token_overlap(pred: str, gold: str) -> float:
    """Recall of gold content tokens in the prediction (0..1): a soft 'semantic overlap' score."""
    g = set(tokens(gold)) - {"yes", "no"}
    if not g:
        return 0.0
    p = set(tokens(pred))
    return round(len(g & p) / len(g), 4)


def brier(confs: Sequence[float], outcomes: Sequence[int]) -> float | None:
    if not confs:
        return None
    return round(sum((c - y) ** 2 for c, y in zip(confs, outcomes, strict=True)) / len(confs), 4)


def ece(confs: Sequence[float], outcomes: Sequence[int], bins: int = 5) -> float | None:
    """Expected calibration error with equal-width bins."""
    if not confs:
        return None
    total = 0.0
    n = len(confs)
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(confs) if (lo <= c < hi) or (b == bins - 1 and c == 1.0)]
        if not idx:
            continue
        acc = sum(outcomes[i] for i in idx) / len(idx)
        conf = sum(confs[i] for i in idx) / len(idx)
        total += len(idx) / n * abs(acc - conf)
    return round(total, 4)


def quote_in_chunk(quote: str, chunk_text: str) -> bool:
    q = " ".join((quote or "").split())
    c = " ".join((chunk_text or "").split())
    return bool(q) and q in c


def mean(xs: Sequence[float]) -> float | None:
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return round(sum(xs) / len(xs), 4) if xs else None
