"""SELECT QUESTION: expected-value scoring of open questions.

``value = uncertainty x importance x novelty + curiosity_bonus``

* **uncertainty** (0..1): mean of ``1 - |2p - 1|`` over the beliefs of the question's claims
  (``p`` = belief confidence); questions with no linked claims count as fairly uncertain
  (``unknown_uncertainty``, default 0.8) -- we know nothing about them yet.
* **importance** (0..1): the question's stored priority, blended with relevance to the run goal
  (TF-IDF similarity) and concept centrality (how many chunks mention its concepts/claims).
* **novelty** (0..1): ``1 - max similarity`` to questions already asked in this run or already
  resolved, times ``0.5 ** attempts`` (inconclusive earlier attempts) and a penalty if the
  question was already picked in this run.
* **curiosity bonus**: added for questions born from contradictions (+0.15) or gaps (+0.10) and
  for model curiosity (+0.05), scaled by uncertainty so settled conflicts stop attracting attention.

Resolved questions (status ``answered``/``dropped``) are never candidates, and near-duplicates
(similarity >= ``dup_threshold``) of resolved questions are excluded too.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .. import nlp
from ..types import Belief, Question

__all__ = ["SelectionConfig", "ScoredQuestion", "uncertainty", "score_question", "rank_questions", "is_duplicate"]

CURIOSITY_BONUS = {"contradiction": 0.15, "gap": 0.10, "curiosity": 0.05, "user": 0.0}


@dataclass
class SelectionConfig:
    unknown_uncertainty: float = 0.8
    w_priority: float = 0.6
    w_goal: float = 0.25
    w_centrality: float = 0.15
    dup_threshold: float = 0.85
    repeat_penalty: float = 0.3
    attempt_decay: float = 0.5
    #: convergence: stop when no candidate's value exceeds this
    min_value: float = 0.05


@dataclass
class ScoredQuestion:
    question: Question
    value: float
    uncertainty: float
    importance: float
    novelty: float
    bonus: float
    parts: dict[str, Any] = field(default_factory=dict)

    def explain(self) -> str:
        return (
            f"value {self.value:.3f} = uncertainty {self.uncertainty:.2f} x importance {self.importance:.2f} "
            f"x novelty {self.novelty:.2f} + curiosity {self.bonus:.2f}"
        )


def uncertainty(p: float) -> float:
    """1 at p=0.5, 0 at p in {0, 1}."""
    return max(0.0, 1.0 - abs(2.0 * float(p) - 1.0))


def is_duplicate(text: str, others: Sequence[str], threshold: float = 0.85) -> bool:
    norm = " ".join(nlp.content_words(text))
    for o in others:
        if norm and norm == " ".join(nlp.content_words(o)):
            return True
        if nlp.text_similarity(text, o) >= threshold:
            return True
    return False


def score_question(
    q: Question,
    *,
    beliefs: dict[str, Belief],
    goal: str | None = None,
    asked: Sequence[str] = (),
    resolved: Sequence[str] = (),
    picked_ids: Sequence[str] = (),
    centrality: float = 0.0,
    cfg: SelectionConfig | None = None,
) -> ScoredQuestion:
    cfg = cfg or SelectionConfig()
    # uncertainty
    ps = [beliefs[c].confidence for c in q.claim_ids if c in beliefs]
    if ps:
        unc = sum(uncertainty(p) for p in ps) / len(ps)
    elif q.answer_confidence is not None:
        unc = uncertainty(q.answer_confidence)
    else:
        unc = cfg.unknown_uncertainty
    # importance
    goal_rel = nlp.text_similarity(goal, q.text) if goal else 0.0
    if goal:
        imp = cfg.w_priority * q.priority + cfg.w_goal * min(1.0, 2.0 * goal_rel) + cfg.w_centrality * centrality
    else:
        tot = cfg.w_priority + cfg.w_centrality
        imp = (cfg.w_priority * q.priority + cfg.w_centrality * centrality) / tot
    imp = max(0.05, min(1.0, imp))
    # novelty
    prior = list(asked) + list(resolved)
    max_sim = max((nlp.text_similarity(q.text, o) for o in prior if o != q.text), default=0.0)
    nov = 1.0 - max_sim
    attempts = int(q.metadata.get("attempts", 0) or 0)
    nov *= cfg.attempt_decay**attempts
    if q.id in picked_ids:
        nov *= cfg.repeat_penalty
    nov = max(0.0, min(1.0, nov))
    bonus = CURIOSITY_BONUS.get(q.origin, 0.0) * unc * (1.0 if nov > 0.2 else nov / 0.2)
    value = unc * imp * nov + bonus
    return ScoredQuestion(
        q, round(value, 6), round(unc, 4), round(imp, 4), round(nov, 4), round(bonus, 4),
        {"goal_relevance": round(goal_rel, 4), "centrality": round(centrality, 4), "attempts": attempts},
    )


def rank_questions(
    questions: Sequence[Question],
    *,
    beliefs: dict[str, Belief],
    goal: str | None = None,
    asked: Sequence[str] = (),
    resolved: Sequence[str] = (),
    picked_ids: Sequence[str] = (),
    centrality: dict[str, float] | None = None,
    cfg: SelectionConfig | None = None,
) -> list[ScoredQuestion]:
    """Score open, non-duplicate questions; best first (ties broken by text for determinism)."""
    cfg = cfg or SelectionConfig()
    out = []
    for q in questions:
        if q.status in ("answered", "dropped"):
            continue
        if resolved and is_duplicate(q.text, resolved, cfg.dup_threshold):
            continue
        out.append(
            score_question(
                q, beliefs=beliefs, goal=goal, asked=asked, resolved=resolved, picked_ids=picked_ids,
                centrality=(centrality or {}).get(q.id, 0.0), cfg=cfg,
            )
        )
    out.sort(key=lambda s: (-round(s.value, 6), canonical_text(s.question.text)))
    return out


def canonical_text(text: str) -> str:
    """Order-insensitive key for tie-breaking ("A or B" == "B or A" for quoted alternatives)."""
    quoted = re.findall(r'"([^"]+)"', text)
    return " | ".join(sorted(quoted)) if len(quoted) >= 2 else text
