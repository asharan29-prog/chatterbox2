"""Calibrated belief updates from evidence.

Model
-----
Confidence that a claim is true is ``sigmoid(L)`` where the log-odds ``L`` is::

    L = prior_logodds + adjustment + sum_g  sum_i  decay**i * c_{g,i}

* each evidence item contributes ``c = sign(stance) * strength * w(reliability)``, with
  ``w(r) = scale * log(r / (1 - r))`` clamped to ``[0, max_item_logodds]`` -- a source at
  reliability 0.5 carries no information and *less* reliable sources never count as
  counter-evidence (unreliability is not anti-truth);
* ``context`` evidence contributes 0;
* **duplicate-source discounting**: items are grouped by independence group (the source's
  ``independence_group`` or its id). Within a group and stance direction, contributions are
  sorted by magnitude and the i-th is multiplied by ``decay**i`` (default 0.5), so ten chunks
  from one blog count for at most ~2x a single chunk, while ten independent sources add up;
* ``adjustment`` is a persistent log-odds offset written by reasoning/user judgments via
  :func:`adjust_belief` (kept across recomputations);
* ``L`` is clamped to ``+-max_logodds`` so confidence never reaches exactly 0 or 1.

Status thresholds (configurable, :class:`BeliefConfig`): ``believed`` >= 0.75,
``rejected`` <= 0.15, ``doubted`` <= 0.35, otherwise ``hypothesis`` (also used when there is
no non-context evidence at all).

Every change is appended to ``Belief.history`` with the reason, actor and evidence ids.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from .store import KnowledgeStore
from .types import Belief, BeliefChange, BeliefStatus, Claim, Evidence, Source

__all__ = [
    "BeliefConfig",
    "EvidenceItem",
    "Contribution",
    "logit",
    "sigmoid",
    "reliability_weight",
    "aggregate",
    "status_for",
    "evidence_items",
    "update_belief",
    "update_all_beliefs",
    "adjust_belief",
    "set_belief",
    "explain_belief",
]

DEFAULT_NONDOC_RELIABILITY = 0.55


@dataclass(frozen=True)
class BeliefConfig:
    prior: float = 0.5
    scale: float = 1.0
    max_item_logodds: float = 3.0
    decay: float = 0.5
    max_logodds: float = 6.0
    believed: float = 0.75
    doubted: float = 0.35
    rejected: float = 0.15
    #: minimum |confidence change| that is recorded as a new history entry (status changes always are)
    min_change: float = 1e-3


DEFAULT_CONFIG = BeliefConfig()


@dataclass(frozen=True)
class EvidenceItem:
    """Pure-data input to :func:`aggregate` (decoupled from the store for testing)."""

    stance: str
    strength: float
    reliability: float
    group: str
    evidence_id: str = ""


@dataclass
class Contribution:
    evidence_id: str
    group: str
    stance: str
    raw: float
    discount: float
    value: float


@dataclass
class Aggregate:
    confidence: float
    logodds: float
    status: BeliefStatus
    contributions: list[Contribution] = field(default_factory=list)
    n_support: int = 0
    n_contradict: int = 0
    n_groups: int = 0

    def rationale(self) -> str:
        sup = sum(c.value for c in self.contributions if c.value > 0)
        con = -sum(c.value for c in self.contributions if c.value < 0)
        return (
            f"{self.n_support} supporting / {self.n_contradict} contradicting items from {self.n_groups} "
            f"independent source group(s); log-odds +{sup:.2f} / -{con:.2f} -> {self.confidence:.2f}"
        )


def logit(p: float) -> float:
    p = min(max(p, 1e-9), 1 - 1e-9)
    return math.log(p / (1 - p))


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    e = math.exp(x)
    return e / (1 + e)


def reliability_weight(r: float, cfg: BeliefConfig = DEFAULT_CONFIG) -> float:
    """Log-odds weight of one full-strength item from a source of reliability ``r``."""
    return min(cfg.max_item_logodds, max(0.0, cfg.scale * logit(r)))


def status_for(confidence: float, has_evidence: bool = True, cfg: BeliefConfig = DEFAULT_CONFIG) -> BeliefStatus:
    if not has_evidence:
        return "hypothesis"
    if confidence >= cfg.believed:
        return "believed"
    if confidence <= cfg.rejected:
        return "rejected"
    if confidence <= cfg.doubted:
        return "doubted"
    return "hypothesis"


def aggregate(
    items: Iterable[EvidenceItem], *, adjustment: float = 0.0, cfg: BeliefConfig = DEFAULT_CONFIG
) -> Aggregate:
    """Combine evidence into a calibrated confidence (pure function)."""
    by_key: dict[tuple[str, int], list[tuple[float, EvidenceItem]]] = defaultdict(list)
    n_sup = n_con = 0
    groups: set[str] = set()
    for it in items:
        if it.stance == "supports":
            sign = 1
            n_sup += 1
        elif it.stance == "contradicts":
            sign = -1
            n_con += 1
        else:
            continue
        raw = sign * max(0.0, min(1.0, it.strength)) * reliability_weight(it.reliability, cfg)
        by_key[(it.group, sign)].append((raw, it))
        groups.add(it.group)
    contributions: list[Contribution] = []
    total = 0.0
    for (group, _sign), lst in by_key.items():
        lst.sort(key=lambda x: -abs(x[0]))
        for i, (raw, it) in enumerate(lst):
            disc = cfg.decay**i
            val = raw * disc
            total += val
            contributions.append(Contribution(it.evidence_id, group, it.stance, raw, disc, val))
    L = logit(cfg.prior) + adjustment + total
    L = max(-cfg.max_logodds, min(cfg.max_logodds, L))
    conf = sigmoid(L)
    has_ev = (n_sup + n_con) > 0 or abs(adjustment) > 1e-9
    return Aggregate(
        confidence=conf,
        logodds=L,
        status=status_for(conf, has_ev, cfg),
        contributions=contributions,
        n_support=n_sup,
        n_contradict=n_con,
        n_groups=len(groups),
    )


# ----------------------------------------------------------------------------- store-backed
def evidence_items(store: KnowledgeStore, evidence: Sequence[Evidence]) -> list[EvidenceItem]:
    """Resolve each evidence row's source reliability and independence group."""
    cache: dict[str, Source | None] = {}
    out = []
    for ev in evidence:
        if ev.source_id:
            if ev.source_id not in cache:
                cache[ev.source_id] = store.get_source(ev.source_id)
            src = cache[ev.source_id]
            rel = src.reliability if src else DEFAULT_NONDOC_RELIABILITY
            group = src.group if src else ev.source_id
        else:
            rel = float(ev.metadata.get("reliability", DEFAULT_NONDOC_RELIABILITY))
            group = str(ev.metadata.get("group", f"{ev.method}:{ev.id}"))
        out.append(EvidenceItem(ev.stance, ev.strength, rel, group, ev.id))
    return out


def _record(
    belief: Belief,
    agg_conf: float,
    status: BeliefStatus,
    reason: str,
    evidence_ids: list[str],
    actor: str,
    cfg: BeliefConfig,
    force: bool = False,
) -> bool:
    old_c = belief.confidence if belief.history else None
    old_s = belief.status if belief.history else None
    changed = force or old_c is None or abs(agg_conf - old_c) >= cfg.min_change or status != old_s
    if changed:
        belief.history.append(
            BeliefChange(
                old_confidence=old_c,
                new_confidence=round(agg_conf, 6),
                old_status=old_s,
                new_status=status,
                reason=reason,
                evidence_ids=evidence_ids,
                actor=actor,
            )
        )
    belief.confidence = round(agg_conf, 6)
    belief.status = status
    return changed


def update_belief(
    store: KnowledgeStore,
    claim_id: str,
    *,
    reason: str = "evidence update",
    actor: str = "system",
    cfg: BeliefConfig = DEFAULT_CONFIG,
) -> Belief:
    """(Re)compute the belief for ``claim_id`` from all its evidence and persist it.

    Creates the belief if needed. Appends a history entry when confidence or status changed.
    """
    if store.get_claim(claim_id) is None:
        raise KeyError(f"claim {claim_id!r} not found")
    belief = store.get_belief_for_claim(claim_id) or Belief(claim_id=claim_id, confidence=cfg.prior)
    evs = store.list_evidence(claim_id)
    agg = aggregate(evidence_items(store, evs), adjustment=belief.adjustment, cfg=cfg)
    new_ids = [e.id for e in evs if e.id not in set(belief.evidence_ids)]
    _record(belief, agg.confidence, agg.status, reason, new_ids, actor, cfg)
    belief.evidence_ids = [e.id for e in evs]
    belief.rationale = agg.rationale()
    belief.metadata["logodds"] = round(agg.logodds, 4)
    belief.metadata["n_groups"] = agg.n_groups
    return store.save_belief(belief)


def update_all_beliefs(
    store: KnowledgeStore, *, reason: str = "recompute", actor: str = "system", cfg: BeliefConfig = DEFAULT_CONFIG
) -> list[Belief]:
    return [update_belief(store, c.id, reason=reason, actor=actor, cfg=cfg) for c in store.list(Claim)]


def adjust_belief(
    store: KnowledgeStore,
    claim_id: str,
    delta_logodds: float,
    *,
    reason: str,
    actor: str = "reasoning",
    cfg: BeliefConfig = DEFAULT_CONFIG,
) -> Belief:
    """Add a persistent log-odds adjustment (e.g. from a JUDGE step) and recompute.

    Prefer adding :class:`Evidence` (with provenance) when possible; use this for judgments that
    are not attributable to a single document. ``reason`` is mandatory and is logged.
    """
    belief = store.get_belief_for_claim(claim_id) or Belief(claim_id=claim_id, confidence=cfg.prior)
    belief.adjustment = float(belief.adjustment + delta_logodds)
    store.save_belief(belief)
    return update_belief(store, claim_id, reason=f"{reason} (adjustment {delta_logodds:+.2f})", actor=actor, cfg=cfg)


def set_belief(
    store: KnowledgeStore,
    claim_id: str,
    confidence: float,
    *,
    reason: str,
    actor: str = "user",
    cfg: BeliefConfig = DEFAULT_CONFIG,
) -> Belief:
    """Force the belief to ``confidence`` by setting the adjustment that achieves it."""
    belief = store.get_belief_for_claim(claim_id) or Belief(claim_id=claim_id, confidence=cfg.prior)
    evs = store.list_evidence(claim_id)
    base = aggregate(evidence_items(store, evs), adjustment=0.0, cfg=cfg)
    target = max(-cfg.max_logodds, min(cfg.max_logodds, logit(confidence)))
    unclamped = logit(cfg.prior) + sum(c.value for c in base.contributions)
    belief.adjustment = target - unclamped
    store.save_belief(belief)
    return update_belief(store, claim_id, reason=f"{reason} (set to {confidence:.2f})", actor=actor, cfg=cfg)


def explain_belief(store: KnowledgeStore, belief_or_claim_id: str, cfg: BeliefConfig = DEFAULT_CONFIG) -> dict[str, Any]:
    """Structured explanation: claim, confidence, per-evidence contributions, history."""
    belief = store.get_belief(belief_or_claim_id) or store.get_belief_for_claim(belief_or_claim_id)
    if belief is None:
        raise KeyError(f"no belief for {belief_or_claim_id!r}")
    claim = store.get_claim(belief.claim_id)
    evs = store.list_evidence(belief.claim_id)
    agg = aggregate(evidence_items(store, evs), adjustment=belief.adjustment, cfg=cfg)
    by_id = {e.id: e for e in evs}
    contribs = []
    for c in sorted(agg.contributions, key=lambda c: -abs(c.value)):
        ev = by_id.get(c.evidence_id)
        src = store.get_source(ev.source_id) if ev and ev.source_id else None
        contribs.append(
            {
                "evidence_id": c.evidence_id,
                "stance": c.stance,
                "logodds": round(c.value, 4),
                "discount": c.discount,
                "source_id": ev.source_id if ev else None,
                "source_title": src.title if src else None,
                "reliability": src.reliability if src else None,
                "quote": ev.quote if ev else "",
            }
        )
    return {
        "belief_id": belief.id,
        "claim_id": belief.claim_id,
        "claim": claim.text if claim else "",
        "confidence": belief.confidence,
        "status": belief.status,
        "rationale": belief.rationale,
        "adjustment": belief.adjustment,
        "contributions": contribs,
        "context_evidence": [e.id for e in evs if e.stance == "context"],
        "history": [h.model_dump(mode="json") for h in belief.history],
    }
