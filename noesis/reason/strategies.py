"""JUDGE strategies: ``simple``, ``self_consistency``, ``debate``.

A strategy turns (claim, evidence documents) into a :class:`Judgment`. Judgments are only a
*minor* input to belief confidence: :class:`noesis.reason.Mind` converts them into a small,
bounded log-odds adjustment scaled by ``agreement``; the bulk of a belief's confidence always
comes from evidence aggregation in :mod:`noesis.beliefs`.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from ..llm.base import LLMError
from ..store import KnowledgeStore
from . import prompts as P
from .citations import validate_citations

__all__ = ["Judgment", "JudgeContext", "Strategy", "SimpleStrategy", "SelfConsistencyStrategy", "DebateStrategy", "get_strategy"]


@dataclass
class Judgment:
    verdict: str = "uncertain"  # true|false|uncertain
    llm_confidence: float = 0.5  # model-reported P(claim true)
    agreement: float = 1.0  # 0..1 strength of the strategy's internal agreement
    citations: list[str] = field(default_factory=list)  # validated chunk ids
    dropped_citations: list[str] = field(default_factory=list)
    rationale: str = ""
    samples: list[dict[str, Any]] = field(default_factory=list)
    arguments: dict[str, Any] = field(default_factory=dict)
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "llm_confidence": round(self.llm_confidence, 4),
            "agreement": round(self.agreement, 4),
            "citations": self.citations,
            "dropped_citations": self.dropped_citations,
            "rationale": self.rationale,
            "samples": self.samples,
            "arguments": self.arguments,
            "error": self.error,
        }


@dataclass
class JudgeContext:
    llm: Any
    store: KnowledgeStore
    rng: random.Random


def _norm_verdict(v: Any) -> str:
    v = str(v or "").strip().lower()
    if v in ("true", "yes", "supported"):
        return "true"
    if v in ("false", "no", "refuted", "unsupported"):
        return "false"
    return "uncertain"


def _conf(x: Any) -> float:
    try:
        f = float(x)
    except (TypeError, ValueError):
        return 0.5
    return 0.5 if math.isnan(f) else max(0.0, min(1.0, f))


def _judge_once(ctx: JudgeContext, claim: str, docs: list[dict], *, temperature: float = 0.0, extra: dict[str, str] | None = None) -> dict:
    fields = {"claim": claim, **(extra or {})}
    data = P.call(ctx.llm, "judge", P.make("judge", documents=docs, fields=fields), temperature=temperature)
    chk = validate_citations(ctx.store, data.get("citations") or [], allowed=[d["id"] for d in docs])
    return {
        "verdict": _norm_verdict(data.get("verdict")),
        "confidence": _conf(data.get("confidence")),
        "rationale": str(data.get("rationale") or ""),
        "citations": chk.valid,
        "dropped": chk.dropped,
    }


class Strategy:
    name = "base"

    def judge(self, ctx: JudgeContext, claim: str, docs: list[dict]) -> Judgment:  # pragma: no cover
        raise NotImplementedError


class SimpleStrategy(Strategy):
    """One judgment per hypothesis; agreement = 1."""

    name = "simple"

    def judge(self, ctx: JudgeContext, claim: str, docs: list[dict]) -> Judgment:
        try:
            r = _judge_once(ctx, claim, docs)
        except LLMError as e:
            return Judgment(error=str(e), agreement=0.0, rationale="judge failed")
        return Judgment(r["verdict"], r["confidence"], 1.0, r["citations"], r["dropped"], r["rationale"], [r])


class SelfConsistencyStrategy(Strategy):
    """k independent judgments, each over a seeded bootstrap subset of the evidence (and with
    temperature > 0 for samples after the first). Majority vote; ``agreement`` = share of the
    majority verdict, so a split jury barely moves the belief."""

    name = "self_consistency"

    def __init__(self, k: int = 3, subset: float = 0.7, temperature: float = 0.7) -> None:
        self.k = max(1, k)
        self.subset = subset
        self.temperature = temperature

    def judge(self, ctx: JudgeContext, claim: str, docs: list[dict]) -> Judgment:
        samples: list[dict] = []
        errors = 0
        for i in range(self.k):
            if i == 0 or len(docs) <= 1:
                sub = list(docs)
            else:
                n = max(1, math.ceil(self.subset * len(docs)))
                sub = ctx.rng.sample(list(docs), n)
            try:
                r = _judge_once(ctx, claim, sub, temperature=0.0 if i == 0 else self.temperature)
            except LLMError:
                errors += 1
                continue
            r["docs"] = [d["id"] for d in sub]
            samples.append(r)
        if not samples:
            return Judgment(error=f"{errors} judge samples failed", agreement=0.0)
        votes = Counter(s["verdict"] for s in samples)
        top = votes.most_common()
        verdict = top[0][0] if len(top) == 1 or top[0][1] > top[1][1] else "uncertain"
        agreement = votes[verdict] / len(samples) if verdict != "uncertain" or votes["uncertain"] else 0.0
        conf = sum(s["confidence"] for s in samples) / len(samples)
        cites = list(dict.fromkeys(c for s in samples for c in s["citations"]))
        dropped = list(dict.fromkeys(c for s in samples for c in s["dropped"]))
        rationale = f"{dict(votes)} over {len(samples)} samples; " + (samples[0]["rationale"] or "")
        return Judgment(verdict, conf, agreement, cites, dropped, rationale, samples)


class DebateStrategy(Strategy):
    """Two advocates (pro / con) argue from the evidence, then a judge rules on the claim with
    both arguments in view. Agreement is 1 when the verdict sides with the advocate who backed
    their case with more *valid* citations, 0.5 otherwise (and for 'uncertain')."""

    name = "debate"

    def _advocate(self, ctx: JudgeContext, claim: str, docs: list[dict], role: str) -> dict:
        side = "FOR" if role == "pro" else "AGAINST"
        data = P.call(ctx.llm, "advocate", P.make("advocate", documents=docs, fields={"claim": claim, "role": role}, role=role, side=side))
        chk = validate_citations(ctx.store, data.get("citations") or [], allowed=[d["id"] for d in docs])
        return {"arguments": [str(a) for a in (data.get("arguments") or [])][:5], "citations": chk.valid, "dropped": chk.dropped}

    def judge(self, ctx: JudgeContext, claim: str, docs: list[dict]) -> Judgment:
        try:
            pro = self._advocate(ctx, claim, docs, "pro")
            con = self._advocate(ctx, claim, docs, "con")
            extra = {
                "pro_arguments": " | ".join(pro["arguments"]) or "(none)",
                "con_arguments": " | ".join(con["arguments"]) or "(none)",
            }
            r = _judge_once(ctx, claim, docs, extra=extra)
        except LLMError as e:
            return Judgment(error=str(e), agreement=0.0, rationale="debate failed")
        np_, nc = len(pro["citations"]), len(con["citations"])
        stronger = "true" if np_ > nc else "false" if nc > np_ else "uncertain"
        agreement = 1.0 if r["verdict"] != "uncertain" and r["verdict"] == stronger else 0.5
        cites = list(dict.fromkeys(r["citations"] + pro["citations"] + con["citations"]))
        dropped = list(dict.fromkeys(r["dropped"] + pro["dropped"] + con["dropped"]))
        return Judgment(
            r["verdict"], r["confidence"], agreement, cites, dropped,
            r["rationale"], [r], {"pro": pro, "con": con},
        )


def get_strategy(name: str | Strategy, **kwargs: Any) -> Strategy:
    if isinstance(name, Strategy):
        return name
    key = (name or "simple").lower().replace("-", "_")
    if key == "simple":
        return SimpleStrategy()
    if key in ("self_consistency", "sc", "consistency"):
        return SelfConsistencyStrategy(**kwargs)
    if key == "debate":
        return DebateStrategy()
    raise ValueError(f"unknown strategy {name!r}; expected simple | self_consistency | debate")
