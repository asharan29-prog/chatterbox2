"""Core data model for Noesis.

Every object is a pydantic v2 model with a string ``id`` (prefixed, e.g. ``clm_3f2a...``)
so it can be stored in the :class:`noesis.store.KnowledgeStore`, serialized to JSON, and
referenced from thoughts, questions and beliefs.

Provenance rule: a :class:`Claim` is never trusted on its own. Its support comes from
:class:`Evidence` rows, each of which points at a :class:`Chunk` (with a character span
inside its :class:`Source`). :class:`Belief` confidence is computed from that evidence by
:mod:`noesis.beliefs`.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "new_id",
    "utcnow",
    "Stance",
    "BeliefStatus",
    "QuestionOrigin",
    "QuestionStatus",
    "ThoughtStep",
    "NoesisModel",
    "Source",
    "Chunk",
    "Concept",
    "Claim",
    "Evidence",
    "BeliefChange",
    "Belief",
    "Question",
    "Thought",
    "ReasoningRun",
    "Summary",
]

Stance = Literal["supports", "contradicts", "context"]
BeliefStatus = Literal["hypothesis", "believed", "doubted", "rejected"]
QuestionOrigin = Literal["user", "gap", "contradiction", "curiosity"]
QuestionStatus = Literal["open", "in_progress", "answered", "dropped"]
# Free-form string so the reasoning engine can add step types; these are the canonical ones.
ThoughtStep = str
CANONICAL_THOUGHT_STEPS = (
    "observe",
    "select_question",
    "hypothesize",
    "gather",
    "test",
    "judge",
    "update",
    "reflect",
    "answer",
    "note",
)


def new_id(prefix: str) -> str:
    """Return a new random id such as ``clm_1a2b3c4d5e6f``."""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class NoesisModel(BaseModel):
    """Base class: permissive on extra keys when loading old exports, strict on types."""

    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    #: id prefix used by :func:`new_id`; overridden per subclass.
    ID_PREFIX: ClassVar[str] = "obj"


class Source(NoesisModel):
    """A document / page / file / packet stream that was ingested."""

    ID_PREFIX = "src"
    id: str = Field(default_factory=lambda: new_id("src"))
    uri: str = ""
    title: str = ""
    kind: str = "text"  # text|markdown|html|url|pdf|code|json|csv|packets
    #: Prior probability that a claim asserted by this source is correct (0..1).
    #: 0.5 = uninformative; values <= 0.5 contribute no evidential weight.
    reliability: float = Field(default=0.6, ge=0.0, le=1.0)
    reliability_reason: str = "default prior"
    #: Sources sharing an independence group are treated as non-independent
    #: (e.g. syndicated copies of one press release). Defaults to the source id.
    independence_group: str | None = None
    content_hash: str = ""
    text: str = ""  # full normalized text (spans in chunks index into this)
    metadata: dict[str, Any] = Field(default_factory=dict)
    added_at: datetime = Field(default_factory=utcnow)

    @property
    def group(self) -> str:
        return self.independence_group or self.id


class Chunk(NoesisModel):
    """A contiguous span of a source's text: ``source.text[start:end] == chunk.text``."""

    ID_PREFIX = "chk"
    id: str = Field(default_factory=lambda: new_id("chk"))
    source_id: str
    index: int = 0
    text: str
    start: int = 0
    end: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class Concept(NoesisModel):
    """A named entity / topic / idea. ``parent_id`` forms the concept hierarchy."""

    ID_PREFIX = "cpt"
    id: str = Field(default_factory=lambda: new_id("cpt"))
    name: str
    description: str = ""
    aliases: list[str] = Field(default_factory=list)
    chunk_ids: list[str] = Field(default_factory=list)
    parent_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Claim(NoesisModel):
    """An atomic proposition. Merged claims list the variant wordings in ``variants``."""

    ID_PREFIX = "clm"
    id: str = Field(default_factory=lambda: new_id("clm"))
    text: str
    concept_ids: list[str] = Field(default_factory=list)
    variants: list[str] = Field(default_factory=list)
    #: who created it: "extraction" | "user" | "reasoning" | "hypothesis"
    origin: str = "extraction"
    created_at: datetime = Field(default_factory=utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Evidence(NoesisModel):
    """Link between a claim and a chunk, with stance and strength.

    ``chunk_id``/``source_id`` may be ``None`` only for non-document evidence (e.g. a
    user assertion or a reasoning-derived argument); such evidence uses
    ``metadata["reliability"]`` (default 0.55) as its reliability.
    """

    ID_PREFIX = "evd"
    id: str = Field(default_factory=lambda: new_id("evd"))
    claim_id: str
    chunk_id: str | None = None
    source_id: str | None = None
    stance: Stance = "supports"
    strength: float = Field(default=0.7, ge=0.0, le=1.0)
    quote: str = ""  # exact supporting text (substring of the chunk when chunk_id set)
    span: tuple[int, int] | None = None  # char span inside the *source* text
    rationale: str = ""
    method: str = "heuristic"  # heuristic|llm|user|reasoning
    created_at: datetime = Field(default_factory=utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


class BeliefChange(NoesisModel):
    """One entry in a belief's audit trail."""

    at: datetime = Field(default_factory=utcnow)
    old_confidence: float | None = None
    new_confidence: float
    old_status: BeliefStatus | None = None
    new_status: BeliefStatus
    reason: str
    evidence_ids: list[str] = Field(default_factory=list)
    actor: str = "system"  # system|understand|reasoning|user


class Belief(NoesisModel):
    """Calibrated confidence that a claim is true, with full history."""

    ID_PREFIX = "bel"
    id: str = Field(default_factory=lambda: new_id("bel"))
    claim_id: str
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    status: BeliefStatus = "hypothesis"
    rationale: str = ""
    #: persistent log-odds adjustment from reasoning/user judgments (kept across recomputes)
    adjustment: float = 0.0
    evidence_ids: list[str] = Field(default_factory=list)
    history: list[BeliefChange] = Field(default_factory=list)
    updated_at: datetime = Field(default_factory=utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Question(NoesisModel):
    ID_PREFIX = "qst"
    id: str = Field(default_factory=lambda: new_id("qst"))
    text: str
    origin: QuestionOrigin = "user"
    priority: float = Field(default=0.5, ge=0.0, le=1.0)
    status: QuestionStatus = "open"
    claim_ids: list[str] = Field(default_factory=list)
    concept_ids: list[str] = Field(default_factory=list)
    answer: str | None = None
    answer_confidence: float | None = None
    created_at: datetime = Field(default_factory=utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Thought(NoesisModel):
    """One step of the reasoning trace."""

    ID_PREFIX = "thg"
    id: str = Field(default_factory=lambda: new_id("thg"))
    run_id: str | None = None
    step: ThoughtStep = "note"
    content: str = ""
    refs: list[str] = Field(default_factory=list)  # ids of claims/evidence/chunks/questions
    timestamp: datetime = Field(default_factory=utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ReasoningRun(NoesisModel):
    ID_PREFIX = "run"
    id: str = Field(default_factory=lambda: new_id("run"))
    goal: str | None = None
    strategy: str = "simple"
    model: str = ""
    status: str = "running"  # running|completed|stopped|failed
    stop_reason: str = ""
    steps: int = 0
    budget: dict[str, Any] = Field(default_factory=dict)
    usage: dict[str, Any] = Field(default_factory=dict)
    summary: str = ""
    started_at: datetime = Field(default_factory=utcnow)
    ended_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Summary(NoesisModel):
    """Hierarchical (RAPTOR-style) summary. level 0 = concept, 1 = topic."""

    ID_PREFIX = "sum"
    id: str = Field(default_factory=lambda: new_id("sum"))
    level: int = 0
    scope: str = "topic"  # "topic" or a concept id
    title: str = ""
    text: str
    child_ids: list[str] = Field(default_factory=list)  # claim ids (level 0) / summary ids
    created_at: datetime = Field(default_factory=utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)
