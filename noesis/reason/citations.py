"""Citation guardrails: every citation must point at a real, stored chunk that was actually
shown to the model. Anything else is dropped and reported as hallucinated."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from ..store import KnowledgeStore
from ..types import Chunk

__all__ = ["Citation", "CitationCheck", "validate_citations", "citation_for_chunk"]

_ID_RE = re.compile(r"\bchk_[0-9a-f]{6,32}\b")


@dataclass
class Citation:
    """A resolved citation: ``(source, span)`` plus the quote and ids for auditing."""

    chunk_id: str
    source_id: str
    source: str  # source title or uri
    uri: str
    span: tuple[int, int]
    quote: str = ""

    def as_tuple(self) -> tuple[str, tuple[int, int]]:
        return (self.source, self.span)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "source_id": self.source_id,
            "source": self.source,
            "uri": self.uri,
            "span": list(self.span),
            "quote": self.quote,
        }


@dataclass
class CitationCheck:
    valid: list[str] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)  # hallucinated / not-shown / unknown ids

    @property
    def ok(self) -> bool:
        return not self.dropped


def _normalize(raw: Iterable[Any]) -> list[str]:
    out: list[str] = []
    for r in raw or []:
        if isinstance(r, dict):
            r = r.get("id") or r.get("chunk_id") or ""
        s = str(r).strip().strip("[]()\"' ")
        if s:
            out.append(s)
    return out


def validate_citations(
    store: KnowledgeStore, cited: Iterable[Any], allowed: Iterable[str] | None = None
) -> CitationCheck:
    """Keep only citations that (a) exist as chunks in ``store`` and (b) are in ``allowed``
    (the ids that were actually put in front of the model), when ``allowed`` is given.

    Free-text citations containing a ``chk_...`` id are reduced to that id. Order is preserved
    and duplicates removed.
    """
    allow = set(allowed) if allowed is not None else None
    res = CitationCheck()
    seen: set[str] = set()
    for c in _normalize(cited):
        m = _ID_RE.search(c)
        cid = m.group(0) if m else c
        if cid in seen:
            continue
        seen.add(cid)
        if (allow is not None and cid not in allow) or store.get(Chunk, cid) is None:
            res.dropped.append(c)
        else:
            res.valid.append(cid)
    return res


def citation_for_chunk(store: KnowledgeStore, chunk_id: str, quote: str = "") -> Citation | None:
    ch = store.get(Chunk, chunk_id)
    if ch is None:
        return None
    src = store.get_source(ch.source_id)
    span = (ch.start, ch.end)
    q = quote or ch.text[:240]
    if quote:
        i = ch.text.find(quote)
        if i >= 0:
            span = (ch.start + i, ch.start + i + len(quote))
    return Citation(
        chunk_id=ch.id,
        source_id=ch.source_id,
        source=(src.title or src.uri) if src else ch.source_id,
        uri=src.uri if src else "",
        span=span,
        quote=q,
    )
