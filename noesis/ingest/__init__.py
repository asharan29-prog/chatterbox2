"""Ingestion: load files/URLs/text -> Source + Chunks (+ vectors) in a KnowledgeStore.

>>> from noesis.ingest import ingest, ingest_text
>>> sources = ingest(store, "examples/corpus")          # files, dirs (recursive) or URLs
>>> src = ingest_text(store, "Some notes...", title="notes", reliability=0.7)
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ..embed import Embedder, get_embedder, index_chunks
from ..store import KnowledgeStore
from ..types import Source
from .chunker import chunk_text, make_chunks
from .loaders import SUPPORTED_EXTS, LoadedDocument, load_path, load_url, parse_front_matter
from .reliability import reliability_prior

__all__ = [
    "ingest",
    "ingest_path",
    "ingest_url",
    "ingest_text",
    "ingest_document",
    "chunk_text",
    "make_chunks",
    "reliability_prior",
    "LoadedDocument",
    "load_path",
    "load_url",
]

_SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv", ".mypy_cache", ".pytest_cache"}
_SKIP_FILES = {"questions.json"}


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def ingest_document(
    store: KnowledgeStore,
    doc: LoadedDocument,
    *,
    reliability: float | None = None,
    independence_group: str | None = None,
    max_chars: int = 1200,
    overlap: int = 150,
    embedder: Embedder | None = None,
    dedupe: bool = True,
) -> Source:
    """Store a loaded document as a Source with chunks and chunk vectors.

    Identical content (sha256 of normalized text) is not stored twice: the existing Source is
    returned (with reliability updated if one was given explicitly).
    """
    text = doc.text
    h = _hash(text)
    if dedupe:
        existing = store.find_source_by_hash(h)
        if existing is not None:
            if reliability is not None and abs(existing.reliability - reliability) > 1e-9:
                existing = store.set_source_reliability(existing.id, reliability, "explicit at re-ingest")
            return existing
    md = dict(doc.metadata)
    if reliability is not None:
        rel, why = float(reliability), "explicit at ingest"
    else:
        rel, why = reliability_prior(doc.uri, md)
    group = independence_group or md.get("independence_group") or md.get("origin")
    src = Source(
        uri=doc.uri,
        title=doc.title or doc.uri,
        kind=doc.kind,
        reliability=rel,
        reliability_reason=why,
        independence_group=str(group) if group else None,
        content_hash=h,
        text=text,
        metadata=md,
    )
    store.add_source(src)
    chunks = make_chunks(src.id, text, max_chars=max_chars, overlap=overlap)
    store.add_chunks(chunks)
    index_chunks(store, chunks, embedder or get_embedder(None))
    return src


def ingest_text(
    store: KnowledgeStore,
    text: str,
    *,
    title: str = "",
    uri: str = "",
    kind: str = "text",
    metadata: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Source:
    """Ingest a raw string (front matter, if present, is parsed into metadata)."""
    meta, body = parse_front_matter(text)
    meta.update(metadata or {})
    body = re.sub(r"\n{3,}", "\n\n", body.replace("\r\n", "\n")).strip()
    doc = LoadedDocument(body, title or str(meta.get("title") or uri or "untitled"), kind, uri or f"text:{_hash(body)[:12]}", meta)
    return ingest_document(store, doc, **kwargs)


def ingest_path(store: KnowledgeStore, path: str | Path, **kwargs: Any) -> list[Source]:
    """Ingest a file or (recursively) a directory of supported files."""
    p = Path(path)
    if p.is_dir():
        out: list[Source] = []
        for f in sorted(p.rglob("*")):
            if any(part in _SKIP_DIRS or part.startswith(".") for part in f.relative_to(p).parts):
                continue
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS and f.name not in _SKIP_FILES:
                out.append(ingest_document(store, load_path(f), **kwargs))
        return out
    if not p.exists():
        raise FileNotFoundError(str(p))
    return [ingest_document(store, load_path(p), **kwargs)]


def ingest_url(store: KnowledgeStore, url: str, *, transport: Any = None, **kwargs: Any) -> Source:
    return ingest_document(store, load_url(url, transport=transport), **kwargs)


def ingest(store: KnowledgeStore, *targets: str | Path | Iterable[str | Path], **kwargs: Any) -> list[Source]:
    """Ingest any mix of file paths, directories and http(s) URLs. Returns the Sources."""
    flat: list[str | Path] = []
    for t in targets:
        if isinstance(t, (str, Path)):
            flat.append(t)
        else:
            flat.extend(t)
    out: list[Source] = []
    for t in flat:
        if isinstance(t, str) and re.match(r"^https?://", t):
            out.append(ingest_url(store, t, **kwargs))
        else:
            out.extend(ingest_path(store, t, **kwargs))
    return out
