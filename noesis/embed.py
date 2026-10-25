"""Text embeddings: provider embeddings when available, else local hashing vectors.

The local :class:`HashingEmbedder` needs no downloads and is stable across processes
(blake2b feature hashing of stemmed unigrams + bigrams, sublinear TF, L2 norm), so vectors
stored in a ``.noesis.db`` stay comparable forever. It is lexical, not semantic -- good
enough for retrieval and duplicate detection on small corpora.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

import numpy as np

from . import nlp

__all__ = [
    "Embedder",
    "HashingEmbedder",
    "LLMEmbedder",
    "get_embedder",
    "vector_namespace",
    "index_chunks",
    "index_claims",
    "ensure_index",
    "hybrid_search",
]


@runtime_checkable
class Embedder(Protocol):
    name: str

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return a float32 array of shape (len(texts), dim), rows L2-normalized."""
        ...


class HashingEmbedder:
    def __init__(self, dim: int = 512) -> None:
        self.dim = dim
        self.name = f"hashing-{dim}"

    def _features(self, text: str) -> Counter[str]:
        words = nlp.content_words(text)
        feats = Counter(words)
        feats.update(f"{a}_{b}" for a, b in zip(words, words[1:]))
        return feats

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for feat, c in self._features(t).items():
                h = hashlib.blake2b(feat.encode("utf-8"), digest_size=8).digest()
                idx = int.from_bytes(h[:4], "little") % self.dim
                sign = 1.0 if h[4] & 1 else -1.0
                w = (1.0 + math.log(c)) * (1.5 if "_" in feat else 1.0)
                out[i, idx] += sign * w
            n = np.linalg.norm(out[i])
            if n > 0:
                out[i] /= n
        return out


class LLMEmbedder:
    """Wrap an LLM adapter's ``embed`` method (normalizes rows)."""

    def __init__(self, llm: Any) -> None:
        self.llm = llm
        self.name = f"llm:{getattr(llm, 'name', 'unknown')}"

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        v = np.asarray(self.llm.embed(list(texts)), dtype=np.float32)
        n = np.linalg.norm(v, axis=1, keepdims=True)
        n[n == 0] = 1.0
        return v / n


def get_embedder(llm: Any = None, *, prefer_local: bool = False, dim: int = 512) -> Embedder:
    """Provider embeddings if ``llm.supports_embeddings`` (and it is not the mock), else hashing.

    MockLLM's embeddings are the hashing vectors anyway, so it maps to :class:`HashingEmbedder`
    (keeps vectors compatible between mock and fully-offline runs).
    """
    if (
        llm is not None
        and not prefer_local
        and getattr(llm, "supports_embeddings", False)
        and not str(getattr(llm, "name", "")).startswith("mock")
    ):
        return LLMEmbedder(llm)
    return HashingEmbedder(dim=dim)


# ----------------------------------------------------------------------------- store indexing
def vector_namespace(kind: str, embedder: Embedder) -> str:
    """Namespace for vectors of ``kind`` ("chunk"/"claim") made by ``embedder``."""
    return f"{kind}:{embedder.name}"


def index_chunks(store: Any, chunks: Sequence[Any], embedder: Embedder) -> str:
    """Embed and store vectors for ``chunks``; records the embedder in store meta."""
    ns = vector_namespace("chunk", embedder)
    if chunks:
        store.set_vectors(ns, [c.id for c in chunks], embedder.embed([c.text for c in chunks]))
    if store.get_meta("embedder") is None:
        store.set_meta("embedder", embedder.name)
    return ns


def index_claims(store: Any, claims: Sequence[Any], embedder: Embedder) -> str:
    ns = vector_namespace("claim", embedder)
    if claims:
        store.set_vectors(ns, [c.id for c in claims], embedder.embed([c.text for c in claims]))
    return ns


def ensure_index(store: Any, embedder: Embedder, kind: str = "chunk") -> str:
    """Embed any chunks/claims that have no vector in this embedder's namespace yet."""
    from .types import Chunk, Claim

    cls = Chunk if kind == "chunk" else Claim
    ns = vector_namespace(kind, embedder)
    have, _ = store.get_vectors(ns)
    have_set = set(have)
    missing = [o for o in store.list(cls) if o.id not in have_set]
    if missing:
        (index_chunks if kind == "chunk" else index_claims)(store, missing, embedder)
    return ns


def hybrid_search(
    store: Any,
    query: str,
    *,
    kind: str = "chunk",
    k: int = 8,
    embedder: Embedder | None = None,
    alpha: float = 0.6,
) -> list[tuple[str, float]]:
    """Blend vector cosine (weight ``alpha``) and FTS5 BM25 (rank-normalized) scores.

    Returns ``[(item_id, score)]`` best first, score in [0, 1]. This is the recommended
    retrieval call for the reasoning engine's GATHER step.
    """
    embedder = embedder or get_embedder(None)
    ns = vector_namespace(kind, embedder)
    scores: dict[str, float] = {}
    try:
        vec_hits = store.vector_search(ns, embedder.embed([query])[0], k=k * 3)
    except ValueError:
        vec_hits = []
    for item_id, sim in vec_hits:
        scores[item_id] = scores.get(item_id, 0.0) + alpha * max(0.0, sim)
    fts_hits = store.search_text(query, kind=kind, k=k * 3)
    if fts_hits:
        top = max(s for _, s in fts_hits) or 1.0
        for item_id, s in fts_hits:
            scores[item_id] = scores.get(item_id, 0.0) + (1 - alpha) * max(0.0, s / top)
    return sorted(scores.items(), key=lambda kv: -kv[1])[:k]
