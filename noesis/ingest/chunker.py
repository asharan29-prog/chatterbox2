"""Paragraph/sentence-aware chunking with exact character spans."""

from __future__ import annotations

import re

from ..nlp import split_sentences
from ..types import Chunk

__all__ = ["chunk_text", "make_chunks"]

_PARA_RE = re.compile(r"\n\s*\n")


def chunk_text(text: str, max_chars: int = 1200, overlap: int = 150) -> list[tuple[int, int]]:
    """Return ``(start, end)`` spans covering ``text``.

    Paragraphs are packed greedily up to ``max_chars``; a paragraph longer than ``max_chars``
    is split at sentence boundaries (and hard-split as a last resort). When a long paragraph
    is split, consecutive pieces overlap by up to ``overlap`` chars (whole sentences).
    """
    if not text.strip():
        return []
    # paragraph spans
    paras: list[tuple[int, int]] = []
    pos = 0
    for m in _PARA_RE.finditer(text):
        paras.append((pos, m.start()))
        pos = m.end()
    paras.append((pos, len(text)))
    paras = [_trim(text, s, e) for s, e in paras]
    paras = [(s, e) for s, e in paras if e > s]

    pieces: list[tuple[int, int]] = []
    for s, e in paras:
        if e - s <= max_chars:
            pieces.append((s, e))
        else:
            pieces.extend(_split_long(text, s, e, max_chars, overlap))

    chunks: list[tuple[int, int]] = []
    cur: tuple[int, int] | None = None
    for s, e in pieces:
        if cur is None:
            cur = (s, e)
        elif e - cur[0] <= max_chars and s >= cur[1]:
            cur = (cur[0], e)
        else:
            chunks.append(cur)
            cur = (s, e)
    if cur:
        chunks.append(cur)
    return chunks


def _trim(text: str, s: int, e: int) -> tuple[int, int]:
    while s < e and text[s].isspace():
        s += 1
    while e > s and text[e - 1].isspace():
        e -= 1
    return s, e


def _split_long(text: str, s: int, e: int, max_chars: int, overlap: int) -> list[tuple[int, int]]:
    sents = [(a + s, b + s) for _, a, b in split_sentences(text[s:e])]
    # hard-split any single sentence longer than max_chars
    flat: list[tuple[int, int]] = []
    for a, b in sents:
        while b - a > max_chars:
            flat.append((a, a + max_chars))
            a += max_chars
        if b > a:
            flat.append((a, b))
    out: list[tuple[int, int]] = []
    i = 0
    while i < len(flat):
        start = flat[i][0]
        j = i
        while j + 1 < len(flat) and flat[j + 1][1] - start <= max_chars:
            j += 1
        out.append((start, flat[j][1]))
        if j + 1 >= len(flat):
            break
        # step back for overlap
        k = j + 1
        while k - 1 > i and flat[j][1] - flat[k - 1][0] <= overlap:
            k -= 1
        i = max(k, i + 1)
    return out


def make_chunks(source_id: str, text: str, max_chars: int = 1200, overlap: int = 150) -> list[Chunk]:
    return [
        Chunk(source_id=source_id, index=i, text=text[s:e], start=s, end=e)
        for i, (s, e) in enumerate(chunk_text(text, max_chars=max_chars, overlap=overlap))
    ]
