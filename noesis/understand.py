"""Build the understanding map: claims, concepts, evidence links, summaries, questions.

Entry point::

    from noesis.understand import understand
    result = understand(store, llm)        # llm may be None -> fully heuristic/offline

Pipeline (incremental: only chunks not yet marked ``understood`` are processed):

1. **extract**   per chunk: claims (+quote spans) and concepts -- LLM JSON task
                 ``extract_claims`` or :func:`noesis.nlp.extract_claims`.
2. **integrate** merge each candidate into an existing claim when TF-IDF similarity is high and
                 polarity matches (LLM ``same_claim`` check in the ambiguous band); the source
                 chunk becomes ``supports`` evidence. Otherwise a new Claim is created.
3. **relate**    cross-source evidence linking: for each (new claim x all chunks) and
                 (old claim x new chunks) pair that is lexically/vectorially close, classify
                 stance (LLM ``classify_stance`` or :func:`noesis.nlp.lexical_stance`).
4. **calibrate** recompute beliefs (:mod:`noesis.beliefs`) for touched claims.
5. **abstract**  concept hierarchy + RAPTOR-style summaries (concept level 0, topic level 1).
6. **question**  contradictions and gaps -> :class:`~noesis.types.Question` rows.

Also exported for the reasoning engine: :func:`assess_claim` (add a hypothesis and link
evidence for it), :func:`classify_stance`, :func:`extract_candidates`.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import nlp
from .beliefs import update_belief
from .embed import Embedder, ensure_index, get_embedder, index_claims, vector_namespace
from .llm.base import LLMError
from .llm.prompts import TASK_SCHEMAS, build_prompt, system_prompt
from .store import KnowledgeStore
from .types import Belief, Chunk, Claim, Concept, Evidence, Question, Source, Summary

__all__ = [
    "UnderstandConfig",
    "UnderstandResult",
    "understand",
    "extract_candidates",
    "classify_stance",
    "same_claim",
    "link_evidence",
    "detect_contradictions",
    "generate_questions",
    "build_summaries",
    "assess_claim",
    "locate_quote",
]

log = logging.getLogger("noesis.understand")


@dataclass
class UnderstandConfig:
    #: TF-IDF cosine at/above which two same-polarity claims are merged without asking
    merge_threshold: float = 0.6
    #: with an LLM, similarities in [merge_ask_low, merge_threshold) are checked with ``same_claim``
    merge_ask_low: float = 0.4
    #: chunks considered per claim during evidence linking
    link_top_k: int = 6
    #: minimum chunk/claim lexical similarity to run a stance check
    link_min_sim: float = 0.12
    #: minimum stance strength to store supports/contradicts evidence
    min_evidence_strength: float = 0.45
    #: hard cap on stance checks per run (LLM cost control)
    max_stance_checks: int = 600
    max_claims_per_chunk: int = 25
    summaries: bool = True
    max_concept_summaries: int = 12
    questions: bool = True
    max_new_questions: int = 20
    #: use LLM for summaries (else extractive)
    llm_summaries: bool = True


@dataclass
class UnderstandResult:
    method: str
    chunks_processed: int = 0
    claims_created: list[str] = field(default_factory=list)
    claims_merged: int = 0
    evidence_added: list[str] = field(default_factory=list)
    concepts: list[str] = field(default_factory=list)
    contradictions: list[tuple[str, str]] = field(default_factory=list)
    questions: list[str] = field(default_factory=list)
    summaries: list[str] = field(default_factory=list)
    beliefs_updated: list[str] = field(default_factory=list)
    llm_errors: int = 0
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "chunks_processed": self.chunks_processed,
            "claims_created": len(self.claims_created),
            "claims_merged": self.claims_merged,
            "evidence_added": len(self.evidence_added),
            "concepts": len(self.concepts),
            "contradictions": len(self.contradictions),
            "questions": len(self.questions),
            "summaries": len(self.summaries),
            "beliefs_updated": len(self.beliefs_updated),
            "llm_errors": self.llm_errors,
            "warnings": self.warnings[:20],
        }


# ============================================================================ helpers
def locate_quote(quote: str, chunk: Chunk) -> tuple[int, int] | None:
    """Find ``quote`` in ``chunk`` and return its span in *source* coordinates."""
    if not quote:
        return None
    text = chunk.text
    i = text.find(quote)
    if i < 0:
        i = text.lower().find(quote.lower())
    if i < 0:
        # whitespace-insensitive match
        pattern = r"\s+".join(re.escape(w) for w in quote.split())
        m = re.search(pattern, text, re.I) if pattern else None
        if m:
            return chunk.start + m.start(), chunk.start + m.end()
        best = nlp.best_sentences(quote, text, k=1)
        if not best:
            return None
        i = text.find(best[0])
        if i < 0:
            return None
        return chunk.start + i, chunk.start + i + len(best[0])
    return chunk.start + i, chunk.start + i + len(quote)


def _call_json(llm: Any, task: str, prompt: str, guidance: str | None = None) -> Any:
    return llm.complete(prompt, system=system_prompt(guidance), json_schema=TASK_SCHEMAS[task], temperature=0.0, max_tokens=2048)


def extract_candidates(chunk: Chunk, llm: Any = None, *, source: Source | None = None, max_claims: int = 25) -> list[dict]:
    """Candidate claims from one chunk: ``[{"text", "quote", "concepts", "strength", "span"}]``.

    Uses the LLM ``extract_claims`` task when ``llm`` is given (raises LLMError on failure),
    else the heuristic extractor.
    """
    if llm is None:
        raw = nlp.extract_claims(chunk.text, max_claims=max_claims)
        cands = [
            {"text": c["text"], "quote": c["quote"], "concepts": c["concepts"], "strength": 0.6 if c["hedged"] else 0.8}
            for c in raw
        ]
    else:
        prompt = build_prompt(
            "extract_claims",
            "Extract the distinct factual claims asserted in the document. For each claim give: "
            "'text' (a self-contained, declarative sentence understandable without the document), "
            "'quote' (the exact sentence(s) from the document that assert it, copied verbatim), "
            "'concepts' (1-4 key noun phrases), and 'strength' (0-1: how firmly the document asserts "
            "it; hedged or speculative statements get lower values). Do not include claims the "
            "document only mentions in order to reject them as its own; extract what the document "
            "itself asserts. Return {\"claims\": [...]}.",
            documents=[{"text": chunk.text, "id": chunk.id, "source": source.title if source else None}],
        )
        data = _call_json(llm, "extract_claims", prompt)
        cands = []
        for c in (data.get("claims") or [])[:max_claims]:
            text = str(c.get("text", "")).strip()
            if len(text) < 8:
                continue
            cands.append(
                {
                    "text": text,
                    "quote": str(c.get("quote") or text),
                    "concepts": [str(x) for x in (c.get("concepts") or []) if str(x).strip()][:6],
                    "strength": float(c.get("strength", 0.8) or 0.8),
                }
            )
    for c in cands:
        c["strength"] = max(0.05, min(1.0, float(c["strength"])))
        c["span"] = locate_quote(c["quote"], chunk)
    return cands


def same_claim(a: str, b: str, llm: Any = None, sim: float | None = None, cfg: UnderstandConfig | None = None) -> bool:
    """Are ``a`` and ``b`` the same proposition (same polarity)?"""
    cfg = cfg or UnderstandConfig()
    sim = nlp.text_similarity(a, b) if sim is None else sim
    if not _same_polarity(a, b):
        return False
    if sim >= cfg.merge_threshold:
        return True
    if llm is not None and sim >= cfg.merge_ask_low:
        prompt = build_prompt(
            "same_claim",
            "Do claim_a and claim_b assert the same proposition (same meaning and same direction), "
            "such that evidence for one is evidence for the other? Return {\"same\": bool, \"reason\": str}.",
            fields={"claim_a": a, "claim_b": b},
        )
        try:
            return bool(_call_json(llm, "same_claim", prompt).get("same"))
        except LLMError:
            return False
    return False


def _same_polarity(a: str, b: str) -> bool:
    pa, da = nlp.polarity(a)
    pb, db = nlp.polarity(b)
    sign = pa * pb
    for k, v in da.items():
        if k in db and db[k] != v:
            sign *= -1
    return sign > 0


def classify_stance(claim: str, passage: str, llm: Any = None, *, idf: dict[str, float] | None = None) -> dict[str, Any]:
    """Stance of ``passage`` toward ``claim``: ``{"stance", "strength", "quote", "rationale", "method"}``.

    ``idf`` (from :func:`noesis.nlp.build_idf`) sharpens the heuristic path.
    """
    if llm is None:
        best_q, best = "", ("context", 0.0)
        for sent, _, _ in nlp.split_sentences(passage):
            if sent.rstrip().endswith("?"):
                continue
            st = nlp.lexical_stance(claim, sent, idf=idf)
            if st[0] == "context":
                if best[0] == "context" and st[1] > best[1]:
                    best, best_q = st, sent
            elif best[0] == "context" or st[1] > best[1]:
                best, best_q = st, sent
        return {"stance": best[0], "strength": best[1], "quote": best_q, "rationale": "lexical overlap + polarity", "method": "heuristic"}
    prompt = build_prompt(
        "classify_stance",
        "Decide whether the document SUPPORTS the claim (asserts it or gives evidence for it), "
        "CONTRADICTS it (asserts the opposite or gives evidence against it), or is only CONTEXT "
        "(related but neither). 'strength' (0-1) reflects how directly and firmly it does so. "
        "'quote' must be the exact decisive sentence from the document. "
        "Return {\"stance\", \"strength\", \"quote\", \"rationale\"}.",
        fields={"claim": claim},
        documents=[{"text": passage}],
    )
    data = _call_json(llm, "classify_stance", prompt)
    stance = str(data.get("stance", "context")).lower()
    if stance not in ("supports", "contradicts", "context"):
        stance = "context"
    return {
        "stance": stance,
        "strength": max(0.0, min(1.0, float(data.get("strength", 0.5) or 0.0))),
        "quote": str(data.get("quote") or ""),
        "rationale": str(data.get("rationale") or ""),
        "method": "llm",
    }


# ============================================================================ concepts
def _upsert_concepts(store: KnowledgeStore, names: Iterable[str], chunk_id: str, cache: dict[str, Concept]) -> list[str]:
    ids = []
    for raw in names:
        name = re.sub(r"\s+", " ", str(raw)).strip(" .,;:")
        if len(name) < 3 or name.lower() in nlp.STOPWORDS:
            continue
        key = name.lower()
        c = cache.get(key) or store.find_concept(name)
        if c is None:
            c = Concept(name=name)
        if chunk_id and chunk_id not in c.chunk_ids:
            c.chunk_ids.append(chunk_id)
        cache[key] = c
        ids.append(c.id)
    return list(dict.fromkeys(ids))


def _build_hierarchy(store: KnowledgeStore) -> None:
    """Parent = the shortest other concept whose words are a strict subset (e.g. 'blood pressure'
    is the parent of 'systolic blood pressure')."""
    concepts = store.list(Concept)
    words = {c.id: set(nlp.stem(w) for w in nlp.tokenize(c.name)) for c in concepts}
    changed = []
    for c in concepts:
        cands = [o for o in concepts if o.id != c.id and words[o.id] and words[o.id] < words[c.id]]
        parent = min(cands, key=lambda o: (len(words[o.id]), -len(o.chunk_ids)), default=None)
        pid = parent.id if parent else None
        if c.parent_id != pid:
            c.parent_id = pid
            changed.append(c)
    if changed:
        store.put_many(changed)


# ============================================================================ main
def understand(
    store: KnowledgeStore,
    llm: Any = None,
    *,
    config: UnderstandConfig | None = None,
    embedder: Embedder | None = None,
    source_ids: Sequence[str] | None = None,
    reprocess: bool = False,
) -> UnderstandResult:
    """Run the full understanding pipeline over not-yet-understood chunks.

    ``llm=None`` runs fully offline (heuristics). If an LLM call fails for a chunk, that chunk
    falls back to heuristics and the failure is counted in ``result.llm_errors``.
    """
    cfg = config or UnderstandConfig()
    res = UnderstandResult(method="llm" if llm is not None else "heuristic")
    if embedder is None:
        embedder = get_embedder(llm)
    try:
        chunk_ns = ensure_index(store, embedder, "chunk")
    except Exception as e:  # noqa: BLE001 - provider embedding failure -> local vectors
        res.warnings.append(f"embedding with {embedder.name} failed ({e}); using local hashing vectors")
        embedder = get_embedder(None)
        chunk_ns = ensure_index(store, embedder, "chunk")

    all_chunks = store.list(Chunk)
    if source_ids is not None:
        sid = set(source_ids)
        all_chunks_scope = [c for c in all_chunks if c.source_id in sid]
    else:
        all_chunks_scope = all_chunks
    todo = [c for c in all_chunks_scope if reprocess or not c.metadata.get("understood")]
    sources = {s.id: s for s in store.list(Source)}
    concept_cache: dict[str, Concept] = {}

    # ---------------------------------------------------------------- 1. extract
    candidates: list[tuple[Chunk, dict]] = []
    for ch in todo:
        try:
            cands = extract_candidates(ch, llm, source=sources.get(ch.source_id), max_claims=cfg.max_claims_per_chunk)
        except LLMError as e:
            res.llm_errors += 1
            res.warnings.append(f"extract_claims failed for {ch.id}: {e}; used heuristic")
            cands = extract_candidates(ch, None, max_claims=cfg.max_claims_per_chunk)
        chunk_concepts = nlp.extract_concepts(ch.text, max_concepts=8)
        _upsert_concepts(store, chunk_concepts, ch.id, concept_cache)
        for c in cands:
            candidates.append((ch, c))
        res.chunks_processed += 1

    # ---------------------------------------------------------------- 2. integrate
    existing = store.list(Claim)
    texts = [c.text for c in existing] + [c["text"] for _, c in candidates]
    tfidf = nlp.tfidf_matrix(texts) if texts else np.zeros((0, 1))
    rep_rows: list[int] = list(range(len(existing)))  # matrix rows of claim representatives
    rep_claims: list[Claim] = list(existing)
    touched: dict[str, Claim] = {}
    new_evidence: list[Evidence] = []
    for k, (ch, cand) in enumerate(candidates):
        row = len(existing) + k
        target: Claim | None = None
        if rep_rows:
            sims = tfidf[rep_rows] @ tfidf[row]
            order = np.argsort(-sims)[:3]
            for j in order:
                s = float(sims[j])
                if s < cfg.merge_ask_low:
                    break
                if same_claim(rep_claims[j].text, cand["text"], llm, sim=s, cfg=cfg):
                    target = rep_claims[j]
                    break
        concept_ids = _upsert_concepts(store, cand["concepts"], ch.id, concept_cache)
        if target is None:
            target = Claim(text=cand["text"], concept_ids=concept_ids, origin="extraction")
            rep_rows.append(row)
            rep_claims.append(target)
            res.claims_created.append(target.id)
        else:
            if cand["text"] != target.text and cand["text"] not in target.variants:
                target.variants.append(cand["text"])
            target.concept_ids = list(dict.fromkeys(target.concept_ids + concept_ids))
            res.claims_merged += 1
        touched[target.id] = target
        # one 'asserted-by' evidence per (claim, chunk)
        if any(e.claim_id == target.id and e.chunk_id == ch.id for e in new_evidence):
            continue
        if not res.claims_created or target.id not in res.claims_created:
            if store.list_evidence(target.id, chunk_id=ch.id):
                continue
        new_evidence.append(
            Evidence(
                claim_id=target.id,
                chunk_id=ch.id,
                source_id=ch.source_id,
                stance="supports",
                strength=cand["strength"],
                quote=_quote_text(cand, ch),
                span=cand["span"],
                rationale="asserted by source",
                method="llm" if llm is not None else "heuristic",
                metadata={"relation": "asserted"},
            )
        )
    store.put_many(list(concept_cache.values()))
    store.put_many(list(touched.values()))
    store.put_many(new_evidence)
    res.evidence_added += [e.id for e in new_evidence]
    res.concepts = [c.id for c in concept_cache.values()]
    new_claims = [touched[i] for i in res.claims_created]
    index_claims(store, new_claims, embedder)

    # ---------------------------------------------------------------- 3. relate
    todo_ids = {c.id for c in todo}
    old_claims = [c for c in store.list(Claim) if c.id not in set(res.claims_created)]
    pairs: list[tuple[Claim, list[Chunk]]] = []
    if new_claims:
        pairs += [(c, all_chunks) for c in new_claims]
    new_chunks = [c for c in all_chunks if c.id in todo_ids]
    if old_claims and new_chunks and not reprocess:
        pairs += [(c, new_chunks) for c in old_claims]
    linked = link_evidence(store, llm, pairs, cfg=cfg, embedder=embedder, chunk_ns=chunk_ns, result=res)
    for cid in linked:
        if cid not in touched:
            cl = store.get_claim(cid)
            if cl:
                touched[cid] = cl

    # ---------------------------------------------------------------- 4. calibrate
    for cid in touched:
        b = update_belief(store, cid, reason="understand: evidence integrated", actor="understand")
        res.beliefs_updated.append(b.id)

    # ---------------------------------------------------------------- 5/6. abstract + question
    res.contradictions = detect_contradictions(store, claim_ids=list(touched))
    _build_hierarchy(store)
    if cfg.summaries and (todo or reprocess):
        res.summaries = build_summaries(store, llm if cfg.llm_summaries else None, cfg=cfg, result=res)
    if cfg.questions:
        res.questions = generate_questions(store, contradictions=res.contradictions, max_new=cfg.max_new_questions)

    for ch in todo:
        ch.metadata["understood"] = True
        ch.metadata["understood_by"] = res.method
    store.put_many(todo)
    store.set_meta("understood_by", getattr(llm, "name", None) or "heuristic")
    return res


def _quote_text(cand: dict, ch: Chunk) -> str:
    span = cand.get("span")
    if span:
        return ch.text[span[0] - ch.start : span[1] - ch.start]
    return cand.get("quote", "")[:500]


# ============================================================================ relate
def link_evidence(
    store: KnowledgeStore,
    llm: Any,
    pairs: Sequence[tuple[Claim, Sequence[Chunk]]],
    *,
    cfg: UnderstandConfig | None = None,
    embedder: Embedder | None = None,
    chunk_ns: str | None = None,
    result: UnderstandResult | None = None,
) -> set[str]:
    """For each ``(claim, candidate_chunks)``: pick the closest chunks and classify stance.

    Adds Evidence rows (supports/contradicts with strength >= ``min_evidence_strength``).
    Returns ids of claims that received new evidence.
    """
    cfg = cfg or UnderstandConfig()
    embedder = embedder or get_embedder(None)
    chunk_ns = chunk_ns or vector_namespace("chunk", embedder)
    touched: set[str] = set()
    checks = 0
    sources = {s.id: s for s in store.list(Source)}
    idf = nlp.build_idf([c.text for c in store.list(Chunk)])
    for claim, chunks in pairs:
        if checks >= cfg.max_stance_checks:
            if result is not None:
                result.warnings.append("max_stance_checks reached; some evidence links skipped")
            break
        have = {e.chunk_id for e in store.list_evidence(claim.id)}
        cand_chunks = [c for c in chunks if c.id not in have]
        if not cand_chunks:
            continue
        # rank by blend of vector similarity and lexical overlap
        try:
            qv = embedder.embed([claim.text])[0]
            hits = dict(store.vector_search(chunk_ns, qv, k=len(cand_chunks), restrict_to=[c.id for c in cand_chunks]))
        except ValueError:
            hits = {}
        scored = []
        for c in cand_chunks:
            lex = max((nlp.text_similarity(claim.text, s) for s, _, _ in nlp.split_sentences(c.text)), default=0.0)
            scored.append((0.5 * hits.get(c.id, 0.0) + 0.5 * lex, lex, c))
        scored.sort(key=lambda x: -x[0])
        new_evs = []
        for _score, lex, ch in scored[: cfg.link_top_k]:
            if lex < cfg.link_min_sim:
                continue
            checks += 1
            try:
                st = classify_stance(claim.text, ch.text, llm, idf=idf)
            except LLMError as e:
                if result is not None:
                    result.llm_errors += 1
                    result.warnings.append(f"classify_stance failed ({e}); used heuristic")
                st = classify_stance(claim.text, ch.text, None, idf=idf)
            if st["stance"] == "context" or st["strength"] < cfg.min_evidence_strength:
                continue
            span = locate_quote(st["quote"], ch) if st["quote"] else None
            quote = ch.text[span[0] - ch.start : span[1] - ch.start] if span else st["quote"][:500]
            src = sources.get(ch.source_id)
            new_evs.append(
                Evidence(
                    claim_id=claim.id,
                    chunk_id=ch.id,
                    source_id=ch.source_id,
                    stance=st["stance"],
                    strength=st["strength"],
                    quote=quote,
                    span=span,
                    rationale=st["rationale"] or f"stance from {src.title if src else ch.source_id}",
                    method=st["method"],
                    metadata={"relation": "linked"},
                )
            )
        if new_evs:
            store.put_many(new_evs)
            touched.add(claim.id)
            if result is not None:
                result.evidence_added += [e.id for e in new_evs]
    return touched


def detect_contradictions(store: KnowledgeStore, claim_ids: Sequence[str] | None = None) -> list[tuple[str, str]]:
    """Find contradicting claim pairs.

    A pair (a, b) is reported when the claims are lexically close and of opposite polarity,
    or when one claim's asserting chunk was linked as *contradicting* evidence of the other.
    Stores the partner ids in ``claim.metadata["contradicts"]``.
    """
    claims = store.list(Claim)
    if len(claims) < 2:
        return []
    focus = set(claim_ids) if claim_ids is not None else {c.id for c in claims}
    by_id = {c.id: c for c in claims}
    pairs: set[tuple[str, str]] = set()
    idf = nlp.build_idf([c.text for c in store.list(Chunk)])
    # (1) evidence-based: the sentence asserting b was linked as contradicting a
    asserted: dict[str, list[tuple[str, tuple[int, int] | None, str]]] = defaultdict(list)
    for ev in store.list(Evidence, stance="supports"):
        if ev.chunk_id and ev.metadata.get("relation") == "asserted":
            asserted[ev.chunk_id].append((ev.claim_id, ev.span, ev.quote))
    for ev in store.list(Evidence, stance="contradicts"):
        if not ev.chunk_id or ev.claim_id not in focus:
            continue
        for other, span, quote in asserted.get(ev.chunk_id, ()):
            if other == ev.claim_id or other not in by_id:
                continue
            same_sentence = (
                (span and ev.span and span[0] < ev.span[1] and ev.span[0] < span[1])
                or (quote and ev.quote and (quote in ev.quote or ev.quote in quote))
            )
            if same_sentence:
                pairs.add(tuple(sorted((ev.claim_id, other))))  # type: ignore[arg-type]
    # (2) direct lexical
    texts = [c.text for c in claims]
    m = nlp.tfidf_matrix(texts)
    sims = m @ m.T
    for i, a in enumerate(claims):
        if a.id not in focus:
            continue
        for j in np.argsort(-sims[i])[:8]:
            b = claims[int(j)]
            if b.id == a.id or sims[i, j] < 0.3:
                continue
            if nlp.lexical_stance(a.text, b.text, min_overlap=0.4, idf=idf)[0] == "contradicts":
                pairs.add(tuple(sorted((a.id, b.id))))  # type: ignore[arg-type]
    changed: dict[str, Claim] = {}
    for a, b in pairs:
        for x, y in ((a, b), (b, a)):
            c = changed.get(x) or by_id[x]
            lst = list(c.metadata.get("contradicts", []))
            if y not in lst:
                lst.append(y)
                c.metadata = {**c.metadata, "contradicts": lst}
                changed[x] = c
    if changed:
        store.put_many(list(changed.values()))
    return sorted(pairs)


# ============================================================================ summaries
def build_summaries(
    store: KnowledgeStore,
    llm: Any = None,
    *,
    cfg: UnderstandConfig | None = None,
    result: UnderstandResult | None = None,
) -> list[str]:
    """RAPTOR-style: level-0 summaries for the most-supported concepts, level-1 topic summary.

    Replaces previous summaries. Returns new summary ids.
    """
    cfg = cfg or UnderstandConfig()
    for s in store.list(Summary):
        store.delete(Summary, s.id)
    claims = store.list(Claim)
    beliefs = {b.claim_id: b for b in store.list(Belief)}
    by_concept: dict[str, list[Claim]] = defaultdict(list)
    for c in claims:
        for cid in c.concept_ids:
            by_concept[cid].append(c)
    concepts = {c.id: c for c in store.list(Concept)}
    ranked = sorted(by_concept.items(), key=lambda kv: (-len(kv[1]), concepts[kv[0]].name if kv[0] in concepts else ""))
    out: list[Summary] = []
    for cid, cls in ranked[: cfg.max_concept_summaries]:
        if cid not in concepts or len(cls) < 2:
            continue
        cls = sorted(cls, key=lambda c: -(beliefs[c.id].confidence if c.id in beliefs else 0.5))
        name = concepts[cid].name
        text = _summarize([c.text for c in cls[:8]], name, llm, [beliefs.get(c.id) for c in cls[:8]], result)
        out.append(Summary(level=0, scope=cid, title=name, text=text, child_ids=[c.id for c in cls]))
    topic = store.get_meta("topic") or "Topic overview"
    if out:
        top_text = _summarize([f"{s.title}: {s.text}" for s in out], str(topic), llm, None, result)
    else:
        top_claims = sorted(claims, key=lambda c: -(beliefs[c.id].confidence if c.id in beliefs else 0.5))[:6]
        top_text = _summarize([c.text for c in top_claims], str(topic), llm, None, result) if top_claims else "No claims yet."
    out.append(Summary(level=1, scope="topic", title=str(topic), text=top_text, child_ids=[s.id for s in out]))
    store.put_many(out)
    return [s.id for s in out]


def _summarize(items: list[str], title: str, llm: Any, beliefs: list[Belief | None] | None, result: UnderstandResult | None) -> str:
    if llm is not None:
        prompt = build_prompt(
            "summarize",
            "Write a concise, neutral summary (2-4 sentences) of the documents about the given title. "
            "Mention disagreements if the documents conflict. Return {\"title\", \"summary\"}.",
            fields={"title": title},
            documents=[{"text": t, "id": f"item{i}"} for i, t in enumerate(items)],
        )
        try:
            s = str(_call_json(llm, "summarize", prompt).get("summary", "")).strip()
            if s:
                return s
        except LLMError as e:
            if result is not None:
                result.llm_errors += 1
                result.warnings.append(f"summarize failed: {e}")
    parts = []
    for i, t in enumerate(items[:4]):
        conf = beliefs[i].confidence if beliefs and i < len(beliefs) and beliefs[i] else None
        parts.append(t.rstrip(".") + (f" (confidence {conf:.2f})" if conf is not None else "") + ".")
    return " ".join(parts)


# ============================================================================ questions
def generate_questions(
    store: KnowledgeStore, *, contradictions: Sequence[tuple[str, str]] = (), max_new: int = 20
) -> list[str]:
    """Create open questions from contradictions (high priority) and evidence gaps.

    Skips questions that already exist (same normalized text). Returns new question ids.
    """
    claims = {c.id: c for c in store.list(Claim)}
    beliefs = {b.claim_id: b for b in store.list(Belief)}
    new: list[Question] = []

    def unc(cid: str) -> float:
        b = beliefs.get(cid)
        p = b.confidence if b else 0.5
        return 1 - abs(2 * p - 1)

    def add(q: Question) -> None:
        if len(new) >= max_new or store.find_question(q.text) or any(n.text == q.text for n in new):
            return
        new.append(q)

    for a, b in contradictions:
        if a in claims and b in claims:
            add(
                Question(
                    text=f'Which is correct: "{claims[a].text.rstrip(".")}" or "{claims[b].text.rstrip(".")}"?',
                    origin="contradiction",
                    priority=round(min(1.0, 0.6 + 0.4 * max(unc(a), unc(b))), 3),
                    claim_ids=[a, b],
                )
            )
    # contested claims: evidence on both sides
    ev_by_claim: dict[str, list[Evidence]] = defaultdict(list)
    for ev in store.list(Evidence):
        ev_by_claim[ev.claim_id].append(ev)
    for cid, evs in ev_by_claim.items():
        if cid not in claims:
            continue
        stances = {e.stance for e in evs}
        groups = {e.source_id for e in evs if e.stance != "context"}
        if {"supports", "contradicts"} <= stances:
            add(
                Question(
                    text=f'Is it true that {_lc(claims[cid].text)}?',
                    origin="contradiction",
                    priority=round(min(1.0, 0.5 + 0.4 * unc(cid)), 3),
                    claim_ids=[cid],
                    metadata={"reason": "evidence on both sides"},
                )
            )
        elif len(groups) <= 1 and 0.3 < (beliefs[cid].confidence if cid in beliefs else 0.5) < 0.8:
            add(
                Question(
                    text=f'Is there independent evidence that {_lc(claims[cid].text)}?',
                    origin="gap",
                    priority=round(0.3 + 0.3 * unc(cid), 3),
                    claim_ids=[cid],
                    metadata={"reason": "single source"},
                )
            )
    # concepts mentioned across chunks but with no claims about them
    with_claims = {cid for c in claims.values() for cid in c.concept_ids}
    for con in store.list(Concept):
        if len(con.chunk_ids) >= 2 and con.id not in with_claims:
            add(
                Question(
                    text=f"What is known about {con.name}?",
                    origin="gap",
                    priority=0.3,
                    concept_ids=[con.id],
                    metadata={"reason": "concept without claims"},
                )
            )
    new.sort(key=lambda q: -q.priority)
    store.put_many(new)
    return [q.id for q in new]


def _lc(text: str) -> str:
    t = text.strip().rstrip(".")
    if len(t) > 1 and t[0].isupper() and not t[1].isupper():
        t = t[0].lower() + t[1:]
    return t


# ============================================================================ hypotheses
def assess_claim(
    store: KnowledgeStore,
    text: str,
    llm: Any = None,
    *,
    origin: str = "hypothesis",
    top_k: int = 8,
    embedder: Embedder | None = None,
    reason: str = "assessed against corpus",
) -> tuple[Claim, list[Evidence], Belief]:
    """Add (or find) a claim and link evidence for it from the whole corpus.

    This is the reasoning engine's way to TEST a hypothesis against stored evidence: the claim
    is created with ``origin`` (default "hypothesis"), its top-k closest chunks are stance-
    classified, evidence rows are stored, and the belief is computed. If an existing claim is
    essentially identical (TF-IDF >= 0.9), that claim is reused.
    """
    embedder = embedder or get_embedder(llm)
    try:
        chunk_ns = ensure_index(store, embedder, "chunk")
    except Exception:  # noqa: BLE001
        embedder = get_embedder(None)
        chunk_ns = ensure_index(store, embedder, "chunk")
    existing = None
    for c in store.list(Claim):
        if nlp.text_similarity(c.text, text) >= 0.9 and _same_polarity(c.text, text):
            existing = c
            break
    claim = existing or store.add_claim(Claim(text=text.strip(), origin=origin))
    if existing is None:
        index_claims(store, [claim], embedder)
    cfg = UnderstandConfig(link_top_k=top_k)
    link_evidence(store, llm, [(claim, store.list(Chunk))], cfg=cfg, embedder=embedder, chunk_ns=chunk_ns)
    belief = update_belief(store, claim.id, reason=reason, actor="reasoning")
    return claim, store.list_evidence(claim.id), belief
