"""The independent reasoning engine: :class:`Mind`.

Loop (one question per step)::

    OBSERVE -> SELECT QUESTION -> HYPOTHESIZE -> GATHER -> TEST -> JUDGE -> UPDATE -> REFLECT

Key guarantees

* **Evidence first.** A hypothesis' confidence is computed by :mod:`noesis.beliefs` from
  Evidence rows linked (stance-classified) against the chunks retrieved for this step. The
  JUDGE step only contributes a small, bounded, *replaced-not-accumulated* log-odds offset
  (``MindConfig.judge_weight`` x agreement, capped at ``max_judge_adjust``), and none at all
  for a hypothesis without evidence.
* **Grounded citations.** Every citation the model returns is validated against the chunk
  ids actually shown to it (and the store); others are dropped and counted.
* **Untrusted data.** Retrieved text and tool results are only ever placed inside
  ``<document>`` wrappers built by :func:`noesis.llm.prompts.build_prompt`.
* **Full trace.** Every phase is persisted as a :class:`~noesis.types.Thought` tied to a
  :class:`~noesis.types.ReasoningRun`; ``on_thought`` streams them live.
* **Deterministic** with :class:`~noesis.llm.MockLLM` (seeded RNG for sampling strategies).
"""

from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from .. import nlp
from ..beliefs import adjust_belief, explain_belief, logit, update_belief
from ..embed import ensure_index, get_embedder, hybrid_search, index_claims, vector_namespace
from ..ingest import ingest_text
from ..llm.base import LLMError
from ..store import KnowledgeStore
from ..types import Belief, Chunk, Claim, Evidence, Question, ReasoningRun, Source, Thought, utcnow
from ..understand import UnderstandConfig, link_evidence
from . import _mock_tasks
from . import prompts as P
from .budget import Budget, BudgetExceeded, BudgetTracker, CountingLLM
from .citations import Citation, citation_for_chunk, validate_citations
from .safety import looks_like_injection, strip_injection
from .selection import ScoredQuestion, SelectionConfig, is_duplicate, rank_questions
from .strategies import JudgeContext, Judgment, Strategy, get_strategy

__all__ = ["Mind", "MindConfig", "Answer", "HypothesisResult"]

ToolFn = Callable[[str], Any]


@dataclass
class MindConfig:
    k_hypotheses: int = 3
    retrieve_k: int = 6
    claim_k: int = 6
    #: hybrid-search score at/above which a chunk/claim counts as relevant
    min_relevance: float = 0.12
    #: max documents shown to a judge / critic per hypothesis
    max_docs: int = 8
    #: log-odds per unit of |logit(model confidence)| from a judgment (scaled by agreement)
    judge_weight: float = 0.15
    #: hard cap on the total judgment adjustment per belief (log-odds)
    max_judge_adjust: float = 0.4
    min_evidence_strength: float = 0.45
    #: generate new questions when fewer open questions than this remain
    min_queue: int = 2
    max_new_questions: int = 5
    #: inconclusive attempts after which a question is dropped
    max_attempts: int = 3
    #: answer threshold: a hypothesis at/above this belief confidence resolves its question
    resolve_confidence: float = 0.75
    #: tools are consulted when fewer relevant chunks than this were retrieved
    tool_trigger_hits: int = 2
    tool_reliability: float = 0.55
    max_tool_calls: int = 3
    sc_samples: int = 3
    #: ask(): below this confidence the answer is flagged "insufficient evidence"
    answer_min_confidence: float = 0.35
    seed: int = 0
    selection: SelectionConfig = field(default_factory=SelectionConfig)


@dataclass
class HypothesisResult:
    text: str
    claim_id: str | None
    before: float | None
    evidence_confidence: float
    final_confidence: float
    status: str
    n_evidence: int
    judgment: Judgment
    discarded: bool = False
    conflicts: list[str] = field(default_factory=list)


@dataclass
class Answer:
    question: str
    text: str
    confidence: float
    citations: list[Citation] = field(default_factory=list)
    supporting_claims: list[dict[str, Any]] = field(default_factory=list)
    contradicting_claims: list[dict[str, Any]] = field(default_factory=list)
    trace: list[Thought] = field(default_factory=list)
    insufficient: bool = False
    dropped_citations: list[str] = field(default_factory=list)
    run_id: str | None = None
    question_id: str | None = None

    @property
    def citation_pairs(self) -> list[tuple[str, tuple[int, int]]]:
        """Citations as ``(source, span)`` tuples."""
        return [c.as_tuple() for c in self.citations]

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.text,
            "confidence": round(self.confidence, 4),
            "insufficient_evidence": self.insufficient,
            "citations": [c.to_dict() for c in self.citations],
            "supporting_claims": self.supporting_claims,
            "contradicting_claims": self.contradicting_claims,
            "dropped_citations": self.dropped_citations,
            "run_id": self.run_id,
            "question_id": self.question_id,
            "trace": [{"step": t.step, "content": t.content, "refs": t.refs} for t in self.trace],
        }

    def __str__(self) -> str:
        lines = [f"{self.text}", f"confidence: {self.confidence:.2f}" + ("  [INSUFFICIENT EVIDENCE]" if self.insufficient else "")]
        for i, c in enumerate(self.citations, 1):
            lines.append(f"  [{i}] {c.source} {c.span[0]}-{c.span[1]}: \"{c.quote[:120]}\"")
        return "\n".join(lines)


class Mind:
    """Independent reasoning over a :class:`KnowledgeStore`.

    >>> mind = Mind(store, get_llm("mock"), strategy="self_consistency", budget=Budget(steps=5))
    >>> run = mind.think(goal="Is coffee good for health?")
    >>> print(mind.ask("Does coffee raise blood pressure?"))
    >>> print(mind.explain(store.list_beliefs()[0].id))
    """

    def __init__(
        self,
        store: KnowledgeStore,
        llm: Any = None,
        *,
        strategy: str | Strategy = "simple",
        budget: Budget | None = None,
        tools: Mapping[str, ToolFn] | Sequence[ToolFn] | None = None,
        on_thought: Callable[[Thought], None] | None = None,
        config: MindConfig | None = None,
        embedder: Any = None,
    ) -> None:
        if llm is None:
            from ..llm.mock import MockLLM

            llm = MockLLM(name="mock")
        self.store = store
        self.raw_llm = llm
        self.llm = CountingLLM(llm)
        _mock_tasks.install(llm)
        self.cfg = config or MindConfig()
        self.strategy = (
            get_strategy(strategy, k=self.cfg.sc_samples)
            if isinstance(strategy, str) and strategy.lower().replace("-", "_") in ("self_consistency", "sc", "consistency")
            else get_strategy(strategy)
        )
        self.budget = budget or Budget()
        if tools is None:
            self.tools: dict[str, ToolFn] = {}
        elif isinstance(tools, Mapping):
            self.tools = dict(tools)
        else:
            self.tools = {getattr(t, "__name__", f"tool{i}"): t for i, t in enumerate(tools)}
        self.on_thought = on_thought
        self.embedder = embedder or get_embedder(llm)
        self.rng = random.Random(self.cfg.seed)
        self._run: ReasoningRun | None = None
        self._thoughts: list[Thought] = []
        self._dropped = 0
        self._tool_calls = 0

    # ================================================================== public API
    @property
    def model_name(self) -> str:
        return str(getattr(self.raw_llm, "name", type(self.raw_llm).__name__))

    def think(self, goal: str | None = None, steps: int | None = None) -> ReasoningRun:
        """Run the reasoning loop until budget, convergence or an empty question queue.

        Returns the persisted :class:`ReasoningRun` (``status``, ``stop_reason``, ``summary``,
        ``usage``; ``metadata["questions"]``/``["belief_changes"]``). Its thoughts are in
        ``store.list_thoughts(run.id)``.
        """
        budget = replace(self.budget, steps=steps) if steps is not None else self.budget
        tracker = BudgetTracker(budget)
        self.llm.tracker = tracker
        goal = goal or self.store.get_meta("topic")
        run = ReasoningRun(goal=goal, strategy=self.strategy.name, model=self.model_name, budget=budget.as_dict())
        self.store.add_run(run)
        self._begin(run)
        picked: list[str] = []
        asked: list[str] = []
        processed: list[dict[str, Any]] = []
        changes: dict[str, dict[str, Any]] = {}
        stop = ""
        try:
            while True:
                stop = tracker.exhausted() or ""
                if stop:
                    break
                self._observe(goal)
                self._maybe_generate_questions(goal)
                ranked = self._rank(goal, asked, picked)
                if not ranked:
                    stop = "empty_queue"
                    self._emit("note", "No open questions remain; stopping.")
                    break
                best = ranked[0]
                if best.value < self.cfg.selection.min_value:
                    stop = "converged"
                    self._emit(
                        "note",
                        f"Best remaining question has expected value {best.value:.3f} < {self.cfg.selection.min_value}; converged.",
                        [best.question.id],
                    )
                    break
                tracker.steps += 1
                picked.append(best.question.id)
                asked.append(best.question.text)
                outcome = self._step(best, goal, ranked[1:4])
                processed.append(outcome)
                for h in outcome["hypotheses"]:
                    if h["claim_id"] and not h["discarded"]:
                        prev = changes.get(h["claim_id"])
                        changes[h["claim_id"]] = {
                            "text": h["text"],
                            "before": prev["before"] if prev else h["before"],
                            "after": h["final_confidence"],
                            "status": h["status"],
                        }
            status = "stopped" if stop in ("budget:llm_calls", "budget:seconds") else "completed"
        except BudgetExceeded as e:
            stop, status = e.reason, "stopped"
            self._emit("note", f"Budget exhausted mid-step ({e.reason}); stopping.")
            self._release_in_progress()
        except Exception as e:  # pragma: no cover - persisted then re-raised
            run.status, run.stop_reason, run.ended_at = "failed", f"error: {e}", utcnow()
            self.store.update_run(run)
            self._release_in_progress()
            raise
        run.status = status
        run.stop_reason = stop
        run.steps = tracker.steps
        run.ended_at = utcnow()
        run.usage = {**tracker.usage(), "dropped_citations": self._dropped, "tool_calls": self._tool_calls}
        run.metadata = {**run.metadata, "questions": processed, "belief_changes": changes}
        run.summary = self._run_summary(processed, changes, stop)
        self.store.update_run(run)
        self._emit("note", f"Run finished: {run.summary}")
        self.llm.tracker = None
        return run

    def ask(self, question: str) -> Answer:
        """Retrieval-grounded answer with validated citations and evidence-based confidence.

        Flags ``insufficient=True`` (and says so in ``text``) when relevant evidence is thin,
        no valid citation survives, or confidence is below ``answer_min_confidence``. An
        insufficiently answered question stays *open* so a later :meth:`think` can pursue it.
        """
        tracker = BudgetTracker(Budget(steps=None, max_llm_calls=self.budget.max_llm_calls, seconds=self.budget.seconds))
        self.llm.tracker = tracker
        run = ReasoningRun(goal=question, strategy="ask", model=self.model_name, metadata={"kind": "ask"})
        self.store.add_run(run)
        self._begin(run)
        try:
            ans = self._ask(question)
            run.status = "completed"
        except BudgetExceeded as e:
            ans = Answer(question, f"Insufficient evidence: stopped early ({e.reason}).", 0.0, insufficient=True)
            run.status, run.stop_reason = "stopped", e.reason
        finally:
            self.llm.tracker = None
        ans.trace = list(self._thoughts)
        ans.run_id = run.id
        run.ended_at = utcnow()
        run.usage = {**tracker.usage(), "dropped_citations": self._dropped, "tool_calls": self._tool_calls}
        run.summary = f"confidence {ans.confidence:.2f}{' (insufficient evidence)' if ans.insufficient else ''}: {ans.text[:200]}"
        run.metadata = {**run.metadata, "answer": ans.to_dict() | {"trace": None}}
        self.store.update_run(run)
        return ans

    def explain(self, belief_id: str) -> str:
        """Human-readable chain for a belief (or claim id): evidence for/against with sources
        and spans, judgment adjustments, the full confidence history, and reasoning-trace mentions."""
        d = self.explain_data(belief_id)
        lines = [
            f"Belief {d['belief_id']} on claim {d['claim_id']} (origin: {d['origin']})",
            f'  "{d["claim"]}"',
            f"Confidence {d['confidence']:.2f} -> status {d['status'].upper()}",
            f"Why: {d['rationale']}",
        ]
        for title, key, sign in (("Evidence FOR", "for", "+"), ("Evidence AGAINST", "against", "-")):
            lines.append(f"{title} ({len(d[key])}):")
            if not d[key]:
                lines.append("  (none)")
            for e in d[key]:
                span = f" chars {e['span'][0]}-{e['span'][1]}" if e.get("span") else ""
                disc = f", duplicate-source discount x{e['discount']:.2f}" if e["discount"] < 1 else ""
                lines.append(
                    f"  {sign} {abs(e['logodds']):.2f} log-odds  \"{_clip(e['quote'], 160)}\"\n"
                    f"      -- {e['source_title'] or e['source_id'] or 'non-document'} (reliability {e['reliability'] if e['reliability'] is not None else 'n/a'}){span}{disc} [{e['evidence_id']}]"
                )
        if d["context_evidence"]:
            lines.append(f"Context-only evidence (no weight): {len(d['context_evidence'])} item(s)")
        if abs(d["adjustment"]) > 1e-9:
            lines.append(
                f"Judgment/user adjustment: {d['adjustment']:+.2f} log-odds "
                f"(reasoning judgments are bounded to +-{self.cfg.max_judge_adjust:.2f})"
            )
        lines.append(f"History ({len(d['history'])} change(s)):")
        for h in d["history"]:
            old = f"{h['old_confidence']:.2f}" if h.get("old_confidence") is not None else "--"
            lines.append(
                f"  {str(h['at'])[:19]}  {old} -> {h['new_confidence']:.2f}  "
                f"({h.get('old_status') or 'new'} -> {h['new_status']}) by {h['actor']}: {h['reason']}"
            )
        if d["thoughts"]:
            lines.append("Reasoning trace mentions:")
            for t in d["thoughts"][-8:]:
                lines.append(f"  [{t['step']}] {_clip(t['content'], 180)} (run {t['run_id']})")
        return "\n".join(lines)

    def explain_data(self, belief_id: str) -> dict[str, Any]:
        """Structured form of :meth:`explain`."""
        base = explain_belief(self.store, belief_id)
        claim = self.store.get_claim(base["claim_id"])
        evs = {e.id: e for e in self.store.list_evidence(base["claim_id"])}
        pro, con = [], []
        for c in base["contributions"]:
            ev = evs.get(c["evidence_id"])
            c = {**c, "span": list(ev.span) if ev and ev.span else None, "chunk_id": ev.chunk_id if ev else None}
            (pro if c["stance"] == "supports" else con).append(c)
        thoughts = [
            {"step": t.step, "content": t.content, "run_id": t.run_id, "at": t.timestamp.isoformat()}
            for t in self.store.list(Thought)
            if base["claim_id"] in t.refs or base["belief_id"] in t.refs
        ]
        return {
            **base,
            "origin": claim.origin if claim else "?",
            "for": pro,
            "against": con,
            "thoughts": thoughts,
        }

    # ================================================================== internals: trace
    def _begin(self, run: ReasoningRun) -> None:
        self._run = run
        self._thoughts = []
        self._dropped = 0
        self._tool_calls = 0

    def _emit(self, step: str, content: str, refs: Sequence[str] = (), **meta: Any) -> Thought:
        t = Thought(
            run_id=self._run.id if self._run else None,
            step=step,
            content=content,
            refs=list(dict.fromkeys(r for r in refs if r)),
            metadata=meta,
        )
        self.store.add_thought(t)
        self._thoughts.append(t)
        if self.on_thought is not None:
            self.on_thought(t)
        return t

    def _release_in_progress(self) -> None:
        for q in self.store.list(Question, status="in_progress"):
            q.status = "open"
            self.store.update_question(q)

    # ================================================================== OBSERVE
    def _observe(self, goal: str | None) -> dict[str, Any]:
        ensure_index(self.store, self.embedder, "chunk")
        ensure_index(self.store, self.embedder, "claim")
        beliefs = self.store.list(Belief)
        statuses = Counter(b.status for b in beliefs)
        uncs = sorted(((1 - abs(2 * b.confidence - 1)), b.claim_id) for b in beliefs)
        mean_unc = sum(u for u, _ in uncs) / len(uncs) if uncs else 1.0
        claims = self.store.list(Claim)
        n_contra = sum(len(c.metadata.get("contradicts", [])) for c in claims) // 2
        open_q = self.store.list_questions(status="open")
        state = {
            "claims": len(claims),
            "beliefs": dict(statuses),
            "mean_uncertainty": round(mean_unc, 4),
            "contradictions": n_contra,
            "open_questions": len(open_q),
            "sources": self.store.count(Source),
        }
        most_uncertain = [cid for _, cid in sorted(uncs, key=lambda x: -x[0])[:5]]
        self._emit(
            "observe",
            f"Map: {state['sources']} sources, {len(claims)} claims; beliefs {dict(sorted(statuses.items()))}; "
            f"mean uncertainty {mean_unc:.2f}; {n_contra} contradiction pair(s); {len(open_q)} open question(s)."
            + (f" Goal: {goal}" if goal else ""),
            most_uncertain,
            **state,
        )
        return state

    # ================================================================== question generation
    def _all_question_texts(self) -> list[str]:
        return [q.text for q in self.store.list(Question)]

    def _add_question(self, text: str, *, origin: str, priority: float, claim_ids: Sequence[str] = (), reason: str = "") -> Question | None:
        text = " ".join(str(text).split())
        if len(text) < 8 or looks_like_injection(text):
            return None
        if not text.endswith("?"):
            text = text.rstrip(".") + "?"
        if self.store.find_question(text) or is_duplicate(text, self._all_question_texts(), self.cfg.selection.dup_threshold):
            return None
        q = Question(
            text=text, origin=origin, priority=round(max(0.0, min(1.0, priority)), 3), claim_ids=list(claim_ids),
            metadata={"reason": reason, "run_id": self._run.id if self._run else None},
        )
        return self.store.add_question(q)

    def _link_claims(self, text: str, k: int = 2) -> list[str]:
        hits = self._search(text, "claim", k)
        return [cid for cid, s in hits if s >= 0.3 and self.store.get_claim(cid)]

    def _maybe_generate_questions(self, goal: str | None) -> list[str]:
        open_q = self.store.list_questions(status="open")
        if len(open_q) >= self.cfg.min_queue:
            return []
        beliefs = self.store.list(Belief)
        claims = {c.id: c for c in self.store.list(Claim)}
        ranked = sorted(beliefs, key=lambda b: (-(1 - abs(2 * b.confidence - 1)), claims[b.claim_id].text if b.claim_id in claims else ""))
        docs = [
            {"text": f"{claims[b.claim_id].text} (confidence {b.confidence:.2f})", "id": b.claim_id}
            for b in ranked[:8]
            if b.claim_id in claims
        ]
        if not docs:
            return []
        asked = "; ".join(self._all_question_texts()[-15:]) or "(none)"
        new: list[Question] = []
        try:
            data = P.call(self.llm, "generate_questions", P.make("generate_questions", documents=docs, fields={"goal": goal or "the topic", "asked": asked}))
            items = data.get("questions") or []
        except LLMError as e:
            items = []
            self._emit("note", f"Question generation failed ({e}); using heuristic gap questions.")
        for it in items[: self.cfg.max_new_questions]:
            text = it.get("text") if isinstance(it, dict) else str(it)
            pr = float(it.get("priority", 0.5) or 0.5) if isinstance(it, dict) else 0.5
            q = self._add_question(str(text or ""), origin="curiosity", priority=pr, claim_ids=self._link_claims(str(text or "")), reason="generated: queue thin")
            if q:
                new.append(q)
        if not new:  # heuristic fallback: most uncertain claims
            for d in docs[:3]:
                c = claims[d["id"]]
                q = self._add_question(f"Is it true that {_lc(c.text)}?", origin="gap", priority=0.4, claim_ids=[c.id], reason="uncertain claim")
                if q:
                    new.append(q)
        if new:
            self._emit(
                "generate_questions",
                f"Question queue was thin ({len(open_q)} open); added {len(new)}: " + " | ".join(q.text for q in new),
                [q.id for q in new],
            )
        return [q.id for q in new]

    # ================================================================== SELECT
    def _rank(self, goal: str | None, asked: Sequence[str], picked: Sequence[str]) -> list[ScoredQuestion]:
        questions = self.store.list_questions(status="open")
        resolved = [q.text for q in self.store.list(Question) if q.status in ("answered", "dropped")]
        beliefs = {b.claim_id: b for b in self.store.list(Belief)}
        ev_count: Counter[str] = Counter(e.claim_id for e in self.store.list(Evidence) if e.stance != "context")
        mx = max(ev_count.values(), default=0)
        cent = {}
        for q in questions:
            n = sum(ev_count.get(c, 0) for c in q.claim_ids)
            cent[q.id] = math.log1p(n) / math.log1p(mx) if mx else 0.0
        return rank_questions(
            questions, beliefs=beliefs, goal=goal, asked=asked, resolved=resolved, picked_ids=picked,
            centrality=cent, cfg=self.cfg.selection,
        )

    # ================================================================== GATHER helpers
    def _retrieve(self, text: str, k: int | None = None) -> list[tuple[str, float]]:
        return self._search(text, "chunk", k or self.cfg.retrieve_k)

    def _search(self, text: str, kind: str, k: int) -> list[tuple[str, float]]:
        """Hybrid (vector + FTS) search with deterministic tie-breaking by item text.

        Store row order follows random ids, so equal scores are re-ordered by text to keep
        runs reproducible; we over-fetch so ties at the cut-off are resolved the same way.
        """
        hits = hybrid_search(self.store, text, kind=kind, k=2 * k + 4, embedder=self.embedder)
        cls = Chunk if kind == "chunk" else Claim
        keyed = []
        for item_id, score in hits:
            obj = self.store.get(cls, item_id)
            if obj is not None:
                keyed.append((-round(score, 6), obj.text, item_id, score))
        keyed.sort()
        return [(item_id, score) for _, _, item_id, score in keyed[:k]]

    def _docs(self, chunk_ids: Sequence[str]) -> list[dict[str, Any]]:
        out = []
        srcs: dict[str, Source | None] = {}
        for cid in chunk_ids:
            ch = self.store.get(Chunk, cid)
            if ch is None:
                continue
            if ch.source_id not in srcs:
                srcs[ch.source_id] = self.store.get_source(ch.source_id)
            out.append(P.doc_for_chunk(ch, srcs[ch.source_id]))
        return out

    def _use_tools(self, query: str) -> list[str]:
        """Call tools with ``query``; ingest returned docs as lower-reliability sources.

        A tool returns ``str`` / ``dict(text, title?, uri|url?, reliability?)`` or a list of them.
        Reliability is capped at ``tool_reliability``. Returns new chunk ids.
        """
        new_chunks: list[str] = []
        for name, fn in self.tools.items():
            if self._tool_calls >= self.cfg.max_tool_calls:
                break
            self._tool_calls += 1
            try:
                res = fn(query)
            except Exception as e:  # noqa: BLE001 - tools are user code
                self._emit("note", f"Tool {name} failed: {e}", tool=name)
                continue
            items = res if isinstance(res, list) else [res] if res else []
            srcs = []
            for it in items[:10]:
                d = {"text": it} if isinstance(it, str) else dict(it or {})
                text = str(d.get("text") or d.get("content") or d.get("snippet") or "").strip()
                if not text:
                    continue
                rel = min(float(d.get("reliability", self.cfg.tool_reliability)), self.cfg.tool_reliability)
                src = ingest_text(
                    self.store, text, title=str(d.get("title") or f"{name}: {query[:60]}"),
                    uri=str(d.get("uri") or d.get("url") or ""),
                    metadata={"origin": f"tool:{name}", "query": query},
                    reliability=rel, embedder=self.embedder,
                )
                srcs.append(src)
                new_chunks += [c.id for c in self.store.list_chunks(src.id)]
            self._emit(
                "gather",
                f"Tool {name}({query!r}) returned {len(items)} item(s); ingested {len(srcs)} source(s) at reliability <= {self.cfg.tool_reliability}.",
                [s.id for s in srcs] + new_chunks,
                tool=name,
            )
        return list(dict.fromkeys(new_chunks))

    # ================================================================== one step
    def _step(self, sq: ScoredQuestion, goal: str | None, runners_up: Sequence[ScoredQuestion]) -> dict[str, Any]:
        q = sq.question
        q.status = "in_progress"
        self.store.update_question(q)
        self._emit(
            "select_question",
            f"Selected [{q.origin}] \"{q.text}\" -- {sq.explain()}"
            + (". Runners-up: " + "; ".join(f"{r.question.text[:60]} ({r.value:.3f})" for r in runners_up) if runners_up else ""),
            [q.id, *q.claim_ids],
            value=sq.value, uncertainty=sq.uncertainty, importance=sq.importance, novelty=sq.novelty, bonus=sq.bonus, **sq.parts,
        )
        # -- initial retrieval for the question
        hits = self._retrieve(q.text)
        relevant = [cid for cid, s in hits if s >= self.cfg.min_relevance]
        if self.tools and len(relevant) < self.cfg.tool_trigger_hits:
            relevant += self._use_tools(q.text)
        # -- HYPOTHESIZE
        hyps = self._hypothesize(q, relevant or [c for c, _ in hits])
        self._emit(
            "hypothesize",
            f"{len(hyps)} hypothesis(es): " + " | ".join(f"H{i + 1}: {h[0]}" for i, h in enumerate(hyps)),
            [q.id] + [h[1] for h in hyps if h[1]],
            hypotheses=[{"text": h[0], "claim_id": h[1]} for h in hyps],
        )
        # -- GATHER
        scores: dict[str, float] = {cid: s for cid, s in hits}
        for cid in relevant:
            scores.setdefault(cid, self.cfg.min_relevance)
        per_hyp: dict[int, list[str]] = {}
        for i, (htext, _cid, _p) in enumerate(hyps):
            hh = self._retrieve(htext)
            per_hyp[i] = [cid for cid, s in hh if s >= self.cfg.min_relevance]
            for cid, s in hh:
                scores[cid] = max(scores.get(cid, 0.0), s)
        gathered = [cid for cid, s in sorted(scores.items(), key=lambda kv: -kv[1]) if s >= self.cfg.min_relevance]
        claim_hits = [
            (cid, s) for cid, s in self._search(q.text, "claim", self.cfg.claim_k)
            if s >= self.cfg.min_relevance
        ]
        n_src = len({self.store.get(Chunk, c).source_id for c in gathered if self.store.get(Chunk, c)})
        self._emit(
            "gather",
            f"Retrieved {len(gathered)} relevant chunk(s) from {n_src} source(s) and {len(claim_hits)} related claim(s) (vector + full-text).",
            gathered + [c for c, _ in claim_hits],
            chunk_scores={c: round(scores[c], 4) for c in gathered},
        )
        # -- TEST / JUDGE / UPDATE per hypothesis
        ctx = JudgeContext(self.llm, self.store, self.rng)
        results: list[HypothesisResult] = []
        for i, (htext, hcid, _prior) in enumerate(hyps):
            self.llm.tracker.check() if self.llm.tracker else None
            focus = list(dict.fromkeys(per_hyp.get(i, []) + gathered))[: self.cfg.max_docs]
            results.append(self._test_and_judge(q, htext, hcid, focus, gathered, claim_hits, ctx))
        # -- UPDATE question
        live = [r for r in results if not r.discarded and r.claim_id]
        best = max(live, key=lambda r: (r.final_confidence, r.text), default=None)
        attempts = int(q.metadata.get("attempts", 0) or 0) + 1
        q.metadata = {**q.metadata, "attempts": attempts, "last_run": self._run.id if self._run else None}
        if best is not None and best.final_confidence >= self.cfg.resolve_confidence:
            q.status, q.answer, q.answer_confidence = "answered", best.text, round(best.final_confidence, 4)
            outcome = f"answered: \"{best.text}\" ({best.final_confidence:.2f})"
        elif attempts >= self.cfg.max_attempts:
            q.status = "dropped"
            q.metadata = {**q.metadata, "reason": f"inconclusive after {attempts} attempts"}
            outcome = f"dropped after {attempts} inconclusive attempts"
        else:
            q.status = "open"
            if best is not None:
                q.answer_confidence = round(best.final_confidence, 4)
            outcome = "still open (inconclusive)"
        self.store.update_question(q)
        self._emit(
            "update",
            f"Question {outcome}. Beliefs: "
            + "; ".join(
                f"\"{_clip(r.text, 70)}\" {f'{r.before:.2f}' if r.before is not None else 'new'} -> {r.final_confidence:.2f} ({r.status})"
                + (" [discarded: no evidence]" if r.discarded else "")
                for r in results
            ),
            [q.id] + [r.claim_id for r in results if r.claim_id and not r.discarded],
            question_status=q.status,
        )
        # -- REFLECT
        self._reflect(q, results, goal)
        return {
            "question_id": q.id,
            "question": q.text,
            "status": q.status,
            "value": sq.value,
            "hypotheses": [
                {
                    "text": r.text, "claim_id": r.claim_id, "before": r.before, "evidence_confidence": round(r.evidence_confidence, 4),
                    "final_confidence": round(r.final_confidence, 4), "status": r.status, "n_evidence": r.n_evidence,
                    "verdict": r.judgment.verdict, "agreement": round(r.judgment.agreement, 3), "discarded": r.discarded,
                    "conflicts": r.conflicts,
                }
                for r in results
            ],
        }

    # ================================================================== HYPOTHESIZE
    def _hypothesize(self, q: Question, chunk_ids: Sequence[str]) -> list[tuple[str, str | None, float]]:
        k = self.cfg.k_hypotheses
        hyps: list[tuple[str, str | None, float]] = []
        linked = sorted((c for c in (self.store.get_claim(cid) for cid in q.claim_ids) if c), key=lambda c: c.text)
        for c in linked[: max(k, 2)]:
            hyps.append((c.text, c.id, 0.5))
        if len(hyps) < k:
            docs = self._docs(list(chunk_ids)[: self.cfg.max_docs])
            try:
                data = P.call(self.llm, "generate_hypotheses", P.make("generate_hypotheses", documents=docs, fields={"question": q.text}, k=k))
                items = data.get("hypotheses") or []
            except LLMError as e:
                items = []
                self._emit("note", f"Hypothesis generation failed ({e}).", [q.id])
            for it in items:
                text = " ".join(str(it.get("text") if isinstance(it, dict) else it).split())
                if len(text) < 8 or is_duplicate(text, [h[0] for h in hyps], 0.9):
                    continue
                if looks_like_injection(text):
                    self._emit("note", f"Ignored instruction-like hypothesis text (treated as data): {_clip(text, 120)}", [q.id])
                    continue
                prior = float(it.get("prior", 0.5) or 0.5) if isinstance(it, dict) else 0.5
                hyps.append((text, None, prior))
                if len(hyps) >= max(k, len(q.claim_ids[:2])):
                    break
        if not hyps:  # last resort: the question itself as a declarative claim
            hyps.append((_declarative(q.text), None, 0.5))
        return hyps

    # ================================================================== TEST + JUDGE + UPDATE
    def _find_or_create_claim(self, text: str, q: Question) -> tuple[Claim, bool]:
        pol = nlp.polarity(text)[0]
        for c in self.store.list(Claim):
            if nlp.text_similarity(c.text, text) >= 0.9 and nlp.polarity(c.text)[0] == pol:
                return c, False
        claim = Claim(
            text=text.strip(), origin="hypothesis",
            metadata={"run_id": self._run.id if self._run else None, "question_id": q.id},
        )
        self.store.add_claim(claim)
        index_claims(self.store, [claim], self.embedder)
        return claim, True

    def _test_and_judge(
        self,
        q: Question,
        htext: str,
        hcid: str | None,
        focus: list[str],
        gathered: list[str],
        claim_hits: list[tuple[str, float]],
        ctx: JudgeContext,
    ) -> HypothesisResult:
        claim = self.store.get_claim(hcid) if hcid else None
        created = False
        if claim is None:
            claim, created = self._find_or_create_claim(htext, q)
        prev = self.store.get_belief_for_claim(claim.id)
        before = prev.confidence if prev else None
        # evidence linking restricted to the retrieved chunks (TEST uses only gathered evidence)
        chunks = [c for c in (self.store.get(Chunk, cid) for cid in gathered) if c is not None]
        ucfg = UnderstandConfig(link_top_k=max(1, len(chunks)), min_evidence_strength=self.cfg.min_evidence_strength)
        if chunks:
            link_evidence(self.store, self.llm, [(claim, chunks)], cfg=ucfg, embedder=self.embedder, chunk_ns=vector_namespace("chunk", self.embedder))
        run_tag = self._run.id if self._run else "?"
        belief = update_belief(self.store, claim.id, reason=f"run {run_tag}: tested against {len(chunks)} retrieved chunk(s) for question {q.id}", actor="reasoning")
        evs = [e for e in self.store.list_evidence(claim.id) if e.stance != "context"]
        ev_conf = belief.confidence
        # critique (for / against) + devil's advocate, grounded in retrieved docs only
        docs = self._docs(focus)
        allowed = [d["id"] for d in docs]
        crit: dict[str, Any] = {}
        devil: dict[str, Any] = {}
        dropped: list[str] = []
        if docs:
            try:
                crit = P.call(self.llm, "critique", P.make("critique", documents=docs, fields={"claim": claim.text}))
                chk = validate_citations(self.store, crit.get("citations") or [], allowed)
                crit["citations"], dropped = chk.valid, dropped + chk.dropped
                devil = P.call(self.llm, "devils_advocate", P.make("devils_advocate", documents=docs, fields={"claim": claim.text}))
                chk = validate_citations(self.store, devil.get("citations") or [], allowed)
                devil["citations"], dropped = chk.valid, dropped + chk.dropped
            except LLMError as e:
                self._emit("note", f"Critique failed for \"{_clip(claim.text, 60)}\": {e}", [claim.id])
        conflicts, consistent = self._consistency(claim, claim_hits, belief)
        sup = [e for e in evs if e.stance == "supports"]
        con = [e for e in evs if e.stance == "contradicts"]
        self._emit(
            "test",
            f"H: \"{claim.text}\" -- evidence {len(sup)} for / {len(con)} against -> evidence-based confidence {ev_conf:.2f}. "
            f"For: {_clip('; '.join(map(str, crit.get('arguments_for', []))), 200) or '-'}. "
            f"Against: {_clip('; '.join(map(str, crit.get('arguments_against', []))), 200) or '-'}. "
            f"Devil's advocate: {_clip(str(devil.get('counterargument', '-')), 160)}"
            + (f". CONFLICTS with {len(conflicts)} held belief(s)" if conflicts else "")
            + (f". Consistent with {len(consistent)} held belief(s)" if consistent else "")
            + (f". Dropped {len(dropped)} invalid citation(s)" if dropped else ""),
            [claim.id] + [e.chunk_id for e in evs if e.chunk_id] + [e.id for e in evs] + conflicts,
            evidence_confidence=ev_conf, n_support=len(sup), n_contradict=len(con), critique=crit, devils_advocate=devil,
            conflicts=conflicts, consistent=consistent, dropped_citations=dropped,
        )
        self._dropped += len(dropped)
        # discard brand-new hypotheses that found no evidence at all (keeps the map clean)
        if created and not evs:
            self.store.delete(Belief, belief.id)
            self.store.delete(Claim, claim.id)
            self._emit("judge", f"Discarded hypothesis \"{_clip(claim.text, 80)}\": no supporting or contradicting evidence retrieved.", [q.id])
            return HypothesisResult(claim.text, None, before, 0.5, 0.5, "hypothesis", 0, Judgment(rationale="no evidence"), discarded=True)
        # JUDGE
        # per-hypothesis seeded RNG: sampling does not depend on the order hypotheses are visited
        hctx = JudgeContext(ctx.llm, ctx.store, random.Random(f"{self.cfg.seed}:{claim.text}"))
        judgment = self.strategy.judge(hctx, claim.text, docs) if docs else Judgment(rationale="no documents")
        self._dropped += len(judgment.dropped_citations)
        delta = self._judge_delta(judgment, len(evs))
        self._emit(
            "judge",
            f"[{self.strategy.name}] verdict {judgment.verdict.upper()} (model P(true)={judgment.llm_confidence:.2f}, "
            f"agreement {judgment.agreement:.2f}, {len(judgment.citations)} valid citation(s)"
            + (f", {len(judgment.dropped_citations)} dropped" if judgment.dropped_citations else "")
            + f") -> bounded adjustment {delta:+.2f} log-odds. {_clip(judgment.rationale, 200)}",
            [claim.id, *judgment.citations],
            judgment=judgment.to_dict(), adjustment=delta,
        )
        belief = self._apply_judgment(claim.id, delta, judgment)
        return HypothesisResult(
            claim.text, claim.id, before, ev_conf, belief.confidence, belief.status, len(evs), judgment, conflicts=conflicts
        )

    def _judge_delta(self, j: Judgment, n_evidence: int) -> float:
        """Target total judgment adjustment (log-odds) for a belief. Zero without evidence."""
        if n_evidence == 0 or j.error or j.verdict == "uncertain":
            return 0.0
        sign = 1.0 if j.verdict == "true" else -1.0
        mag = min(2.0, abs(logit(j.llm_confidence))) or 0.5
        d = sign * mag * self.cfg.judge_weight * j.agreement
        if not j.citations:
            d *= 0.5
        return max(-self.cfg.max_judge_adjust, min(self.cfg.max_judge_adjust, d))

    def _apply_judgment(self, claim_id: str, target: float, j: Judgment) -> Belief:
        """Replace (not accumulate) this belief's judgment adjustment with ``target``."""
        b = self.store.get_belief_for_claim(claim_id)
        assert b is not None
        old = float(b.metadata.get("judge_adjustment", 0.0))
        delta = target - old
        if abs(delta) > 1e-9:
            b = adjust_belief(
                self.store, claim_id, delta,
                reason=f"{self.strategy.name} judgment '{j.verdict}' (model P(true) {j.llm_confidence:.2f}, agreement {j.agreement:.2f}, "
                f"{len(j.citations)} valid citations); judgment term now {target:+.2f}",
                actor="reasoning",
            )
            b.metadata = {**b.metadata, "judge_adjustment": round(target, 6)}
            b = self.store.save_belief(b)
        return b

    def _consistency(self, claim: Claim, claim_hits: list[tuple[str, float]], belief: Belief) -> tuple[list[str], list[str]]:
        """Check a hypothesis against held beliefs; flag and record contradictions."""
        conflicts, consistent = [], []
        extra = self._search(claim.text, "claim", self.cfg.claim_k)
        seen: set[str] = set()
        for cid, s in list(claim_hits) + list(extra):
            if cid == claim.id or cid in seen or s < self.cfg.min_relevance:
                continue
            seen.add(cid)
            other = self.store.get_claim(cid)
            ob = self.store.get_belief_for_claim(cid)
            if other is None or ob is None or ob.confidence < 0.6:
                continue
            st, _strength = nlp.lexical_stance(other.text, claim.text, min_overlap=0.35)
            if st == "contradicts":
                conflicts.append(cid)
            elif st == "supports":
                consistent.append(cid)
        if conflicts:
            fresh = self.store.get_claim(claim.id) or claim
            lst = list(fresh.metadata.get("contradicts", []))
            for cid in conflicts:
                other = self.store.get_claim(cid)
                if other is None:
                    continue
                if cid not in lst:
                    lst.append(cid)
                olst = list(other.metadata.get("contradicts", []))
                if claim.id not in olst:
                    other.metadata = {**other.metadata, "contradicts": olst + [claim.id]}
                    self.store.add_claim(other)
                if belief.confidence >= 0.6:  # both sides currently held: a genuine open conflict
                    self._add_question(
                        f'Which is correct: "{other.text.rstrip(".")}" or "{fresh.text.rstrip(".")}"',
                        origin="contradiction", priority=0.8, claim_ids=[cid, claim.id], reason="reasoning found conflict",
                    )
            fresh.metadata = {**fresh.metadata, "contradicts": lst}
            self.store.add_claim(fresh)
        return conflicts, consistent

    # ================================================================== REFLECT
    def _reflect(self, q: Question, results: list[HypothesisResult], goal: str | None) -> None:
        docs = [
            {
                "text": f"{r.text} -- {f'{r.before:.2f}' if r.before is not None else 'new'} -> {r.final_confidence:.2f} ({r.status}); "
                f"verdict {r.judgment.verdict}",
                "id": r.claim_id or f"h{i}",
            }
            for i, r in enumerate(results)
            if not r.discarded
        ]
        learned, next_qs = "", []
        if docs:
            try:
                data = P.call(self.llm, "reflect", P.make("reflect", documents=docs, fields={"question": q.text}))
                learned = str(data.get("learned") or "")
                next_qs = [str(x.get("text") if isinstance(x, dict) else x) for x in (data.get("next_questions") or [])]
            except LLMError as e:
                learned = f"(reflection failed: {e})"
        new_ids = []
        for t in next_qs[:3]:
            nq = self._add_question(t, origin="curiosity", priority=0.45, claim_ids=self._link_claims(t), reason=f"reflection on {q.id}")
            if nq:
                new_ids.append(nq.id)
        # heuristic follow-ups: uncertain hypotheses resting on a single independent source group
        for r in results:
            if r.discarded or not r.claim_id or not (0.35 < r.final_confidence < self.cfg.resolve_confidence):
                continue
            b = self.store.get_belief_for_claim(r.claim_id)
            if b and int(b.metadata.get("n_groups", 0)) <= 1:
                nq = self._add_question(
                    f"Is there independent evidence that {_lc(r.text)}?", origin="gap", priority=0.4,
                    claim_ids=[r.claim_id], reason="single independent source group",
                )
                if nq:
                    new_ids.append(nq.id)
        moved = [r for r in results if not r.discarded and r.before is not None and abs(r.final_confidence - r.before) >= 0.01]
        summary = (
            f"Learned: {learned or 'no change'}. "
            f"{len(moved)} belief(s) moved, {sum(1 for r in results if r.before is None and not r.discarded)} new; "
            f"{len(new_ids)} follow-up question(s) queued."
        )
        self._emit("reflect", summary, [q.id] + new_ids, next_questions=new_ids)

    # ================================================================== ASK
    def _ask(self, question: str) -> Answer:
        ensure_index(self.store, self.embedder, "chunk")
        ensure_index(self.store, self.embedder, "claim")
        hits = self._retrieve(question, k=self.cfg.max_docs)
        relevant = [cid for cid, s in hits if s >= self.cfg.min_relevance]
        if self.tools and len(relevant) < self.cfg.tool_trigger_hits:
            relevant += self._use_tools(question)
        self._emit("gather", f"Retrieved {len(relevant)} relevant chunk(s) for the question.", relevant,
                   chunk_scores={c: round(s, 4) for c, s in hits})
        qrow = self.store.find_question(question)
        if not relevant:
            ans = Answer(question, "Insufficient evidence: no relevant passages were found in the knowledge store.", 0.0, insufficient=True)
            self._emit("answer", ans.text, [])
            ans.question_id = self._record_user_question(qrow, question, ans).id
            return ans
        docs = self._docs(relevant[: self.cfg.max_docs])
        allowed = [d["id"] for d in docs]
        try:
            data = P.call(self.llm, "answer_question", P.make("answer_question", documents=docs, fields={"question": question}))
        except LLMError as e:
            data = {"answer": "", "confidence": 0.0, "citations": [], "reasoning": f"model failed: {e}"}
        text = " ".join(str(data.get("answer") or "").split())
        text, injected = strip_injection(text)
        if injected:
            self._emit("note", f"Removed {len(injected)} instruction-like sentence(s) from the answer; source text is data, not instructions.",
                       allowed, removed=injected)
        llm_conf = max(0.0, min(1.0, float(data.get("confidence", 0.0) or 0.0)))
        chk = validate_citations(self.store, data.get("citations") or [], allowed)
        self._dropped += len(chk.dropped)
        # supporting / contradicting claims from the belief map
        sup, contra = [], []
        for cid, s in self._search(question, "claim", self.cfg.claim_k + 2):
            if s < self.cfg.min_relevance:
                continue
            c = self.store.get_claim(cid)
            b = self.store.get_belief_for_claim(cid)
            if c is None or b is None or not text:
                continue
            st, strength = nlp.lexical_stance(c.text, text)
            row = {"claim_id": c.id, "belief_id": b.id, "text": c.text, "confidence": round(b.confidence, 4), "status": b.status, "relevance": round(s, 4)}
            if st == "supports":
                sup.append(row)
            elif st == "contradicts":
                contra.append(row)
        # evidence-based confidence (the model's own confidence is a minor 20% input)
        coverage = min(1.0, len(chk.valid) / 2.0)
        if sup:
            top = sorted(sup, key=lambda r: -r["relevance"])[:3]
            wsum = sum(r["relevance"] for r in top) or 1.0
            ev_conf = sum(r["confidence"] * r["relevance"] for r in top) / wsum
        else:
            ev_conf = 0.4 * coverage  # cited passages but no corroborating belief
        contra_conf = max((r["confidence"] for r in contra if r["confidence"] >= 0.5), default=0.0)
        conf = (0.8 * ev_conf * (0.5 + 0.5 * coverage) + 0.2 * llm_conf) * (1.0 - 0.5 * contra_conf)
        insufficient = False
        if not text or not chk.valid:
            insufficient = True
            conf = min(conf, 0.2)
            text = "Insufficient evidence: the retrieved passages do not support a grounded answer." + (
                f" (Ungrounded model answer withheld: {_clip(text, 160)})" if text else ""
            )
        elif conf < self.cfg.answer_min_confidence:
            insufficient = True
            text = f"Insufficient evidence for a confident answer (confidence {conf:.2f}). Tentatively: {text}"
        if injected:
            text += " (Note: instruction-like text found in a source was ignored as data.)"
        cites = []
        for cid in chk.valid:
            ch = self.store.get(Chunk, cid)
            best = nlp.best_sentences(text, ch.text, k=1) if ch else []
            c = citation_for_chunk(self.store, cid, best[0] if best else "")
            if c:
                cites.append(c)
        ans = Answer(
            question, text, round(conf, 4), cites, sup, contra, insufficient=insufficient, dropped_citations=chk.dropped,
        )
        self._emit(
            "answer",
            f"{text} [confidence {conf:.2f}: evidence {ev_conf:.2f}, coverage {coverage:.2f}, model {llm_conf:.2f}, "
            f"contradicting {contra_conf:.2f}]" + (f" Dropped {len(chk.dropped)} invalid citation(s)." if chk.dropped else ""),
            chk.valid + [r["claim_id"] for r in sup + contra],
            confidence=conf, insufficient=insufficient, dropped_citations=chk.dropped,
        )
        ans.question_id = self._record_user_question(qrow, question, ans).id
        return ans

    def _record_user_question(self, qrow: Question | None, question: str, ans: Answer) -> Question:
        q = qrow or Question(text=question, origin="user", priority=0.8)
        if ans.insufficient:
            if q.status != "answered":
                q.status = "open"  # leave it for think() to pursue
        else:
            q.status, q.answer, q.answer_confidence = "answered", ans.text, ans.confidence
        q.metadata = {**q.metadata, "asked_run": self._run.id if self._run else None}
        return self.store.update_question(q)

    # ================================================================== misc
    def _run_summary(self, processed: list[dict], changes: dict[str, dict], stop: str) -> str:
        answered = sum(1 for p in processed if p["status"] == "answered")
        moved = sorted(
            ((c["after"] - (c["before"] if c["before"] is not None else 0.5)), c["text"]) for c in changes.values()
        )
        parts = [f"{len(processed)} question(s) investigated, {answered} answered", f"{len(changes)} belief(s) touched"]
        if moved:
            up, down = moved[-1], moved[0]
            if up[0] > 0.01:
                parts.append(f"largest rise {up[0]:+.2f} \"{_clip(up[1], 60)}\"")
            if down[0] < -0.01:
                parts.append(f"largest drop {down[0]:+.2f} \"{_clip(down[1], 60)}\"")
        parts.append(f"stop: {stop}")
        return "; ".join(parts)


def _clip(s: str, n: int) -> str:
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[: n - 1] + "…"


def _lc(text: str) -> str:
    t = text.strip().rstrip(".")
    if len(t) > 1 and t[0].isupper() and not t[1].isupper():
        t = t[0].lower() + t[1:]
    return t


def _declarative(question: str) -> str:
    q = question.strip().rstrip("?").strip()
    for pre in ("Is it true that ", "Is there evidence that ", "Is there independent evidence that "):
        if q.lower().startswith(pre.lower()):
            q = q[len(pre):]
            break
    return (q[:1].upper() + q[1:] + ".") if q else "Unknown."
