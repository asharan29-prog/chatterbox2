"""Noesis Research Agent: an internet-connected agent built on the Noesis engine.

Pipeline of :meth:`ResearchAgent.research`::

    plan sub-questions (model) -> search each (provider) -> pick diverse, reputable sources
    (one per domain by default, reliability-ranked, deny/allow lists) -> fetch with guards ->
    ingest with domain reliability priors -> understand (claims, evidence, contradictions)
    -> Mind.think(goal) with a ``web_search`` tool so the loop can go back to the internet when
    evidence is thin -> ask() the main and sub-questions -> cited research brief (markdown) with
    a sources table, confidence per conclusion and an explicit "what we could not verify" list,
    plus the standard Noesis HTML report.

>>> from noesis import KnowledgeStore, get_llm
>>> from noesis.agent import ResearchAgent
>>> agent = ResearchAgent(KnowledgeStore("research.db"), get_llm("anthropic:claude-sonnet-5-5"), search="brave")
>>> result = agent.research("Does intermittent fasting improve longevity?", depth=2)
>>> result.write("brief.md")            # + brief.report.html
>>> agent.watch("Does intermittent fasting improve longevity?")   # later: only new URLs
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .. import nlp
from ..ingest import LoadedDocument, ingest_document
from ..llm.base import LLMError
from ..llm.prompts import build_prompt, system_prompt
from ..reason import Budget, Mind, MindConfig
from ..store import KnowledgeStore
from ..types import Belief, Claim, Evidence, Source
from ..understand import understand
from .web import (
    DomainPolicy,
    FetchedPage,
    FetchError,
    SearchError,
    SearchResult,
    WebToolkit,
    registrable_domain,
)

__all__ = ["ResearchAgent", "ResearchBudget", "ResearchResult", "SourceRecord", "Conclusion", "WatchResult", "plan_heuristic"]

PLAN_SCHEMA = {
    "type": "object",
    "required": ["sub_questions"],
    "properties": {
        "sub_questions": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["question"],
                "properties": {"question": {"type": "string"}, "query": {"type": "string"}, "why": {"type": "string"}},
            },
        }
    },
}
PLAN_INSTRUCTIONS = (
    "You are planning web research for the question above. Break it into {n} distinct, specific "
    "sub-questions that together would answer it, covering: the strongest evidence (systematic "
    "reviews, large studies, official data), mechanisms or definitions, risks/criticisms/"
    "counter-evidence, and recent developments. For each give a concise web search query "
    "(keywords, no operators). Return {\"sub_questions\": [{\"question\", \"query\", \"why\"}]}."
)


# ============================================================================ data
@dataclass
class ResearchBudget:
    """Limits for one research run. ``depth`` scales sub-questions and reasoning steps."""

    max_sources: int = 8
    results_per_query: int = 6
    per_domain: int = 1
    #: sources whose prior is below this are skipped (content farms sit at 0.3)
    min_reliability: float = 0.35
    think_steps: int | None = None  # default 2 x depth
    max_llm_calls: int | None = None
    seconds: float | None = None
    #: times the reasoning loop may call the web_search tool per think()/ask() run
    max_tool_searches: int = 3
    #: pages fetched per web_search tool call
    tool_fetches: int = 2


@dataclass
class SourceRecord:
    url: str
    title: str = ""
    domain: str = ""
    final_url: str = ""
    reliability: float = 0.0
    reliability_reason: str = ""
    query: str = ""
    provider: str = ""
    via: str = "plan"  # plan | reasoning | watch
    status: str = "pending"  # ingested | failed | skipped
    reason: str = ""
    fetched_at: str = ""
    source_id: str | None = None
    injection_suspect: bool = False
    truncated: bool = False
    n: int = 0  # citation number in the brief


@dataclass
class Conclusion:
    question: str
    answer: str
    confidence: float
    insufficient: bool
    citations: list[int] = field(default_factory=list)
    quotes: list[tuple[int, str]] = field(default_factory=list)
    dropped_citations: int = 0


@dataclass
class ResearchResult:
    question: str
    sub_questions: list[dict[str, str]]
    sources: list[SourceRecord]
    main: Conclusion
    conclusions: list[Conclusion]
    beliefs: list[dict[str, Any]]
    contested: list[dict[str, Any]]
    unverified: list[str]
    run: dict[str, Any]
    markdown: str = ""
    model: str = ""
    provider: str = ""
    store: KnowledgeStore | None = None

    @property
    def cited_sources(self) -> list[SourceRecord]:
        return [s for s in self.sources if s.status == "ingested"]

    def write(self, path: str | Path, *, html: bool = True) -> list[Path]:
        """Write the markdown brief to ``path`` and (``html=True``) the Noesis HTML report next to it
        as ``<stem>.report.html``. Returns the written paths."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.markdown, encoding="utf-8")
        out = [p]
        if html and self.store is not None:
            from ..report import write_report

            out.append(write_report(self.store, p.with_name(p.stem + ".report.html"), fmt="html", title=f"Research: {self.question}"))
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "model": self.model,
            "search_provider": self.provider,
            "answer": asdict(self.main),
            "sub_questions": self.sub_questions,
            "conclusions": [asdict(c) for c in self.conclusions],
            "beliefs": self.beliefs,
            "contested": self.contested,
            "could_not_verify": self.unverified,
            "sources": [asdict(s) for s in self.sources],
            "run": self.run,
            "brief_markdown": self.markdown,
        }


@dataclass
class WatchResult:
    topic: str
    new_sources: list[SourceRecord]
    changes: list[dict[str, Any]]
    markdown: str


# ============================================================================ planning
_AUX = re.compile(r"^(is|are|was|were|does|do|did|can|could|should|would|will|has|have|how|what|why|which|when|where|who)\s+", re.I)


def _core(question: str) -> str:
    q = " ".join(question.strip().rstrip("?.!").split())
    prev = None
    while prev != q:
        prev = q
        q = _AUX.sub("", q)
    return q or question.strip()


def plan_heuristic(question: str, n: int = 3) -> list[dict[str, str]]:
    """Offline fallback planner: template sub-questions with keyword queries."""
    core = _core(question)
    templates = [
        ("What does the strongest evidence (systematic reviews, large studies) say about {c}?", "{c} systematic review evidence"),
        ("What are the main risks, criticisms or counter-evidence regarding {c}?", "{c} risks criticism"),
        ("What mechanisms or definitions explain {c}?", "{c} mechanism explained"),
        ("What do official guidelines or recent studies say about {c}?", "{c} guidelines recent study"),
        ("Where do reputable sources disagree about {c}?", "{c} debate conflicting evidence"),
        ("Which groups or conditions change the answer about {c}?", "{c} differences by group"),
    ]
    return [{"question": t.format(c=core), "query": s.format(c=core), "why": "template"} for t, s in templates[:n]]


def _mock_plan_handler(prompt: str, schema: dict | None) -> dict:
    from ..llm.prompts import parse_field

    q = parse_field(prompt, "question") or ""
    m = re.search(r"into (\d+) distinct", prompt)
    return {"sub_questions": plan_heuristic(q, int(m.group(1)) if m else 3)}


def _install_mock(llm: Any) -> None:
    from ..llm.mock import MockLLM

    target = getattr(llm, "_llm", llm)
    if isinstance(target, MockLLM) and "plan_research" not in target._handlers:
        target.register_task("plan_research", _mock_plan_handler)


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60] or "topic"


def _label(p: float) -> str:
    return "high" if p >= 0.8 else "moderate" if p >= 0.6 else "low" if p >= 0.4 else "very low"


def _md(s: str) -> str:
    return " ".join(str(s).split()).replace("|", "\\|")


def _clip(s: str, n: int) -> str:
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[: n - 1] + "…"


# ============================================================================ agent
class ResearchAgent:
    """Internet-connected research on top of a :class:`KnowledgeStore` and any Noesis model.

    ``search``: a provider name (``ddg``, ``brave``, ``tavily``, ``searxng``, ``mock``), a
    :class:`WebToolkit`, or ``None`` (``$NOESIS_SEARCH`` or ddg). ``on_event`` receives short
    progress strings. ``on_thought`` streams the reasoning trace.
    """

    def __init__(
        self,
        store: KnowledgeStore,
        llm: Any = None,
        *,
        search: str | WebToolkit | None = None,
        budget: ResearchBudget | None = None,
        policy: DomainPolicy | None = None,
        strategy: str = "simple",
        on_event: Callable[[str], None] | None = None,
        on_thought: Callable[[Any], None] | None = None,
        **web_kwargs: Any,
    ) -> None:
        if llm is None:
            from ..llm.mock import MockLLM

            llm = MockLLM(name="mock")
        self.store = store
        self.llm = llm
        _install_mock(llm)
        self.web = search if isinstance(search, WebToolkit) else WebToolkit.create(search, policy=policy, **web_kwargs)
        if policy is not None and isinstance(search, WebToolkit):
            self.web.fetcher.policy = policy
        self.policy = self.web.policy
        self.budget = budget or ResearchBudget()
        self.strategy = strategy
        self.on_event = on_event
        self.on_thought = on_thought
        self.records: list[SourceRecord] = []

    # ------------------------------------------------------------------ helpers
    @property
    def provider_name(self) -> str:
        if getattr(self.web, "mock_web", None) is not None:
            return "mock (offline fixture web)"
        return self.web.search_provider.name

    @property
    def model_name(self) -> str:
        return str(getattr(self.llm, "name", type(self.llm).__name__))

    def _event(self, msg: str) -> None:
        if self.on_event is not None:
            self.on_event(msg)

    def _known_uris(self) -> set[str]:
        return {s.uri for s in self.store.list_sources() if s.uri}

    # ------------------------------------------------------------------ plan
    def plan(self, question: str, n: int = 3) -> list[dict[str, str]]:
        """Sub-questions (each with a search ``query``). Falls back to templates on model failure."""
        subs: list[dict[str, str]] = []
        try:
            prompt = build_prompt("plan_research", PLAN_INSTRUCTIONS.replace("{n}", str(n)), fields={"question": question})
            data = self.llm.complete(prompt, system=system_prompt(), json_schema=PLAN_SCHEMA, temperature=0.2, max_tokens=800)
            for it in (data or {}).get("sub_questions", []):
                q = " ".join(str(it.get("question", "")).split())
                if len(q) < 8 or any(nlp.text_similarity(q, s["question"]) > 0.9 for s in subs):
                    continue
                query = " ".join(str(it.get("query") or q).split())[:200]
                subs.append({"question": q[:300], "query": query, "why": str(it.get("why", ""))[:200]})
        except (LLMError, AttributeError, TypeError, ValueError) as e:
            self._event(f"planner failed ({e}); using template sub-questions")
        if not subs:
            subs = plan_heuristic(question, n)
        return subs[:n]

    # ------------------------------------------------------------------ search + select
    def _search_all(self, queries: list[str]) -> list[list[SearchResult]]:
        out = []
        for q in queries:
            try:
                res = self.web.search(q, max_results=self.budget.results_per_query)
                self._event(f"search[{self.provider_name}] {q!r}: {len(res)} result(s)")
            except SearchError as e:
                self._event(f"search failed for {q!r}: {e}")
                res = []
            out.append(res)
        return out

    def _score(self, r: SearchResult, question: str) -> float:
        prior, _ = self.policy.reliability(r.url)
        overlap = nlp.text_similarity(question, f"{r.title} {r.snippet}") if (r.title or r.snippet) else 0.0
        return 0.6 * prior + 0.25 / max(r.rank, 1) + 0.15 * overlap

    def _select_and_ingest(self, question: str, per_query: list[list[SearchResult]], *, limit: int, via: str) -> list[SourceRecord]:
        """Round-robin over queries, best-scored first, one source per domain; fetch + ingest."""
        known = self._known_uris()
        seen_urls = {r.url for r in self.records} | {r.final_url for r in self.records if r.final_url}
        # research: diversity counts what the knowledge base already has; watch: only this run's picks
        domain_count: Counter[str] = Counter(
            registrable_domain(u) for u in known if u.startswith("http") and via != "watch"
        )
        queues = [sorted(rs, key=lambda r: -self._score(r, question)) for rs in per_query]
        ingested: list[SourceRecord] = []
        while len(ingested) < limit and any(queues):
            for queue in queues:
                if len(ingested) >= limit:
                    break
                while queue:
                    r = queue.pop(0)
                    if r.url in seen_urls or r.url in known:
                        continue
                    seen_urls.add(r.url)
                    rec = self._consider(r, domain_count, via)
                    if rec.status == "ingested":
                        ingested.append(rec)
                        domain_count[rec.domain] += 1
                        break
        return ingested

    def _consider(self, r: SearchResult, domain_count: Counter[str], via: str) -> SourceRecord:
        rec = SourceRecord(url=r.url, title=r.title, domain=r.domain, query=r.query, provider=r.provider, via=via)
        self.records.append(rec)
        ok, why = self.policy.permitted(r.url)
        prior, prior_why = self.policy.reliability(r.url)
        if not ok:
            rec.status, rec.reason = "skipped", why
        elif domain_count[rec.domain] >= self.budget.per_domain:
            rec.status, rec.reason = "skipped", f"already have {domain_count[rec.domain]} source(s) from {rec.domain} (diversity)"
        elif prior < self.budget.min_reliability:
            rec.status, rec.reason = "skipped", f"low reliability prior {prior:.2f}: {prior_why}"
        if rec.status == "skipped":
            rec.reliability, rec.reliability_reason = prior, prior_why
            return rec
        try:
            page = self.web.fetch(r.url)
        except FetchError as e:
            rec.status, rec.reason = "failed", f"{e.reason}: {e}"
            self._event(f"skip {r.url} ({e.reason})")
            return rec
        final_known = page.final_url in self._known_uris()
        if final_known:
            rec.status, rec.reason = "skipped", "already in the knowledge base"
            return rec
        self._ingest_page(page, rec)
        return rec

    def _ingest_page(self, page: FetchedPage, rec: SourceRecord) -> Source:
        rel, why = self.policy.reliability(page.final_url, page.metadata)
        if page.injection_suspect:
            rel, why = max(0.3, rel - 0.15), why + "; -0.15: page contains instruction-like text aimed at AI (treated as data)"
        doc = LoadedDocument(
            page.text, page.title or page.final_url, "url", page.final_url,
            {
                "origin": "web", "url": page.url, "query": rec.query, "provider": rec.provider, "via": rec.via,
                "fetched_at": page.fetched_at, "content_type": page.content_type, "truncated": page.truncated,
                "injection_suspect": page.injection_suspect,
                **{k: v for k, v in page.metadata.items() if k in ("doi", "citation_journal_title", "article:published_time", "description")},
            },
        )
        src = ingest_document(self.store, doc, reliability=rel, independence_group=f"web:{registrable_domain(page.final_url)}")
        if src.reliability_reason != why:
            src = self.store.set_source_reliability(src.id, src.reliability, why)
        rec.status, rec.source_id = "ingested", src.id
        rec.final_url, rec.title, rec.domain = page.final_url, page.title or rec.title, registrable_domain(page.final_url)
        rec.reliability, rec.reliability_reason = src.reliability, src.reliability_reason
        rec.fetched_at, rec.injection_suspect, rec.truncated = page.fetched_at, page.injection_suspect, page.truncated
        self._event(f"ingested {page.final_url} (reliability {src.reliability:.2f})")
        return src

    # ------------------------------------------------------------------ tool for the reasoning loop
    def web_search_tool(self, query: str) -> list[dict[str, Any]]:
        """``Mind`` tool: search the web for ``query`` and return a few fresh, guarded pages."""
        self._event(f"reasoning loop searched the web: {query!r}")
        try:
            results = self.web.search(query, max_results=self.budget.results_per_query)
        except SearchError as e:
            self._event(f"tool search failed: {e}")
            return []
        known = self._known_uris()
        seen = {r.url for r in self.records} | {r.final_url for r in self.records if r.final_url}
        domain_count: Counter[str] = Counter(registrable_domain(u) for u in known if u.startswith("http"))
        docs: list[dict[str, Any]] = []
        for r in sorted(results, key=lambda r: -self._score(r, query)):
            if len(docs) >= self.budget.tool_fetches:
                break
            if r.url in seen or r.url in known:
                continue
            rec = SourceRecord(url=r.url, title=r.title, domain=r.domain, query=query, provider=r.provider, via="reasoning")
            self.records.append(rec)
            ok, why = self.policy.permitted(r.url)
            prior, prior_why = self.policy.reliability(r.url)
            if not ok or prior < self.budget.min_reliability or domain_count[rec.domain] >= self.budget.per_domain + 1:
                rec.status, rec.reason = "skipped", why or (f"low reliability prior {prior:.2f}" if prior < self.budget.min_reliability else "domain diversity")
                rec.reliability, rec.reliability_reason = prior, prior_why
                continue
            try:
                page = self.web.fetch(r.url)
            except FetchError as e:
                rec.status, rec.reason = "failed", f"{e.reason}: {e}"
                continue
            if page.final_url in known or page.final_url in seen:
                rec.status, rec.reason = "skipped", "duplicate of a page already used"
                continue
            seen.add(page.final_url)
            rel, rwhy = self.policy.reliability(page.final_url, page.metadata)
            rec.status, rec.final_url, rec.title, rec.fetched_at = "ingested", page.final_url, page.title, page.fetched_at
            rec.reliability, rec.reliability_reason = rel, rwhy  # Mind caps tool sources at 0.55
            rec.domain, rec.injection_suspect = registrable_domain(page.final_url), page.injection_suspect
            domain_count[rec.domain] += 1
            docs.append({"text": page.text, "title": page.title, "url": page.final_url, "reliability": rel})
        return docs

    def _resolve_tool_sources(self) -> None:
        by_uri = {s.uri: s for s in self.store.list_sources() if s.uri}
        for rec in self.records:
            if rec.status == "ingested" and rec.source_id is None:
                src = by_uri.get(rec.final_url or rec.url)
                if src is not None:
                    why = (f"found by the reasoning loop: capped at {src.reliability:.2f} until corroborated "
                           f"(domain prior {rec.reliability:.2f}: {rec.reliability_reason})")
                    if src.reliability_reason != why and src.metadata.get("origin", "").startswith("tool:"):
                        src = self.store.set_source_reliability(src.id, src.reliability, why)
                    rec.source_id = src.id
                    rec.reliability, rec.reliability_reason = src.reliability, src.reliability_reason
                else:
                    rec.status, rec.reason = "failed", "tool result was not ingested"

    def _mind(self) -> Mind:
        b = self.budget
        return Mind(
            self.store, self.llm, strategy=self.strategy,
            budget=Budget(steps=b.think_steps or 4, max_llm_calls=b.max_llm_calls, seconds=b.seconds),
            tools={"web_search": self.web_search_tool},
            config=MindConfig(max_tool_calls=b.max_tool_searches),
            on_thought=self.on_thought,
        )

    # ------------------------------------------------------------------ research
    def research(self, question: str, depth: int = 2) -> ResearchResult:
        """Run the full pipeline for ``question``; ``depth`` 1 (quick) .. 5 (thorough)."""
        question = " ".join(question.split())
        if not question:
            raise ValueError("empty research question")
        depth = max(1, min(int(depth), 5))
        self.records = []
        if not self.store.get_meta("topic"):
            self.store.set_meta("topic", question)
        n_sub = min(1 + depth, 6)
        steps = self.budget.think_steps or 2 * depth
        self._event(f"planning {n_sub} sub-question(s) for: {question}")
        subs = self.plan(question, n_sub)
        queries = list(dict.fromkeys([question] + [s["query"] for s in subs]))
        per_query = self._search_all(queries)
        new = self._select_and_ingest(question, per_query, limit=self.budget.max_sources, via="plan")
        self._event(f"fetched and ingested {len(new)} source(s); building understanding")
        if new:
            understand(self.store, self.llm, source_ids=[r.source_id for r in new if r.source_id])
        mind = self._mind()
        self._event(f"reasoning ({steps} step(s), web_search tool enabled)")
        run = mind.think(goal=question, steps=steps)
        main_ans = mind.ask(question)
        sub_ans = [mind.ask(s["question"]) for s in subs]
        self._resolve_tool_sources()
        if any(r.via == "reasoning" and r.status == "ingested" for r in self.records):
            understand(self.store, self.llm)  # claims/evidence for pages the loop fetched
        self._number_sources()
        main = self._conclusion(main_ans)
        concl = [self._conclusion(a) for a in sub_ans]
        beliefs, contested = self._belief_tables()
        unverified = self._unverified(main, concl, subs, per_query[1:], beliefs)
        run_info = {"id": run.id, "status": run.status, "stop_reason": run.stop_reason, "steps": run.steps,
                    "summary": run.summary, "usage": run.usage}
        self.store.set_meta(f"agent:watch:{_slug(question)}", {
            "question": question, "queries": queries, "sub_questions": subs, "last_run": _now(),
        })
        res = ResearchResult(question, subs, list(self.records), main, concl, beliefs, contested, unverified, run_info,
                             model=self.model_name, provider=self.provider_name, store=self.store)
        res.markdown = self.render_brief(res)
        return res

    # ------------------------------------------------------------------ assembly
    def _number_sources(self) -> None:
        n = 0
        numbered: dict[str, int] = {}
        for rec in self.records:
            if rec.status == "ingested" and rec.source_id:
                if rec.source_id in numbered:  # same page reached twice (e.g. via a redirect)
                    rec.status, rec.reason = "skipped", f"duplicate of source [{numbered[rec.source_id]}]"
                    continue
                n += 1
                rec.n = numbered[rec.source_id] = n

    def _num_by_source(self) -> dict[str, int]:
        return {r.source_id: r.n for r in self.records if r.n and r.source_id}

    def _conclusion(self, ans: Any) -> Conclusion:
        nums = self._num_by_source()
        cites: list[int] = []
        quotes: list[tuple[int, str]] = []
        for c in ans.citations:
            n = nums.get(c.source_id)
            if n is None:  # cited a source from an earlier session: add it to the table
                n = self._add_existing_source(c.source_id)
            if n and n not in cites:
                cites.append(n)
                if c.quote:
                    quotes.append((n, _clip(c.quote, 220)))
        return Conclusion(ans.question, ans.text, float(ans.confidence), bool(ans.insufficient), cites, quotes, len(ans.dropped_citations))

    def _add_existing_source(self, source_id: str) -> int | None:
        src = self.store.get_source(source_id)
        if src is None:
            return None
        n = max((r.n for r in self.records), default=0) + 1
        self.records.append(SourceRecord(
            url=src.uri or src.title, title=src.title, domain=registrable_domain(src.uri) if src.uri.startswith("http") else "",
            final_url=src.uri, reliability=src.reliability, reliability_reason=src.reliability_reason,
            via="earlier", status="ingested", fetched_at=str(src.metadata.get("fetched_at", "")), source_id=src.id, n=n,
        ))
        return n

    def _belief_tables(self, limit: int = 12) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        nums = self._num_by_source()
        claims = {c.id: c for c in self.store.list(Claim)}
        ev: dict[str, list[Evidence]] = defaultdict(list)
        for e in self.store.list(Evidence):
            ev[e.claim_id].append(e)
        rows = []
        for b in self.store.list(Belief):
            c = claims.get(b.claim_id)
            if c is None:
                continue
            sup = sorted({nums[e.source_id] for e in ev[c.id] if e.stance == "supports" and e.source_id in nums})
            con = sorted({nums[e.source_id] for e in ev[c.id] if e.stance == "contradicts" and e.source_id in nums})
            if not sup and not con:
                continue
            groups = {self._group(e.source_id) for e in ev[c.id] if e.stance != "context"}
            rows.append({"claim_id": c.id, "belief_id": b.id, "claim": c.text, "confidence": round(b.confidence, 3),
                         "status": b.status, "supporting_sources": sup, "contradicting_sources": con,
                         "independent_groups": len(groups), "contradicts": list(c.metadata.get("contradicts", []))})
        rows.sort(key=lambda r: (-r["confidence"], r["claim"]))
        contested = [r for r in rows if r["contradicting_sources"] or r["status"] in ("doubted", "rejected")]
        return rows[:limit], contested[:limit]

    def _group(self, source_id: str | None) -> str:
        src = self.store.get_source(source_id) if source_id else None
        return (src.independence_group or src.id) if src else str(source_id)

    def _unverified(self, main: Conclusion, concl: list[Conclusion], subs: list[dict[str, str]],
                    sub_results: list[list[SearchResult]], beliefs: list[dict[str, Any]]) -> list[str]:
        out: list[str] = []
        for c in [main, *concl]:
            if c.insufficient:
                out.append(f"Not enough evidence to answer: \"{c.question}\" (confidence {c.confidence:.2f}).")
        for s, rs in zip(subs, sub_results):
            if not any(r.status == "ingested" and r.query == s["query"] for r in self.records):
                out.append(f"No source was selected specifically for sub-question \"{s['question']}\" (search: {s['query']!r}, {len(rs)} result(s)).")
        for b in beliefs:
            if b["independent_groups"] <= 1 and b["confidence"] < 0.8:
                out.append(f"Single-source claim (not independently corroborated): \"{_clip(b['claim'], 140)}\" [{', '.join(map(str, b['supporting_sources'] or b['contradicting_sources']))}].")
        reasons = Counter(r.reason.split(":")[0] for r in self.records if r.status == "failed")
        if reasons:
            out.append("Could not fetch " + ", ".join(f"{n} page(s) ({k})" for k, n in reasons.most_common()) + " -- their content was not used.")
        low = [r for r in self.records if r.status == "skipped" and r.reason.startswith("low reliability")]
        if low:
            out.append(f"Skipped {len(low)} low-reliability result(s) (e.g. {low[0].domain}); claims only found there are unverified.")
        inj = [r for r in self.records if r.injection_suspect and r.n]
        for r in inj:
            out.append(f"Source [{r.n}] ({r.domain}) contains instruction-like text aimed at AI systems; it was treated strictly as data and down-weighted.")
        return out[:25]

    # ------------------------------------------------------------------ brief
    def render_brief(self, res: ResearchResult) -> str:
        def cite(nums: list[int]) -> str:
            return "".join(f"[{n}]" for n in nums)

        L = [f"# Research brief: {_md(res.question)}", ""]
        L.append(f"_Generated {_now()} by the Noesis Research Agent · model `{res.model}` · search `{res.provider}` · "
                 f"{len(res.cited_sources)} source(s) used._")
        L += ["", "## Answer", ""]
        m = res.main
        L.append(f"{_md(m.answer)} {cite(m.citations)}".strip())
        L += ["", f"**Confidence: {m.confidence:.2f} ({_label(m.confidence)})**" + (" — insufficient evidence" if m.insufficient else ""), ""]
        for n, q in m.quotes[:4]:
            L.append(f"> [{n}] \"{_md(q)}\"")
        L += ["", "## Conclusions by sub-question", "", "| # | Sub-question | Conclusion | Confidence | Sources |", "|---|---|---|---|---|"]
        for i, c in enumerate(res.conclusions, 1):
            conf = f"{c.confidence:.2f} ({_label(c.confidence)})" + (" ⚠ insufficient" if c.insufficient else "")
            L.append(f"| {i} | {_md(c.question)} | {_md(_clip(c.answer, 400))} | {conf} | {cite(c.citations) or '—'} |")
        if res.beliefs:
            L += ["", "## Key findings (calibrated beliefs)", "", "| Claim | Confidence | Status | Supported by | Contradicted by |", "|---|---|---|---|---|"]
            for b in res.beliefs:
                L.append(f"| {_md(_clip(b['claim'], 200))} | {b['confidence']:.2f} | {b['status']} | {cite(b['supporting_sources']) or '—'} | {cite(b['contradicting_sources']) or '—'} |")
        if res.contested:
            L += ["", "## Contested or doubted", ""]
            for b in res.contested:
                L.append(f"- {_md(_clip(b['claim'], 200))} — confidence {b['confidence']:.2f}; for {cite(b['supporting_sources']) or '—'}, against {cite(b['contradicting_sources']) or '—'}")
        L += ["", "## What we could not verify", ""]
        L += [f"- {_md(u)}" for u in res.unverified] or ["- Nothing flagged; still check the sources before relying on this brief."]
        L += ["", "## Sources", "", "| # | Source | Domain | Reliability | Why | Found via | Fetched |", "|---|---|---|---|---|---|---|"]
        for s in res.cited_sources:
            via = f"{s.via}: {_md(_clip(s.query, 60))}" if s.query else s.via
            flag = " ⚠ injection text" if s.injection_suspect else ""
            L.append(f"| {s.n} | [{_md(_clip(s.title or s.final_url, 90))}]({s.final_url or s.url}){flag} | {s.domain} | {s.reliability:.2f} | {_md(s.reliability_reason)} | {via} | {s.fetched_at[:10]} |")
        skipped = [s for s in res.sources if s.status in ("skipped", "failed")]
        if skipped:
            L += ["", "<details><summary>Results not used (" + str(len(skipped)) + ")</summary>", ""]
            L += [f"- {s.url} — {s.status}: {_md(_clip(s.reason, 160))}" for s in skipped[:40]]
            L += ["", "</details>"]
        L += ["", "## Method", ""]
        L.append(f"Planned {len(res.sub_questions)} sub-question(s), searched the web, kept at most {self.budget.per_domain} source(s) per domain "
                 f"ranked by a domain reliability prior, fetched pages (robots.txt respected, no JavaScript), ingested them into Noesis, "
                 f"extracted claims and supporting/contradicting evidence, then ran the reasoning loop (run `{res.run.get('id')}`: "
                 f"{_md(res.run.get('summary', ''))}). Pages fetched by the reasoning loop itself are capped at reliability 0.55 until corroborated.")
        L += ["", "Confidence is computed from evidence (source reliability, independence, stance); the model's own judgment is a bounded minor input. "
              "Web pages are untrusted data. This brief is a starting point for verification, not a final authority."]
        return "\n".join(L) + "\n"

    # ------------------------------------------------------------------ watch
    def watch(self, topic: str, depth: int = 1) -> WatchResult:
        """Incremental refresh: re-run the saved searches, ingest only URLs not seen before,
        reason briefly and report which beliefs moved."""
        topic = " ".join(topic.split())
        key = f"agent:watch:{_slug(topic)}"
        state = self.store.get_meta(key) or {}
        queries = state.get("queries") or list(dict.fromkeys([topic] + [s["query"] for s in self.plan(topic, min(1 + depth, 6))]))
        before = {b.claim_id: b.confidence for b in self.store.list(Belief)}
        self.records = []
        per_query = self._search_all(queries)
        new = self._select_and_ingest(topic, per_query, limit=self.budget.max_sources, via="watch")
        self._event(f"watch: {len(new)} new source(s)")
        if new:
            understand(self.store, self.llm, source_ids=[r.source_id for r in new if r.source_id])
            self._mind().think(goal=topic, steps=max(1, depth))
            self._resolve_tool_sources()
        claims = {c.id: c for c in self.store.list(Claim)}
        changes = []
        for b in self.store.list(Belief):
            old = before.get(b.claim_id)
            if (old is None or abs(b.confidence - old) >= 0.05) and b.claim_id in claims:
                changes.append({"claim": claims[b.claim_id].text, "before": old, "after": round(b.confidence, 3), "status": b.status})
        changes.sort(key=lambda c: -abs(c["after"] - (c["before"] if c["before"] is not None else 0.5)))
        self.store.set_meta(key, {**state, "question": state.get("question", topic), "queries": queries, "last_run": _now()})
        L = [f"# Watch update: {_md(topic)}", "", f"_{_now()} · {len(new)} new source(s) · {len(changes)} belief change(s)._", ""]
        if new:
            L += ["## New sources", ""] + [f"- [{_md(r.title)}]({r.final_url}) — {r.domain}, reliability {r.reliability:.2f}" for r in new]
        else:
            L.append("No new sources since the last run.")
        if changes:
            L += ["", "## Belief changes", "", "| Claim | Before | After | Status |", "|---|---|---|---|"]
            for c in changes[:25]:
                old = "new" if c["before"] is None else f"{c['before']:.2f}"
                L.append(f"| {_md(_clip(c['claim'], 180))} | {old} | {c['after']:.2f} | {c['status']} |")
        return WatchResult(topic, new, changes, "\n".join(L) + "\n")
