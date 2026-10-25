"""ResearchAgent end to end against the offline mock web, plus CLI and MCP smoke tests."""

from __future__ import annotations

import asyncio
import re

import pytest

from noesis import KnowledgeStore
from noesis.agent import DomainPolicy, ResearchAgent, ResearchBudget, plan_heuristic
from noesis.agent.mockweb import mock_toolkit
from noesis.llm import MockLLM
from noesis.types import Source

PRIVATE_OR_BLOCKED = ("127.0.0.1", "169.254.169.254", "intranet.corp-internal.com", "blocked.example.org/coffee-report",
                      "setup.exe")


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for var in ("NOESIS_DB", "NOESIS_MODEL", "NOESIS_SEARCH", "NOESIS_WEBCACHE"):
        monkeypatch.delenv(var, raising=False)


def _agent(store=None, **kw):
    tk = kw.pop("toolkit", None) or mock_toolkit(policy=kw.pop("policy", None))
    agent = ResearchAgent(store or KnowledgeStore(":memory:"), MockLLM(), search=tk, **kw)
    return agent, tk.mock_web


def test_research_end_to_end_brief_has_valid_citations(tmp_path):
    agent, web = _agent(budget=ResearchBudget(max_sources=6))
    res = agent.research("Is coffee good for health?", depth=2)
    fetched = {u for m, u in web.requests if m == "GET"}

    # sources: diverse domains, all actually fetched, priors applied
    used = res.cited_sources
    assert len(used) >= 4
    plan_used = [s for s in used if s.via == "plan"]
    assert len({s.domain for s in plan_used}) == len(plan_used)  # one per domain
    for s in used:
        assert s.final_url in fetched, s.final_url
        assert agent.store.get_source(s.source_id).uri == s.final_url
    nih = next(s for s in used if s.domain == "nih.gov")
    assert nih.reliability >= 0.8 and "public-health" in nih.reliability_reason
    plos = next(s for s in used if s.domain == "plos.org")
    assert "scholarly" in plos.reliability_reason or "journal" in plos.reliability_reason

    # hostile results were never requested; the content farm was skipped by its prior
    assert not any(any(b in u for b in PRIVATE_OR_BLOCKED) for u in fetched)
    farm = [s for s in res.sources if s.domain == "ehow.com"]
    assert farm and all(s.status == "skipped" and "low reliability" in s.reason for s in farm)
    assert not any("ehow.com" in u for u in fetched)

    # every [n] in the brief refers to a row of the sources table
    md = res.markdown
    nums = {int(n) for n in re.findall(r"\[(\d+)\]", md)}
    table = {s.n: s for s in used}
    assert nums and nums <= set(table)
    for n in nums:
        assert f"| {n} | [" in md and table[n].final_url in md
    assert res.main.citations and not res.main.insufficient
    assert all(0.0 <= c.confidence <= 1.0 for c in [res.main, *res.conclusions])
    for section in ("## Answer", "## Conclusions by sub-question", "## What we could not verify", "## Sources", "Confidence"):
        assert section in md
    assert res.unverified

    # reasoning ran with the web tool wired in
    assert res.run["steps"] >= 1
    tool = agent.web_search_tool("caffeine sleep half-life")
    assert all(d["url"].startswith("https://") and d["text"] for d in tool)

    # outputs
    paths = res.write(tmp_path / "out" / "brief.md")
    assert paths[0].read_text().startswith("# Research brief") and paths[1].name == "brief.report.html"
    assert "<html" in paths[1].read_text().lower()
    d = res.to_dict()
    assert d["sources"] and d["could_not_verify"] and d["answer"]["confidence"] == res.main.confidence


def test_reasoning_loop_goes_back_to_the_web_when_evidence_is_thin():
    # one up-front source only -> the loop must use the web_search tool
    agent, web = _agent(budget=ResearchBudget(max_sources=1, think_steps=3))
    res = agent.research("Does caffeine disrupt sleep?", depth=1)
    via_tool = [s for s in res.cited_sources if s.via == "reasoning"]
    assert via_tool, [vars(s) for s in res.sources]
    for s in via_tool:
        src = agent.store.get_source(s.source_id)
        assert src.reliability <= 0.55 and "reasoning loop" in src.reliability_reason
    assert len({s.source_id for s in res.cited_sources}) == len(res.cited_sources)


def test_injection_page_is_down_weighted_and_flagged():
    agent, _ = _agent(budget=ResearchBudget(max_sources=10, min_reliability=0.0))
    res = agent.research("Does coffee raise blood pressure and cause dehydration?", depth=1)
    blog = [s for s in res.cited_sources if s.domain == "medium.com"]
    assert blog and blog[0].injection_suspect
    assert blog[0].reliability < 0.4 and "instruction-like" in blog[0].reliability_reason
    assert "instruction-like text" in res.markdown
    assert "confidence 1.0" not in res.main.answer.lower()


def test_allow_and_deny_lists():
    agent, web = _agent(policy=DomainPolicy(allow=["nih.gov", "wikipedia.org"]), budget=ResearchBudget(per_domain=2))
    res = agent.research("coffee blood pressure heart", depth=1)
    assert res.cited_sources and {s.domain for s in res.cited_sources} <= {"nih.gov", "wikipedia.org"}
    agent2, web2 = _agent(policy=DomainPolicy(deny=["nih.gov"]))
    res2 = agent2.research("coffee blood pressure heart", depth=1)
    assert all(s.domain != "nih.gov" for s in res2.cited_sources)
    assert not any("nih.gov" in u for _, u in web2.requests)


def test_watch_ingests_only_new_urls(tmp_path):
    store = KnowledgeStore(tmp_path / "w.db")
    from noesis.agent.mockweb import MockPage

    agent, web = _agent(store, budget=ResearchBudget(max_sources=2))
    agent.research("Is coffee good for health?", depth=1)
    before = {s.uri for s in store.list(Source)}
    new_url = "https://www.nih.gov/news-events/new-coffee-trial-2026"
    web.pages[new_url] = MockPage(new_url, "New randomized trial: coffee and health", [
        "A new randomized trial found that moderate coffee drinking did not raise long-term blood pressure.",
        "Coffee drinking was associated with a lower risk of type 2 diabetes in the trial population.",
    ], "New randomized trial on coffee good for health")
    n_before = len(web.requests)
    w = agent.watch("Is coffee good for health?")
    assert new_url in [r.final_url for r in w.new_sources]
    assert all(r.final_url not in before for r in w.new_sources)
    assert not any(u in before for _, u in web.requests[n_before:])  # old pages not refetched
    assert w.changes
    assert "Watch update" in w.markdown
    uris = [s.uri for s in store.list(Source)]
    assert len(uris) == len(set(uris))
    w2 = agent.watch("Is coffee good for health?")
    assert all(r.final_url not in before for r in w2.new_sources)
    store.close()


def test_planner_uses_model_and_falls_back():
    llm = MockLLM({"plan_research": {"sub_questions": [
        {"question": "What do cohort studies show about coffee and mortality?", "query": "coffee mortality cohort"},
        {"question": "x"},  # too short -> dropped
    ]}})
    agent = ResearchAgent(KnowledgeStore(":memory:"), llm, search=mock_toolkit())
    subs = agent.plan("Is coffee good for health?", 3)
    assert subs == [{"question": "What do cohort studies show about coffee and mortality?", "query": "coffee mortality cohort", "why": ""}]
    bad = ResearchAgent(KnowledgeStore(":memory:"), MockLLM({"plan_research": "not json"}, fallback="error"), search=mock_toolkit())
    assert len(bad.plan("Is coffee good for health?", 2)) == 2
    assert plan_heuristic("Does coffee raise blood pressure?", 2)[0]["query"].startswith("coffee raise blood pressure")


def test_search_failure_is_reported_not_fatal():
    from noesis.agent.web import SearchError

    tk = mock_toolkit()

    def broken(query, max_results=8):
        raise SearchError("provider down")

    tk.search_provider.search = broken
    agent = ResearchAgent(KnowledgeStore(":memory:"), MockLLM(), search=tk)
    res = agent.research("Is coffee good for health?", depth=1)
    assert not res.cited_sources and res.main.insufficient
    assert any("Not enough evidence" in u for u in res.unverified)


# ------------------------------------------------------------------ CLI
def test_cli_research_mock_end_to_end(tmp_path, capsys):
    from noesis.cli import main

    code = main(["--model", "mock", "research", "Is coffee good for health?", "--depth", "1", "-o", "brief.md",
                 "--max-sources", "4", "--trust", "wikipedia.org=0.75"])
    out, err = capsys.readouterr()
    assert code == 0, err
    assert "OFFLINE mock web" in err
    brief = (tmp_path / "brief.md").read_text()
    assert brief.startswith("# Research brief") and "## Sources" in brief
    assert (tmp_path / "brief.report.html").exists()
    assert (tmp_path / ".noesis" / "research-is-coffee-good-for-health.db").exists()
    assert "0.75 | user trust override" in brief

    code = main(["--model", "mock", "research", "Is coffee good for health?", "--watch", "--quiet"])
    out, err = capsys.readouterr()
    assert code == 0 and "# Watch update" in out and err == ""


def test_noesis_agent_script_json_and_errors(tmp_path, capsys):
    import json

    from noesis.agent.cli import main

    assert main(["--model", "mock", "--db", str(tmp_path / "a.db"), "Is coffee good for health?", "--depth", "1", "--json", "--quiet"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["sources"] and data["answer"]["answer"] and "brief_markdown" in data
    assert main(["--model", "mock", "q?", "--depth", "9"]) == 2
    assert main(["--model", "mock", "q?", "--trust", "nonsense"]) == 2
    capsys.readouterr()


def test_cli_brave_without_key_is_usage_error(monkeypatch, capsys):
    from noesis.agent.cli import main

    monkeypatch.delenv("BRAVE_API_KEY", raising=False)
    assert main(["--model", "mock", "q?", "--search", "brave", "--quiet"]) == 2
    assert "BRAVE_API_KEY" in capsys.readouterr().err


# ------------------------------------------------------------------ MCP
def test_mcp_service_web_tools(tmp_path, monkeypatch):
    from noesis.mcp_server import NoesisService

    monkeypatch.setenv("NOESIS_SEARCH", "mock")
    svc = NoesisService(tmp_path / "m.db", "mock")
    try:
        s = svc.web_search("coffee mortality meta-analysis", 3)
        assert s["provider"] == "mock" and len(s["results"]) == 3
        assert {"url", "title", "snippet", "reliability_prior"} <= set(s["results"][0])
        with pytest.raises(ValueError):
            svc.web_search("  ")
        r = svc.web_research("Is coffee good for health?", depth=1, max_sources=3)
        assert r["sources"] and r["answer"] and 0 <= r["confidence"] <= 1
        assert {src["n"] for src in r["sources"]} >= set(r["answer_sources"])
        assert svc.list_beliefs(limit=3)["total"] > 0  # research landed in the knowledge base
    finally:
        svc.close()


def test_mcp_lists_web_tools(tmp_path, monkeypatch):
    pytest.importorskip("mcp")
    from noesis.mcp_server import NoesisService, create_server

    async def go(server):
        try:
            from mcp import Client

            cm = Client(server)
        except ImportError:
            from mcp.shared.memory import create_connected_server_and_client_session

            cm = create_connected_server_and_client_session(server._mcp_server)
        async with cm as c:
            return {t.name: t.description for t in (await c.list_tools()).tools}

    svc = NoesisService(tmp_path / "t.db", "mock")
    try:
        tools = asyncio.run(go(create_server(svc)))
    finally:
        svc.close()
    assert {"web_search", "web_research"} <= set(tools)
    assert "untrusted" in tools["web_search"]
