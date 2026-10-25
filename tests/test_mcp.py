"""MCP server: tool functions directly (no SDK needed) + an in-process client session."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from noesis.mcp_server import NoesisService

CORPUS = Path(__file__).resolve().parent.parent / "examples" / "corpus"
TOOLS = {
    "ingest_source",
    "build_understanding",
    "ask",
    "think",
    "list_beliefs",
    "list_open_questions",
    "explain_belief",
    "add_evidence",
    "search_knowledge",
}


@pytest.fixture()
def svc(tmp_path):
    s = NoesisService(tmp_path / "mcp.db", "mock")
    s.store.set_meta("topic", "Is coffee good for health?")
    yield s
    s.close()


@pytest.fixture()
def built(svc):
    svc.ingest_source(str(CORPUS))
    svc.build_understanding()
    return svc


def test_service_pipeline(built):
    svc = built
    beliefs = svc.list_beliefs(limit=5)
    assert beliefs["total"] > 0 and len(beliefs["beliefs"]) <= 5
    confs = [b["confidence"] for b in beliefs["beliefs"]]
    assert confs == sorted(confs, reverse=True)
    assert all(b["confidence"] >= 0.7 for b in svc.list_beliefs(min_confidence=0.7)["beliefs"])

    qs = svc.list_open_questions(limit=3)
    assert len(qs["questions"]) <= 3 and qs["total"] >= len(qs["questions"])

    ans = svc.ask("Does coffee raise blood pressure?")
    assert {"answer", "confidence", "citations", "insufficient_evidence"} <= set(ans)
    assert "trace" not in ans

    run = svc.think(steps=2)
    assert run["status"] in ("completed", "stopped") and run["steps"] >= 1 and run["trace"]

    exp = svc.explain_belief(beliefs["beliefs"][0]["belief_id"])
    assert "Confidence" in exp["explanation"] and exp["data"]["claim_id"]

    hits = svc.search_knowledge("blood pressure", k=3)
    assert hits["passages"] and len(hits["passages"]) <= 3
    assert "blood pressure" in " ".join(p["text"].lower() for p in hits["passages"])
    assert hits["claims"]

    md = svc.summary_markdown()
    assert md.startswith("# Is coffee good for health?")
    assert "belief" in svc.report_markdown().lower()
    json.dumps([beliefs, qs, ans, run, exp, hits])  # all JSON-serializable


def test_add_evidence_assistant_mode(svc):
    """The calling assistant reasons; Noesis only stores cited evidence and calibrates."""
    claim = "Green tea contains caffeine."
    r1 = svc.add_evidence(
        claim, "Green tea naturally contains caffeine, about 30 mg per cup.", "supports", "Tea facts", 0.85
    )
    assert r1["claim_created"] and r1["confidence"] > 0.5
    r2 = svc.add_evidence(
        "green tea contains caffeine",
        "Lab analysis found 25-35 mg caffeine per cup of green tea.",
        "supports",
        "Lab report",
        0.9,
    )
    assert not r2["claim_created"] and r2["claim_id"] == r1["claim_id"]
    assert r2["confidence"] > r1["confidence"]
    r3 = svc.add_evidence(
        claim, "This green tea brand is completely caffeine free.", "contradicts", "Ad copy", 0.3
    )
    assert r3["claim_id"] == r1["claim_id"]
    exp = svc.explain_belief(r1["belief_id"])
    assert exp["data"]["for"] and exp["data"]["against"] is not None
    hits = svc.search_knowledge("green tea caffeine")
    assert any(c["claim_id"] == r1["claim_id"] for c in hits["claims"])


def test_service_errors(svc, tmp_path):
    with pytest.raises(ValueError):
        svc.ingest_source()
    with pytest.raises(FileNotFoundError):
        svc.ingest_source(str(tmp_path / "missing.md"))
    with pytest.raises(ValueError):
        svc.ingest_source(text="x", reliability=2.0)
    with pytest.raises(ValueError):
        svc.explain_belief("bel_nope")
    with pytest.raises(ValueError):
        svc.add_evidence("a claim", "a passage", "maybe")


def test_ingest_text(svc):
    r = svc.ingest_source(
        text="Decaf coffee contains a small amount of caffeine.", title="Decaf note", reliability=0.7
    )
    assert r["ingested"][0]["title"] == "Decaf note" and r["ingested"][0]["reliability"] == 0.7
    assert r["store"]["chunks"] >= 1


# ---------------------------------------------------------------- real MCP session (SDK)
def _attr(obj, snake, camel):
    return getattr(obj, snake, None) if hasattr(obj, snake) else getattr(obj, camel, None)


async def _session_roundtrip(server, corpus: str):
    """Works with mcp 2.x (Client(server)) and 1.x (memory transport)."""
    try:
        from mcp import Client  # mcp >= 2

        cm = Client(server)
    except ImportError:
        from mcp.shared.memory import create_connected_server_and_client_session

        cm = create_connected_server_and_client_session(server._mcp_server)
    async with cm as c:
        tools = {t.name for t in (await c.list_tools()).tools}
        r = await c.call_tool("ingest_source", {"path_or_url": corpus})
        assert not _attr(r, "is_error", "isError")
        await c.call_tool("build_understanding", {})
        lb = await c.call_tool("list_beliefs", {"limit": 3})
        payload = _attr(lb, "structured_content", "structuredContent") or json.loads(lb.content[0].text)
        bad = await c.call_tool("explain_belief", {"belief_id": "bel_nope"})
        resources = {str(x.uri) for x in (await c.list_resources()).resources}
        summary = (await c.read_resource("noesis://summary")).contents[0].text
        return tools, payload, bad, resources, summary


def test_mcp_session_in_process(tmp_path):
    pytest.importorskip("mcp")
    from noesis.mcp_server import create_server

    svc = NoesisService(tmp_path / "s.db", "mock")
    try:
        server = create_server(svc)
        tools, payload, bad, resources, summary = asyncio.run(_session_roundtrip(server, str(CORPUS)))
    finally:
        svc.close()
    assert TOOLS <= tools
    assert payload["total"] > 0 and len(payload["beliefs"]) == 3
    assert _attr(bad, "is_error", "isError")
    assert "no belief or claim" in bad.content[0].text
    assert {"noesis://summary", "noesis://report"} <= resources
    assert "Most confident beliefs" in summary


def test_tool_descriptions_present():
    pytest.importorskip("mcp")
    from noesis.mcp_server import create_server

    async def go(server):
        try:
            from mcp import Client

            cm = Client(server)
        except ImportError:
            from mcp.shared.memory import create_connected_server_and_client_session

            cm = create_connected_server_and_client_session(server._mcp_server)
        async with cm as c:
            return (await c.list_tools()).tools

    tools = asyncio.run(go(create_server(NoesisService(":memory:", "mock"))))
    for t in tools:
        assert t.description and len(t.description) > 60, t.name


def test_missing_sdk_message(monkeypatch):
    import builtins

    from noesis import mcp_server

    real_import = builtins.__import__

    def fake(name, *a, **k):
        if name == "mcp" or name.startswith("mcp."):
            raise ImportError("no mcp")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake)
    with pytest.raises(mcp_server.MCPUnavailable, match=r"noesis\[mcp\]"):
        mcp_server.create_server(NoesisService(":memory:", "mock"))


def test_stdio_subprocess(tmp_path):
    """`python -m noesis serve-mcp` speaks MCP over stdio."""
    pytest.importorskip("mcp")
    import os
    import sys

    from mcp import StdioServerParameters

    root = str(Path(__file__).resolve().parent.parent)
    env = {
        **os.environ,
        "PYTHONPATH": root + os.pathsep + os.environ.get("PYTHONPATH", ""),
        "NOESIS_MODEL": "mock",
    }
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "noesis", "--db", str(tmp_path / "io.db"), "serve-mcp"], env=env
    )

    async def go():
        try:
            from mcp import Client

            async with Client(params) as c:
                r = await c.call_tool("ingest_source", {"text": "Coffee contains caffeine.", "title": "t"})
                return {t.name for t in (await c.list_tools()).tools}, r
        except ImportError:
            from mcp import ClientSession
            from mcp.client.stdio import stdio_client

            async with stdio_client(params) as (read, write), ClientSession(read, write) as c:
                await c.initialize()
                r = await c.call_tool("ingest_source", {"text": "Coffee contains caffeine.", "title": "t"})
                return {t.name for t in (await c.list_tools()).tools}, r

    tools, r = asyncio.run(asyncio.wait_for(go(), 60))
    assert TOOLS <= tools and not _attr(r, "is_error", "isError")
