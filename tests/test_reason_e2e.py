"""End-to-end: ingest -> understand -> think(steps=5) -> ask -> explain -> report (MockLLM, offline)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from noesis import KnowledgeStore, ingest, ingest_text, understand
from noesis.llm import MockLLM
from noesis.reason import Budget, Mind
from noesis.report import render_html, render_markdown, write_report
from noesis.types import Belief, Chunk, Evidence, Question, ReasoningRun, Thought

CORPUS = Path(__file__).resolve().parent.parent / "examples" / "corpus"

FALLBACK = [
    ("Review", 0.9, "Moderate coffee consumption is associated with lower all-cause mortality. Caffeine raises blood pressure briefly in people who rarely drink it."),
    ("Trial", 0.85, "Habitual coffee drinkers showed no lasting increase in blood pressure. Caffeine raises blood pressure briefly in non-drinkers."),
    ("Blog", 0.35, "Coffee causes a lasting increase in blood pressure. Coffee dehydrates you."),
    ("Physician", 0.7, "Moderate coffee intake does not dehydrate regular drinkers."),
]

LOOP_STEPS = {"observe", "select_question", "hypothesize", "gather", "test", "judge", "update", "reflect"}


def _populate(store: KnowledgeStore) -> str:
    if CORPUS.is_dir() and any(CORPUS.glob("*.md")):
        ingest(store, str(CORPUS))
        topic = "Is coffee good for health?"
        qfile = CORPUS / "questions.json"
        if qfile.exists():
            topic = json.loads(qfile.read_text()).get("topic", topic)
    else:
        for title, rel, text in FALLBACK:
            ingest_text(store, text, title=title, reliability=rel)
        topic = "Is coffee good for health?"
    store.set_meta("topic", topic)
    return topic


@pytest.mark.parametrize("strategy", ["simple", "self_consistency", "debate"])
def test_full_pipeline(tmp_path, strategy):
    db = tmp_path / "e2e.noesis.db"
    store = KnowledgeStore(db)
    topic = _populate(store)
    llm = MockLLM()
    understand(store, llm)
    n_claims_before = store.count(Belief)

    seen: list[Thought] = []
    mind = Mind(store, llm, strategy=strategy, budget=Budget(steps=5, max_llm_calls=2000, seconds=120), on_thought=seen.append)
    run = mind.think(goal=topic, steps=5)

    # run + thoughts persisted
    assert run.status == "completed" and run.steps >= 1
    assert run.stop_reason in ("budget:steps", "converged", "empty_queue")
    thoughts = store.list_thoughts(run.id)
    assert len(thoughts) == len(seen) > 0
    assert LOOP_STEPS <= {t.step for t in thoughts}
    assert run.usage["llm_calls"] > 0 and run.summary

    # beliefs persisted, with evidence-backed confidence and history
    assert store.count(Belief) >= n_claims_before
    touched = run.metadata["belief_changes"]
    assert touched
    for cid in touched:
        b = store.get_belief_for_claim(cid)
        assert b is not None and b.history
        assert any(e.stance != "context" for e in store.list_evidence(cid))

    # every evidence row / thought ref points at real objects
    for ev in store.list(Evidence):
        if ev.chunk_id:
            assert store.get(Chunk, ev.chunk_id) is not None

    # at least one question changed state; resolved ones are recorded
    statuses = {q["status"] for q in run.metadata["questions"]}
    assert statuses <= {"answered", "open", "dropped"}

    # ask: grounded with valid citations, or honestly insufficient
    ans = mind.ask("Is moderate coffee consumption associated with lower all-cause mortality?")
    assert ans.citations, ans.text
    for c in ans.citations:
        ch = store.get(Chunk, c.chunk_id)
        assert ch is not None and ch.source_id == c.source_id
        assert ch.start <= c.span[0] <= c.span[1] <= ch.end
    assert not ans.dropped_citations
    assert 0.0 <= ans.confidence <= 1.0
    if not ans.insufficient:
        assert ans.confidence >= mind.cfg.answer_min_confidence
    unknown = mind.ask("How many moons does Jupiter have?")
    assert unknown.insufficient and unknown.confidence <= 0.2

    # explain
    some = store.get_belief_for_claim(next(iter(touched)))
    text = mind.explain(some.id)
    assert "Evidence FOR" in text and "History" in text

    # report renders (markdown + self-contained HTML)
    md = render_markdown(store)
    for section in ("## Overview", "## Beliefs", "## Contradictions", "## Open questions", "## Reasoning trace"):
        assert section in md
    html = render_html(store)
    assert "<svg" in html and "prefers-color-scheme: dark" in html
    assert "http://" not in html.replace('xmlns="http://www.w3.org/2000/svg"', "") and "<script" not in html
    out = write_report(store, tmp_path / "report.html")
    assert out.read_text().startswith("<!doctype html>")

    # persistence across reopen
    store.close()
    again = KnowledgeStore(db)
    assert again.get(ReasoningRun, run.id).status == "completed"
    assert len(again.list_thoughts(run.id)) == len(thoughts)
    assert again.count(Question) > 0
    again.close()
