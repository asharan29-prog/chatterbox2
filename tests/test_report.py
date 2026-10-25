"""Report rendering tests (offline)."""

from __future__ import annotations

from noesis import KnowledgeStore, ingest_text, understand
from noesis.llm import MockLLM
from noesis.reason import Mind
from noesis.report import build_report_data, concept_graph_svg, render_html, render_markdown, write_report


def _store() -> KnowledgeStore:
    s = KnowledgeStore(":memory:")
    s.set_meta("topic", "Green tea <script>alert(1)</script>")
    ingest_text(s, "Green tea lowers blood pressure in adults. Green tea contains caffeine.", title="Review", reliability=0.9)
    ingest_text(s, "Green tea does not lower blood pressure in adults.", title="Blog <b>x</b>", reliability=0.4)
    ingest_text(s, "A trial found green tea lowers blood pressure in adults.", title="Trial", reliability=0.85)
    understand(s, MockLLM())
    return s


def test_empty_store_renders():
    s = KnowledgeStore(":memory:")
    md = render_markdown(s)
    html = render_html(s)
    assert "# Noesis report" in md and "_No reasoning runs yet._" in md
    assert "No concepts yet" in html and "</html>" in html


def test_report_sections_and_escaping(tmp_path):
    s = _store()
    Mind(s, MockLLM()).think(steps=2)
    d = build_report_data(s)
    assert d["beliefs"] and d["trace"] and d["runs"]
    assert all(0 <= b["confidence"] <= 1 for b in d["beliefs"])
    html = render_html(s)
    assert "<script>" not in html and "&lt;script&gt;" in html  # untrusted text is escaped
    assert "Blog &lt;b&gt;x&lt;/b&gt;" in html
    assert 'class="bar"' in html and "<svg" in html and "timeline" in html
    assert "@media (prefers-color-scheme: dark)" in html
    md = render_markdown(s)
    assert "| confidence | status | claim |" in md and "## Reasoning trace (run `" in md
    p = write_report(s, tmp_path / "r.md")
    assert p.read_text().startswith("# ")
    p2 = write_report(s, tmp_path / "r.txt", fmt="html")
    assert "<!doctype html>" in p2.read_text()


def test_concept_graph_svg():
    svg = concept_graph_svg(
        {
            "nodes": [
                {"id": "a", "name": "tea & <health>", "claims": 3, "confidence": 0.9},
                {"id": "b", "name": "caffeine", "claims": 1, "confidence": 0.2},
                {"id": "c", "name": "sleep", "claims": 0, "confidence": None},
            ],
            "edges": [{"a": "a", "b": "b", "w": 2}],
        }
    )
    assert svg.startswith("<svg") and svg.count("<circle") == 3 and svg.count("<line") == 1
    assert "tea &amp; &lt;health&gt;" in svg
