"""Reports: Markdown and a self-contained HTML page (no external assets, dark-mode aware).

>>> from noesis.report import render_markdown, render_html, write_report
>>> write_report(store, "coffee_report.html")     # format from the suffix (.html / .md)

Sections: overview (stats, topic summary, runs), beliefs (confidence bars + citations),
contradictions, open questions (and answered ones), reasoning-trace timeline, and the concept
graph as inline SVG. All store text is HTML-escaped (ingested text is untrusted data).
"""

from __future__ import annotations

import html
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from .store import KnowledgeStore
from .types import Belief, Claim, Concept, Evidence, Question, ReasoningRun, Source, Thought

__all__ = ["build_report_data", "render_markdown", "render_html", "concept_graph_svg", "write_report"]


# ============================================================================ data
def _clip(s: str, n: int) -> str:
    s = " ".join(str(s or "").split())
    return s if len(s) <= n else s[: n - 1] + "\u2026"


def build_report_data(
    store: KnowledgeStore,
    *,
    title: str | None = None,
    max_beliefs: int = 40,
    max_thoughts: int = 80,
    max_citations: int = 3,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Collect everything a report needs into plain dicts (also handy for JSON/UIs)."""
    claims = {c.id: c for c in store.list(Claim)}
    beliefs = {b.claim_id: b for b in store.list(Belief)}
    sources = {s.id: s for s in store.list(Source)}
    ev_by_claim: dict[str, list[Evidence]] = defaultdict(list)
    for e in store.list(Evidence):
        ev_by_claim[e.claim_id].append(e)

    def cites(cid: str, stance: str = "supports") -> list[dict[str, Any]]:
        evs = [e for e in ev_by_claim.get(cid, []) if e.stance == stance and e.source_id]
        evs.sort(key=lambda e: (-(sources[e.source_id].reliability if e.source_id in sources else 0) * e.strength, e.id))
        out, seen = [], set()
        for e in evs:
            if e.source_id in seen:
                continue
            seen.add(e.source_id)
            src = sources.get(e.source_id)
            out.append(
                {
                    "source": (src.title or src.uri) if src else e.source_id,
                    "source_id": e.source_id,
                    "chunk_id": e.chunk_id,
                    "span": list(e.span) if e.span else None,
                    "quote": _clip(e.quote, 200),
                    "reliability": src.reliability if src else None,
                }
            )
            if len(out) >= max_citations:
                break
        return out

    brows = []
    for b in sorted(beliefs.values(), key=lambda b: (-b.confidence, claims[b.claim_id].text if b.claim_id in claims else "")):
        c = claims.get(b.claim_id)
        if c is None:
            continue
        evs = ev_by_claim.get(c.id, [])
        brows.append(
            {
                "belief_id": b.id,
                "claim_id": c.id,
                "claim": c.text,
                "confidence": b.confidence,
                "status": b.status,
                "origin": c.origin,
                "n_for": sum(1 for e in evs if e.stance == "supports"),
                "n_against": sum(1 for e in evs if e.stance == "contradicts"),
                "citations": cites(c.id),
                "changes": len(b.history),
            }
        )
    status_counts = Counter(r["status"] for r in brows)
    # contradictions
    pairs = set()
    for c in claims.values():
        for o in c.metadata.get("contradicts", []) or []:
            if o in claims:
                pairs.add(tuple(sorted((c.id, o))))
    contra = []
    for a, b in sorted(pairs):
        ba, bb = beliefs.get(a), beliefs.get(b)
        contra.append(
            {
                "a": {"claim_id": a, "text": claims[a].text, "confidence": ba.confidence if ba else None, "citations": cites(a, "supports")[:1]},
                "b": {"claim_id": b, "text": claims[b].text, "confidence": bb.confidence if bb else None, "citations": cites(b, "supports")[:1]},
                "resolved": bool(ba and bb and abs(ba.confidence - bb.confidence) >= 0.4),
            }
        )
    contra.sort(key=lambda x: (x["resolved"], x["a"]["text"]))
    questions = store.list(Question)
    open_q = sorted((q for q in questions if q.status in ("open", "in_progress")), key=lambda q: (-q.priority, q.text))
    answered = [q for q in questions if q.status == "answered"]
    # runs + trace
    runs = store.list(ReasoningRun)
    think_runs = [r for r in runs if r.strategy != "ask"]
    focus = None
    if run_id:
        focus = store.get_run(run_id)
    elif think_runs:
        focus = think_runs[-1]
    thoughts = store.list_thoughts(focus.id) if focus else store.list(Thought)
    thoughts = thoughts[:max_thoughts]
    summary = store.topic_summary()
    stats = store.stats()
    return {
        "title": title or store.get_meta("topic") or "Noesis report",
        "generated_at": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z"),
        "stats": stats,
        "summary": summary.text if summary else "",
        "status_counts": dict(status_counts),
        "beliefs": brows[:max_beliefs],
        "beliefs_total": len(brows),
        "contradictions": contra,
        "open_questions": [
            {"id": q.id, "text": q.text, "origin": q.origin, "priority": q.priority, "attempts": int(q.metadata.get("attempts", 0) or 0)}
            for q in open_q
        ],
        "answered_questions": [
            {"id": q.id, "text": q.text, "answer": q.answer or "", "confidence": q.answer_confidence, "origin": q.origin} for q in answered
        ],
        "runs": [
            {
                "id": r.id, "goal": r.goal, "strategy": r.strategy, "model": r.model, "status": r.status,
                "stop_reason": r.stop_reason, "steps": r.steps, "summary": r.summary,
                "started_at": r.started_at.strftime("%Y-%m-%d %H:%M:%S"), "usage": r.usage,
            }
            for r in runs
        ],
        "trace_run": focus.id if focus else None,
        "trace": [
            {"id": t.id, "step": t.step, "content": t.content, "refs": t.refs, "at": t.timestamp.strftime("%H:%M:%S")} for t in thoughts
        ],
        "graph": _graph_data(store, claims, beliefs),
        "sources": [
            {"id": s.id, "title": s.title or s.uri, "uri": s.uri, "reliability": s.reliability, "reason": s.reliability_reason}
            for s in sorted(sources.values(), key=lambda s: (-s.reliability, s.title))
        ],
    }


def _graph_data(store: KnowledgeStore, claims: dict[str, Claim], beliefs: dict[str, Belief], max_nodes: int = 16) -> dict[str, Any]:
    concepts = {c.id: c for c in store.list(Concept)}
    by_concept: dict[str, list[str]] = defaultdict(list)
    for c in claims.values():
        for cid in c.concept_ids:
            if cid in concepts:
                by_concept[cid].append(c.id)
    ranked = sorted(concepts.values(), key=lambda c: (-len(by_concept.get(c.id, [])), -len(c.chunk_ids), c.name))[:max_nodes]
    keep = {c.id for c in ranked}
    nodes = []
    for c in ranked:
        cl = by_concept.get(c.id, [])
        confs = [beliefs[x].confidence for x in cl if x in beliefs]
        nodes.append({"id": c.id, "name": c.name, "claims": len(cl), "confidence": (sum(confs) / len(confs)) if confs else None})
    edges: Counter[tuple[str, str]] = Counter()
    for c in claims.values():
        ids = sorted({x for x in c.concept_ids if x in keep})
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                edges[(ids[i], ids[j])] += 1
    for c in ranked:
        if c.parent_id and c.parent_id in keep:
            edges[tuple(sorted((c.id, c.parent_id)))] += 1  # type: ignore[index]
    return {"nodes": nodes, "edges": [{"a": a, "b": b, "w": w} for (a, b), w in sorted(edges.items())]}


# ============================================================================ markdown
def _bar(p: float, width: int = 10) -> str:
    n = int(round(max(0.0, min(1.0, p)) * width))
    return "\u2588" * n + "\u2591" * (width - n)


def _md(s: str) -> str:
    return str(s or "").replace("|", "\\|").replace("\n", " ")


def _cite_md(c: dict[str, Any]) -> str:
    span = f" @{c['span'][0]}-{c['span'][1]}" if c.get("span") else ""
    return f"{_md(c['source'])}{span}"


def render_markdown(store: KnowledgeStore, **kwargs: Any) -> str:
    d = build_report_data(store, **kwargs)
    st = d["stats"]
    out = [f"# {_md(d['title'])}", "", f"_Generated {d['generated_at']} by Noesis._", "", "## Overview", ""]
    out.append(
        f"{st.get('sources', 0)} sources, {st.get('chunks', 0)} chunks, {st.get('claims', 0)} claims, "
        f"{st.get('evidence', 0)} evidence links, {st.get('questions', 0)} questions, {st.get('runs', 0)} reasoning runs."
    )
    if d["status_counts"]:
        out.append("Beliefs by status: " + ", ".join(f"{k} {v}" for k, v in sorted(d["status_counts"].items())) + ".")
    if d["summary"]:
        out += ["", "> " + _md(d["summary"])]
    if d["runs"]:
        out += ["", "| run | strategy | status | steps | summary |", "|---|---|---|---|---|"]
        for r in d["runs"][-8:]:
            out.append(f"| `{r['id']}` | {r['strategy']} | {r['status']} ({_md(r['stop_reason'])}) | {r['steps']} | {_md(_clip(r['summary'], 160))} |")
    out += ["", f"## Beliefs (top {len(d['beliefs'])} of {d['beliefs_total']})", "", "| confidence | status | claim | evidence | citations |", "|---|---|---|---|---|"]
    for b in d["beliefs"]:
        cits = "; ".join(_cite_md(c) for c in b["citations"]) or "_none_"
        out.append(
            f"| `{_bar(b['confidence'])}` {b['confidence']:.2f} | {b['status']} | {_md(b['claim'])}"
            + (" _(hypothesis)_" if b["origin"] == "hypothesis" else "")
            + f" | +{b['n_for']} / -{b['n_against']} | {cits} |"
        )
    out += ["", "## Contradictions", ""]
    if not d["contradictions"]:
        out.append("_None detected._")
    for c in d["contradictions"]:
        fa = f"{c['a']['confidence']:.2f}" if c["a"]["confidence"] is not None else "?"
        fb = f"{c['b']['confidence']:.2f}" if c["b"]["confidence"] is not None else "?"
        out.append(f"- {'(resolved) ' if c['resolved'] else ''}**{_md(c['a']['text'])}** ({fa}) vs **{_md(c['b']['text'])}** ({fb})")
    out += ["", "## Open questions", ""]
    if not d["open_questions"]:
        out.append("_None._")
    for q in d["open_questions"]:
        out.append(f"- [{q['origin']}, priority {q['priority']:.2f}] {_md(q['text'])}")
    if d["answered_questions"]:
        out += ["", "## Answered questions", ""]
        for q in d["answered_questions"]:
            conf = f" ({q['confidence']:.2f})" if q["confidence"] is not None else ""
            out.append(f"- **{_md(q['text'])}** {_md(_clip(q['answer'], 240))}{conf}")
    out += ["", "## Reasoning trace" + (f" (run `{d['trace_run']}`)" if d["trace_run"] else ""), ""]
    if not d["trace"]:
        out.append("_No reasoning runs yet._")
    for i, t in enumerate(d["trace"], 1):
        out.append(f"{i}. `{t['at']}` **{t['step']}** {_md(_clip(t['content'], 400))}")
    out += ["", "## Sources", "", "| reliability | source | reason |", "|---|---|---|"]
    for s in d["sources"]:
        out.append(f"| {s['reliability']:.2f} | {_md(s['title'])} | {_md(s['reason'])} |")
    out.append("")
    return "\n".join(out)


# ============================================================================ SVG graph
def _conf_color(p: float | None) -> str:
    """Diverging scale (red-ish = doubted, grey = uncertain, teal = believed)."""
    if p is None:
        return "var(--muted)"
    p = max(0.0, min(1.0, p))
    if p >= 0.5:
        t = (p - 0.5) * 2
        h, s, light = 172, 20 + 50 * t, 55 - 12 * t
    else:
        t = (0.5 - p) * 2
        h, s, light = 8, 20 + 55 * t, 58 - 8 * t
    return f"hsl({h:.0f} {s:.0f}% {light:.0f}%)"


def concept_graph_svg(graph: dict[str, Any], width: int = 720, height: int = 440) -> str:
    nodes = graph.get("nodes", [])
    if not nodes:
        return '<p class="muted">No concepts yet.</p>'
    cx, cy = width / 2, height / 2
    rx, ry = cx - 190, cy - 50  # leave room for side labels (<= 24 chars)
    pos = {}
    n = len(nodes)
    for i, nd in enumerate(nodes):
        ang = -math.pi / 2 + 2 * math.pi * i / n
        pos[nd["id"]] = (cx + rx * math.cos(ang), cy + ry * math.sin(ang))
    maxc = max((nd["claims"] for nd in nodes), default=1) or 1
    maxw = max((e["w"] for e in graph.get("edges", [])), default=1) or 1
    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Concept graph" class="graph" xmlns="http://www.w3.org/2000/svg">'
    ]
    for e in graph.get("edges", []):
        if e["a"] in pos and e["b"] in pos:
            (x1, y1), (x2, y2) = pos[e["a"]], pos[e["b"]]
            sw = 0.8 + 3.2 * e["w"] / maxw
            parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="var(--edge)" stroke-width="{sw:.1f}" stroke-linecap="round"/>')
    for nd in nodes:
        x, y = pos[nd["id"]]
        rad = 7 + 15 * math.sqrt(nd["claims"] / maxc)
        conf = f"{nd['confidence']:.2f}" if nd["confidence"] is not None else "n/a"
        name = html.escape(_clip(nd["name"], 24))
        anchor = "middle" if abs(x - cx) < 20 else ("start" if x > cx else "end")
        lx = x + (0 if anchor == "middle" else (rad + 5) * (1 if x > cx else -1))
        ly = y + (rad + 14 if anchor == "middle" and y > cy else -rad - 6 if anchor == "middle" else 4)
        parts.append(
            f'<g><title>{html.escape(nd["name"])}: {nd["claims"]} claim(s), mean confidence {conf}</title>'
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{rad:.1f}" fill="{_conf_color(nd["confidence"])}" stroke="var(--bg)" stroke-width="2"/>'
            f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="{anchor}" class="glabel">{name}</text></g>'
        )
    parts.append("</svg>")
    return "".join(parts)


# ============================================================================ HTML
_CSS = """
:root{--bg:#fbfaf7;--fg:#1d1f21;--muted:#6b6f76;--card:#ffffff;--line:#e4e1da;--accent:#0f766e;
--bar-bg:#ece9e2;--good:#0f766e;--mid:#8a8f98;--bad:#b4442e;--edge:rgba(60,64,72,.22);--chip:#f1efe9;}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#141518;--fg:#e7e5e0;--muted:#9aa0a8;
--card:#1c1e22;--line:#2c2f35;--accent:#5eead4;--bar-bg:#2a2d33;--good:#2dd4bf;--mid:#8a8f98;--bad:#f0866e;
--edge:rgba(220,220,230,.18);--chip:#25282d;}}
:root[data-theme="dark"]{--bg:#141518;--fg:#e7e5e0;--muted:#9aa0a8;--card:#1c1e22;--line:#2c2f35;--accent:#5eead4;
--bar-bg:#2a2d33;--good:#2dd4bf;--mid:#8a8f98;--bad:#f0866e;--edge:rgba(220,220,230,.18);--chip:#25282d;}
*{box-sizing:border-box}html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:1080px;margin:0 auto;padding:28px 16px 64px}
h1{font-size:1.75rem;line-height:1.2;margin:0 0 4px;letter-spacing:-.01em}
h2{font-size:1.1rem;margin:36px 0 12px;padding-bottom:6px;border-bottom:1px solid var(--line)}
.muted{color:var(--muted)}.small{font-size:.85rem}
nav{display:flex;flex-wrap:wrap;gap:6px;margin:14px 0 4px}nav a{color:var(--accent);text-decoration:none;font-size:.85rem;
padding:3px 10px;border:1px solid var(--line);border-radius:999px;background:var(--card)}
.stats{display:grid;grid-template-columns:repeat(auto-fill,minmax(130px,1fr));gap:10px;margin:16px 0}
.stat{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 12px}
.stat b{display:block;font-size:1.35rem;font-variant-numeric:tabular-nums}
blockquote{margin:12px 0;padding:10px 14px;border-left:3px solid var(--accent);background:var(--card);border-radius:0 8px 8px 0}
.tablewrap{overflow-x:auto;border:1px solid var(--line);border-radius:10px;background:var(--card)}
table{border-collapse:collapse;width:100%;font-size:.9rem}th,td{text-align:left;vertical-align:top;padding:8px 10px;border-bottom:1px solid var(--line)}
th{font-weight:600;color:var(--muted);font-size:.78rem;text-transform:uppercase;letter-spacing:.04em}
tr:last-child td{border-bottom:none}
.bar{position:relative;width:92px;height:8px;background:var(--bar-bg);border-radius:4px;overflow:hidden;margin-top:6px}
.bar span{position:absolute;inset:0 auto 0 0;border-radius:4px}
.num{font-variant-numeric:tabular-nums;font-weight:600}
.chip{display:inline-block;font-size:.75rem;padding:1px 8px;border-radius:999px;background:var(--chip);color:var(--muted);white-space:nowrap}
.st-believed{color:var(--good)}.st-rejected,.st-doubted{color:var(--bad)}
.cite{display:block;font-size:.8rem;color:var(--muted);margin-top:2px}
.cite q{color:var(--fg);opacity:.85}
ul.plain{list-style:none;padding:0;margin:0}ul.plain li{padding:8px 0;border-bottom:1px solid var(--line)}
.vs{display:grid;grid-template-columns:1fr auto 1fr;gap:10px;align-items:start;background:var(--card);border:1px solid var(--line);
border-radius:10px;padding:10px 12px;margin-bottom:8px}.vs .mid{color:var(--muted);font-size:.8rem;padding-top:2px}
.timeline{position:relative;margin:0;padding:0 0 0 18px;list-style:none;border-left:2px solid var(--line)}
.timeline li{position:relative;margin:0 0 10px;padding-left:6px}
.timeline li::before{content:"";position:absolute;left:-25px;top:7px;width:10px;height:10px;border-radius:50%;background:var(--card);border:2px solid var(--accent)}
.timeline .step{font-weight:600;font-size:.8rem;text-transform:uppercase;letter-spacing:.05em;color:var(--accent)}
.graph{width:100%;height:auto;background:var(--card);border:1px solid var(--line);border-radius:10px}
.glabel{font-size:12px;fill:var(--fg)}
.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:.8rem;color:var(--muted);margin-top:6px}
.legend i{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px;vertical-align:-1px}
@media (max-width:640px){.vs{grid-template-columns:1fr}.vs .mid{padding:0}th:nth-child(4),td:nth-child(4){display:none}}
"""


def _e(s: Any) -> str:
    return html.escape(str(s if s is not None else ""), quote=True)


def _bar_html(p: float) -> str:
    color = "var(--good)" if p >= 0.75 else "var(--bad)" if p <= 0.35 else "var(--mid)"
    return f'<div class="bar" aria-hidden="true"><span style="width:{max(0, min(100, p * 100)):.0f}%;background:{color}"></span></div>'


def _cite_html(c: dict[str, Any]) -> str:
    span = f" &middot; chars {c['span'][0]}&ndash;{c['span'][1]}" if c.get("span") else ""
    rel = f" &middot; rel {c['reliability']:.2f}" if c.get("reliability") is not None else ""
    q = f' <q>{_e(_clip(c["quote"], 140))}</q>' if c.get("quote") else ""
    return f'<span class="cite" title="{_e(c.get("chunk_id"))}">{_e(c["source"])}{span}{rel}{q}</span>'


def render_html(store: KnowledgeStore, **kwargs: Any) -> str:
    d = build_report_data(store, **kwargs)
    st = d["stats"]
    h: list[str] = [
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">",
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        '<meta name="color-scheme" content="light dark">',
        f"<title>{_e(_clip(d['title'], 60))} - Noesis</title><style>{_CSS}</style></head><body><main>",
        f"<h1>{_e(d['title'])}</h1><div class=\"muted small\">Noesis understanding report &middot; generated {_e(d['generated_at'])}</div>",
        '<nav><a href="#overview">Overview</a><a href="#beliefs">Beliefs</a><a href="#contradictions">Contradictions</a>'
        '<a href="#questions">Questions</a><a href="#trace">Reasoning trace</a><a href="#graph">Concept graph</a><a href="#sources">Sources</a></nav>',
        '<section id="overview"><h2>Overview</h2><div class="stats">',
    ]
    for label, key in (("Sources", "sources"), ("Claims", "claims"), ("Evidence links", "evidence"), ("Questions", "questions"), ("Reasoning runs", "runs"), ("Thoughts", "thoughts")):
        h.append(f'<div class="stat"><b>{_e(st.get(key, 0))}</b><span class="muted small">{label}</span></div>')
    h.append("</div>")
    if d["status_counts"]:
        h.append('<p class="small">Beliefs: ' + " &middot; ".join(
            f'<span class="st-{_e(k)}">{_e(k)}</span> {_e(v)}' for k, v in sorted(d["status_counts"].items())) + "</p>")
    if d["summary"]:
        h.append(f"<blockquote>{_e(d['summary'])}</blockquote>")
    if d["runs"]:
        h.append('<div class="tablewrap"><table><thead><tr><th>Run</th><th>Strategy</th><th>Status</th><th>Steps</th><th>Summary</th></tr></thead><tbody>')
        for r in d["runs"][-8:]:
            h.append(
                f'<tr><td><code class="small">{_e(r["id"])}</code><div class="muted small">{_e(r["started_at"])}</div></td>'
                f'<td>{_e(r["strategy"])}</td><td>{_e(r["status"])}<div class="muted small">{_e(r["stop_reason"])}</div></td>'
                f'<td class="num">{_e(r["steps"])}</td><td class="small">{_e(_clip(r["summary"], 220))}</td></tr>'
            )
        h.append("</tbody></table></div>")
    h.append("</section>")
    # beliefs
    h.append(f'<section id="beliefs"><h2>Beliefs <span class="muted small">top {len(d["beliefs"])} of {d["beliefs_total"]}</span></h2>')
    h.append('<div class="tablewrap"><table><thead><tr><th>Confidence</th><th>Claim &amp; citations</th><th>Status</th><th>Evidence</th></tr></thead><tbody>')
    for b in d["beliefs"]:
        chip = ' <span class="chip">reasoning hypothesis</span>' if b["origin"] == "hypothesis" else ""
        h.append(
            f'<tr><td><span class="num">{b["confidence"]:.2f}</span>{_bar_html(b["confidence"])}</td>'
            f'<td>{_e(b["claim"])}{chip}{"".join(_cite_html(c) for c in b["citations"]) or "<span class=cite>no citations</span>"}</td>'
            f'<td class="st-{_e(b["status"])}">{_e(b["status"])}</td>'
            f'<td class="num small">+{b["n_for"]} / &minus;{b["n_against"]}</td></tr>'
        )
    h.append("</tbody></table></div></section>")
    # contradictions
    h.append('<section id="contradictions"><h2>Contradictions</h2>')
    if not d["contradictions"]:
        h.append('<p class="muted">None detected.</p>')
    for c in d["contradictions"]:
        def side(x: dict[str, Any]) -> str:
            conf = f'<span class="num">{x["confidence"]:.2f}</span>{_bar_html(x["confidence"])}' if x["confidence"] is not None else ""
            return f'<div>{_e(x["text"])}{"".join(_cite_html(ci) for ci in x["citations"])}{conf}</div>'
        h.append(f'<div class="vs">{side(c["a"])}<div class="mid">{"resolved" if c["resolved"] else "vs"}</div>{side(c["b"])}</div>')
    h.append("</section>")
    # questions
    h.append('<section id="questions"><h2>Open questions</h2>')
    if not d["open_questions"]:
        h.append('<p class="muted">None.</p>')
    else:
        h.append('<ul class="plain">')
        for q in d["open_questions"]:
            h.append(f'<li><span class="chip">{_e(q["origin"])}</span> <span class="num small">{q["priority"]:.2f}</span> {_e(q["text"])}</li>')
        h.append("</ul>")
    if d["answered_questions"]:
        h.append('<h2>Answered questions</h2><ul class="plain">')
        for q in d["answered_questions"]:
            conf = f' <span class="num small">{q["confidence"]:.2f}</span>' if q["confidence"] is not None else ""
            h.append(f'<li><b>{_e(q["text"])}</b><div class="small">{_e(_clip(q["answer"], 300))}{conf}</div></li>')
        h.append("</ul>")
    h.append("</section>")
    # trace
    h.append('<section id="trace"><h2>Reasoning trace' + (f' <span class="muted small">run {_e(d["trace_run"])}</span>' if d["trace_run"] else "") + "</h2>")
    if not d["trace"]:
        h.append('<p class="muted">No reasoning runs yet.</p>')
    else:
        h.append('<ol class="timeline">')
        for t in d["trace"]:
            h.append(f'<li><span class="step">{_e(t["step"])}</span> <span class="muted small">{_e(t["at"])}</span><div class="small">{_e(_clip(t["content"], 600))}</div></li>')
        h.append("</ol>")
    h.append("</section>")
    # graph
    h.append('<section id="graph"><h2>Concept graph</h2>' + concept_graph_svg(d["graph"]))
    h.append(
        '<div class="legend"><span><i style="background:hsl(172 70% 43%)"></i>well supported</span>'
        '<span><i style="background:hsl(172 20% 55%)"></i>uncertain</span><span><i style="background:hsl(8 75% 50%)"></i>doubted</span>'
        "<span>node size = number of claims; edge = concepts share claims</span></div></section>"
    )
    # sources
    h.append('<section id="sources"><h2>Sources</h2><div class="tablewrap"><table><thead><tr><th>Reliability</th><th>Source</th><th>Why</th></tr></thead><tbody>')
    for s in d["sources"]:
        h.append(f'<tr><td><span class="num">{s["reliability"]:.2f}</span>{_bar_html(s["reliability"])}</td><td>{_e(s["title"])}<div class="muted small">{_e(s["uri"])}</div></td><td class="small">{_e(s["reason"])}</td></tr>')
    h.append("</tbody></table></div></section>")
    h.append('<p class="muted small" style="margin-top:32px">Confidence is computed from evidence (source reliability, independence, stance); model judgments are a bounded minor input. Noesis structures and audits reasoning; it does not make a model infallible.</p>')
    h.append("</main></body></html>")
    return "".join(h)


def write_report(store: KnowledgeStore, path: str | Path, *, fmt: str | None = None, **kwargs: Any) -> Path:
    """Write a report; ``fmt`` is ``"html"``/``"md"`` or inferred from the suffix."""
    p = Path(path)
    fmt = (fmt or ("html" if p.suffix.lower() in (".html", ".htm") else "md")).lower()
    text = render_html(store, **kwargs) if fmt == "html" else render_markdown(store, **kwargs)
    p.write_text(text, encoding="utf-8")
    return p
