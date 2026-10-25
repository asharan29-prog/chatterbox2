"""MockLLM heuristics for the reasoning-only task tags (``devils_advocate``, ``advocate``).

The core :class:`noesis.llm.mock.MockLLM` already handles ``generate_questions``,
``generate_hypotheses``, ``critique``, ``judge``, ``reflect`` and ``answer_question``. The two
extra tasks are registered per instance through the public :meth:`MockLLM.register_task` hook
by :func:`install` (called from :class:`noesis.reason.Mind`), without touching core code.
Handlers are deterministic and derive output only from the prompt.
"""

from __future__ import annotations

from typing import Any

from .. import nlp
from ..llm.mock import MockLLM
from ..llm.prompts import parse_documents, parse_field


def _claim(prompt: str) -> str:
    return parse_field(prompt, "claim") or parse_field(prompt, "question") or ""


def _scored(claim: str, prompt: str) -> list[tuple[str, float, str, dict]]:
    out = []
    for d in parse_documents(prompt):
        best = nlp.best_sentences(claim, d["text"], k=1)
        sent = best[0] if best else d["text"][:200]
        st, s = nlp.lexical_stance(claim, sent)
        out.append((st, s, sent, d))
    return out


def h_devils_advocate(prompt: str, schema: dict | None) -> dict:
    claim = _claim(prompt)
    st = _scored(claim, prompt)
    con = sorted([x for x in st if x[0] == "contradicts"], key=lambda x: -x[1])
    if con:
        _, s, sent, d = con[0]
        return {"counterargument": f"A source states otherwise: {sent}", "citations": [d.get("id", "")], "severity": round(s, 3)}
    sup_sources = {d.get("source", d.get("id")) for k, _, _, d in st if k == "supports"}
    if len(sup_sources) <= 1:
        cites = [d.get("id", "") for k, _, _, d in st if k == "supports"][:1]
        return {
            "counterargument": "The support rests on a single source; it may not be independently replicated.",
            "citations": cites,
            "severity": 0.4,
        }
    return {
        "counterargument": "Supporting sources may share study designs or confounders (e.g. observational data).",
        "citations": [],
        "severity": 0.2,
    }


def h_advocate(prompt: str, schema: dict | None) -> dict:
    claim = _claim(prompt)
    role = (parse_field(prompt, "role") or "pro").strip().lower()
    want = "supports" if role.startswith("pro") else "contradicts"
    st = sorted([x for x in _scored(claim, prompt) if x[0] == want], key=lambda x: -x[1])
    args = [sent for _, _, sent, _ in st[:3]]
    cites = [d.get("id", "") for _, _, _, d in st[:3] if d.get("id")]
    if not args:
        args = ["The documents provide no direct evidence on this side."]
    return {"arguments": args, "citations": cites}


HANDLERS: dict[str, Any] = {"devils_advocate": h_devils_advocate, "advocate": h_advocate}


def install(llm: Any) -> None:
    """Register the reasoning task heuristics on a :class:`MockLLM` (no-op for other LLMs).

    Handlers already registered by the user (``llm.register_task``) are kept.
    """
    target = getattr(llm, "_llm", llm)  # unwrap CountingLLM
    if not isinstance(target, MockLLM):
        return
    existing = getattr(target, "_handlers", {})
    for task, handler in HANDLERS.items():
        if task not in existing:
            target.register_task(task, handler)
