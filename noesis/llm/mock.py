"""Deterministic offline LLM for tests, demos and benchmarks.

Two modes (combinable):

* **heuristic** (default): recognizes the prompt type from the ``<task>name</task>`` tag that
  :func:`noesis.llm.prompts.build_prompt` puts in every prompt, parses the ``<document>``
  blocks and named fields, and returns plausible, *input-derived* JSON (sentence-split claims,
  noun-phrase concepts, lexical stance, extractive answers...). Unknown tasks with a JSON
  schema get a minimal schema-conforming object.
* **scripted**: exact responses for fixtures. ``script`` may be

  - a list: consumed in order for every call;
  - a dict ``{task_name: response | [responses...] | callable}``;
  - a callable ``f(prompt: str, task: str | None, call: dict) -> response``.

  A response may be a ``str`` (returned verbatim -- use this to simulate malformed JSON) or a
  ``dict``/``list`` (JSON-encoded). When the script has no entry for a call, the mock falls
  back to heuristic mode (``fallback="heuristic"``) or raises (``fallback="error"``).

Every call is recorded in ``mock.calls`` (task, prompt, system, json_schema, response).
Reasoning engineers can add heuristics for new tasks with :meth:`MockLLM.register_task`.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from .. import nlp
from .base import BaseLLM, LLMError, Message, task_of
from .prompts import parse_documents, parse_field

Handler = Callable[[str, dict | None], Any]


class MockLLM(BaseLLM):
    supports_embeddings = True

    def __init__(
        self,
        script: list | dict | Callable | None = None,
        *,
        mode: str | None = None,
        fallback: str = "heuristic",
        name: str = "mock",
        embed_dim: int = 512,
    ) -> None:
        super().__init__()
        self.name = name
        self.mode = mode or ("scripted" if script is not None else "heuristic")
        self.fallback = fallback
        self._script = script
        self._queue: list | None = list(script) if isinstance(script, list) else None
        self._task_queues: dict[str, list] = {}
        if isinstance(script, dict):
            for k, v in script.items():
                self._task_queues[k] = list(v) if isinstance(v, list) else [v]
        self.calls: list[dict[str, Any]] = []
        self._handlers: dict[str, Handler] = dict(_DEFAULT_HANDLERS)
        self._embed_dim = embed_dim

    # -------------------------------------------------------------- extension points
    def register_task(self, task: str, handler: Handler) -> None:
        """Register ``handler(prompt, json_schema) -> response`` for heuristic mode."""
        self._handlers[task] = handler

    def push(self, *responses: Any, task: str | None = None) -> None:
        """Append scripted responses (globally, or for one task)."""
        if task:
            self._task_queues.setdefault(task, []).extend(responses)
        else:
            if self._queue is None:
                self._queue = []
            self._queue.extend(responses)
        self.mode = "scripted"

    # -------------------------------------------------------------- core
    def _complete_text(
        self,
        messages: list[Message],
        system: str | None,
        temperature: float,
        max_tokens: int,
        json_schema: dict | None,
        json_mode: bool,
    ) -> str:
        prompt = messages[-1]["content"] if messages else ""
        # On a JSON repair retry, the original prompt is the first user message.
        original = next((m["content"] for m in messages if m["role"] == "user"), prompt)
        task = task_of(original)
        call = {"task": task, "prompt": original, "system": system, "json_schema": json_schema, "json_mode": json_mode}
        resp = self._scripted(original, task, call)
        if resp is _NO_SCRIPT:
            if self.mode == "scripted" and self.fallback == "error":
                raise LLMError(f"MockLLM: no scripted response for task={task!r}")
            resp = self._heuristic(original, task, json_schema, json_mode)
        text = resp if isinstance(resp, str) else json.dumps(resp)
        call["response"] = text
        self.calls.append(call)
        return text

    def _scripted(self, prompt: str, task: str | None, call: dict) -> Any:
        if self._script is None and not self._task_queues and not self._queue:
            return _NO_SCRIPT
        if callable(self._script) and not isinstance(self._script, (list, dict)):
            r = self._script(prompt, task, call)
            return _NO_SCRIPT if r is None else r
        if task and self._task_queues.get(task):
            q = self._task_queues[task]
            item = q.pop(0) if len(q) > 1 else q[0]  # last item repeats
            return item(prompt, task, call) if callable(item) else item
        if self._queue:
            item = self._queue.pop(0)
            return item(prompt, task, call) if callable(item) else item
        return _NO_SCRIPT

    def _heuristic(self, prompt: str, task: str | None, schema: dict | None, json_mode: bool) -> Any:
        handler = self._handlers.get(task or "")
        if handler is not None:
            return handler(prompt, schema)
        if json_mode:
            return synth_from_schema(schema, prompt) if schema else {"text": _first_sentence(prompt)}
        docs = parse_documents(prompt)
        if docs:
            return " ".join(_first_sentence(d["text"]) for d in docs[:3])
        return "MockLLM response."

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        from ..embed import HashingEmbedder

        return HashingEmbedder(dim=self._embed_dim).embed(texts)


_NO_SCRIPT = object()


# ============================================================== heuristic handlers
def _first_sentence(text: str) -> str:
    sents = nlp.split_sentences(text or "")
    return sents[0][0] if sents else (text or "").strip()[:200]


def _h_extract_claims(prompt: str, schema: dict | None) -> dict:
    out = []
    for d in parse_documents(prompt):
        for c in nlp.extract_claims(d["text"]):
            out.append(
                {
                    "text": c["text"],
                    "quote": c["quote"],
                    "concepts": c["concepts"],
                    "strength": 0.6 if c["hedged"] else 0.8,
                }
            )
    return {"claims": out}


def _h_extract_concepts(prompt: str, schema: dict | None) -> dict:
    text = "\n".join(d["text"] for d in parse_documents(prompt))
    names = nlp.extract_concepts(text)
    return {"concepts": [{"name": n, "description": _describe(n, text)} for n in names]}


def _describe(name: str, text: str) -> str:
    for s, _, _ in nlp.split_sentences(text):
        if name.lower() in s.lower():
            return s[:200]
    return ""


def _claims_from_fields(prompt: str) -> list[str]:
    vals = []
    for m in re.finditer(r"<claim[^>]*>(.*?)</claim>", prompt, re.S):
        vals.append(m.group(1))
    for name in ("claim_a", "claim_b"):
        v = parse_field(prompt, name)
        if v:
            vals.append(v)
    return vals


def _h_same_claim(prompt: str, schema: dict | None) -> dict:
    a, b = parse_field(prompt, "claim_a") or "", parse_field(prompt, "claim_b") or ""
    sim = nlp.text_similarity(a, b)
    pa, da = nlp.polarity(a)
    pb, db = nlp.polarity(b)
    opposite = pa != pb or any(k in db and db[k] != v for k, v in da.items())
    same = sim >= 0.5 and not opposite
    return {"same": bool(same), "reason": f"tfidf similarity {sim:.2f}; polarity {'differs' if opposite else 'matches'}"}


def _h_classify_stance(prompt: str, schema: dict | None) -> dict:
    claims = _claims_from_fields(prompt)
    claim = claims[0] if claims else ""
    docs = parse_documents(prompt)
    passage = docs[0]["text"] if docs else (parse_field(prompt, "passage") or "")
    best = nlp.best_sentences(claim, passage, k=1)
    target = best[0] if best else passage
    stance, strength = nlp.lexical_stance(claim, target)
    return {"stance": stance, "strength": strength, "quote": target[:300], "rationale": "lexical overlap + polarity"}


def _h_summarize(prompt: str, schema: dict | None) -> dict:
    docs = parse_documents(prompt)
    sents = [_first_sentence(d["text"]) for d in docs if d.get("text")]
    title = parse_field(prompt, "title") or (nlp.extract_concepts(" ".join(sents), 1) or ["Summary"])[0]
    summary = " ".join(sents[:4])
    return {"title": str(title), "summary": summary}


def _h_generate_questions(prompt: str, schema: dict | None) -> dict:
    qs = []
    for d in parse_documents(prompt):
        s = _first_sentence(d["text"]).rstrip(".")
        if s:
            qs.append({"text": f"Is it true that {s[0].lower() + s[1:]}?", "priority": 0.5})
    return {"questions": qs[:5]}


def _rank_docs(query: str, docs: list[dict]) -> list[tuple[float, dict, str]]:
    ranked = []
    for d in docs:
        best = nlp.best_sentences(query, d["text"], k=1)
        sent = best[0] if best else ""
        score = nlp.text_similarity(query, sent) if sent else 0.0
        ranked.append((score, d, sent))
    ranked.sort(key=lambda x: -x[0])
    return ranked


def _h_answer_question(prompt: str, schema: dict | None) -> dict:
    q = parse_field(prompt, "question") or ""
    docs = parse_documents(prompt)
    ranked = [r for r in _rank_docs(q, docs) if r[2]]
    if not ranked:
        return {"answer": "The provided evidence does not answer this question.", "confidence": 0.1, "citations": []}
    top = ranked[:2]
    answer = " ".join(s for _, _, s in top)
    conf = round(min(0.9, 0.3 + 0.6 * top[0][0]), 3)
    cites = [d.get("id", "") for _, d, _ in top if d.get("id")]
    return {"answer": answer, "confidence": conf, "citations": cites, "reasoning": "extractive: highest-overlap sentences"}


def _negate(sentence: str) -> str:
    s = sentence.strip().rstrip(".")
    m = re.search(r"\b(is|are|was|were|does|do|did|can|will|has|have)\b", s)
    if m:
        return s[: m.end()] + " not" + s[m.end() :] + "."
    return "It is not the case that " + s[0].lower() + s[1:] + "."


def _h_generate_hypotheses(prompt: str, schema: dict | None) -> dict:
    q = parse_field(prompt, "question") or ""
    docs = parse_documents(prompt)
    ranked = [r for r in _rank_docs(q, docs) if r[2]]
    base = ranked[0][2] if ranked else (q.rstrip("?") + ".")
    hyps = [{"text": base, "prior": 0.5}, {"text": _negate(base), "prior": 0.5}]
    if len(ranked) > 1 and ranked[1][2] != base:
        hyps.append({"text": ranked[1][2], "prior": 0.4})
    return {"hypotheses": hyps}


def _stances(claim: str, docs: list[dict]) -> list[tuple[str, float, str, dict]]:
    res = []
    for d in docs:
        best = nlp.best_sentences(claim, d["text"], k=1)
        sent = best[0] if best else d["text"][:200]
        st, s = nlp.lexical_stance(claim, sent)
        res.append((st, s, sent, d))
    return res


def _h_critique(prompt: str, schema: dict | None) -> dict:
    claims = _claims_from_fields(prompt)
    claim = claims[0] if claims else (parse_field(prompt, "question") or "")
    st = _stances(claim, parse_documents(prompt))
    pro = [s for k, _, s, _ in st if k == "supports"]
    con = [s for k, _, s, _ in st if k == "contradicts"]
    if not con:
        con = ["The supporting evidence may come from a limited number of sources."]
    verdict = "supported" if len(pro) > len(con) else "contested" if pro else "unsupported"
    return {"arguments_for": pro, "arguments_against": con, "verdict": verdict}


def _h_judge(prompt: str, schema: dict | None) -> dict:
    claims = _claims_from_fields(prompt)
    claim = claims[0] if claims else (parse_field(prompt, "question") or "")
    st = _stances(claim, parse_documents(prompt))
    score = sum(s if k == "supports" else -s if k == "contradicts" else 0 for k, s, _, _ in st)
    n = sum(1 for k, *_ in st if k != "context")
    if n == 0:
        return {"verdict": "uncertain", "confidence": 0.5, "rationale": "no relevant evidence", "citations": []}
    p = 1 / (1 + np.exp(-1.5 * score))
    verdict = "true" if p >= 0.6 else "false" if p <= 0.4 else "uncertain"
    cites = [d.get("id", "") for k, _, _, d in st if k != "context" and d.get("id")]
    return {"verdict": verdict, "confidence": round(float(p), 3), "rationale": f"{n} relevant passages, net score {score:.2f}", "citations": cites}


def _h_reflect(prompt: str, schema: dict | None) -> dict:
    docs = parse_documents(prompt)
    learned = " ".join(_first_sentence(d["text"]) for d in docs[:2]) or "No new information."
    return {"learned": learned, "next_questions": []}


_DEFAULT_HANDLERS: dict[str, Handler] = {
    "extract_claims": _h_extract_claims,
    "extract_concepts": _h_extract_concepts,
    "same_claim": _h_same_claim,
    "classify_stance": _h_classify_stance,
    "summarize": _h_summarize,
    "generate_questions": _h_generate_questions,
    "answer_question": _h_answer_question,
    "answer": _h_answer_question,
    "generate_hypotheses": _h_generate_hypotheses,
    "hypothesize": _h_generate_hypotheses,
    "critique": _h_critique,
    "test": _h_critique,
    "judge": _h_judge,
    "reflect": _h_reflect,
}


def synth_from_schema(schema: dict | None, context: str = "") -> Any:
    """Produce a minimal value conforming to ``schema`` (used for unknown tasks)."""
    if not schema:
        return {}
    if "enum" in schema:
        return schema["enum"][0]
    t = schema.get("type")
    if isinstance(t, list):
        t = next((x for x in t if x != "null"), t[0])
    if t == "object":
        props = schema.get("properties") or {}
        keys = list(dict.fromkeys(list(schema.get("required", [])) + list(props)))
        return {k: synth_from_schema(props.get(k, {"type": "string"}), context) for k in keys}
    if t == "array":
        items = schema.get("items")
        return [synth_from_schema(items, context)] if isinstance(items, dict) and schema.get("minItems", 0) > 0 else []
    if t == "string":
        return _first_sentence(re.sub(r"<[^>]+>", " ", context))[:200] if context else ""
    if t in ("number", "integer"):
        lo, hi = schema.get("minimum", 0), schema.get("maximum", 1)
        v = (lo + hi) / 2
        return int(v) if t == "integer" else v
    if t == "boolean":
        return False
    return None
