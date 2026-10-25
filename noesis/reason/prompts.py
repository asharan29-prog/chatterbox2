"""Prompt instructions and JSON schemas for the reasoning tasks.

All prompts are assembled with :func:`noesis.llm.prompts.build_prompt`, so retrieved text is
always wrapped as untrusted ``<document>`` data and the safety system prompt applies.
Document ids in reasoning prompts are real chunk ids; models must cite those ids only.
"""

from __future__ import annotations

from typing import Any

from ..llm.prompts import TASK_SCHEMAS, build_prompt, system_prompt
from ..types import Chunk, Source

_STR = {"type": "string"}
_NUM01 = {"type": "number", "minimum": 0, "maximum": 1}
_STRS = {"type": "array", "items": _STR}

SCHEMAS: dict[str, dict] = {
    "generate_questions": TASK_SCHEMAS["generate_questions"],
    "generate_hypotheses": TASK_SCHEMAS["generate_hypotheses"],
    "answer_question": TASK_SCHEMAS["answer_question"],
    "reflect": TASK_SCHEMAS["reflect"],
    "judge": TASK_SCHEMAS["judge"],
    "critique": {
        "type": "object",
        "required": ["arguments_for", "arguments_against"],
        "properties": {
            "arguments_for": _STRS,
            "arguments_against": _STRS,
            "verdict": _STR,
            "citations": _STRS,
        },
    },
    "devils_advocate": {
        "type": "object",
        "required": ["counterargument"],
        "properties": {"counterargument": _STR, "citations": _STRS, "severity": _NUM01},
    },
    "advocate": {
        "type": "object",
        "required": ["arguments"],
        "properties": {"arguments": _STRS, "citations": _STRS},
    },
}

CITE_RULE = (
    "Use ONLY the documents provided. Cite documents by their exact id attribute in 'citations'; "
    "never cite ids that are not present, never invent sources or quotes."
)

INSTRUCTIONS: dict[str, str] = {
    "generate_questions": (
        "You are exploring a topic to improve understanding. Given the topic goal and the claims "
        "below (with current confidence), propose up to 5 new, specific, answerable research questions "
        "that would most reduce uncertainty or resolve conflicts. Do not repeat questions listed as "
        "already asked. Return {\"questions\": [{\"text\", \"priority\"}]}."
    ),
    "generate_hypotheses": (
        "Propose up to {k} distinct candidate answers (hypotheses) to the question, each a single "
        "declarative sentence that can be true or false. Include at least one hypothesis that "
        "contradicts the most obvious answer. " + CITE_RULE + " Return {\"hypotheses\": [{\"text\", \"prior\"}]}."
    ),
    "critique": (
        "Argue FOR and AGAINST the claim using only the documents. Each argument must be grounded in a "
        "document. " + CITE_RULE + " Return {\"arguments_for\", \"arguments_against\", \"verdict\", \"citations\"}."
    ),
    "devils_advocate": (
        "Act as a devil's advocate: give the single strongest reason the claim could be WRONG or "
        "overstated given the documents (bias, weak design, single source, confounding, scope). "
        + CITE_RULE + " Return {\"counterargument\", \"citations\", \"severity\"}."
    ),
    "advocate": (
        "You are the {role} advocate in a debate about the claim. Make the strongest case {side} the claim "
        "using only the documents. " + CITE_RULE + " Return {\"arguments\", \"citations\"}."
    ),
    "judge": (
        "Judge whether the claim is true given ONLY the documents (and the debate arguments if given). "
        "Weigh source reliability and independence. Say 'uncertain' if the documents do not settle it. "
        "'confidence' is your probability (0-1) that the claim is TRUE. " + CITE_RULE + " Return {\"verdict\": true|false|uncertain, \"confidence\", \"rationale\", \"citations\"}."
    ),
    "answer_question": (
        "Answer the question using only the documents. If they do not contain the answer, say so. "
        + CITE_RULE + " Return {\"answer\", \"confidence\", \"citations\", \"reasoning\"}."
    ),
    "reflect": (
        "Reflect on this reasoning step: in one or two sentences, what was learned (what changed in "
        "the beliefs below), and list up to 3 follow-up questions worth investigating next. "
        "Return {\"learned\", \"next_questions\"}."
    ),
}


def doc_for_chunk(chunk: Chunk, source: Source | None) -> dict[str, Any]:
    """Document dict for :func:`build_prompt`; ``id`` is the real chunk id (citable)."""
    d: dict[str, Any] = {"text": chunk.text, "id": chunk.id}
    if source is not None:
        d["source"] = source.title or source.uri
        d["reliability"] = f"{source.reliability:.2f}"
    return d


def make(task: str, *, documents: list[dict] = (), fields: dict[str, str] | None = None, **fmt: Any) -> str:  # type: ignore[assignment]
    instr = INSTRUCTIONS[task]
    for key, val in fmt.items():  # simple token replacement (instructions contain JSON braces)
        instr = instr.replace("{" + key + "}", str(val))
    return build_prompt(task, instr, documents=documents, fields=fields)


def call(llm: Any, task: str, prompt: str, *, temperature: float = 0.0, max_tokens: int = 1024) -> Any:
    return llm.complete(
        prompt,
        system=system_prompt("You are the reasoning engine of Noesis. Be skeptical and evidence-bound."),
        json_schema=SCHEMAS[task],
        temperature=temperature,
        max_tokens=max_tokens,
    )
