"""Noesis: evidence-backed understanding + independent, auditable reasoning for any AI model.

Quick start (fully offline)::

    from noesis import KnowledgeStore, ingest, understand, get_llm
    store = KnowledgeStore("coffee.noesis.db")
    ingest(store, "examples/corpus")
    result = understand(store, get_llm("mock"))   # or llm=None for pure heuristics
    for b in store.list_beliefs(order_by_confidence=True):
        print(b.confidence, store.get_claim(b.claim_id).text)

See ``noesis/API_NOTES.md`` for the full public API.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .beliefs import (
    BeliefConfig,
    adjust_belief,
    explain_belief,
    set_belief,
    update_all_beliefs,
    update_belief,
)
from .embed import HashingEmbedder, get_embedder, hybrid_search
from .ingest import ingest, ingest_text
from .llm import LLM, LLMError, MockLLM, get_llm
from .store import KnowledgeStore
from .types import (
    Belief,
    BeliefChange,
    Chunk,
    Claim,
    Concept,
    Evidence,
    Question,
    ReasoningRun,
    Source,
    Summary,
    Thought,
)
from .understand import UnderstandConfig, UnderstandResult, assess_claim, understand

__all__ = [
    "__version__",
    "KnowledgeStore",
    "ingest",
    "ingest_text",
    "understand",
    "UnderstandConfig",
    "UnderstandResult",
    "assess_claim",
    "update_belief",
    "update_all_beliefs",
    "adjust_belief",
    "set_belief",
    "explain_belief",
    "BeliefConfig",
    "get_llm",
    "LLM",
    "LLMError",
    "MockLLM",
    "get_embedder",
    "HashingEmbedder",
    "hybrid_search",
    "Source",
    "Chunk",
    "Concept",
    "Claim",
    "Evidence",
    "Belief",
    "BeliefChange",
    "Question",
    "Thought",
    "ReasoningRun",
    "Summary",
    # lazily imported (see __getattr__): reasoning engine, reports, MCP
    "Mind",
    "MindConfig",
    "Budget",
    "Answer",
    "write_report",
    "render_markdown",
    "render_html",
    "NoesisService",
    "create_mcp_server",
]

_LAZY = {
    "Mind": ("noesis.reason", "Mind"),
    "MindConfig": ("noesis.reason", "MindConfig"),
    "Budget": ("noesis.reason", "Budget"),
    "Answer": ("noesis.reason", "Answer"),
    "write_report": ("noesis.report", "write_report"),
    "render_markdown": ("noesis.report", "render_markdown"),
    "render_html": ("noesis.report", "render_html"),
    "NoesisService": ("noesis.mcp_server", "NoesisService"),
    "create_mcp_server": ("noesis.mcp_server", "create_server"),
}


def __getattr__(name: str):  # PEP 562: keep `import noesis` light
    if name in _LAZY:
        import importlib

        mod, attr = _LAZY[name]
        value = getattr(importlib.import_module(mod), attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'noesis' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))
