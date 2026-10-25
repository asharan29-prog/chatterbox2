"""Noesis Research Agent: an internet-connected agent on top of the Noesis engine.

>>> from noesis import KnowledgeStore, get_llm
>>> from noesis.agent import ResearchAgent
>>> agent = ResearchAgent(KnowledgeStore("r.db"), get_llm("mock"), search="mock")   # offline demo web
>>> result = agent.research("Is coffee good for health?", depth=1)
>>> print(result.markdown)

Search providers: ``ddg`` (no key, fragile), ``brave`` (BRAVE_API_KEY), ``tavily``
(TAVILY_API_KEY), ``searxng`` (SEARXNG_URL), ``mock`` (bundled offline fixture web).
See docs/research-agent.md.
"""

from __future__ import annotations

from .research import (
    Conclusion,
    ResearchAgent,
    ResearchBudget,
    ResearchResult,
    SourceRecord,
    WatchResult,
    plan_heuristic,
)
from .web import (
    BraveProvider,
    DiskCache,
    DomainPolicy,
    DuckDuckGoProvider,
    FetchedPage,
    FetchError,
    RateLimiter,
    SearchError,
    SearchProvider,
    SearchResult,
    SearxngProvider,
    TavilyProvider,
    WebFetcher,
    WebToolkit,
    extract_main_text,
    get_search_provider,
    is_public_address,
    registrable_domain,
)

__all__ = [
    "ResearchAgent",
    "ResearchBudget",
    "ResearchResult",
    "SourceRecord",
    "Conclusion",
    "WatchResult",
    "plan_heuristic",
    "WebToolkit",
    "WebFetcher",
    "DomainPolicy",
    "DiskCache",
    "RateLimiter",
    "SearchProvider",
    "SearchResult",
    "FetchedPage",
    "FetchError",
    "SearchError",
    "DuckDuckGoProvider",
    "BraveProvider",
    "TavilyProvider",
    "SearxngProvider",
    "get_search_provider",
    "extract_main_text",
    "is_public_address",
    "registrable_domain",
]
