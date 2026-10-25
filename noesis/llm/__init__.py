"""Model adapters behind one small protocol.

>>> from noesis.llm import get_llm
>>> llm = get_llm("mock")                       # offline, deterministic
>>> llm = get_llm("anthropic:claude-sonnet-5-5")  # needs ANTHROPIC_API_KEY
>>> llm = get_llm("openai:gpt-4o-mini")         # needs OPENAI_API_KEY (or base_url=...)
>>> llm = get_llm("ollama:llama3.1")            # local Ollama at $OLLAMA_HOST
"""

from __future__ import annotations

import os
from typing import Any

from .base import LLM, BaseLLM, JSONParseError, LLMError, Message, Usage, extract_json, validate_schema
from .mock import MockLLM
from .prompts import SAFETY_SYSTEM, TASK_SCHEMAS, build_prompt, system_prompt, wrap_document

__all__ = [
    "LLM",
    "BaseLLM",
    "Message",
    "Usage",
    "LLMError",
    "JSONParseError",
    "MockLLM",
    "get_llm",
    "extract_json",
    "validate_schema",
    "build_prompt",
    "wrap_document",
    "system_prompt",
    "SAFETY_SYSTEM",
    "TASK_SCHEMAS",
]

PROVIDERS = ("mock", "anthropic", "openai", "ollama", "none")


def get_llm(spec: str | None = None, **kwargs: Any) -> BaseLLM | None:
    """Create an LLM from a ``"provider[:model]"`` string.

    ``spec`` defaults to ``$NOESIS_MODEL`` or ``"mock"``. ``"none"``/``"offline"`` returns
    ``None`` (pure heuristic pipeline). Extra ``kwargs`` go to the adapter constructor
    (e.g. ``base_url=``, ``api_key=``, ``timeout=``, ``script=`` for mock).
    """
    spec = (spec or os.environ.get("NOESIS_MODEL") or "mock").strip()
    provider, _, model = spec.partition(":")
    provider = provider.lower()
    if provider in ("none", "offline", "heuristic"):
        return None
    if provider == "mock":
        return MockLLM(name=spec, **kwargs)
    if provider in ("anthropic", "claude"):
        from .anthropic import AnthropicLLM

        return AnthropicLLM(model or "claude-sonnet-5-5", **kwargs)
    if provider in ("openai", "openai-compatible"):
        from .openai import OpenAILLM

        return OpenAILLM(model or "gpt-4o-mini", **kwargs)
    if provider == "ollama":
        from .ollama import OllamaLLM

        return OllamaLLM(model or "llama3.1", **kwargs)
    raise LLMError(f"unknown model provider {provider!r}; expected one of {', '.join(PROVIDERS)}")
