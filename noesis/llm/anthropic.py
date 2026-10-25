"""Anthropic Messages API adapter (plain HTTP via httpx; no SDK needed).

Docs: POST https://api.anthropic.com/v1/messages with headers ``x-api-key``,
``anthropic-version`` and a body of ``model``, ``max_tokens``, ``system``, ``messages``.
The key is read from ``ANTHROPIC_API_KEY`` (or passed explicitly). Anthropic has no
embeddings endpoint, so :meth:`embed` raises and Noesis falls back to local vectors.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

import httpx

from .base import BaseLLM, LLMError, Message, post_with_retries

DEFAULT_MODEL = "claude-sonnet-5-5"
API_URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"


class AnthropicLLM(BaseLLM):
    supports_embeddings = False

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 4,
        backoff: float = 1.0,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        super().__init__()
        self.model = model or DEFAULT_MODEL
        self.name = f"anthropic:{self.model}"
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        # NOTE: deliberately NOT reading ANTHROPIC_BASE_URL implicitly; pass base_url or set
        # NOESIS_ANTHROPIC_BASE_URL to target a proxy/compatible endpoint.
        self.url = (base_url or os.environ.get("NOESIS_ANTHROPIC_BASE_URL") or "").rstrip("/")
        self.url = f"{self.url}/v1/messages" if self.url else API_URL
        self.max_retries = max_retries
        self.backoff = backoff
        self._sleep = sleep
        self._client = httpx.Client(timeout=httpx.Timeout(timeout, connect=15.0), transport=transport)

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            raise LLMError("ANTHROPIC_API_KEY is not set")
        return {
            "x-api-key": self.api_key,
            "anthropic-version": API_VERSION,
            "content-type": "application/json",
        }

    def _complete_text(
        self,
        messages: list[Message],
        system: str | None,
        temperature: float,
        max_tokens: int,
        json_schema: dict | None,
        json_mode: bool,
    ) -> str:
        body: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": m["role"], "content": m["content"]} for m in messages],
        }
        if system:
            body["system"] = system
        kw: dict[str, Any] = {"max_retries": self.max_retries, "backoff": self.backoff, "provider": "anthropic"}
        if self._sleep is not None:
            kw["sleep"] = self._sleep
        data = post_with_retries(self._client, self.url, json_body=body, headers=self._headers(), **kw)
        usage = data.get("usage") or {}
        self.usage.input_tokens += int(usage.get("input_tokens", 0) or 0)
        self.usage.output_tokens += int(usage.get("output_tokens", 0) or 0)
        parts = [b.get("text", "") for b in data.get("content", []) if b.get("type") == "text"]
        if not parts:
            raise LLMError(f"anthropic: response had no text content: {str(data)[:200]}")
        return "".join(parts)
