"""OpenAI Chat Completions adapter, also usable with any OpenAI-compatible server.

POST {base_url}/chat/completions with ``Authorization: Bearer $OPENAI_API_KEY``.
JSON mode uses ``response_format`` (``json_schema`` when ``strict_schema=True``, else
``json_object``, which more compatible servers accept). Embeddings use POST /embeddings.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from typing import Any

import httpx
import numpy as np

from .base import BaseLLM, LLMError, Message, post_with_retries

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_EMBED_MODEL = "text-embedding-3-small"


class OpenAILLM(BaseLLM):
    supports_embeddings = True

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        embed_model: str | None = DEFAULT_EMBED_MODEL,
        strict_schema: bool = False,
        timeout: float = 120.0,
        max_retries: int = 4,
        backoff: float = 1.0,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        super().__init__()
        self.model = model or DEFAULT_MODEL
        self.name = f"openai:{self.model}"
        self.api_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY", "")
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.embed_model = embed_model
        self.supports_embeddings = bool(embed_model)
        self.strict_schema = strict_schema
        self.max_retries = max_retries
        self.backoff = backoff
        self._sleep = sleep
        self._client = httpx.Client(timeout=httpx.Timeout(timeout, connect=15.0), transport=transport)

    def _headers(self) -> dict[str, str]:
        h = {"content-type": "application/json"}
        if self.api_key:
            h["authorization"] = f"Bearer {self.api_key}"
        elif self.base_url == DEFAULT_BASE_URL:
            raise LLMError("OPENAI_API_KEY is not set")
        return h

    def _post(self, path: str, body: dict) -> dict:
        kw: dict[str, Any] = {"max_retries": self.max_retries, "backoff": self.backoff, "provider": "openai"}
        if self._sleep is not None:
            kw["sleep"] = self._sleep
        return post_with_retries(self._client, f"{self.base_url}{path}", json_body=body, headers=self._headers(), **kw)

    def _complete_text(
        self,
        messages: list[Message],
        system: str | None,
        temperature: float,
        max_tokens: int,
        json_schema: dict | None,
        json_mode: bool,
    ) -> str:
        msgs: list[dict[str, str]] = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs += [{"role": m["role"], "content": m["content"]} for m in messages]
        body: dict[str, Any] = {
            "model": self.model,
            "messages": msgs,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            if json_schema and self.strict_schema:
                body["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": "noesis_output", "schema": json_schema},
                }
            else:
                body["response_format"] = {"type": "json_object"}
        data = self._post("/chat/completions", body)
        usage = data.get("usage") or {}
        self.usage.input_tokens += int(usage.get("prompt_tokens", 0) or 0)
        self.usage.output_tokens += int(usage.get("completion_tokens", 0) or 0)
        try:
            content = data["choices"][0]["message"].get("content")
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(f"openai: unexpected response: {str(data)[:200]}") from e
        if content is None:
            raise LLMError("openai: empty message content")
        return content

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not self.embed_model:
            raise NotImplementedError("no embed_model configured")
        out: list[list[float]] = []
        texts = list(texts)
        for i in range(0, len(texts), 256):
            data = self._post("/embeddings", {"model": self.embed_model, "input": texts[i : i + 256]})
            rows = sorted(data.get("data", []), key=lambda r: r.get("index", 0))
            out += [r["embedding"] for r in rows]
        return np.asarray(out, dtype=np.float32)
