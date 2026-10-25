"""Ollama adapter (local models). POST {host}/api/chat and /api/embed.

Host from ``OLLAMA_HOST`` (default ``http://localhost:11434``). JSON mode passes
``format`` = the JSON schema (Ollama >= 0.5 structured outputs) or ``"json"``.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from typing import Any

import httpx
import numpy as np

from .base import BaseLLM, LLMError, Message, post_with_retries

DEFAULT_MODEL = "llama3.1"
DEFAULT_HOST = "http://localhost:11434"


def _normalize_host(host: str) -> str:
    host = host.strip().rstrip("/")
    if not host.startswith(("http://", "https://")):
        host = "http://" + host
    return host


class OllamaLLM(BaseLLM):
    supports_embeddings = True

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        host: str | None = None,
        embed_model: str | None = None,
        timeout: float = 300.0,
        max_retries: int = 2,
        backoff: float = 1.0,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        super().__init__()
        self.model = model or DEFAULT_MODEL
        self.name = f"ollama:{self.model}"
        self.host = _normalize_host(host or os.environ.get("OLLAMA_HOST") or DEFAULT_HOST)
        self.embed_model = embed_model or self.model
        self.max_retries = max_retries
        self.backoff = backoff
        self._sleep = sleep
        self._client = httpx.Client(timeout=httpx.Timeout(timeout, connect=10.0), transport=transport)

    def _post(self, path: str, body: dict) -> dict:
        kw: dict[str, Any] = {"max_retries": self.max_retries, "backoff": self.backoff, "provider": "ollama"}
        if self._sleep is not None:
            kw["sleep"] = self._sleep
        return post_with_retries(self._client, f"{self.host}{path}", json_body=body, **kw)

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
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if json_mode:
            body["format"] = json_schema if json_schema else "json"
        data = self._post("/api/chat", body)
        self.usage.input_tokens += int(data.get("prompt_eval_count", 0) or 0)
        self.usage.output_tokens += int(data.get("eval_count", 0) or 0)
        content = (data.get("message") or {}).get("content")
        if content is None:
            raise LLMError(f"ollama: unexpected response: {str(data)[:200]}")
        return content

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        data = self._post("/api/embed", {"model": self.embed_model, "input": list(texts)})
        emb = data.get("embeddings")
        if emb is None:
            raise LLMError(f"ollama: no embeddings in response: {str(data)[:200]}")
        return np.asarray(emb, dtype=np.float32)
