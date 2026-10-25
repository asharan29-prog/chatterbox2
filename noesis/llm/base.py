"""LLM protocol, shared base class, JSON helpers and HTTP retry logic."""

from __future__ import annotations

import json
import random
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, TypedDict, runtime_checkable

import httpx
import numpy as np

__all__ = [
    "Message",
    "LLM",
    "BaseLLM",
    "LLMError",
    "JSONParseError",
    "Usage",
    "extract_json",
    "validate_schema",
    "schema_instructions",
    "post_with_retries",
]


class Message(TypedDict):
    role: str  # "user" | "assistant"
    content: str


class LLMError(RuntimeError):
    """Provider/transport failure (after retries) or misconfiguration."""


class JSONParseError(LLMError):
    """The model did not return valid JSON matching the schema after all retries."""


@runtime_checkable
class LLM(Protocol):
    """What every Noesis model adapter provides.

    ``complete`` returns ``str`` when ``json_schema is None``; otherwise it returns the parsed
    JSON value (``dict``/``list``). Pass ``json_schema=True`` for "any JSON object", or a JSON
    Schema dict to request (and lightly validate) a specific shape.
    """

    name: str
    supports_embeddings: bool

    def complete(
        self,
        messages: Sequence[Message] | str,
        system: str | None = None,
        json_schema: dict | bool | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> Any: ...

    def embed(self, texts: Sequence[str]) -> np.ndarray: ...


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    json_retries: int = 0
    by_task: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "json_retries": self.json_retries,
            "by_task": dict(self.by_task),
        }


_TASK_RE = re.compile(r"<task>\s*([\w\-]+)\s*</task>")


def task_of(text: str) -> str | None:
    m = _TASK_RE.search(text or "")
    return m.group(1) if m else None


class BaseLLM:
    """Shared implementation: message normalization, JSON mode with repair-retry, usage.

    Subclasses implement :meth:`_complete_text` (and optionally :meth:`embed`).
    """

    name: str = "base"
    supports_embeddings: bool = False
    json_retries: int = 2

    def __init__(self) -> None:
        self.usage = Usage()

    # -- to implement ------------------------------------------------------------
    def _complete_text(
        self,
        messages: list[Message],
        system: str | None,
        temperature: float,
        max_tokens: int,
        json_schema: dict | None,
        json_mode: bool,
    ) -> str:
        raise NotImplementedError

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError(f"{self.name} does not provide embeddings")

    # -- public ------------------------------------------------------------------
    def complete(
        self,
        messages: Sequence[Message] | str,
        system: str | None = None,
        json_schema: dict | bool | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> Any:
        msgs: list[Message] = (
            [{"role": "user", "content": messages}] if isinstance(messages, str) else [dict(m) for m in messages]  # type: ignore[misc]
        )
        task = task_of(msgs[-1]["content"]) if msgs else None
        if task:
            self.usage.by_task[task] = self.usage.by_task.get(task, 0) + 1
        if json_schema is None or json_schema is False:
            self.usage.calls += 1
            return self._complete_text(msgs, system, temperature, max_tokens, None, False)

        schema = json_schema if isinstance(json_schema, dict) else None
        sys_full = (system + "\n\n" if system else "") + schema_instructions(schema)
        attempt_msgs = list(msgs)
        last_err = ""
        for attempt in range(self.json_retries + 1):
            self.usage.calls += 1
            raw = self._complete_text(attempt_msgs, sys_full, temperature, max_tokens, schema, True)
            try:
                obj = extract_json(raw)
                problems = validate_schema(obj, schema) if schema else []
                if problems:
                    raise ValueError("; ".join(problems[:5]))
                return obj
            except ValueError as e:
                last_err = str(e)
                if attempt >= self.json_retries:
                    break
                self.usage.json_retries += 1
                attempt_msgs = list(msgs) + [
                    {"role": "assistant", "content": raw[:4000]},
                    {
                        "role": "user",
                        "content": (
                            f"Your previous reply was not valid JSON for the requested schema ({last_err}). "
                            "Reply again with ONLY the corrected JSON value, no prose, no code fences."
                        ),
                    },
                ]
        raise JSONParseError(f"{self.name}: could not obtain valid JSON: {last_err}")

    def __repr__(self) -> str:  # pragma: no cover
        return f"<{type(self).__name__} {self.name}>"


# ----------------------------------------------------------------------------- JSON helpers
_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.S)


def extract_json(text: str) -> Any:
    """Robustly pull a JSON value out of model output.

    Handles code fences, leading/trailing prose, trailing commas and single surrounding
    quotes. Raises ``ValueError`` if nothing parseable is found.
    """
    if text is None:
        raise ValueError("empty response")
    if not isinstance(text, str):
        return text
    s = text.strip()
    candidates: list[str] = [s]
    candidates += [m.strip() for m in _FENCE_RE.findall(s)]
    # first balanced {...} or [...] region
    for opener, closer in (("{", "}"), ("[", "]")):
        i = s.find(opener)
        while i != -1:
            region = _balanced(s, i, opener, closer)
            if region:
                candidates.append(region)
                break
            i = s.find(opener, i + 1)
    for c in candidates:
        for variant in (c, re.sub(r",\s*([}\]])", r"\1", c)):
            try:
                return json.loads(variant)
            except (json.JSONDecodeError, TypeError):
                continue
    raise ValueError("no valid JSON found in response")


def _balanced(s: str, start: int, opener: str, closer: str) -> str | None:
    depth = 0
    in_str = False
    esc = False
    for j in range(start, len(s)):
        ch = s[j]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return s[start : j + 1]
    return None


_PY_TYPES: dict[str, tuple[type, ...]] = {
    "object": (dict,),
    "array": (list,),
    "string": (str,),
    "number": (int, float),
    "integer": (int,),
    "boolean": (bool,),
    "null": (type(None),),
}


def validate_schema(obj: Any, schema: dict | None, path: str = "$") -> list[str]:
    """Minimal JSON-Schema check (type, required, properties, items, enum, min/max).

    Returns a list of problems (empty = OK). Deliberately lenient: unknown keywords ignored,
    extra properties allowed.
    """
    if not schema:
        return []
    problems: list[str] = []
    t = schema.get("type")
    if t:
        types = t if isinstance(t, list) else [t]
        ok = False
        for tt in types:
            py = _PY_TYPES.get(tt)
            if py and isinstance(obj, py) and not (tt in ("number", "integer") and isinstance(obj, bool)):
                ok = True
        if not ok:
            return [f"{path}: expected {t}, got {type(obj).__name__}"]
    if "enum" in schema and obj not in schema["enum"]:
        problems.append(f"{path}: {obj!r} not in {schema['enum']}")
    if isinstance(obj, (int, float)) and not isinstance(obj, bool):
        if "minimum" in schema and obj < schema["minimum"]:
            problems.append(f"{path}: {obj} < minimum {schema['minimum']}")
        if "maximum" in schema and obj > schema["maximum"]:
            problems.append(f"{path}: {obj} > maximum {schema['maximum']}")
    if isinstance(obj, dict):
        for req in schema.get("required", []):
            if req not in obj:
                problems.append(f"{path}: missing required key '{req}'")
        for k, sub in (schema.get("properties") or {}).items():
            if k in obj:
                problems += validate_schema(obj[k], sub, f"{path}.{k}")
    if isinstance(obj, list) and isinstance(schema.get("items"), dict):
        for i, item in enumerate(obj[:200]):
            problems += validate_schema(item, schema["items"], f"{path}[{i}]")
    return problems


def schema_instructions(schema: dict | None) -> str:
    base = "Respond with a single valid JSON value only: no prose, no markdown code fences."
    if schema:
        return base + "\nThe JSON must conform to this JSON Schema:\n" + json.dumps(schema, separators=(",", ":"))
    return base + " Use a JSON object."


# ----------------------------------------------------------------------------- HTTP
RETRY_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 529})


def post_with_retries(
    client: httpx.Client,
    url: str,
    *,
    json_body: dict,
    headers: dict[str, str] | None = None,
    max_retries: int = 4,
    backoff: float = 1.0,
    max_backoff: float = 30.0,
    sleep: Callable[[float], None] = time.sleep,
    provider: str = "llm",
) -> dict:
    """POST JSON with exponential backoff (+jitter) on 429/5xx and transport errors.

    Honors ``Retry-After`` (seconds). Returns the decoded JSON body or raises
    :class:`LLMError` with the provider's error message.
    """
    last: str = ""
    for attempt in range(max_retries + 1):
        try:
            resp = client.post(url, json=json_body, headers=headers)
        except (httpx.TransportError, httpx.TimeoutException) as e:
            last = f"transport error: {e}"
            if attempt >= max_retries:
                break
            sleep(min(max_backoff, backoff * 2**attempt) * (1 + random.random() * 0.1))
            continue
        if resp.status_code < 300:
            try:
                return resp.json()
            except ValueError as e:
                raise LLMError(f"{provider}: non-JSON response body: {resp.text[:200]}") from e
        detail = resp.text[:500]
        last = f"HTTP {resp.status_code}: {detail}"
        if resp.status_code in RETRY_STATUS and attempt < max_retries:
            delay = min(max_backoff, backoff * 2**attempt)
            ra = resp.headers.get("retry-after")
            if ra:
                try:
                    delay = min(max_backoff, max(delay, float(ra)))
                except ValueError:
                    pass
            sleep(delay * (1 + random.random() * 0.1))
            continue
        break
    raise LLMError(f"{provider}: request failed: {last}")
