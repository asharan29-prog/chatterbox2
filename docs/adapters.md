# Model adapters

Source: `noesis/llm/`. Noesis talks to models through one small protocol, so any chat model can
drive understanding and reasoning.

## Built-in providers

| spec | adapter | configuration |
|---|---|---|
| `mock` | `MockLLM` | Offline and deterministic. Heuristic JSON for every task, or scripted responses. The default. |
| `none` (`offline`, `heuristic`) | none (`get_llm` returns `None`) | `understand` uses the pure heuristic pipeline. `Mind` falls back to `MockLLM` for its reasoning calls. |
| `anthropic:<model>` (`claude:`) | `AnthropicLLM` | Messages API over httpx (no SDK needed). Key from `ANTHROPIC_API_KEY`. Optional base URL via `NOESIS_ANTHROPIC_BASE_URL` or `base_url=` (`ANTHROPIC_BASE_URL` is deliberately *not* read implicitly). Default model `claude-sonnet-5-5`. |
| `openai:<model>` (`openai-compatible:`) | `OpenAILLM` | Chat Completions plus Embeddings. Key from `OPENAI_API_KEY`. Base URL from `OPENAI_BASE_URL` or `base_url=`, which works with any OpenAI-compatible server (vLLM, LM Studio, llama.cpp server, OpenRouter, ...). Default model `gpt-4o-mini`. |
| `ollama:<model>` | `OllamaLLM` | `POST /api/chat` and `/api/embed` on `OLLAMA_HOST` (default `http://localhost:11434`). JSON mode passes the schema as `format`. Default model `llama3.1`. |

Select a model with `--model SPEC` on the CLI, `$NOESIS_MODEL`, or `get_llm(spec, **kwargs)` in
Python. Extra kwargs (`api_key=`, `base_url=`, `timeout=`, `max_retries=`, ...) go to the adapter
constructor.

```python
from noesis import get_llm
llm = get_llm("openai:qwen2.5-14b-instruct", base_url="http://localhost:8000/v1", api_key="local")
```

You can also pass any object that satisfies the protocol straight to `understand(store, llm)` /
`Mind(store, llm)`. Registering it in `get_llm` is optional.

## The protocol

```python
class LLM(Protocol):
    name: str
    supports_embeddings: bool

    def complete(self, messages: Sequence[Message] | str, system: str | None = None,
                 json_schema: dict | bool | None = None,
                 temperature: float = 0.0, max_tokens: int = 1024) -> Any: ...
    def embed(self, texts: Sequence[str]) -> np.ndarray: ...   # optional
```

- With `json_schema=None`, `complete` returns `str`.
- With `json_schema=True` (any JSON object) or a JSON Schema dict, it returns the **parsed**
  `dict`/`list`.
- `embed` returns a float32 array of shape `(n, dim)`. If `supports_embeddings` is false (or the
  model is the mock), Noesis uses its local hashing embedder instead.

## Writing an adapter

Subclass `BaseLLM` and implement one method. `BaseLLM.complete` already handles message
normalization, usage accounting per task, JSON extraction from messy output (code fences, prose
around the JSON, trailing commas), light schema validation, and **repair retries** (it sends the
bad output back with the error, up to `json_retries` = 2 times, then raises `JSONParseError`).

```python
from noesis.llm.base import BaseLLM, LLMError, Message, post_with_retries
import httpx, os

class AcmeLLM(BaseLLM):
    supports_embeddings = False

    def __init__(self, model: str = "acme-large", *, api_key: str | None = None, timeout: float = 120.0):
        super().__init__()
        self.model = model
        self.name = f"acme:{model}"
        self.api_key = api_key or os.environ.get("ACME_API_KEY", "")
        self._client = httpx.Client(timeout=timeout)

    def _complete_text(self, messages: list[Message], system: str | None, temperature: float,
                       max_tokens: int, json_schema: dict | None, json_mode: bool) -> str:
        if not self.api_key:
            raise LLMError("acme: set ACME_API_KEY")
        body = {"model": self.model, "system": system or "", "messages": messages,
                "temperature": temperature, "max_tokens": max_tokens}
        if json_mode:
            body["response_format"] = {"type": "json_object"}   # if the provider supports it
        data = post_with_retries(self._client, "https://api.acme.example/v1/chat",
                                 json_body=body, headers={"Authorization": f"Bearer {self.api_key}"},
                                 provider="acme")
        self.usage.input_tokens += data.get("usage", {}).get("input_tokens", 0)
        self.usage.output_tokens += data.get("usage", {}).get("output_tokens", 0)
        return data["output_text"]
```

Notes:

- **Always pass `system` through.** It carries `SAFETY_SYSTEM` (ingested text is data, not
  instructions) and the JSON-schema instructions. Do not drop or rewrite it.
- `post_with_retries` retries transient failures (connection errors, 429, 5xx) with backoff and
  raises `LLMError` with a readable message otherwise. Raise `LLMError` for configuration
  problems too. Core code catches `LLMError` and falls back to heuristics where it can.
- `json_mode=True` means "the caller wants JSON". Use the provider's native JSON or structured
  output mode if it has one. `BaseLLM` validates and repairs either way.
- Never log or persist API keys.

### Register the prefix (optional)

Add a branch to `noesis.llm.get_llm` so `--model acme:acme-large` works from the CLI and MCP
server, and add the name to `PROVIDERS`.

### Test it offline

Adapters take an `httpx` `transport=` argument (see `OllamaLLM`/`OpenAILLM`) so tests can use
`httpx.MockTransport` to assert on the request body and feed canned responses, including
malformed JSON to exercise repair retries. Put live tests behind `@pytest.mark.integration`, and
skip them unless the key is set:

```python
import os, pytest
pytestmark = pytest.mark.integration

@pytest.mark.skipif(not os.environ.get("ACME_API_KEY"), reason="no ACME_API_KEY")
def test_live_roundtrip():
    from noesis import get_llm
    out = get_llm("acme:acme-large").complete("Reply with JSON {\"ok\": true}", json_schema=True)
    assert out["ok"] is True
```

Then run the benchmark with it: `python -m benchmarks.run --model acme:acme-large --llm-judge`.

## MockLLM

`MockLLM` recognizes every prompt by the `<task>name</task>` tag that `build_prompt` adds, and
returns plausible JSON derived from the input (sentence-split claims, lexical stance, extractive
answers). That is how the entire pipeline, the tests and the benchmarks run offline. For fixtures:

```python
from noesis import MockLLM
llm = MockLLM(script={"judge": {"verdict": "true", "confidence": 0.9, "citations": [], "rationale": "x"}})
llm = MockLLM(script=["not json at all", {"claims": []}])   # consumed in order; tests repair retries
llm.register_task("my_task", lambda prompt, schema: {"ok": True})
```

Unscripted calls fall back to the heuristics (`fallback="heuristic"`), or raise if you pass
`fallback="error"`. Every call is recorded in `llm.calls`.

Mock output is **not** reasoning. It is a deterministic stand-in so that structure, provenance,
budgets and guardrails can be tested without a model.
