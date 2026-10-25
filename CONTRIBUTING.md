# Contributing to Noesis

Thanks for helping. Noesis is a small codebase with a few firm rules. Knowing them up front
saves review time.

## Ground rules

1. **Everything runs offline.** Every test must pass with no network and no API keys, using
   `MockLLM` (`--model mock`) or the heuristic pipeline (`--model none`). Real-provider tests are
   opt-in: mark them `@pytest.mark.integration` and skip them when the key is missing.
2. **Ingested text is data, never instructions.** Untrusted content (documents, tool results,
   model output derived from them) only enters a prompt through `noesis.llm.prompts.wrap_document`
   / `build_prompt`, under `SAFETY_SYSTEM`. See [SECURITY.md](SECURITY.md) and
   [docs/security.md](docs/security.md).
3. **No uncited conclusions.** A belief's confidence comes from `Evidence` rows with provenance
   (source, chunk, span, quote). Model judgments may only nudge it within the documented bounds
   ([docs/belief-math.md](docs/belief-math.md)).
4. **Deterministic with the mock.** Seed any randomness and break ties in a stable order (for
   example by text, not by random id), so `mock` runs are reproducible.
5. **Keep core dependencies light.** The core needs only `pydantic`, `httpx` and `numpy`. Anything
   else goes in an optional extra and is imported lazily.

## Development setup

```bash
git clone https://github.com/asharan29-prog/chatterbox2.git noesis && cd noesis
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev,mcp]"
ruff check .
python -m pytest -q
python -m benchmarks.run --model mock --no-write   # offline baseline; see docs/benchmarks.md
```

Python 3.10 to 3.13 are supported and tested in CI. Do not use syntax or stdlib APIs newer than
3.10. For example, avoid `tomllib`, `typing.Self`, `ExceptionGroup`/`except*`, `datetime.UTC`
and PEP 695 generics, and do not nest same-quote strings inside f-strings (PEP 701).

## Where things live

| Area | Module | Docs |
|---|---|---|
| Data model | `noesis/types.py` | [architecture](docs/architecture.md) |
| Storage (SQLite, FTS5, vectors) | `noesis/store.py` | [architecture](docs/architecture.md) |
| Ingestion and reliability priors | `noesis/ingest/` | [architecture](docs/architecture.md) |
| Understanding map | `noesis/understand.py`, `noesis/nlp.py`, `noesis/embed.py` | [architecture](docs/architecture.md) |
| Beliefs | `noesis/beliefs.py` | [belief math](docs/belief-math.md) |
| Reasoning loop | `noesis/reason/` | [reasoning loop](docs/reasoning-loop.md) |
| Model adapters | `noesis/llm/` | [adapters](docs/adapters.md) |
| MCP server | `noesis/mcp_server.py` | [mcp](docs/mcp.md) |
| CLI and reports | `noesis/cli.py`, `noesis/report.py` | [README](README.md) |
| Benchmarks | `benchmarks/` | [benchmarks](docs/benchmarks.md) |

## Pull requests

- Keep PRs focused, and add or adjust tests with every behavior change.
- If you touch understanding, beliefs or reasoning, run the benchmark with `--model mock` and
  mention any metric that moved. Re-commit `benchmarks/results/mock.json`, `mock.md`,
  `none.json` and `none.md` when the offline baselines change on purpose.
- Add a line to `CHANGELOG.md` under **Unreleased**.
- By contributing you agree that your contribution is licensed under the MIT License.

## Adding a model provider

See [docs/adapters.md](docs/adapters.md). In short: subclass `BaseLLM`, implement
`_complete_text` (and `embed` if the provider has embeddings), register the prefix in
`noesis.llm.get_llm`, and add mocked-transport unit tests plus an opt-in integration test.

## Code of conduct

Participation is governed by our [Code of Conduct](CODE_OF_CONDUCT.md).
