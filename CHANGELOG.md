# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- **Noesis Research Agent** (`noesis/agent/`): a separate, internet-connected agent on top of the
  engine. It plans sub-questions, searches, selects diverse and reputable sources, fetches them,
  ingests them with domain reliability priors, builds the understanding, and runs `Mind.think`
  with a `web_search` tool. It writes a cited markdown brief (confidence per conclusion, a
  sources table, "what we could not verify") plus the HTML report. `watch()` gives an
  incremental refresh.
- Search providers: DuckDuckGo HTML (no key, fragile), Brave, Tavily and SearXNG, plus a bundled
  offline mock web.
- Guarded fetcher: robots.txt (RFC 9309), an SSRF guard re-checked on every redirect,
  timeout/size/content-type limits, no executables, no JavaScript, readability-style extraction,
  per-host rate limiting, a disk cache in `.noesis/webcache`, and injection flagging.
- `DomainPolicy` with allow/deny lists, trust overrides and content-farm/scholarly priors.
- Interfaces: `noesis research` subcommand, `noesis-agent` console script, MCP tools
  `web_search` and `web_research`, the `agent` extra, `examples/agent/research_agent.py`, and
  `docs/research-agent.md`.

## [0.1.0] - 2026-10-25

First public release.

### Added

- **Knowledge store**: one SQLite file per topic, with CRUD for sources, chunks, concepts, claims,
  evidence, beliefs (with full history), questions, thoughts and runs. Also FTS5 full-text search,
  numpy vector search and JSON export/import.
- **Ingestion**: text/markdown (with front matter), HTML and URLs, PDF (optional), code, JSON/CSV,
  and EmotionLens `packets.jsonl` behavior packets. Chunking keeps character spans. Each source gets
  a reliability prior (from declared metadata, source type or domain rules) and an independence
  group.
- **Understanding map** (`understand`): claim and concept extraction, cross-source claim merging,
  supports/contradicts evidence linking, contradiction and gap detection that produces questions,
  and RAPTOR-style concept/topic summaries. Works with a model or fully heuristically.
- **Calibrated beliefs**: log-odds aggregation of evidence weighted by strength and source
  reliability, with duplicate-source discounting, status thresholds and an audit trail.
- **Reasoning engine** (`Mind`): `think()` runs an observe, select-question, hypothesize, gather,
  test, judge, update, reflect loop with budgets and stop rules. It offers `simple`,
  `self_consistency` and `debate` strategies, `ask()` (grounded answer with validated citations,
  evidence-based confidence and an insufficient-evidence flag) and `explain()`. Every step is kept
  as a persisted thought trace.
- **Model adapters**: Anthropic Messages API, OpenAI Chat Completions (plus any OpenAI-compatible
  `base_url`), Ollama, a deterministic `MockLLM` and `none` (heuristics only). Robust JSON
  extraction with repair retries.
- **Prompt-injection defenses**: document wrapping with escaping, a safety system prompt, an output
  filter for instruction-like text and citation validation.
- **Reports**: markdown and self-contained HTML with belief table, contradictions, open questions,
  reasoning timeline and an inline SVG concept graph.
- **CLI**: `noesis init / ingest / understand / think / ask / beliefs / questions / explain / report /
  export / import / serve-mcp / doctor`.
- **MCP server**: 9 tools and 2 resources over stdio or streamable HTTP. It supports both the
  "assistant reasons, Noesis remembers" and "Noesis reasons with its own model" modes.
- **Benchmarks** (`python -m benchmarks.run`): QA correctness, citation faithfulness, abstention,
  contradiction recall, calibration (Brier/ECE) and prompt-injection resistance on the bundled
  corpus, with committed offline baselines for `mock` and `none`.
- Docs, CI (ruff and pytest on Python 3.10 to 3.13), and community files.

[Unreleased]: https://github.com/asharan29-prog/chatterbox2/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/asharan29-prog/chatterbox2/releases/tag/v0.1.0
