# Security Policy

## Supported versions

Noesis is pre-1.0. Security fixes go to the latest release and to `main`.

| Version | Supported |
|---|---|
| 0.1.x | yes |

## Reporting a vulnerability

Please **do not open a public issue**. Use GitHub's private vulnerability reporting instead:
<https://github.com/asharan29-prog/chatterbox2/security/advisories/new>.

Include a minimal reproduction, ideally with `--model mock` so it is deterministic, and describe
the impact. We aim to acknowledge reports within 7 days.

In scope, among others:

- **prompt-injection bypasses**: ingested text that changes beliefs, answers, questions or tool
  behavior *as an instruction*, rather than being weighed as evidence;
- citation spoofing, where a citation passes validation but does not point at a stored chunk that
  was shown to the model;
- path traversal or arbitrary file reads and writes through ingestion, export/import or the MCP
  tools;
- leakage of API keys (for example into logs, reports, exports or the knowledge base).

## Threat model in brief

The full discussion is in [docs/security.md](docs/security.md).

- **Ingested text is untrusted data.** Documents, URLs and tool results are wrapped in
  `<document>` tags (with tag-like content escaped) under a safety system prompt that says their
  content is never instructions. On the output side, instruction-like sentences are filtered out
  of hypotheses, questions and answers. Citations are checked against the chunks actually shown to
  the model. Belief confidence is computed from evidence weighted by source reliability, so a
  low-reliability source cannot talk its way to high confidence. The bundled corpus contains a
  deliberate injection (`examples/corpus/06_wellness_blog.md`), and the benchmark checks it
  ([docs/benchmarks.md](docs/benchmarks.md)).
  These are mitigations, not guarantees. A sufficiently capable model can still be influenced by
  what it reads, so treat answers about adversarial corpora with care.
- **The MCP HTTP transport has no authentication.** `noesis serve-mcp --transport http` binds to
  `127.0.0.1` by default. Anyone who can reach the port can read and modify the knowledge base and
  spend your model budget. Do not bind it to a public interface without an authenticating reverse
  proxy.
- **Data stays local.** The knowledge base is a local SQLite file. Nothing leaves your machine
  except the requests you configure: calls to the model provider you choose (document excerpts are
  sent to it), URLs you ask Noesis to fetch, and tools you plug in. With `--model mock` or
  `--model none`, Noesis makes no network calls apart from URL ingestion you request.
- **Secrets** are read from environment variables (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, ...) and
  are never written to the database or reports.
