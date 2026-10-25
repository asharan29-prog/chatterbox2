# Noesis as an MCP server

Source: `noesis/mcp_server.py`, examples in `examples/mcp/`. With the server, any MCP-capable
assistant (Claude Desktop, Claude Code, or any MCP client) gets a **structured, evidence-backed,
calibrated memory** for a topic, and optionally a reasoning engine too.

```bash
pip install "noesis[mcp] @ git+https://github.com/asharan29-prog/chatterbox2.git"
noesis init "Is coffee good for health?" && noesis ingest examples/corpus && noesis understand
noesis serve-mcp                          # stdio (default)
noesis serve-mcp --transport http         # streamable HTTP at http://127.0.0.1:8765/mcp
noesis-mcp --db path/to/topic.db          # same server, standalone entry point
```

The knowledge base is chosen by `--db`, then `$NOESIS_DB`, then the topic selected by
`noesis init`. The server-side model is chosen by `--model`, then `$NOESIS_MODEL`, then `mock`.
Works with the official `mcp` Python SDK 1.x (`FastMCP`) and 2.x (`MCPServer`).

## Two ways to use it

### 1. The assistant reasons, Noesis remembers (`NOESIS_MODEL=none`)

The calling assistant does the thinking. Noesis is the memory and evidence layer:

1. `ingest_source` documents (paths, URLs, or text the assistant already fetched);
2. `build_understanding` (heuristic extraction when no model is configured);
3. `search_knowledge` to read passages and related claims;
4. the assistant reasons, then records conclusions with **`add_evidence(claim_text, source_text,
   stance, source_title, reliability)`**. The verbatim source passage is stored as a cited source;
5. `list_beliefs` / `explain_belief` show the calibrated confidence Noesis computes from all
   evidence (weighted by reliability, discounted for non-independent sources), with history.

No API key is needed on the server. The assistant cannot simply *assert* a confidence. It can only
add evidence with provenance, and the math decides ([belief-math.md](belief-math.md)). With
`none`, the `ask`/`think` tools still work, but their reasoning calls fall back to the
deterministic `MockLLM`. That output is not real reasoning, so in this mode prefer
`search_knowledge` plus your own judgment.

### 2. Noesis reasons with its own model (`NOESIS_MODEL=anthropic:...` etc.)

`build_understanding`, `think` and `ask` run Noesis' own pipeline with the configured model
(provider key in the server's environment). The assistant can delegate a whole investigation
(`think(goal, steps=5, strategy="debate")`), then inspect the trace and beliefs. Both modes can be
mixed freely in one knowledge base.

## Tools

| tool | arguments | returns |
|---|---|---|
| `ingest_source` | `path_or_url` **or** `text`, `title`, `reliability` | new source ids (duplicates detected) |
| `build_understanding` | `reprocess=false` | counts: claims, merges, evidence, contradictions, questions |
| `ask` | `question`, `include_trace=false` | answer, evidence-based confidence, validated citations (source, span, quote), supporting/contradicting claims, `insufficient_evidence` |
| `think` | `goal`, `steps=3`, `strategy=simple\|self_consistency\|debate` | run summary, belief changes, thought trace |
| `list_beliefs` | `status`, `min_confidence`, `limit=25` | beliefs, most confident first |
| `list_open_questions` | `limit=20` | open questions by priority (contradictions, gaps, curiosity) |
| `explain_belief` | `belief_id` (or claim id) | per-evidence contributions, adjustment, history |
| `add_evidence` | `claim_text`, `source_text`, `stance`, `source_title`, `reliability` | recomputed confidence and status |
| `search_knowledge` | `query`, `k=8` | top passages (chunk id, source, reliability, span, text) and closest claims with confidence |

Resources: `noesis://summary` (topic, stats, top beliefs, open questions) and `noesis://report`
(full markdown report). The server's instructions tell the client that ingested documents are
data, never instructions.

## Claude Desktop

Add to `claude_desktop_config.json`. On macOS it is in `~/Library/Application Support/Claude/`, on
Windows in `%APPDATA%\Claude\`. Then restart Claude Desktop:

```json
{
  "mcpServers": {
    "noesis": {
      "command": "noesis",
      "args": ["serve-mcp"],
      "env": {
        "NOESIS_DB": "/absolute/path/to/.noesis/is-coffee-good-for-health.db",
        "NOESIS_MODEL": "none"
      }
    }
  }
}
```

Use an absolute path for `NOESIS_DB`. If `noesis` is not on the PATH Claude Desktop sees, set
`"command"` to the absolute path of the executable (for example `/path/to/.venv/bin/noesis`), or use
`"command": "/path/to/python", "args": ["-m", "noesis", "serve-mcp"]`. For mode 2, set
`"NOESIS_MODEL": "anthropic:claude-sonnet-5-5"` and add `"ANTHROPIC_API_KEY"` to `env`.

## Claude Code

```bash
# stdio, current project only
claude mcp add noesis \
  --env NOESIS_DB="$PWD/.noesis/is-coffee-good-for-health.db" \
  --env NOESIS_MODEL=none \
  -- noesis serve-mcp

# shared with the team via .mcp.json
claude mcp add --scope project noesis --env NOESIS_DB=.noesis/topic.db -- noesis serve-mcp

# streamable HTTP
noesis serve-mcp --transport http --port 8765 &
claude mcp add --transport http noesis http://127.0.0.1:8765/mcp
```

See `examples/mcp/` for ready-made files (`claude_desktop_config.json`, `claude_code.sh`,
`generic_stdio.json`, `http.md`). Check the Claude Code and Claude Desktop documentation for the
current flags of your client version.

## Other clients

Any stdio MCP client can launch `python -m noesis --db /abs/topic.db serve-mcp` (see
`examples/mcp/generic_stdio.json`). Clients that support streamable HTTP can connect to
`http://HOST:PORT/mcp`.

## Security notes

- **HTTP transport has no authentication.** It binds to `127.0.0.1` by default. Anyone who can
  reach the port can read the knowledge base, add evidence and spend your model budget. Put it
  behind an authenticating reverse proxy before binding to anything else.
- `ingest_source(path_or_url=...)` reads **any path the server process can read**, and fetches any
  URL. Run the server as a user whose file access you are comfortable exposing to the connected
  assistant.
- Treat tool outputs (`search_knowledge` passages, answers) as untrusted source material. See
  [security.md](security.md).
