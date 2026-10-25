#!/usr/bin/env sh
# Register Noesis with Claude Code (stdio). Run from the project that owns the knowledge base.
#   pip install "noesis[mcp] @ git+https://github.com/asharan29-prog/chatterbox2.git"
#   noesis init "Is coffee good for health?" && noesis ingest examples/corpus && noesis understand
#
# NOESIS_MODEL=none : Claude does the reasoning itself and uses Noesis as structured,
#                     calibrated memory (ingest_source / search_knowledge / add_evidence).
# NOESIS_MODEL=anthropic:claude-sonnet-5-5 (plus ANTHROPIC_API_KEY): the Noesis server also
#                     runs its own understanding/think/ask loop with that model.
claude mcp add noesis \
  --env NOESIS_DB="$PWD/.noesis/is-coffee-good-for-health.db" \
  --env NOESIS_MODEL=none \
  -- noesis serve-mcp

# Project-scoped (writes .mcp.json so the team shares it):
#   claude mcp add --scope project noesis --env NOESIS_DB=.noesis/topic.db -- noesis serve-mcp
# Streamable HTTP instead of stdio:
#   noesis serve-mcp --transport http --port 8765 &
#   claude mcp add --transport http noesis http://127.0.0.1:8765/mcp
