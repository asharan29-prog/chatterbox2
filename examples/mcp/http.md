# Streamable HTTP

    noesis --db topic.noesis.db serve-mcp --transport http --host 127.0.0.1 --port 8765

Endpoint: `http://127.0.0.1:8765/mcp`. Binds to localhost by default; it has no
authentication, so put it behind a reverse proxy with auth before exposing it.
