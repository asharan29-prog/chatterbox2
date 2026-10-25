## What and why

<!-- What does this change, and what problem does it solve? Link issues with "Fixes #123". -->

## How it was tested

- [ ] `ruff check .`
- [ ] `python -m pytest -q` (offline, MockLLM)
- [ ] `python -m benchmarks.run --model mock --no-write` if the change touches understanding, beliefs or reasoning (paste any metric that moved)
- [ ] Opt-in integration tests with a real provider, if adapters changed

## Checklist

- [ ] Ingested text is still treated as data: untrusted content only enters prompts through `wrap_document` / `build_prompt`
- [ ] New beliefs/answers keep provenance (no uncited conclusions)
- [ ] Docs updated (`README.md`, `docs/`) and a line added to `CHANGELOG.md` under *Unreleased*
- [ ] No secrets, private documents or `.noesis/` databases committed
