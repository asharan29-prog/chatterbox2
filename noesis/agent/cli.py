"""Command line for the Research Agent: ``noesis research ...`` and the ``noesis-agent`` script.

``noesis-agent "question"`` is the friendly form (progress on stderr, brief on stdout or ``-o``);
``noesis research "question"`` is the same command inside the main ``noesis`` CLI.

Database: ``--db`` > ``$NOESIS_DB`` > ``./.noesis/research-<slug>.db`` (one per question, so
re-running or ``--watch`` builds on what was already learned).
Search: ``--search`` > ``$NOESIS_SEARCH`` > ``mock`` when the model is ``mock`` (offline demo web)
> ``ddg``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

EXIT_OK, EXIT_ERROR, EXIT_USAGE = 0, 1, 2

AGENT_EPILOG = """\
examples:
  noesis-agent "Does intermittent fasting improve longevity?" -o brief.md
  noesis-agent "Is nuclear power safer than wind?" --search brave --depth 3 --max-sources 12
  noesis-agent "Coffee and health" --deny-domain example.com --trust nih.gov=0.9
  noesis-agent "Coffee and health" --watch          # later: only new pages, show belief changes
  noesis-agent --model mock "Is coffee good for health?"   # offline demo against a mocked web

search providers (--search or $NOESIS_SEARCH):
  ddg      DuckDuckGo HTML, no key (fragile scraping; may be rate-limited)
  brave    Brave Search API   (BRAVE_API_KEY)
  tavily   Tavily API         (TAVILY_API_KEY)
  searxng  your SearXNG       (SEARXNG_URL, JSON format enabled)
  mock     bundled offline fixture web (default when --model mock)
privacy: your queries are sent to the chosen search provider and pages are fetched from
  their sites (robots.txt respected). Respect each site's terms of use.
"""


def add_research_arguments(sp: argparse.ArgumentParser) -> None:
    sp.add_argument("question", help="the research question or topic")
    sp.add_argument("--depth", type=int, default=2, help="1 (quick) .. 5 (thorough); scales sub-questions and reasoning steps (default 2)")
    sp.add_argument("--search", choices=["ddg", "brave", "tavily", "searxng", "mock"], default=None,
                    help="search provider (default: $NOESIS_SEARCH, 'mock' with --model mock, else ddg)")
    sp.add_argument("--max-sources", type=int, default=8, help="pages to fetch and ingest up front (default 8)")
    sp.add_argument("--per-domain", type=int, default=1, help="max sources per domain (default 1, for diversity)")
    sp.add_argument("--allow-domain", action="append", default=[], metavar="DOMAIN",
                    help="only use these domains (repeatable; subdomains included)")
    sp.add_argument("--deny-domain", action="append", default=[], metavar="DOMAIN", help="never use these domains (repeatable)")
    sp.add_argument("--trust", action="append", default=[], metavar="DOMAIN=R", help="reliability override, e.g. nih.gov=0.9 (repeatable)")
    sp.add_argument("--min-reliability", type=float, default=0.35, help="skip sources with a lower prior (default 0.35)")
    sp.add_argument("--steps", type=int, default=None, help="reasoning steps (default 2 x depth)")
    sp.add_argument("--strategy", default="simple", choices=["simple", "self_consistency", "debate"])
    sp.add_argument("--max-llm-calls", type=int, default=None)
    sp.add_argument("--seconds", type=float, default=None, help="wall-clock limit for the reasoning loop")
    sp.add_argument("-o", "--output", metavar="FILE", help="write the markdown brief to FILE (+ FILE-stem.report.html)")
    sp.add_argument("--no-html", action="store_true", help="with -o: do not write the HTML report")
    sp.add_argument("--watch", action="store_true", help="incremental refresh: re-search, ingest only new URLs, show belief changes")
    sp.add_argument("--no-cache", action="store_true", help="do not read/write the web cache (.noesis/webcache)")
    sp.add_argument("--cache-dir", metavar="DIR", help="web cache directory (default .noesis/webcache or $NOESIS_WEBCACHE)")
    sp.add_argument("--json", action="store_true", help="print the full result as JSON instead of the brief")
    sp.add_argument("--quiet", action="store_true", help="no progress output on stderr")


def _slug(text: str) -> str:
    import re

    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60] or "topic"


def _parse_trust(items: Sequence[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for it in items:
        dom, sep, val = it.partition("=")
        try:
            r = float(val)
        except ValueError:
            r = -1.0
        if not sep or not dom.strip() or not 0.0 <= r <= 1.0:
            raise ValueError(f"--trust expects DOMAIN=R with 0<=R<=1, got {it!r}")
        out[dom.strip().lower()] = r
    return out


def search_spec(arg: str | None, model: str) -> str:
    if arg:
        return arg
    if os.environ.get("NOESIS_SEARCH"):
        return os.environ["NOESIS_SEARCH"].strip().lower()
    return "mock" if model.split(":")[0].strip().lower() == "mock" else "ddg"


def run_research(args: argparse.Namespace, *, prog: str = "noesis research") -> int:
    from ..llm import LLMError, get_llm
    from ..store import KnowledgeStore
    from .research import ResearchAgent, ResearchBudget
    from .web import DomainPolicy, SearchError

    model = (getattr(args, "model", None) or os.environ.get("NOESIS_MODEL") or "mock").strip()
    if not args.question.strip():
        print(f"{prog}: error: empty question", file=sys.stderr)
        return EXIT_USAGE
    if not 1 <= args.depth <= 5:
        print(f"{prog}: error: --depth must be between 1 and 5", file=sys.stderr)
        return EXIT_USAGE
    try:
        trust = _parse_trust(args.trust)
        llm = get_llm(model)
    except (ValueError, LLMError) as e:
        print(f"{prog}: error: {e}", file=sys.stderr)
        return EXIT_USAGE
    search = search_spec(args.search, model)

    def say(msg: str) -> None:
        if not args.quiet:
            print(f"· {msg}", file=sys.stderr, flush=True)

    if search == "mock":
        say("using the bundled OFFLINE mock web (demo/test data, not the internet); pass --search ddg|brave|tavily|searxng for real search")
    db = Path(getattr(args, "db", None) or os.environ.get("NOESIS_DB") or Path(".noesis") / f"research-{_slug(args.question)}.db")
    db.parent.mkdir(parents=True, exist_ok=True)
    policy = DomainPolicy(allow=list(args.allow_domain), deny=list(args.deny_domain), trust=trust)
    budget = ResearchBudget(max_sources=max(1, args.max_sources), per_domain=max(1, args.per_domain),
                            min_reliability=args.min_reliability, think_steps=args.steps,
                            max_llm_calls=args.max_llm_calls, seconds=args.seconds)
    store = KnowledgeStore(db)
    try:
        try:
            agent = ResearchAgent(store, llm, search=search, budget=budget, policy=policy, strategy=args.strategy,
                                  on_event=say, cache_dir=args.cache_dir, use_cache=not args.no_cache)
        except SearchError as e:
            print(f"{prog}: error: {e}", file=sys.stderr)
            return EXIT_USAGE
        say(f"knowledge base: {db}")
        if args.watch:
            w = agent.watch(args.question, depth=args.depth)
            text = w.markdown
            payload: Any = {"topic": w.topic, "new_sources": [vars(r) for r in w.new_sources], "changes": w.changes, "markdown": w.markdown}
        else:
            res = agent.research(args.question, depth=args.depth)
            text = res.markdown
            payload = res.to_dict()
            if args.output:
                paths = res.write(args.output, html=not args.no_html)
                say("wrote " + ", ".join(str(p) for p in paths))
        if args.watch and args.output:
            Path(args.output).write_text(text, encoding="utf-8")
            say(f"wrote {args.output}")
        if args.json:
            print(json.dumps(payload, indent=2, default=str, ensure_ascii=False))
        elif not args.output:
            print(text)
        agent.web.close()
    finally:
        store.close()
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="noesis-agent",
        description="Noesis Research Agent: searches the web, reads reputable sources, builds an evidence-backed "
                    "understanding with Noesis and writes a cited research brief.",
        epilog=AGENT_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    from .. import __version__

    p.add_argument("--version", action="version", version=f"noesis-agent {__version__}")
    p.add_argument("--model", metavar="SPEC", help="model (default $NOESIS_MODEL or mock): anthropic:<m>, openai:<m>, ollama:<m>, mock")
    p.add_argument("--db", metavar="PATH", help="knowledge base (default ./.noesis/research-<question>.db)")
    add_research_arguments(p)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run_research(args, prog="noesis-agent")
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except Exception as e:  # noqa: BLE001 - readable error instead of a traceback
        if os.environ.get("NOESIS_DEBUG"):
            raise
        print(f"noesis-agent: error: {type(e).__name__}: {e}  (set NOESIS_DEBUG=1 for a traceback)", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
