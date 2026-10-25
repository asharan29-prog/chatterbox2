"""Command-line interface: ``noesis`` (console script) or ``python -m noesis``.

Stdlib :mod:`argparse` only (no extra dependencies). Every command works offline with
``--model mock`` (the default) or ``--model none`` (pure heuristics, no model at all).

Database resolution (first match wins): ``--db PATH`` > ``$NOESIS_DB`` > the topic recorded
by ``noesis init`` in ``./.noesis/current`` > the only ``*.db`` in ``./.noesis/``.

Exit codes: 0 ok, 1 runtime error (bad id, provider failure, ...), 2 usage error
(bad arguments, no database selected).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from . import __version__

EXIT_OK, EXIT_ERROR, EXIT_USAGE = 0, 1, 2
STATE_DIR = ".noesis"
CURRENT_FILE = "current"


class CLIError(Exception):
    """A user-facing error; ``code`` is the process exit code."""

    def __init__(self, message: str, code: int = EXIT_ERROR) -> None:
        super().__init__(message)
        self.code = code


# ============================================================================ helpers
def slugify(topic: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", topic.lower()).strip("-")
    return (s or "topic")[:60].rstrip("-")


def state_dir() -> Path:
    return Path.cwd() / STATE_DIR


def resolve_db(arg: str | None, *, must_exist: bool = True) -> Path:
    if arg:
        p = Path(arg)
    elif os.environ.get("NOESIS_DB"):
        p = Path(os.environ["NOESIS_DB"])
    else:
        sd = state_dir()
        cur = sd / CURRENT_FILE
        if cur.is_file() and cur.read_text(encoding="utf-8").strip():
            p = Path(cur.read_text(encoding="utf-8").strip())
            if not p.is_absolute():
                p = sd / p
        else:
            dbs = sorted(sd.glob("*.db")) if sd.is_dir() else []
            if len(dbs) == 1:
                p = dbs[0]
            elif not dbs:
                raise CLIError(
                    'no knowledge base selected. Run `noesis init "<topic>"` first, or pass --db PATH '
                    "(or set NOESIS_DB).",
                    EXIT_USAGE,
                )
            else:
                names = ", ".join(d.name for d in dbs)
                raise CLIError(
                    f"several knowledge bases in {sd} ({names}); choose one with --db.", EXIT_USAGE
                )
    if must_exist and not p.exists():
        raise CLIError(
            f'database {p} does not exist. Run `noesis init "<topic>"` (or `noesis --db {p} init TOPIC`).',
            EXIT_USAGE,
        )
    return p


def open_store(args: argparse.Namespace, *, must_exist: bool = True) -> Any:
    from .store import KnowledgeStore

    return KnowledgeStore(resolve_db(args.db, must_exist=must_exist))


def model_spec(args: argparse.Namespace) -> str:
    return (args.model or os.environ.get("NOESIS_MODEL") or "mock").strip()


def make_llm(args: argparse.Namespace) -> Any:
    from .llm import LLMError, get_llm

    try:
        return get_llm(model_spec(args))
    except LLMError as e:
        raise CLIError(str(e), EXIT_USAGE) from e


def emit_json(obj: Any) -> None:
    print(json.dumps(obj, indent=2, default=str, ensure_ascii=False))


def _clip(s: str, n: int) -> str:
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[: n - 1] + "…"


def _belief_rows(store: Any, status: str | None, min_conf: float | None) -> list[dict[str, Any]]:
    rows = []
    for b in store.list_beliefs(status, order_by_confidence=True):
        if min_conf is not None and b.confidence < min_conf:
            continue
        claim = store.get_claim(b.claim_id)
        rows.append(
            {
                "id": b.id,
                "claim_id": b.claim_id,
                "claim": claim.text if claim else "",
                "confidence": round(b.confidence, 4),
                "status": b.status,
                "n_evidence": len(b.evidence_ids),
                "rationale": b.rationale,
            }
        )
    return rows


def _question_rows(store: Any, status: str | None = "open") -> list[dict[str, Any]]:
    qs = store.list_questions(status=status) if status else store.list_questions()
    qs = sorted(qs, key=lambda q: -q.priority)
    return [
        {
            "id": q.id,
            "text": q.text,
            "origin": q.origin,
            "priority": round(q.priority, 3),
            "status": q.status,
            "answer": q.answer,
            "answer_confidence": q.answer_confidence,
        }
        for q in qs
    ]


# ============================================================================ commands
def cmd_init(args: argparse.Namespace) -> int:
    from .store import KnowledgeStore

    if args.db or os.environ.get("NOESIS_DB"):
        db = resolve_db(args.db, must_exist=False)
        point = False
    else:
        db = state_dir() / f"{slugify(args.topic)}.db"
        point = True
    existed = db.exists()
    store = KnowledgeStore(db)
    try:
        store.set_meta("topic", args.topic)
    finally:
        store.close()
    if point:
        (state_dir() / CURRENT_FILE).write_text(db.name + "\n", encoding="utf-8")
    print(f"{'Reopened' if existed else 'Created'} knowledge base for {args.topic!r}: {db}")
    print("Next: noesis ingest <files|dirs|urls>  then  noesis understand")
    return EXIT_OK


def cmd_ingest(args: argparse.Namespace) -> int:
    from .ingest import ingest_path, ingest_url

    store = open_store(args)
    kw: dict[str, Any] = {}
    if args.reliability is not None:
        if not 0.0 <= args.reliability <= 1.0:
            raise CLIError("--reliability must be between 0 and 1", EXIT_USAGE)
        kw["reliability"] = args.reliability
    total, failures = 0, 0
    try:
        for target in args.targets:
            try:
                if re.match(r"^https?://", target, re.I):
                    sources = [ingest_url(store, target, **kw)]
                else:
                    if not Path(target).exists():
                        raise FileNotFoundError(f"no such file or directory: {target}")
                    sources = ingest_path(store, target, **kw)
            except Exception as e:  # noqa: BLE001 - report and continue with the next target
                failures += 1
                print(f"error: {target}: {e}", file=sys.stderr)
                continue
            for s in sources:
                total += 1
                print(f"  + {s.title}  [{s.kind}, reliability {s.reliability:.2f}]  {s.id}")
            if not sources:
                print(f"  (no supported files in {target})")
        st = store.stats()
    finally:
        store.close()
    print(
        f"Ingested {total} source(s); store has {st.get('sources', 0)} sources, {st.get('chunks', 0)} chunks."
    )
    return EXIT_ERROR if failures and not total else EXIT_OK


def cmd_understand(args: argparse.Namespace) -> int:
    from .understand import understand

    store = open_store(args)
    try:
        llm = make_llm(args)
        res = understand(store, llm, reprocess=args.reprocess)
    finally:
        store.close()
    d = res.as_dict()
    if args.json:
        emit_json(d)
        return EXIT_OK
    print(
        f"Understanding ({d['method']}): {d['chunks_processed']} chunks -> {d['claims_created']} new claims "
        f"({d['claims_merged']} merged), {d['evidence_added']} evidence links, {d['concepts']} concepts, "
        f"{d['contradictions']} contradictions, {d['questions']} questions, {d['beliefs_updated']} beliefs updated."
    )
    for w in d["warnings"]:
        print(f"warning: {w}", file=sys.stderr)
    if d["llm_errors"]:
        print(f"warning: {d['llm_errors']} model call(s) failed and fell back to heuristics", file=sys.stderr)
    return EXIT_OK


def _print_thought(t: Any) -> None:
    print(f"[{t.step:>15}] {_clip(t.content, 220)}", flush=True)


def cmd_think(args: argparse.Namespace) -> int:
    from .reason import Budget, Mind

    store = open_store(args)
    try:
        llm = make_llm(args)
        budget = Budget(steps=args.steps, max_llm_calls=args.max_llm_calls, seconds=args.seconds)
        mind = Mind(
            store,
            llm,
            strategy=args.strategy,
            budget=budget,
            on_thought=None if args.quiet else _print_thought,
        )
        goal = args.goal or store.topic
        run = mind.think(goal=goal, steps=args.steps)
        if args.json:
            emit_json(run.model_dump(mode="json"))
        else:
            print(f"\nRun {run.id}: {run.status} after {run.steps} step(s) ({run.stop_reason or 'done'}).")
            if run.summary:
                print(run.summary)
    finally:
        store.close()
    return EXIT_OK if run.status in ("completed", "stopped") else EXIT_ERROR


def cmd_ask(args: argparse.Namespace) -> int:
    from .reason import Budget, Mind

    store = open_store(args)
    try:
        mind = Mind(
            store, make_llm(args), budget=Budget(max_llm_calls=args.max_llm_calls, seconds=args.seconds)
        )
        ans = mind.ask(args.question)
    finally:
        store.close()
    if args.json:
        d = ans.to_dict()
        if not args.trace:
            d.pop("trace", None)
        emit_json(d)
    else:
        print(str(ans))
    return EXIT_OK


def cmd_beliefs(args: argparse.Namespace) -> int:
    store = open_store(args)
    try:
        rows = _belief_rows(store, args.status, args.min_confidence)
    finally:
        store.close()
    if args.limit:
        rows = rows[: args.limit]
    if args.json:
        emit_json(rows)
        return EXIT_OK
    if not rows:
        print("No beliefs yet. Run `noesis understand` (and `noesis think`).")
        return EXIT_OK
    print(f"{'confidence':>10}  {'status':<10}  {'id':<18}  claim")
    for r in rows:
        print(f"{r['confidence']:>10.2f}  {r['status']:<10}  {r['id']:<18}  {_clip(r['claim'], 100)}")
    print(f"\n{len(rows)} belief(s). Details: noesis explain <id>")
    return EXIT_OK


def cmd_questions(args: argparse.Namespace) -> int:
    store = open_store(args)
    try:
        rows = _question_rows(store, None if args.all else "open")
    finally:
        store.close()
    if args.limit:
        rows = rows[: args.limit]
    if args.json:
        emit_json(rows)
        return EXIT_OK
    if not rows:
        print("No open questions.")
        return EXIT_OK
    print(f"{'priority':>8}  {'origin':<13}  {'id':<18}  question")
    for r in rows:
        extra = "" if r["status"] == "open" else f"  [{r['status']}]"
        print(f"{r['priority']:>8.2f}  {r['origin']:<13}  {r['id']:<18}  {_clip(r['text'], 100)}{extra}")
    return EXIT_OK


def cmd_explain(args: argparse.Namespace) -> int:
    from .reason import Mind

    store = open_store(args)
    try:
        mind = Mind(store, make_llm(args))
        try:
            if args.json:
                emit_json(mind.explain_data(args.id))
            else:
                print(mind.explain(args.id))
        except KeyError as e:
            raise CLIError(f"no belief or claim with id {args.id!r} (see `noesis beliefs`)") from e
    finally:
        store.close()
    return EXIT_OK


def cmd_report(args: argparse.Namespace) -> int:
    from .report import render_html, render_markdown, write_report

    fmt = "html" if args.html else "md" if args.md else None
    store = open_store(args)
    try:
        if args.output:
            p = write_report(store, args.output, fmt=fmt)
            print(f"Wrote {p}")
        else:
            print(render_html(store) if fmt == "html" else render_markdown(store))
    finally:
        store.close()
    return EXIT_OK


def cmd_export(args: argparse.Namespace) -> int:
    store = open_store(args)
    try:
        p = store.export_to(args.file)
        st = store.stats()
    finally:
        store.close()
    print(f"Exported {sum(st.values())} records to {p}")
    return EXIT_OK


def cmd_import(args: argparse.Namespace) -> int:
    if not Path(args.file).is_file():
        raise CLIError(f"no such file: {args.file}", EXIT_USAGE)
    store = open_store(args, must_exist=False)
    try:
        try:
            counts = store.import_json(Path(args.file), overwrite=not args.keep_existing)
        except (ValueError, json.JSONDecodeError) as e:
            raise CLIError(f"{args.file} is not a valid Noesis export: {e}") from e
    finally:
        store.close()
    parts = ", ".join(f"{v} {k}" for k, v in counts.items() if v)
    print(f"Imported {parts}" if parts else "Imported nothing")
    return EXIT_OK


def cmd_serve_mcp(args: argparse.Namespace) -> int:
    from .mcp_server import MCPUnavailable, serve

    try:
        db: str | None = str(resolve_db(args.db, must_exist=False))
    except CLIError:
        db = None  # server falls back to ./.noesis/noesis.db
    if args.transport == "http":
        print(f"Noesis MCP server (streamable HTTP) on http://{args.host}:{args.port}/mcp", file=sys.stderr)
    try:
        serve(db_path=db, model=args.model, transport=args.transport, host=args.host, port=args.port)
    except MCPUnavailable as e:
        raise CLIError(str(e), EXIT_USAGE) from e
    return EXIT_OK


def cmd_doctor(args: argparse.Namespace) -> int:
    import importlib.util
    import platform

    ok = True
    print(f"noesis {__version__}  (python {platform.python_version()}, {sys.platform})")
    spec = model_spec(args)
    print(f"\nModel: {spec}")
    try:
        llm = make_llm(args)
    except CLIError as e:
        print(f"  [FAIL] {e}")
        llm, ok = None, False
    else:
        if llm is None:
            print("  [ ok ] offline heuristic mode (no model)")
        elif spec.lower().startswith("mock"):
            print("  [ ok ] deterministic offline MockLLM")
        else:
            key_env = {
                "anthropic": "ANTHROPIC_API_KEY",
                "claude": "ANTHROPIC_API_KEY",
                "openai": "OPENAI_API_KEY",
            }.get(spec.split(":")[0].lower())
            if key_env:
                print(
                    f"  [{' ok ' if os.environ.get(key_env) else 'warn'}] {key_env} {'is set' if os.environ.get(key_env) else 'is NOT set'}"
                )
            if args.offline:
                print("  [skip] connectivity check (--offline)")
            else:
                try:
                    out = llm.complete("Reply with the single word: pong", max_tokens=8)
                    print(f"  [ ok ] reachable; replied {_clip(str(out), 40)!r}")
                except Exception as e:  # noqa: BLE001
                    ok = False
                    print(f"  [FAIL] {type(e).__name__}: {_clip(str(e), 200)}")
    print("\nKnowledge base:")
    try:
        db = resolve_db(args.db)
    except CLIError as e:
        print(f"  [info] {e}")
    else:
        from .store import KnowledgeStore

        store = KnowledgeStore(db)
        try:
            print(f"  path : {db}")
            print(f"  topic: {store.topic or '(none)'}")
            print("  " + ", ".join(f"{k}={v}" for k, v in store.stats().items()))
        finally:
            store.close()
    print("\nOptional dependencies:")
    for mod, extra, purpose in (
        ("mcp", "mcp", "MCP server (noesis serve-mcp)"),
        ("pypdf", "pdf", "PDF ingestion"),
        ("bs4", "web", "better HTML/URL text extraction"),
        ("sklearn", "sklearn", "TF-IDF extras"),
        ("rich", "cli", "pretty terminal output"),
    ):
        have = importlib.util.find_spec(mod) is not None
        print(
            f"  [{' ok ' if have else ' -- '}] {mod:<8} {purpose}"
            + ("" if have else f"  (pip install 'noesis[{extra}]')")
        )
    print("\nAll checks passed." if ok else "\nSome checks failed.")
    return EXIT_OK if ok else EXIT_ERROR


# ============================================================================ parser
EPILOG = """\
examples:
  noesis init "Is coffee good for health?"
  noesis ingest examples/corpus https://example.org/article.html --reliability 0.7
  noesis understand
  noesis think --goal "Does coffee raise blood pressure?" --steps 5 --strategy debate
  noesis ask "Does coffee dehydrate regular drinkers?" --json
  noesis beliefs --status believed --min-confidence 0.7
  noesis explain bel_1a2b3c
  noesis report --html -o report.html
  noesis serve-mcp                       # stdio MCP server for Claude Desktop / Claude Code
  noesis --model anthropic:claude-sonnet-5-5 think --steps 10
  noesis research "Is coffee good for health?" --search ddg -o brief.md   # web research agent

models (--model or $NOESIS_MODEL): mock (default, offline), none (heuristics only),
  anthropic:<model>, openai:<model>, ollama:<model>
database (--db or $NOESIS_DB): defaults to ./.noesis/<topic>.db chosen by `noesis init`
"""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="noesis",
        description="Build an evidence-backed understanding of a topic and reason over it with any AI model.",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--version", action="version", version=f"noesis {__version__}")
    p.add_argument(
        "--db", metavar="PATH", help="knowledge base file (default: $NOESIS_DB or ./.noesis/<topic>.db)"
    )
    p.add_argument(
        "--model", metavar="SPEC", help="model spec (default: $NOESIS_MODEL or 'mock'; 'none' = no model)"
    )
    sub = p.add_subparsers(dest="command", metavar="COMMAND")

    def add(name: str, func: Any, help_: str, example: str = "") -> argparse.ArgumentParser:
        sp = sub.add_parser(
            name,
            help=help_,
            description=help_,
            epilog=f"example:\n  {example}" if example else None,
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        sp.set_defaults(func=func)
        return sp

    sp = add(
        "init",
        cmd_init,
        "create (or reopen) a knowledge base for a topic",
        'noesis init "Is coffee good for health?"',
    )
    sp.add_argument("topic")

    sp = add(
        "ingest",
        cmd_ingest,
        "add files, directories (recursive) or http(s) URLs as sources",
        "noesis ingest notes/ paper.pdf https://example.org/post",
    )
    sp.add_argument("targets", nargs="+", metavar="PATH|URL")
    sp.add_argument(
        "--reliability",
        type=float,
        help="source reliability prior 0..1 (default: domain/metadata heuristics)",
    )

    sp = add(
        "understand",
        cmd_understand,
        "extract concepts, claims, evidence, contradictions and questions",
        "noesis understand",
    )
    sp.add_argument("--reprocess", action="store_true", help="re-run on chunks already processed")
    sp.add_argument("--json", action="store_true", help="machine-readable output")

    sp = add(
        "think",
        cmd_think,
        "run the independent reasoning loop (thoughts stream as they happen)",
        'noesis think --goal "Is decaf different?" --steps 5',
    )
    sp.add_argument("--goal", help="what to reason about (default: the topic)")
    sp.add_argument("--steps", type=int, default=5, help="maximum reasoning steps (default 5)")
    sp.add_argument("--strategy", default="simple", choices=["simple", "self_consistency", "debate"])
    sp.add_argument("--max-llm-calls", type=int, default=None, help="hard cap on model calls")
    sp.add_argument("--seconds", type=float, default=None, help="wall-clock limit")
    sp.add_argument("--quiet", action="store_true", help="do not stream thoughts")
    sp.add_argument("--json", action="store_true", help="print the finished run as JSON")

    sp = add(
        "ask",
        cmd_ask,
        "answer a question with citations and evidence-based confidence",
        'noesis ask "Does coffee raise blood pressure?" --json',
    )
    sp.add_argument("question")
    sp.add_argument("--json", action="store_true")
    sp.add_argument("--trace", action="store_true", help="include the reasoning trace in --json output")
    sp.add_argument("--max-llm-calls", type=int, default=None)
    sp.add_argument("--seconds", type=float, default=None)

    sp = add(
        "beliefs", cmd_beliefs, "list beliefs, most confident first", "noesis beliefs --status doubted --json"
    )
    sp.add_argument("--status", choices=["hypothesis", "believed", "doubted", "rejected"])
    sp.add_argument("--min-confidence", type=float, default=None)
    sp.add_argument("--limit", type=int, default=None)
    sp.add_argument("--json", action="store_true")

    sp = add(
        "questions", cmd_questions, "list open questions, highest priority first", "noesis questions --json"
    )
    sp.add_argument("--all", action="store_true", help="include answered/dropped questions")
    sp.add_argument("--limit", type=int, default=None)
    sp.add_argument("--json", action="store_true")

    sp = add(
        "explain",
        cmd_explain,
        "show why a belief has its confidence (evidence, sources, history)",
        "noesis explain bel_1a2b3c",
    )
    sp.add_argument("id", help="belief id or claim id")
    sp.add_argument("--json", action="store_true")

    sp = add(
        "report",
        cmd_report,
        "render a markdown (default) or self-contained HTML report",
        "noesis report --html -o report.html",
    )
    g = sp.add_mutually_exclusive_group()
    g.add_argument("--html", action="store_true")
    g.add_argument("--md", action="store_true")
    sp.add_argument(
        "-o",
        "--output",
        metavar="FILE",
        help="write to FILE (format inferred from suffix unless --html/--md)",
    )

    sp = add("export", cmd_export, "export the whole knowledge base as JSON", "noesis export backup.json")
    sp.add_argument("file")

    sp = add(
        "import",
        cmd_import,
        "import a JSON export into the knowledge base",
        "noesis --db copy.db import backup.json",
    )
    sp.add_argument("file")
    sp.add_argument("--keep-existing", action="store_true", help="skip records whose id already exists")

    sp = add(
        "serve-mcp",
        cmd_serve_mcp,
        "run the MCP server (needs: pip install 'noesis[mcp]')",
        "noesis serve-mcp --transport http --port 8765",
    )
    sp.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    sp.add_argument("--host", default="127.0.0.1")
    sp.add_argument("--port", type=int, default=8765)

    sp = add(
        "doctor",
        cmd_doctor,
        "check model connectivity, database and optional dependencies",
        "noesis --model ollama:llama3.1 doctor",
    )
    sp.add_argument("--offline", action="store_true", help="skip the live model call")

    from .agent.cli import add_research_arguments

    sp = add(
        "research",
        cmd_research,
        "research a question on the web (Noesis Research Agent; also `noesis-agent`)",
        'noesis research "Is coffee good for health?" --search brave --depth 2 -o brief.md',
    )
    add_research_arguments(sp)
    return p


def cmd_research(args: argparse.Namespace) -> int:
    from .agent.cli import run_research

    return run_research(args)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return EXIT_USAGE
    try:
        return int(args.func(args) or 0)
    except CLIError as e:
        print(f"noesis: error: {e}", file=sys.stderr)
        return e.code
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except BrokenPipeError:
        return EXIT_OK
    except Exception as e:  # noqa: BLE001 - last-resort, readable error instead of a traceback
        if os.environ.get("NOESIS_DEBUG"):
            raise
        print(
            f"noesis: error: {type(e).__name__}: {e}  (set NOESIS_DEBUG=1 for a traceback)", file=sys.stderr
        )
        return EXIT_ERROR


def run() -> None:  # pragma: no cover - console-script shim
    sys.exit(main())


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
