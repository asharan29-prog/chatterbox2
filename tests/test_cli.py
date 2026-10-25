"""CLI end-to-end on the bundled corpus (offline, mock model)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from noesis.cli import main, slugify

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "examples" / "corpus"


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("NOESIS_DB", raising=False)
    monkeypatch.delenv("NOESIS_MODEL", raising=False)


def run(capsys, *argv: str) -> tuple[int, str, str]:
    code = main(list(argv))
    out, err = capsys.readouterr()
    return code, out, err


def test_full_flow(capsys, tmp_path):
    code, out, _ = run(capsys, "init", "Is coffee good for health?")
    assert code == 0
    db = tmp_path / ".noesis" / "is-coffee-good-for-health.db"
    assert db.exists() and (tmp_path / ".noesis" / "current").read_text().strip() == db.name

    code, out, _ = run(capsys, "ingest", str(CORPUS))
    assert code == 0 and "Ingested 8 source(s)" in out

    code, out, _ = run(capsys, "understand")
    assert code == 0 and "claims" in out

    code, out, _ = run(capsys, "think", "--steps", "3")
    assert code == 0
    for step in ("[", "select_question", "hypothesize", "gather", "judge", "reflect"):
        assert step in out
    assert "Run run_" in out

    code, out, _ = run(capsys, "ask", "Does coffee raise blood pressure?", "--json")
    assert code == 0
    ans = json.loads(out)
    assert {"answer", "confidence", "citations"} <= set(ans) and "trace" not in ans

    code, out, _ = run(capsys, "beliefs", "--json", "--min-confidence", "0.6")
    rows = json.loads(out)
    assert code == 0 and rows and all(r["confidence"] >= 0.6 for r in rows)
    code, out, _ = run(capsys, "beliefs", "--limit", "3")
    assert code == 0 and "confidence" in out and out.count("bel_") == 3

    code, out, _ = run(capsys, "questions", "--json")
    assert code == 0 and isinstance(json.loads(out), list)

    code, out, _ = run(capsys, "explain", rows[0]["id"])
    assert code == 0 and "Evidence FOR" in out and "History" in out

    code, out, _ = run(capsys, "report", "--html", "-o", "r.html")
    assert code == 0 and (tmp_path / "r.html").read_text().lstrip().lower().startswith("<!doctype html")
    code, out, _ = run(capsys, "report")
    assert code == 0 and out.lstrip().startswith("#")

    code, out, _ = run(capsys, "export", "dump.json")
    assert code == 0 and json.loads((tmp_path / "dump.json").read_text())["beliefs"]
    code, out, _ = run(capsys, "--db", "copy.db", "import", "dump.json")
    assert code == 0 and "beliefs" in out
    code, out, _ = run(capsys, "--db", "copy.db", "beliefs", "--json")
    assert len(json.loads(out)) >= len(rows)

    code, out, _ = run(capsys, "doctor")
    assert code == 0 and "MockLLM" in out and "sources=8" in out


def test_model_none_heuristic(capsys):
    assert run(capsys, "--model", "none", "init", "t")[0] == 0
    assert run(capsys, "ingest", str(CORPUS / "03_caffeine_trial.md"), "--reliability", "0.8")[0] == 0
    code, out, _ = run(capsys, "--model", "none", "understand", "--json")
    assert code == 0 and json.loads(out)["method"] == "heuristic"
    code, out, _ = run(capsys, "--model", "none", "think", "--steps", "1", "--quiet")
    assert code == 0 and "[" not in out.split("Run ")[0]


def test_errors(capsys, tmp_path):
    code, _, err = run(capsys, "beliefs")
    assert code == 2 and "noesis init" in err
    assert run(capsys, "init", "t")[0] == 0
    code, _, err = run(capsys, "explain", "bel_nope")
    assert code == 1 and "no belief or claim" in err
    code, _, err = run(capsys, "ingest", str(tmp_path / "missing.md"))
    assert code == 1 and "missing.md" in err
    code, _, err = run(capsys, "ingest", str(CORPUS), "--reliability", "3")
    assert code == 2
    code, _, err = run(capsys, "--model", "bogus:x", "understand")
    assert code == 2 and "unknown model provider" in err
    code, _, err = run(capsys, "import", "nope.json")
    assert code == 2
    assert run(capsys)[0] == 2  # no command -> help
    with pytest.raises(SystemExit) as e:
        main(["think", "--strategy", "nope"])
    assert e.value.code == 2


def test_db_env_and_multiple(capsys, tmp_path, monkeypatch):
    assert run(capsys, "init", "alpha")[0] == 0
    assert run(capsys, "init", "beta")[0] == 0  # `current` now points at beta
    code, out, _ = run(capsys, "doctor", "--offline")
    assert "topic: beta" in out
    (tmp_path / ".noesis" / "current").unlink()
    code, _, err = run(capsys, "beliefs")
    assert code == 2 and "several knowledge bases" in err
    monkeypatch.setenv("NOESIS_DB", str(tmp_path / ".noesis" / "alpha.db"))
    code, out, _ = run(capsys, "doctor", "--offline")
    assert "topic: alpha" in out


def test_slugify():
    assert slugify("Is coffee good for health?") == "is-coffee-good-for-health"
    assert slugify("???") == "topic"


def test_help_has_examples(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])
    out = capsys.readouterr().out
    assert "examples:" in out and "serve-mcp" in out and "doctor" in out


def test_python_dash_m(tmp_path):
    env = {**os.environ, "PYTHONPATH": str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")}
    env.pop("NOESIS_DB", None)
    r = subprocess.run(
        [sys.executable, "-m", "noesis", "init", "sub topic"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert r.returncode == 0, r.stderr
    r = subprocess.run(
        [sys.executable, "-m", "noesis", "--version"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert r.returncode == 0 and "noesis" in r.stdout
