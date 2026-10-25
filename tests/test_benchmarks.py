"""Smoke tests for the benchmark harness (offline, ~1-2 s)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import metrics as M  # noqa: E402
from benchmarks.run import main, model_slug, run_benchmark, to_markdown  # noqa: E402


def test_metric_helpers():
    assert M.brier([1.0, 0.0], [1, 0]) == 0.0
    assert M.brier([0.5, 0.5], [1, 0]) == 0.25
    assert M.ece([0.9, 0.9, 0.1], [1, 1, 0]) == round((2 / 3) * 0.1 + (1 / 3) * 0.1, 4)
    assert M.quote_in_chunk("lower  risk", "a lower risk of death")
    assert not M.quote_in_chunk("", "anything")
    assert M.is_negated("Coffee intake was not associated with atrial fibrillation.")
    assert not M.is_negated("Lower risk compared with no coffee.")
    assert M.strip_hedge("Insufficient evidence for a confident answer (confidence 0.30). Tentatively: Yes.") == "Yes."
    assert M.strip_hedge("Insufficient evidence: nothing found.") == ""
    assert model_slug("anthropic:claude-sonnet-5-5") == "anthropic_claude-sonnet-5-5"


def test_score_answer_polarity_and_keywords():
    yes = M.score_answer("q14", "No; the cohort found no association with a higher atrial fibrillation risk.", False)
    assert yes["correct"]
    wrong = M.score_answer("q14", "Coffee raises the risk of atrial fibrillation.", False)
    assert not wrong["correct"]
    hedged = M.score_answer("q08", "Insufficient evidence for a confident answer (confidence 0.2). Tentatively: 200 mg per day.", True)
    assert hedged["correct_if_committed"] and not hedged["correct"]
    mixed = M.score_answer(
        "q02", "Caffeine raises blood pressure briefly, but regular coffee drinking does not cause lasting high blood pressure.", False
    )
    assert mixed["correct"]


def test_run_benchmark_mock(tmp_path):
    res = run_benchmark("mock", think_steps=1, out_dir=tmp_path, quiet=True)
    s = res["summary"]
    assert res["meta"]["offline_heuristic_baseline"] is True
    assert len(res["questions"]) >= 15
    for v in (s["qa"]["accuracy"], s["citations"]["quote_in_chunk_rate"], s["calibration"]["brier"], s["contradictions"]["recall"]):
        assert v is not None and 0.0 <= v <= 1.0
    # invariants the scaffold must keep even offline
    assert s["citations"]["quote_in_chunk_rate"] == 1.0
    assert s["injection"]["passed"] is True
    assert s["abstention"]["ood_abstention_rate"] > 0
    data = json.loads((tmp_path / "mock.json").read_text())
    assert data["meta"]["model"] == "mock"
    md = (tmp_path / "mock.md").read_text()
    assert "NOT measures of real reasoning quality" in md and "| Metric | Value | Notes |" in md
    assert to_markdown(res) == md


def test_cli_none_no_write(tmp_path, capsys):
    assert main(["--model", "none", "--think-steps", "0", "--no-write", "--quiet", "--out-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "Brier score" in out and "Prompt-injection resistance" in out
    assert not list(tmp_path.iterdir())
