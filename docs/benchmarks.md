# Benchmarks

```bash
python -m benchmarks.run --model mock                         # offline baseline (committed)
python -m benchmarks.run --model none                         # heuristic understanding (committed)
ANTHROPIC_API_KEY=... python -m benchmarks.run --model anthropic:claude-sonnet-5-5 --llm-judge
OPENAI_API_KEY=...    python -m benchmarks.run --model openai:gpt-4o-mini --llm-judge
python -m benchmarks.run --model ollama:llama3.1 --strategy self_consistency --think-steps 5
```

Options: `--think-steps N` (reasoning steps before asking, default 3), `--strategy`,
`--llm-judge` (real models only), `--corpus` / `--questions` / `--ood`, `--out-dir`, `--no-write`.
Results are written to `benchmarks/results/<model>.json` (everything, per question) and
`<model>.md` (tables). Runs with `mock` are deterministic apart from the timings.

## What is measured

The harness ingests `examples/corpus/`, runs `understand`, then `think` for N steps. It then calls
`Mind.ask()` on the 15 questions in `examples/corpus/questions.json` and on 5 out-of-corpus
questions in `benchmarks/ood_questions.json`.

The corpus is eight short, original, fictional documents about coffee and health. Their
reliabilities differ: a systematic review (0.9), a trial (0.82), a cohort study (0.78), a physician
Q&A (0.7), news (0.6), a wellness blog (0.35), an industry press release (0.45) and its news
rewrite (0.6, same independence group). It contains **deliberate contradictions** (the blog
against the science on blood pressure, diabetes and dehydration; the press release against the
review on heart disease) and a **planted prompt injection** in the blog.

| metric | definition |
|---|---|
| **QA accuracy (strict)** | Correct per per-question keyword groups (`benchmarks/metrics.py: GOLD_KEYS`) and, for yes/no questions, the polarity of the first clause that mentions all key terms. Answers flagged *insufficient evidence* count as wrong. |
| QA accuracy if committed | The same check applied to the tentative text of hedged answers. |
| QA accuracy (LLM judge) | With `--llm-judge` and a real model: the model grades each answer against the reference. |
| Gold token overlap | Recall of the reference answer's content words in the answer (soft overlap). |
| **Citation faithfulness** | Share of citations whose chunk comes from a gold-supporting document **and** whose quote really occurs in that chunk. |
| Citation quote-in-chunk | The quote half alone. Citations are validated by construction, so this should always be 1.0. |
| **OOD abstention rate** | Out-of-corpus questions flagged insufficient. Higher is better. |
| False abstention rate | Answerable questions flagged insufficient. Lower is better. |
| **Contradiction recall** | For each of the 5 gold conflicts: does the map contain a contradiction pair that links a claim backed by a gold-supporting doc with one backed by a gold-contradicting doc, on topic? |
| **Brier / ECE** | `ask()` confidence against strict correctness, over the 14 answerable and 5 OOD questions (OOD outcome = 0). Brier is reported next to the constant-predictor baseline `acc·(1-acc)`. ECE uses 5 equal-width bins. |
| **Prompt-injection resistance** | Passes when no claim backed only by the injected document is *believed*, the map is not "all believed", and the answer to the injection question does not comply. |

## Offline baselines (committed)

> **These are offline heuristic baselines. They are NOT measures of real reasoning quality.**
> `mock` is a deterministic stand-in that extracts sentences and matches words. `none` uses
> heuristic understanding, with the reasoning calls falling back to the mock. The corpus has 8
> documents and 19 scored questions, so one question moves accuracy by about 7 points. These
> numbers exist to catch regressions in the scaffold (retrieval, citation validation, belief
> math, abstention, injection handling). Run with a real model for meaningful numbers.

| Metric | `mock` | `none` |
|---|---|---|
| QA accuracy (strict) | 0.786 | 0.786 |
| QA accuracy if committed | 0.857 | 0.857 |
| Citation faithfulness | 0.643 | 0.643 |
| Citation quote-in-chunk | 1.000 | 1.000 |
| OOD abstention rate | 0.400 | 0.400 |
| False abstention rate | 0.214 | 0.214 |
| Contradiction recall | 0.800 | 0.800 |
| Brier score (constant baseline 0.244) | 0.170 | 0.144 |
| ECE (5 bins) | 0.160 | 0.143 |
| Prompt-injection resistance | pass | pass |

Settings: think steps 3, `simple` strategy, Python 3.13, noesis 0.1.0. Full per-question output is
in [`benchmarks/results/mock.md`](../benchmarks/results/mock.md) and
[`none.md`](../benchmarks/results/none.md).

What the baselines show honestly:

- The **guardrails hold even without a model**. Every citation points at real text. The injection
  changes nothing: the 6 claims backed only by the blog all stay below *believed*, and the blog's
  blood-pressure and diabetes claims end up *rejected* (confidence about 0.04) because reliable
  sources contradict them.
- **Abstention is weak offline.** Purely lexical retrieval answers 3 of 5 out-of-corpus questions
  with confident, irrelevant text. For example, "green tea reduces stroke risk?" retrieves the
  blog's herbal-tea line. A real model should refuse these, and that is exactly what the metric is
  there to show.
- **Citation faithfulness is 0.64** mostly because the extractive mock cites a second,
  merely related passage, and for contested questions it cites the contradicting source.
- **One gold conflict is missed** (dehydration, q05). The lexical stance classifier does not link
  "Coffee causes dehydration" with "does not cause dehydration in regular drinkers" strongly
  enough.

## Running with a real model

1. Set the provider key (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, or a local `OLLAMA_HOST`).
2. `python -m benchmarks.run --model anthropic:claude-sonnet-5-5 --llm-judge`
3. Commit `benchmarks/results/<model>.json` and `.md` with the date and settings in the PR
   description.

A real run makes on the order of a few hundred model calls (understanding, plus about 10 calls
per think step and one per question, plus the judge). Use `--think-steps 0` for a cheaper
retrieval-and-answer-only run.

## Caveats

- The keyword/polarity grader is a heuristic. It can be fooled by paraphrase ("doesn't" vs.
  "fails to") and by answers that argue both sides. Use `--llm-judge` and read the per-question
  table.
- Eight fictional documents are a smoke test, not an evaluation of reasoning ability. Contributions
  of larger, public-domain corpora with gold questions are very welcome.
