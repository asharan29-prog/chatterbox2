"""Python API example for the Noesis Research Agent.

Offline demo (bundled mock web + MockLLM, no network):

    python examples/agent/research_agent.py

Real internet + real model:

    ANTHROPIC_API_KEY=... BRAVE_API_KEY=... \
        python examples/agent/research_agent.py --model anthropic:claude-sonnet-5-5 --search brave \
        "Does intermittent fasting improve longevity?"
"""

from __future__ import annotations

import argparse

from noesis import KnowledgeStore, get_llm
from noesis.agent import DomainPolicy, ResearchAgent, ResearchBudget


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?", default="Is coffee good for health?")
    ap.add_argument("--model", default="mock")
    ap.add_argument("--search", default="mock", help="mock | ddg | brave | tavily | searxng")
    ap.add_argument("--db", default=":memory:")
    a = ap.parse_args()

    store = KnowledgeStore(a.db)
    agent = ResearchAgent(
        store,
        get_llm(a.model),
        search=a.search,
        budget=ResearchBudget(max_sources=6, per_domain=1),
        policy=DomainPolicy(deny=["pinterest.com"], trust={"cochranelibrary.com": 0.92}),
        on_event=lambda msg: print("·", msg),
    )
    result = agent.research(a.question, depth=2)

    print(f"\nAnswer ({result.main.confidence:.2f}): {result.main.answer}")
    for c in result.conclusions:
        print(f"- {c.question} -> {c.confidence:.2f} {c.citations}")
    print("\nCould not verify:")
    for item in result.unverified:
        print("  -", item)
    print("\nSources:")
    for s in result.cited_sources:
        print(f"  [{s.n}] {s.reliability:.2f} {s.final_url}")
    for path in result.write("research_brief.md"):
        print("wrote", path)

    # Later: re-run the saved searches, ingest only new pages and show belief changes.
    print(agent.watch(a.question).markdown)
    store.close()


if __name__ == "__main__":
    main()
