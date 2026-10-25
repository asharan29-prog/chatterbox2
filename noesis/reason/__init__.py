"""Independent reasoning engine for Noesis.

>>> from noesis.reason import Mind, Budget
>>> mind = Mind(store, llm, strategy="debate", budget=Budget(steps=5, max_llm_calls=200, seconds=120))
>>> run = mind.think(goal="Is coffee good for health?")        # persisted ReasoningRun
>>> thoughts = store.list_thoughts(run.id)                     # full trace
>>> ans = mind.ask("Does coffee raise blood pressure?")        # Answer with citations
>>> print(mind.explain(ans.supporting_claims[0]["belief_id"]))

See :mod:`noesis.reason.mind` for the loop and its guarantees.
"""

from __future__ import annotations

from .budget import Budget, BudgetExceeded, BudgetTracker, CountingLLM
from .citations import Citation, CitationCheck, citation_for_chunk, validate_citations
from .mind import Answer, HypothesisResult, Mind, MindConfig
from .selection import ScoredQuestion, SelectionConfig, rank_questions, score_question, uncertainty
from .strategies import (
    DebateStrategy,
    Judgment,
    SelfConsistencyStrategy,
    SimpleStrategy,
    Strategy,
    get_strategy,
)

__all__ = [
    "Mind",
    "MindConfig",
    "Budget",
    "BudgetExceeded",
    "BudgetTracker",
    "CountingLLM",
    "Answer",
    "HypothesisResult",
    "Citation",
    "CitationCheck",
    "validate_citations",
    "citation_for_chunk",
    "SelectionConfig",
    "ScoredQuestion",
    "score_question",
    "rank_questions",
    "uncertainty",
    "Strategy",
    "SimpleStrategy",
    "SelfConsistencyStrategy",
    "DebateStrategy",
    "Judgment",
    "get_strategy",
]
