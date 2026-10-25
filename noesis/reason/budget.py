"""Budgets and LLM call accounting for the reasoning loop."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

__all__ = ["Budget", "BudgetExceeded", "BudgetTracker", "CountingLLM"]


@dataclass
class Budget:
    """Limits for one :meth:`Mind.think` run. ``None`` = unlimited.

    * ``steps``: maximum reasoning iterations (one question per iteration).
    * ``max_llm_calls``: hard cap on LLM ``complete`` calls (including JSON repair retries and
      the stance checks done while linking evidence).
    * ``seconds``: wall-clock limit, checked between phases.
    """

    steps: int = 5
    max_llm_calls: int | None = None
    seconds: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class BudgetExceeded(Exception):
    """Raised (deliberately *not* an ``LLMError``) when a budget limit is hit mid-step.

    Not subclassing ``LLMError`` matters: core helpers catch ``LLMError`` and fall back to
    heuristics, whereas a budget stop must propagate out to the loop.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class BudgetTracker:
    def __init__(self, budget: Budget) -> None:
        self.budget = budget
        self.started = time.monotonic()
        self.llm_calls = 0
        self.steps = 0

    def elapsed(self) -> float:
        return time.monotonic() - self.started

    def exhausted(self) -> str | None:
        """Return a stop reason if any limit is reached, else ``None``."""
        b = self.budget
        if b.steps is not None and self.steps >= b.steps:
            return "budget:steps"
        if b.max_llm_calls is not None and self.llm_calls >= b.max_llm_calls:
            return "budget:llm_calls"
        if b.seconds is not None and self.elapsed() >= b.seconds:
            return "budget:seconds"
        return None

    def check(self) -> None:
        """Raise :class:`BudgetExceeded` on call/time limits (steps are checked by the loop)."""
        b = self.budget
        if b.max_llm_calls is not None and self.llm_calls >= b.max_llm_calls:
            raise BudgetExceeded("budget:llm_calls")
        if b.seconds is not None and self.elapsed() >= b.seconds:
            raise BudgetExceeded("budget:seconds")

    def usage(self) -> dict[str, Any]:
        return {"llm_calls": self.llm_calls, "steps": self.steps, "seconds": round(self.elapsed(), 3)}


class CountingLLM:
    """Transparent proxy around an LLM that counts calls and enforces the tracker's budget.

    Everything except ``complete`` is delegated, so core helpers (``link_evidence``,
    ``get_embedder``...) can be handed the proxy unchanged.
    """

    def __init__(self, llm: Any, tracker: BudgetTracker | None = None) -> None:
        self._llm = llm
        self.tracker = tracker

    def complete(self, *args: Any, **kwargs: Any) -> Any:
        if self.tracker is not None:
            self.tracker.check()
        usage = getattr(self._llm, "usage", None)
        before = getattr(usage, "calls", None)
        try:
            return self._llm.complete(*args, **kwargs)
        finally:
            if self.tracker is not None:
                after = getattr(usage, "calls", None)
                self.tracker.llm_calls += (after - before) if isinstance(before, int) and isinstance(after, int) else 1

    def __getattr__(self, name: str) -> Any:
        return getattr(self._llm, name)
