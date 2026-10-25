"""Source-reliability priors (the per-source baseline).

A reliability ``r`` is the prior probability that a claim asserted by the source is correct.
Users can always override (``KnowledgeStore.set_source_reliability`` or a ``reliability:``
front-matter field); these heuristics only provide a sensible default.
"""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse

__all__ = ["reliability_prior", "KIND_PRIORS", "DOMAIN_RULES"]

# source_type (front-matter ``source_type`` / ``type``) -> prior
KIND_PRIORS: dict[str, float] = {
    "systematic_review": 0.9,
    "meta_analysis": 0.9,
    "peer_reviewed": 0.82,
    "journal": 0.82,
    "rct": 0.82,
    "clinical_trial": 0.82,
    "cohort_study": 0.78,
    "government": 0.8,
    "encyclopedia": 0.72,
    "textbook": 0.75,
    "expert_qa": 0.7,
    "news": 0.6,
    "press_release": 0.45,
    "industry": 0.45,
    "blog": 0.4,
    "opinion": 0.4,
    "forum": 0.3,
    "social": 0.3,
    "observation": 0.65,
    "code": 0.7,
    "unknown": 0.6,
}

# (regex on hostname, prior, reason)
DOMAIN_RULES: list[tuple[str, float, str]] = [
    (r"(^|\.)(cochranelibrary|cochrane)\.", 0.9, "systematic-review publisher"),
    (r"(^|\.)(nih|ncbi\.nlm\.nih|who|cdc|nice\.org)\.", 0.85, "public-health agency"),
    (r"\.gov(\.[a-z]{2})?$", 0.82, "government domain"),
    (r"(^|\.)(nature|science|thelancet|nejm|bmj|jamanetwork|cell|plos|springer|wiley|sciencedirect)\.", 0.82, "journal"),
    (r"\.edu$|\.ac\.[a-z]{2}$", 0.78, "academic domain"),
    (r"(^|\.)arxiv\.org$", 0.68, "preprint server"),
    (r"(^|\.)wikipedia\.org$", 0.7, "encyclopedia"),
    (r"(^|\.)(reuters|apnews|bbc|nytimes|theguardian|economist|ft)\.", 0.65, "major news outlet"),
    (r"(^|\.)(medium|substack|blogspot|wordpress|tumblr)\.", 0.4, "blog platform"),
    (r"(^|\.)(reddit|twitter|x|facebook|tiktok|quora|instagram)\.(com|net)$", 0.3, "social media / forum"),
    (r"(^|\.)(github|gitlab)\.com$", 0.65, "code hosting"),
]


def reliability_prior(uri: str = "", metadata: dict[str, Any] | None = None) -> tuple[float, str]:
    """Return ``(reliability, reason)`` for a source.

    Precedence: explicit ``metadata["reliability"]`` > ``metadata["source_type"]`` (or ``type``)
    > URL domain rules > 0.6 default.
    """
    md = metadata or {}
    if md.get("reliability") is not None:
        try:
            r = float(md["reliability"])
            return min(max(r, 0.0), 1.0), "declared in source metadata"
        except (TypeError, ValueError):
            pass
    st = str(md.get("source_type") or md.get("type") or "").strip().lower().replace("-", "_").replace(" ", "_")
    if st in KIND_PRIORS:
        return KIND_PRIORS[st], f"source type '{st}'"
    if uri and re.match(r"^https?://", uri):
        host = (urlparse(uri).hostname or "").lower()
        for pattern, prior, reason in DOMAIN_RULES:
            if re.search(pattern, host):
                return prior, f"{reason} ({host})"
        return 0.55, f"unclassified web domain ({host})"
    return 0.6, "default prior"
