"""Dependency-free text heuristics used by the offline pipeline and by ``MockLLM``.

Nothing here is clever; it is deliberately simple, deterministic and fast so that the whole
Noesis pipeline (understand -> beliefs -> reasoning) can run with no model at all.

Public helpers:

* :func:`split_sentences` -> list of ``(sentence, start, end)`` spans
* :func:`tokenize`, :func:`content_words`
* :func:`extract_claims` -> list of claim dicts ``{"text", "start", "end", "concepts"}``
* :func:`extract_concepts` -> list of concept names
* :func:`polarity`, :func:`lexical_stance` -> ``(stance, strength)``
* :func:`tfidf_matrix`, :func:`cosine_matrix`, :func:`text_similarity`
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence

import numpy as np

STOPWORDS = frozenset(
    """a about above after again against all also am an and any are aren't as at be because been
    before being below between both but by can can't cannot could couldn't did didn't do does doesn't
    doing don't down during each few for from further had hadn't has hasn't have haven't having he
    her here hers herself him himself his how however i if in into is isn't it it's its itself just
    let's may me might more most much must mustn't my myself no nor not of off on once only or other
    ought our ours ourselves out over own per same shan't she should shouldn't so some such than that
    that's the their theirs them themselves then there there's these they this those through thus to
    too under until up upon very was wasn't we were weren't what when where which while who whom why
    will with won't would wouldn't you your yours yourself yourselves one two three many several
    often usually generally around among within without people person day days year years cup cups
    study studies found finds suggest suggests suggested report reports reported said says new like
    even still yet rather quite whether either neither less least every other another""".split()
)

NEGATIONS = frozenset(
    """not never none nobody nothing neither nor cannot can't don't doesn't didn't isn't aren't
    wasn't weren't won't wouldn't shouldn't couldn't hasn't haven't hadn't without lack lacks lacking
    fail fails failed""".split()
)

# Directional lexicon: word -> (axis, sign). Two sentences about the same topic whose first
# directional word on a shared axis has opposite sign are treated as opposite polarity
# ("coffee raises blood pressure" vs "coffee lowers blood pressure").
_AXES: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "quantity": (
        ("increase", "increases", "increased", "increasing", "raise", "raises", "raised", "higher",
         "more", "elevate", "elevates", "elevated", "boost", "boosts", "rise", "rises", "greater"),
        ("decrease", "decreases", "decreased", "decreasing", "reduce", "reduces", "reduced", "reducing",
         "lower", "lowers", "lowered", "less", "fewer", "drop", "drops", "cut", "cuts", "smaller"),
    ),
    "valence": (
        ("benefit", "benefits", "beneficial", "protective", "protects", "protect", "safe", "healthy",
         "good", "positive", "improve", "improves", "improved", "helps", "help", "prevents"),
        ("harm", "harms", "harmful", "dangerous", "unsafe", "unhealthy", "bad", "negative", "worsen",
         "worsens", "worsened", "hurts", "damages", "damage", "causes", "toxic"),
    ),
    "truth": (
        ("true", "confirmed", "confirms", "supports", "proven", "real", "established"),
        ("false", "refuted", "refutes", "disproven", "myth", "debunked", "unfounded"),
    ),
    "strength": (
        ("strong", "effective", "lasting", "persistent", "chronic", "sustained", "significant"),
        ("weak", "ineffective", "temporary", "transient", "acute", "short-lived", "negligible"),
    ),
    "hydration": (("hydrating", "hydrates", "hydration"), ("dehydrating", "dehydrates", "dehydration", "diuretic")),
}
_AXIS_MAP: dict[str, tuple[str, int]] = {}
for _axis, (_pos, _neg) in _AXES.items():
    for _w in _pos:
        _AXIS_MAP.setdefault(_w, (_axis, 1))
    for _w in _neg:
        _AXIS_MAP.setdefault(_w, (_axis, -1))
# "myth"/"unfounded"/"false" are also negations; keep them on the truth axis only.
ANTONYMS = tuple((_a, _p[0], _n[0]) for _a, (_p, _n) in _AXES.items())

_NO_NEG_RE = re.compile(
    r"\bno\s+(?:\w+\s+)?(evidence|effect|effects|association|link|increase|decrease|difference|benefit|"
    r"benefits|harm|change|proof|impact|relationship|role|risk|significant|clear|good|lasting|reason)\b",
    re.I,
)
_CLAUSE_RE = re.compile(
    r",?\s+\b(?:but|however|whereas|although|though|while|yet|because)\b\s+|,\s+(?:so|and so|which means)\s+|;\s+",
    re.I,
)
_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'\-]*")
_SENT_RE = re.compile(r"[^.!?\n]+(?:[.!?]+|\n|$)")
_ABBREV = ("e.g.", "i.e.", "et al.", "vs.", "Dr.", "Mr.", "Ms.", "Mrs.", "approx.", "St.", "Fig.", "No.")
_CLAIM_SKIP_PREFIX = re.compile(
    r"^(in this (article|post|piece|report)|this (article|post)|click|subscribe|read more|see also|"
    r"table of contents|keywords?:|title:|author:|published|copyright|disclaimer)",
    re.I,
)
_VERBISH = re.compile(
    r"\b(is|are|was|were|be|been|has|have|had|do|does|did|can|could|may|might|will|would|should|must|"
    r"\w+(?:s|ed|es))\b",
    re.I,
)


# ----------------------------------------------------------------------------- tokens
def tokenize(text: str) -> list[str]:
    """Lower-cased word tokens (keeps apostrophes/hyphens inside words)."""
    return [t.lower().strip("'-") for t in _TOKEN_RE.findall(text) if t.strip("'-")]


def stem(word: str) -> str:
    """Tiny suffix stripper so 'increases'/'increased'/'increase' (and 'cause'/'causes') collide."""
    w = word.lower()
    if len(w) <= 3:
        return w
    for suf in ("ations", "ation", "ingly", "edly", "ing", "ies", "ied", "ed", "es", "s", "ly"):
        if len(w) > len(suf) + 2 and w.endswith(suf):
            if suf in ("ation", "ations"):
                w = w[: -len(suf)] + "at"
                break
            if suf in ("ies", "ied"):
                w = w[: -len(suf)] + "y"
            elif suf == "s" and w.endswith("ss"):
                pass
            else:
                w = w[: -len(suf)]
            break
    if len(w) > 3 and w.endswith("e"):
        w = w[:-1]
    return w


def content_words(text: str) -> list[str]:
    """Stemmed non-stopword, non-negation tokens."""
    return [stem(t) for t in tokenize(text) if t not in STOPWORDS and t not in NEGATIONS and len(t) > 1]


# ----------------------------------------------------------------------------- sentences
def split_sentences(text: str) -> list[tuple[str, int, int]]:
    """Split ``text`` into sentences, returning ``(sentence, start, end)`` with exact spans."""
    protected = text
    for ab in _ABBREV:
        protected = protected.replace(ab, ab.replace(".", "\x00"))
    out: list[tuple[str, int, int]] = []
    for m in _SENT_RE.finditer(protected):
        raw = m.group(0)
        s, e = m.start(), m.end()
        # trim whitespace while keeping spans exact
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        if e > s:
            out.append((text[s:e], s, e))
        del raw
    return out


def _clean_line(s: str) -> str:
    s = re.sub(r"^[#>*\-•\d.)\s]+", "", s)  # markdown bullets/headers/numbering
    return re.sub(r"\s+", " ", s).strip()


def is_claim_like(sentence: str) -> bool:
    raw = sentence.strip()
    if raw.startswith("#") or not raw.endswith((".", "!", '."', ".'", ".)")):
        return False  # headings / titles / fragments
    s = _clean_line(sentence)
    words = tokenize(s)
    if not (5 <= len(words) <= 60):
        return False
    if s.endswith("?") or s.endswith(":"):
        return False
    if _CLAIM_SKIP_PREFIX.match(s):
        return False
    if not _VERBISH.search(s):
        return False
    if sum(1 for w in words if w not in STOPWORDS) < 3:
        return False
    return True


# ----------------------------------------------------------------------------- concepts
_PHRASE_BREAK = frozenset(
    """rose fell showed show shows linked link links associated caused cause causes drink drinks drinking
    drank consume consumes consuming consumed take takes taking make makes making get gets see seen
    appear appears seem seems help helps contain contains containing claim claims claimed""".split()
)


def _is_breaker(low: str) -> bool:
    return (
        low in STOPWORDS
        or low in NEGATIONS
        or low in _PHRASE_BREAK
        or low in _AXIS_MAP
        or (len(low) < 3 and not low.isdigit())
        or (len(low) > 4 and low.endswith("ed"))
    )


def _phrases(text: str) -> list[tuple[str, bool]]:
    """All 1-3 word n-grams inside maximal runs of non-stopword tokens.

    Returns ``(phrase, capitalized)`` where capitalized means a non sentence-initial capital.
    """
    out: list[tuple[str, bool]] = []
    for sent, _, _ in split_sentences(text):
        if sent.lstrip().startswith("#"):
            continue  # headings are titles, not content
        raw_tokens = _TOKEN_RE.findall(sent)
        n_cap = sum(1 for t in raw_tokens if t[:1].isupper())
        title_case = len(raw_tokens) > 2 and n_cap / len(raw_tokens) > 0.5
        run: list[str] = []
        caps: list[bool] = []
        for idx, tok in enumerate(raw_tokens):
            if _is_breaker(tok.lower()) or (tok.isdigit() and len(tok) > 1):
                if run:
                    out.extend(_emit(run, caps))
                run, caps = [], []
                continue
            run.append(tok)
            caps.append(tok[:1].isupper() and idx > 0 and not title_case and not (tok.isupper() and len(tok) > 1))
        if run:
            out.extend(_emit(run, caps))
    return out


def _emit(run: list[str], caps: list[bool]) -> list[tuple[str, bool]]:
    res = []
    for n in (1, 2, 3):
        for i in range(0, len(run) - n + 1):
            gram = run[i : i + n]
            if gram[0].isdigit() or (n > 1 and gram[-1].isdigit()):
                continue
            res.append((" ".join(gram), any(caps[i : i + n])))
    return res


_NUMBER_WORDS = frozenset("zero one two three four five six seven eight nine ten eleven twelve hundred thousand million".split())
_DROP_CONCEPT_WORDS = frozenset(
    "evidence research researchers data result results risk effect effects participants trial "
    "authors article review analysis amount amounts level levels people adults group groups hours minutes "
    "week weeks month months time times question answer note reader way ways thing things lot".split()
)


def extract_concepts(text: str, max_concepts: int = 12) -> list[str]:
    """Return salient noun-phrase-like concepts in ``text``.

    Heuristic: capitalized (non sentence-initial) phrases, phrases repeated in the text, and
    multi-word phrases containing digits or hyphens (e.g. "type 2 diabetes").
    """
    phrases = _phrases(text)
    counts: Counter[str] = Counter()
    display: dict[str, str] = {}
    capitalized: set[str] = set()
    for ph, cap in phrases:
        key = ph.lower()
        if all(w in _DROP_CONCEPT_WORDS for w in key.split()):
            continue
        counts[key] += 1
        display.setdefault(key, ph if cap else key)
        if cap:
            capitalized.add(key)
    scored: list[tuple[float, str]] = []
    for key, c in counts.items():
        n_words = len(key.split())
        score = c + (1.0 if key in capitalized else 0) + 0.25 * (n_words - 1)
        if n_words == 1 and (len(key) < 4 or key in _NUMBER_WORDS):
            continue
        keep = (c >= 2 and (n_words > 1 or len(key) >= 5)) or (key in capitalized and n_words > 1) or (
            n_words > 1 and any(ch.isdigit() for ch in key)
        )
        if keep:
            scored.append((score, key))
    scored.sort(key=lambda x: (-x[0], x[1]))
    result: list[str] = []
    for _, key in scored:
        if any(f" {key} " in f" {r.lower()} " or f" {r.lower()} " in f" {key} " for r in result):
            continue
        result.append(display[key])
        if len(result) >= max_concepts:
            break
    return result


def concepts_in(text: str, concept_names: Sequence[str]) -> list[str]:
    """Which of ``concept_names`` occur (stem-matched) in ``text``."""
    toks = {stem(t) for t in tokenize(text)}
    low = text.lower()
    hits = []
    for name in concept_names:
        parts = [stem(p) for p in tokenize(name)]
        if parts and (name.lower() in low or all(p in toks for p in parts)):
            hits.append(name)
    return hits


# ----------------------------------------------------------------------------- claims
def extract_claims(text: str, offset: int = 0, max_claims: int = 40) -> list[dict]:
    """Sentence-level claims with exact spans (relative to ``text`` + ``offset``)."""
    concepts = extract_concepts(text, max_concepts=20)
    out = []
    for sent, s, e in split_sentences(text):
        if not is_claim_like(sent):
            continue
        clean = _clean_line(sent)
        out.append(
            {
                "text": clean,
                "quote": sent,
                "start": s + offset,
                "end": e + offset,
                "concepts": concepts_in(clean, concepts) or extract_concepts(clean, max_concepts=3),
                "hedged": bool(re.search(r"\b(may|might|could|possibly|suggests?|appears?)\b", clean, re.I)),
            }
        )
        if len(out) >= max_claims:
            break
    return out


# ----------------------------------------------------------------------------- stance
def polarity(text: str) -> tuple[int, dict[str, int]]:
    """Return ``(negation_parity, axis_signs)``.

    negation_parity is +1 or -1; axis_signs maps a directional axis (quantity, valence, truth,
    strength, hydration) to the sign (+1/-1) of the first directional word on that axis.
    """
    toks = tokenize(text)
    neg = sum(1 for t in toks if t in NEGATIONS)
    neg += len(_NO_NEG_RE.findall(text))
    if re.search(r"\b(little|scant|insufficient|weak) evidence\b", text, re.I):
        neg += 1
    dirs: dict[str, int] = {}
    for t in toks:
        if t in _AXIS_MAP:
            axis, sign = _AXIS_MAP[t]
            dirs.setdefault(axis, sign)
    return (-1 if neg % 2 else 1), dirs


def build_idf(texts: Sequence[str]) -> dict[str, float]:
    """Smoothed IDF over stemmed content words, computed at the sentence level.

    Pass the result as ``idf=`` to :func:`lexical_stance` so ubiquitous topic words (e.g.
    "coffee" in a coffee corpus) count less than discriminative ones.
    """
    units = [s for t in texts for s, _, _ in split_sentences(t)] or list(texts)
    df: Counter[str] = Counter()
    for u in units:
        df.update(set(content_words(u)))
    n = len(units)
    return {w: math.log((1 + n) / (1 + c)) + 1.0 for w, c in df.items()}


def _overlap(a: str, b: str, idf: dict[str, float] | None = None) -> float:
    wa, wb = set(content_words(a)), set(content_words(b))
    # directional words should not count as topical overlap (they are what differs)
    wa = {w for w in wa if w not in _ANTONYM_STEMS}
    wb = {w for w in wb if w not in _ANTONYM_STEMS}
    if not wa or not wb:
        return 0.0
    inter = wa & wb
    if idf is None:
        return len(inter) / math.sqrt(len(wa) * len(wb))
    default = max(idf.values(), default=1.0)

    def w(x: str) -> float:
        return idf.get(x, default)

    num = sum(w(x) for x in inter)
    return num / math.sqrt(sum(w(x) for x in wa) * sum(w(x) for x in wb))


_ANTONYM_STEMS = frozenset(stem(w) for w in _AXIS_MAP)


def clauses(text: str) -> list[str]:
    """Split a sentence into contrastive clauses ("A, but B" -> ["A", "B"])."""
    parts = [p.strip(" ,;") for p in _CLAUSE_RE.split(text) if p and p.strip(" ,;")]
    return parts or [text]


def lexical_stance(
    claim: str, passage: str, min_overlap: float = 0.3, idf: dict[str, float] | None = None
) -> tuple[str, float]:
    """Classify whether ``passage`` supports / contradicts / is merely context for ``claim``.

    Works clause-by-clause: the best-overlapping (claim clause, passage clause) pair decides.
    Polarity = negation parity x directional-word agreement. Returns ``(stance, strength)``.
    """
    best: tuple[float, str, str] = (-1.0, claim, passage)
    whole = _overlap(claim, passage, idf)
    for cc in clauses(claim):
        for pc_ in clauses(passage):
            ov = _overlap(cc, pc_, idf)
            # a clause pair must share >= 2 content words unless it is the whole sentences
            if (cc, pc_) != (claim, passage) and len(
                (set(content_words(cc)) & set(content_words(pc_))) - _ANTONYM_STEMS
            ) < 2:
                continue
            # clause matches are discounted toward the whole-sentence overlap
            if (cc, pc_) != (claim, passage):
                ov = 0.5 * ov + 0.5 * max(ov, whole) if whole > 0 else 0.8 * ov
            if ov > best[0]:
                best = (ov, cc, pc_)
    ov, c_cl, p_cl = best
    ov = max(ov, 0.0)
    if ov < min_overlap:
        return "context", round(min(ov, 0.3), 3)
    pc, dc = polarity(c_cl)
    pp, dp = polarity(p_cl)
    sign = pc * pp
    for axis, d in dc.items():
        if axis in dp and dp[axis] != d:
            sign *= -1
    strength = round(min(1.0, 0.35 + 0.65 * ov), 3)
    return ("supports" if sign > 0 else "contradicts"), strength


# ----------------------------------------------------------------------------- similarity
def tfidf_matrix(texts: Sequence[str], ngram: int = 2) -> np.ndarray:
    """Return an L2-normalized TF-IDF matrix (n_texts x vocab) using stemmed content words."""
    docs = []
    for t in texts:
        words = content_words(t)
        grams = list(words)
        if ngram >= 2:
            grams += [f"{a}_{b}" for a, b in zip(words, words[1:])]
        docs.append(Counter(grams))
    vocab: dict[str, int] = {}
    for d in docs:
        for g in d:
            vocab.setdefault(g, len(vocab))
    n = len(texts)
    if not vocab or n == 0:
        return np.zeros((n, 1), dtype=np.float32)
    df = np.zeros(len(vocab), dtype=np.float32)
    for d in docs:
        for g in d:
            df[vocab[g]] += 1
    idf = np.log((1 + n) / (1 + df)) + 1.0
    m = np.zeros((n, len(vocab)), dtype=np.float32)
    for i, d in enumerate(docs):
        for g, c in d.items():
            m[i, vocab[g]] = (1 + math.log(c)) * idf[vocab[g]]
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


def cosine_matrix(a: np.ndarray, b: np.ndarray | None = None) -> np.ndarray:
    b = a if b is None else b
    an = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-12, None)
    bn = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-12, None)
    return an @ bn.T


def text_similarity(a: str, b: str) -> float:
    """TF-IDF cosine similarity of two strings (fit on just the pair)."""
    m = tfidf_matrix([a, b])
    return float(m[0] @ m[1])


def best_sentences(query: str, text: str, k: int = 2, idf: dict[str, float] | None = None) -> list[str]:
    """The ``k`` sentences of ``text`` with highest content-word overlap with ``query``."""
    sents = [s for s, _, _ in split_sentences(text)]
    scored = sorted(((_overlap(query, s, idf), i, s) for i, s in enumerate(sents)), key=lambda x: (-x[0], x[1]))
    return [s for sc, _, s in scored[:k] if sc > 0]
