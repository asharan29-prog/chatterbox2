"""Web tools for the Noesis Research Agent: search providers, a guarded fetcher, domain priors.

Everything here treats the internet as **untrusted data**:

* ``fetch`` only speaks http(s), never runs JavaScript, never submits forms, refuses
  executables/archives, enforces timeouts, a byte limit and a content-type allow-list, honours
  ``robots.txt`` (RFC 9309 semantics), rate-limits per host and blocks private, loopback,
  link-local and otherwise non-public addresses (SSRF guard, re-checked on every redirect hop).
* Fetched text is returned as plain data; :meth:`FetchedPage.as_document` wraps it as an
  untrusted ``<document>`` for prompts, and pages containing instruction-like text are flagged.
* Search providers are pluggable: DuckDuckGo HTML (no key, fragile scraping), Brave Search API,
  Tavily and SearXNG. All accept an ``httpx`` transport so tests run fully offline.

>>> web = WebToolkit.create("ddg")                    # or "brave", "tavily", "searxng", "mock"
>>> results = web.search("is coffee good for the heart", max_results=5)
>>> page = web.fetch(results[0].url)
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import re
import socket
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote_plus, unquote, urljoin, urlparse
from urllib.robotparser import RobotFileParser

import httpx

from ..ingest.reliability import reliability_prior
from ..llm.prompts import wrap_document
from ..reason.safety import looks_like_injection

__all__ = [
    "USER_AGENT",
    "SearchResult",
    "FetchedPage",
    "FetchError",
    "SearchError",
    "SearchProvider",
    "DuckDuckGoProvider",
    "BraveProvider",
    "TavilyProvider",
    "SearxngProvider",
    "get_search_provider",
    "PROVIDERS",
    "DomainPolicy",
    "RateLimiter",
    "DiskCache",
    "WebFetcher",
    "WebToolkit",
    "extract_main_text",
    "registrable_domain",
    "is_public_address",
    "CONTENT_FARMS",
]

USER_AGENT = "NoesisResearchAgent/0.1 (+https://github.com/asharan29-prog/chatterbox2)"
ROBOTS_AGENT = "NoesisResearchAgent"


class FetchError(RuntimeError):
    """A page could not (or must not) be fetched. ``reason`` is a short machine-readable code."""

    def __init__(self, message: str, reason: str = "error") -> None:
        super().__init__(message)
        self.reason = reason


class SearchError(RuntimeError):
    """A search provider failed or is misconfigured."""


# ============================================================================ data
@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str = ""
    provider: str = ""
    rank: int = 0
    query: str = ""

    @property
    def domain(self) -> str:
        return registrable_domain(self.url)

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "domain": self.domain}


@dataclass
class FetchedPage:
    url: str
    final_url: str
    title: str
    text: str
    content_type: str = ""
    status: int = 200
    fetched_at: str = ""
    truncated: bool = False
    from_cache: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def domain(self) -> str:
        return registrable_domain(self.final_url or self.url)

    @property
    def injection_suspect(self) -> bool:
        return bool(self.metadata.get("injection_suspect"))

    def as_document(self, **attrs: Any) -> str:
        """The page as an untrusted ``<document>`` block (data, never instructions)."""
        return wrap_document(self.text, source=self.final_url or self.url, title=self.title, untrusted="true", **attrs)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ============================================================================ helpers
_MULTI_TLD = {"co.uk", "ac.uk", "gov.uk", "org.uk", "com.au", "edu.au", "gov.au", "co.jp", "ac.jp", "co.nz",
              "com.br", "co.in", "ac.in", "gov.in", "com.cn", "edu.cn", "co.za", "ac.za", "com.mx", "co.kr"}


def _host(url: str) -> str:
    return (urlparse(url).hostname or "").lower().rstrip(".")


def registrable_domain(url_or_host: str) -> str:
    """Approximate eTLD+1 (``www.bbc.co.uk`` -> ``bbc.co.uk``) used to diversify sources."""
    host = _host(url_or_host) if "://" in url_or_host else url_or_host.lower().rstrip(".")
    if not host:
        return ""
    try:
        ipaddress.ip_address(host.strip("[]"))
        return host
    except ValueError:
        pass
    parts = host.split(".")
    if len(parts) >= 3 and ".".join(parts[-2:]) in _MULTI_TLD:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])


def domain_matches(host: str, domains: Iterable[str]) -> bool:
    host = host.lower().rstrip(".")
    for d in domains:
        d = d.lower().strip().lstrip(".").rstrip(".")
        d = re.sub(r"^https?://", "", d).split("/")[0]
        if d and (host == d or host.endswith("." + d)):
            return True
    return False


def is_public_address(ip: str) -> bool:
    """True only for globally routable unicast addresses."""
    try:
        addr = ipaddress.ip_address(ip.split("%")[0])
    except ValueError:
        return False
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped:
        addr = addr.ipv4_mapped
    return not (
        addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_multicast
        or addr.is_reserved or addr.is_unspecified or not addr.is_global
    )


def default_resolver(host: str) -> list[str]:
    infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    return sorted({str(i[4][0]) for i in infos})


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


# ============================================================================ policy / priors
#: Known low-quality / SEO content farms and aggregators (prior 0.3). Extend via DomainPolicy.
CONTENT_FARMS = (
    "ehow.com", "answers.com", "ezinearticles.com", "hubpages.com", "livestrong.com", "wikihow.com",
    "buzzfeed.com", "examiner.com", "articlesbase.com", "squidoo.com", "bustle.com", "thoughtcatalog.com",
    "naturalnews.com", "infowars.com", "mercola.com", "beforeitsnews.com",
)
#: Extra high-quality scholarly hosts not covered by the core rules.
SCHOLARLY = (
    ("pubmed.ncbi.nlm.nih.gov", 0.85, "PubMed (indexed literature)"),
    ("doi.org", 0.8, "DOI resolver (scholarly record)"),
    ("jstor.org", 0.8, "scholarly archive"),
    ("semanticscholar.org", 0.75, "scholarly index"),
    ("europepmc.org", 0.82, "Europe PMC (indexed literature)"),
    ("biorxiv.org", 0.66, "preprint server"),
    ("medrxiv.org", 0.66, "preprint server"),
    ("ssrn.com", 0.66, "preprint server"),
    ("britannica.com", 0.74, "encyclopedia"),
    ("oecd.org", 0.8, "intergovernmental organization"),
    ("un.org", 0.8, "intergovernmental organization"),
    ("europa.eu", 0.8, "EU institution"),
    ("worldbank.org", 0.8, "intergovernmental organization"),
)


@dataclass
class DomainPolicy:
    """Which domains may be used and how much each is trusted a priori.

    * ``deny``: never search-select or fetch these domains (subdomains included).
    * ``allow``: when non-empty, ONLY these domains are used (a whitelist).
    * ``trust``: per-domain reliability overrides, e.g. ``{"mylab.org": 0.9}``.
    * ``content_farms``: domains given a low prior (default :data:`CONTENT_FARMS`).

    Priors: user ``trust`` > scholarly metadata on the page (``citation_*`` meta tags, DOI) >
    agent rules (content farms, scholarly hosts) > core Noesis domain rules (``.gov``, ``.edu``,
    journals, news, blogs, social) > 0.55 for unclassified sites.
    """

    allow: list[str] = field(default_factory=list)
    deny: list[str] = field(default_factory=list)
    trust: dict[str, float] = field(default_factory=dict)
    content_farms: Sequence[str] = CONTENT_FARMS

    def permitted(self, url: str) -> tuple[bool, str]:
        host = _host(url)
        if not host:
            return False, "no host"
        if domain_matches(host, self.deny):
            return False, "denied domain"
        if self.allow and not domain_matches(host, self.allow):
            return False, "not in allow list"
        return True, ""

    def reliability(self, url: str, metadata: dict[str, Any] | None = None) -> tuple[float, str]:
        host = _host(url)
        md = metadata or {}
        for d, r in sorted(self.trust.items(), key=lambda kv: -len(kv[0])):
            if domain_matches(host, [d]):
                return min(max(float(r), 0.0), 1.0), f"user trust override ({d})"
        if md.get("citation_journal_title") or md.get("doi"):
            j = md.get("citation_journal_title") or "DOI present"
            return 0.82, f"scholarly article metadata ({j})"
        if domain_matches(host, self.content_farms):
            return 0.3, f"known content farm / low-quality site ({host})"
        for d, r, why in SCHOLARLY:
            if domain_matches(host, [d]):
                return r, f"{why} ({host})"
        r, why = reliability_prior(url if url.startswith("http") else f"https://{host}/", {})
        return r, why


# ============================================================================ rate limit / cache
class RateLimiter:
    """At most one request per ``min_interval`` seconds per key (host or provider)."""

    def __init__(self, min_interval: float = 1.0, *, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.min_interval = float(min_interval)
        self.clock, self.sleep = clock, sleep
        self._last: dict[str, float] = {}
        self._lock = threading.Lock()

    def wait(self, key: str) -> float:
        with self._lock:
            now = self.clock()
            last = self._last.get(key)
            delay = 0.0 if last is None else max(0.0, self.min_interval - (now - last))
            self._last[key] = now + delay
        if delay > 0:
            self.sleep(delay)
        return delay


class DiskCache:
    """Tiny JSON-file cache (``.noesis/webcache`` by default) with a TTL."""

    def __init__(self, directory: str | Path | None = None, *, ttl: float = 7 * 86400, enabled: bool = True) -> None:
        self.dir = Path(directory or os.environ.get("NOESIS_WEBCACHE") or Path(".noesis") / "webcache")
        self.ttl = ttl
        self.enabled = enabled

    def _path(self, namespace: str, key: str) -> Path:
        h = hashlib.sha256(f"{namespace}\0{key}".encode()).hexdigest()
        return self.dir / namespace / f"{h[:40]}.json"

    def get(self, namespace: str, key: str) -> Any:
        if not self.enabled:
            return None
        p = self._path(namespace, key)
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if time.time() - float(data.get("t", 0)) > self.ttl:
            return None
        return data.get("v")

    def set(self, namespace: str, key: str, value: Any) -> None:
        if not self.enabled:
            return
        p = self._path(namespace, key)
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps({"t": time.time(), "k": key, "v": value}), encoding="utf-8")
        except OSError:  # cache is best-effort
            pass


# ============================================================================ extraction
_DROP_TAGS = ["script", "style", "noscript", "template", "iframe", "object", "embed", "svg", "canvas",
              "form", "button", "input", "select", "textarea", "nav", "footer", "header", "aside", "menu", "dialog"]
_NEG_HINT = re.compile(r"comment|footer|sidebar|nav|menu|share|social|advert|promo|related|cookie|banner|popup|subscribe", re.I)
_POS_HINT = re.compile(r"article|content|main|post|entry|story|body|text", re.I)
_META_KEYS = ("citation_journal_title", "citation_title", "citation_doi", "dc.identifier", "description",
              "og:description", "article:published_time", "citation_publication_date", "author", "citation_author")


def extract_main_text(html: str, url: str = "") -> tuple[str, str, dict[str, Any]]:
    """Readability-style extraction. Returns ``(title, text, metadata)``.

    Scores candidate containers by paragraph text length, comma count and class/id hints,
    penalizes link density, keeps the best one (falls back to the whole body). Never executes
    scripts; forms, navigation and boilerplate are removed.
    """
    try:
        from bs4 import BeautifulSoup  # type: ignore[import-not-found]
    except ImportError:
        from ..ingest.loaders import html_to_text

        title, text = html_to_text(html)
        return title, _clean(text), {}
    soup = BeautifulSoup(html, "html.parser")
    meta: dict[str, Any] = {}
    for m in soup.find_all("meta"):
        k = (m.get("name") or m.get("property") or "").strip().lower()
        v = (m.get("content") or "").strip()
        if k in _META_KEYS and v and k not in meta:
            meta[k] = v[:500]
    doi = meta.get("citation_doi") or (meta.get("dc.identifier") if "10." in meta.get("dc.identifier", "") else "")
    if doi:
        meta["doi"] = doi
    title = ""
    if meta.get("citation_title"):
        title = meta["citation_title"]
    elif soup.title and soup.title.get_text(strip=True):
        title = soup.title.get_text(" ", strip=True)
    elif soup.find("h1"):
        title = soup.find("h1").get_text(" ", strip=True)
    for tag in soup(_DROP_TAGS):
        tag.decompose()
    for el in soup.find_all(attrs={"hidden": True}):
        el.decompose()
    for el in soup.find_all(style=re.compile(r"display\s*:\s*none|visibility\s*:\s*hidden", re.I)):
        el.decompose()

    best, best_score = None, 0.0
    nscore: dict[int, float] = {}
    for p in soup.find_all(["p", "pre", "blockquote", "li"]):
        t = p.get_text(" ", strip=True)
        if len(t) < 25:
            continue
        for depth, parent in enumerate((p.parent, p.parent.parent if p.parent else None)):
            if parent is None or parent.name in ("html", "[document]"):
                continue
            score = 1 + t.count(",") + min(len(t) / 100.0, 3.0)
            nscore[id(parent)] = nscore.get(id(parent), 0.0) + (score if depth == 0 else score / 2)
    for el in soup.find_all(True):
        s = nscore.get(id(el), 0.0)
        if not s:
            continue
        hint = " ".join([el.get("id") or "", *(el.get("class") or [])])
        if el.name in ("article", "main"):
            s *= 1.5
        if hint and _POS_HINT.search(hint):
            s *= 1.25
        if hint and _NEG_HINT.search(hint):
            s *= 0.3
        text_len = len(el.get_text(" ", strip=True)) or 1
        link_len = sum(len(a.get_text(" ", strip=True)) for a in el.find_all("a"))
        s *= 1 - min(link_len / text_len, 0.9)
        if s > best_score:
            best, best_score = el, s
    root = best if best is not None else (soup.body or soup)
    blocks: list[str] = []
    for el in root.find_all(["h1", "h2", "h3", "h4", "p", "li", "blockquote", "pre", "td", "figcaption"]):
        if el.find(["p", "li"]) and el.name in ("li", "td", "blockquote"):
            continue
        t = el.get_text(" ", strip=True)
        if t and (not blocks or blocks[-1] != t):
            blocks.append(t)
    if len(blocks) > 1 and title and blocks[0].strip().lower() == title.strip().lower():
        blocks = blocks[1:]  # the headline is returned as the title, not as body text
    text = "\n\n".join(blocks) if blocks else root.get_text("\n", strip=True)
    return title[:300], _clean(text), meta


def _clean(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\x00", "")
    text = re.sub(r"[ \t ]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


# ============================================================================ fetcher
_BLOCKED_EXT = re.compile(
    r"\.(exe|msi|msp|dmg|pkg|app|apk|aab|ipa|deb|rpm|bin|run|sh|bat|cmd|com|scr|ps1|vbs|jar|dll|so|"
    r"iso|img|zip|rar|7z|tar|gz|tgz|bz2|xz)$",
    re.I,
)
_ALLOWED_CT = ("text/html", "application/xhtml+xml", "text/plain", "text/markdown", "application/json",
               "application/ld+json", "application/xml", "text/xml", "application/pdf")


class WebFetcher:
    """Guarded page fetcher (see the module docstring for the safety rules).

    ``transport`` / ``resolver`` are injectable for offline tests. ``respect_robots`` is on by
    default; turning it off is the caller's responsibility (and against most sites' terms).
    """

    def __init__(
        self,
        *,
        policy: DomainPolicy | None = None,
        transport: httpx.BaseTransport | None = None,
        resolver: Callable[[str], list[str]] | None = None,
        cache: DiskCache | None = None,
        rate_limiter: RateLimiter | None = None,
        timeout: float = 15.0,
        max_bytes: int = 2_000_000,
        max_chars: int = 60_000,
        max_redirects: int = 5,
        respect_robots: bool = True,
        user_agent: str = USER_AGENT,
    ) -> None:
        self.policy = policy or DomainPolicy()
        self.transport = transport
        self.resolver = resolver or default_resolver
        self.cache = cache if cache is not None else DiskCache()
        self.rate = rate_limiter or RateLimiter(1.0)
        self.timeout = timeout
        self.max_bytes = max_bytes
        self.max_chars = max_chars
        self.max_redirects = max_redirects
        self.respect_robots = respect_robots
        self.user_agent = user_agent
        self._robots: dict[str, RobotFileParser | None] = {}
        self._client: httpx.Client | None = None

    # -- client -----------------------------------------------------------------
    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                timeout=httpx.Timeout(self.timeout, connect=min(self.timeout, 10.0)),
                follow_redirects=False,  # every hop is re-validated by check_url
                transport=self.transport,
                headers={"user-agent": self.user_agent, "accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.5"},
            )
        return self._client

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    # -- guards -----------------------------------------------------------------
    def check_url(self, url: str) -> None:
        """Raise :class:`FetchError` unless ``url`` is safe and permitted to fetch."""
        host = self._static_checks(url)
        self._check_host(host)

    def _static_checks(self, url: str) -> str:
        """Checks that need no network (scheme, credentials, policy, file type). Returns the host."""
        u = urlparse(url)
        if u.scheme not in ("http", "https"):
            raise FetchError(f"unsupported scheme {u.scheme!r}", "scheme")
        if u.username or u.password:
            raise FetchError("URLs with embedded credentials are refused", "credentials")
        host = (u.hostname or "").rstrip(".")
        if not host:
            raise FetchError("URL has no host", "host")
        ok, why = self.policy.permitted(url)
        if not ok:
            raise FetchError(f"{host}: {why}", "policy")
        if _BLOCKED_EXT.search(u.path or ""):
            raise FetchError(f"refusing to download executables/archives: {u.path}", "executable")
        if host.lower() in ("localhost", "localhost.localdomain") or host.lower().endswith((".localhost", ".local", ".internal")):
            raise FetchError(f"{host}: local hostnames are blocked (SSRF guard)", "ssrf")
        return host

    def _check_host(self, host: str) -> None:
        try:
            ipaddress.ip_address(host.strip("[]"))
            addrs = [host.strip("[]")]
        except ValueError:
            try:
                addrs = self.resolver(host)
            except (OSError, UnicodeError) as e:
                raise FetchError(f"{host}: DNS resolution failed ({e})", "dns") from e
        if not addrs:
            raise FetchError(f"{host}: no addresses", "dns")
        bad = [a for a in addrs if not is_public_address(a)]
        if bad:
            raise FetchError(f"{host} resolves to non-public address {bad[0]} (SSRF guard)", "ssrf")

    def robots_allows(self, url: str) -> bool:
        if not self.respect_robots:
            return True
        u = urlparse(url)
        base = f"{u.scheme}://{u.netloc}"
        if base not in self._robots:
            self._robots[base] = self._load_robots(base)
        rp = self._robots[base]
        return True if rp is None else rp.can_fetch(ROBOTS_AGENT, url)

    def _load_robots(self, base: str) -> RobotFileParser | None:
        cached = self.cache.get("robots", base)
        if cached is None:
            status, body = 0, ""
            try:
                self.check_url(base + "/robots.txt")
                self.rate.wait(_host(base))
                r = self.client.get(base + "/robots.txt")
                status, body = r.status_code, r.text[:200_000] if r.status_code == 200 else ""
            except FetchError:
                raise
            except httpx.HTTPError:
                status = 599
            cached = {"status": status, "body": body}
            self.cache.set("robots", base, cached)
        status, body = int(cached["status"]), str(cached["body"])
        rp = RobotFileParser()
        if 200 <= status < 300:
            rp.parse(body.splitlines())
            return rp
        if 400 <= status < 500:  # RFC 9309: unavailable -> no restrictions
            return None
        rp.parse(["User-agent: *", "Disallow: /"])  # unreachable (5xx/network) -> assume disallow
        return rp

    # -- fetch ------------------------------------------------------------------
    def fetch(self, url: str, *, use_cache: bool = True) -> FetchedPage:
        """Fetch and extract one page. Raises :class:`FetchError` with a ``reason`` code."""
        url = url.strip()
        self._static_checks(url)
        if use_cache:
            hit = self.cache.get("pages", url)
            if hit:
                page = FetchedPage(**hit)
                page.from_cache = True
                return page
        current = url
        for _hop in range(self.max_redirects + 1):
            self.check_url(current)
            if not self.robots_allows(current):
                raise FetchError(f"robots.txt disallows {current}", "robots")
            self.rate.wait(_host(current))
            try:
                with self.client.stream("GET", current) as resp:
                    if resp.is_redirect:
                        loc = resp.headers.get("location")
                        if not loc:
                            raise FetchError("redirect without location", "http")
                        current = urljoin(current, loc)
                        continue
                    if resp.status_code >= 400:
                        raise FetchError(f"HTTP {resp.status_code} for {current}", "http")
                    ctype = resp.headers.get("content-type", "").split(";")[0].strip().lower()
                    disp = resp.headers.get("content-disposition", "")
                    if "attachment" in disp.lower() and _BLOCKED_EXT.search(disp.strip('"; ').split("=")[-1].strip('"')):
                        raise FetchError("refusing executable attachment", "executable")
                    if ctype and not ctype.startswith(_ALLOWED_CT):
                        raise FetchError(f"unsupported content-type {ctype!r}", "content_type")
                    clen = resp.headers.get("content-length")
                    if clen and clen.isdigit() and int(clen) > max(self.max_bytes * 10, 25_000_000):
                        raise FetchError(f"page too large ({clen} bytes)", "too_large")
                    buf = bytearray()
                    truncated = False
                    for chunk in resp.iter_bytes():
                        buf += chunk
                        if len(buf) > self.max_bytes:
                            truncated = True
                            del buf[self.max_bytes:]
                            break
                    status = resp.status_code
                    encoding = resp.encoding or "utf-8"
            except httpx.TimeoutException as e:
                raise FetchError(f"timeout fetching {current}", "timeout") from e
            except httpx.HTTPError as e:
                raise FetchError(f"network error fetching {current}: {e}", "network") from e
            page = self._extract(url, current, bytes(buf), ctype, encoding, status, truncated)
            if use_cache:
                self.cache.set("pages", url, page.to_dict())
            return page
        raise FetchError(f"too many redirects for {url}", "redirects")

    def _extract(self, url: str, final: str, data: bytes, ctype: str, encoding: str, status: int, truncated: bool) -> FetchedPage:
        meta: dict[str, Any] = {}
        if ctype == "application/pdf" or data[:5] == b"%PDF-":
            if truncated:
                raise FetchError("PDF exceeds size limit", "too_large")
            title, text = self._pdf(data, final)
        else:
            try:
                raw = data.decode(encoding, errors="replace")
            except LookupError:
                raw = data.decode("utf-8", errors="replace")
            if ctype in ("text/html", "application/xhtml+xml") or (not ctype and raw.lstrip()[:15].lower().startswith(("<!doctype", "<html"))):
                title, text, meta = extract_main_text(raw, final)
            elif "json" in ctype:
                try:
                    text = json.dumps(json.loads(raw), indent=1, ensure_ascii=False)
                except ValueError:
                    text = raw
                title = ""
            else:
                title, text = "", _clean(raw)
        if len(text) > self.max_chars:
            text, truncated = text[: self.max_chars], True
        if not text.strip():
            raise FetchError(f"no extractable text at {final}", "empty")
        meta["injection_suspect"] = looks_like_injection(text)
        return FetchedPage(url=url, final_url=final, title=title or final, text=text, content_type=ctype,
                           status=status, fetched_at=_now(), truncated=truncated, metadata=meta)

    @staticmethod
    def _pdf(data: bytes, url: str) -> tuple[str, str]:
        try:
            import io

            from pypdf import PdfReader  # type: ignore[import-not-found]
        except ImportError as e:
            raise FetchError("PDF support needs pypdf: pip install 'noesis[pdf]'", "content_type") from e
        reader = PdfReader(io.BytesIO(data))
        text = "\n\n".join((p.extract_text() or "") for p in reader.pages[:60])
        title = ""
        try:
            title = (reader.metadata.title or "") if reader.metadata else ""
        except Exception:  # noqa: BLE001 - malformed metadata
            pass
        return title or url, _clean(text)


# ============================================================================ search providers
class SearchProvider:
    """Base class. Subclasses implement :meth:`_search` -> list of SearchResult."""

    name = "base"
    #: endpoint host used for rate-limiting
    host = ""

    def __init__(self, *, transport: httpx.BaseTransport | None = None, timeout: float = 15.0,
                 rate_limiter: RateLimiter | None = None, cache: DiskCache | None = None) -> None:
        self.transport = transport
        self.timeout = timeout
        self.rate = rate_limiter or RateLimiter(1.0)
        self.cache = cache

    def _client(self) -> httpx.Client:
        return httpx.Client(timeout=self.timeout, transport=self.transport, follow_redirects=True,
                            headers={"user-agent": USER_AGENT})

    def search(self, query: str, max_results: int = 8) -> list[SearchResult]:
        query = " ".join(query.split())[:400]
        if not query:
            return []
        key = f"{query}\0{max_results}"
        if self.cache is not None:
            hit = self.cache.get(f"search-{self.name}", key)
            if hit is not None:
                return [SearchResult(**r) for r in hit]
        self.rate.wait(self.name)
        try:
            results = self._search(query, max_results)
        except httpx.HTTPStatusError as e:
            raise SearchError(f"{self.name}: HTTP {e.response.status_code}") from e
        except httpx.HTTPError as e:
            raise SearchError(f"{self.name}: {type(e).__name__}: {e}") from e
        out, seen = [], set()
        for r in results:
            if not r.url.startswith(("http://", "https://")) or r.url in seen:
                continue
            seen.add(r.url)
            r.provider, r.query, r.rank = self.name, query, len(out) + 1
            r.title, r.snippet = _clean(r.title)[:300], _clean(r.snippet)[:600]
            out.append(r)
            if len(out) >= max_results:
                break
        if self.cache is not None:
            self.cache.set(f"search-{self.name}", key, [asdict(r) for r in out])
        return out

    def _search(self, query: str, max_results: int) -> list[SearchResult]:  # pragma: no cover
        raise NotImplementedError


class DuckDuckGoProvider(SearchProvider):
    """DuckDuckGo's no-JavaScript HTML endpoint (no API key).

    FRAGILE: this scrapes HTML that DuckDuckGo may change at any time, and it may rate-limit or
    serve a challenge page to automated clients. Use a keyed API (Brave, Tavily) or your own
    SearXNG for anything serious. Ads (``result--ad``) are skipped; redirect links
    (``/l/?uddg=...``) are unwrapped.
    """

    name = "ddg"
    endpoint = "https://html.duckduckgo.com/html/"

    def _search(self, query: str, max_results: int) -> list[SearchResult]:
        with self._client() as c:
            r = c.post(self.endpoint, data={"q": query, "kl": "wt-wt"})
            r.raise_for_status()
        results = self.parse(r.text)
        if not results and re.search(r"anomaly|captcha|challenge", r.text, re.I):
            raise SearchError("ddg: DuckDuckGo served a bot challenge; try another --search provider")
        return results

    @staticmethod
    def unwrap(href: str) -> str:
        if href.startswith("//"):
            href = "https:" + href
        u = urlparse(href)
        if u.hostname and u.hostname.endswith("duckduckgo.com") and u.path.startswith("/l/"):
            target = parse_qs(u.query).get("uddg", [""])[0]
            return unquote(target)
        return href

    @classmethod
    def parse(cls, html: str) -> list[SearchResult]:
        out: list[SearchResult] = []
        try:
            from bs4 import BeautifulSoup  # type: ignore[import-not-found]
        except ImportError:
            for m in re.finditer(r'<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', html, re.S):
                out.append(SearchResult(re.sub(r"<[^>]+>", "", m.group(2)), cls.unwrap(m.group(1).replace("&amp;", "&"))))
            return out
        soup = BeautifulSoup(html, "html.parser")
        for res in soup.select("div.result, div.web-result"):
            classes = " ".join(res.get("class") or [])
            if "result--ad" in classes:
                continue
            a = res.select_one("a.result__a")
            if a is None or not a.get("href"):
                continue
            sn = res.select_one(".result__snippet")
            out.append(SearchResult(a.get_text(" ", strip=True), cls.unwrap(a["href"]), sn.get_text(" ", strip=True) if sn else ""))
        return out


class BraveProvider(SearchProvider):
    """Brave Search API (``BRAVE_API_KEY``; https://api.search.brave.com)."""

    name = "brave"
    endpoint = "https://api.search.brave.com/res/v1/web/search"

    def __init__(self, api_key: str | None = None, **kw: Any) -> None:
        super().__init__(**kw)
        self.api_key = api_key or os.environ.get("BRAVE_API_KEY", "")
        if not self.api_key:
            raise SearchError("brave: set BRAVE_API_KEY (https://brave.com/search/api/)")

    def _search(self, query: str, max_results: int) -> list[SearchResult]:
        with self._client() as c:
            r = c.get(self.endpoint, params={"q": query, "count": min(max_results, 20)},
                      headers={"X-Subscription-Token": self.api_key, "accept": "application/json"})
            r.raise_for_status()
        return self.parse(r.json())

    @staticmethod
    def parse(data: dict[str, Any]) -> list[SearchResult]:
        rows = ((data or {}).get("web") or {}).get("results") or []
        return [SearchResult(str(x.get("title", "")), str(x.get("url", "")), re.sub(r"<[^>]+>", "", str(x.get("description", ""))))
                for x in rows if isinstance(x, dict)]


class TavilyProvider(SearchProvider):
    """Tavily search API (``TAVILY_API_KEY``; https://tavily.com)."""

    name = "tavily"
    endpoint = "https://api.tavily.com/search"

    def __init__(self, api_key: str | None = None, **kw: Any) -> None:
        super().__init__(**kw)
        self.api_key = api_key or os.environ.get("TAVILY_API_KEY", "")
        if not self.api_key:
            raise SearchError("tavily: set TAVILY_API_KEY (https://tavily.com)")

    def _search(self, query: str, max_results: int) -> list[SearchResult]:
        with self._client() as c:
            r = c.post(self.endpoint, json={"query": query, "max_results": min(max_results, 20), "search_depth": "basic"},
                       headers={"Authorization": f"Bearer {self.api_key}"})
            r.raise_for_status()
        return self.parse(r.json())

    @staticmethod
    def parse(data: dict[str, Any]) -> list[SearchResult]:
        rows = (data or {}).get("results") or []
        return [SearchResult(str(x.get("title", "")), str(x.get("url", "")), str(x.get("content", "")))
                for x in rows if isinstance(x, dict)]


class SearxngProvider(SearchProvider):
    """A SearXNG instance (``SEARXNG_URL``, e.g. ``http://localhost:8888``) with JSON output enabled
    (``search.formats: [html, json]`` in its settings.yml)."""

    name = "searxng"

    def __init__(self, base_url: str | None = None, **kw: Any) -> None:
        super().__init__(**kw)
        self.base_url = (base_url or os.environ.get("SEARXNG_URL", "")).rstrip("/")
        if not self.base_url:
            raise SearchError("searxng: set SEARXNG_URL to your instance, e.g. http://localhost:8888")

    def _search(self, query: str, max_results: int) -> list[SearchResult]:
        with self._client() as c:
            r = c.get(f"{self.base_url}/search", params={"q": query, "format": "json"})
            r.raise_for_status()
        return self.parse(r.json())

    @staticmethod
    def parse(data: dict[str, Any]) -> list[SearchResult]:
        rows = (data or {}).get("results") or []
        return [SearchResult(str(x.get("title", "")), str(x.get("url", "")), str(x.get("content", "")))
                for x in rows if isinstance(x, dict)]


PROVIDERS: dict[str, type[SearchProvider]] = {
    "ddg": DuckDuckGoProvider,
    "duckduckgo": DuckDuckGoProvider,
    "brave": BraveProvider,
    "tavily": TavilyProvider,
    "searxng": SearxngProvider,
}


def get_search_provider(name: str | None = None, **kw: Any) -> SearchProvider:
    """``name``: ddg | brave | tavily | searxng (default ``$NOESIS_SEARCH`` or ddg)."""
    name = (name or os.environ.get("NOESIS_SEARCH") or "ddg").strip().lower()
    if name not in PROVIDERS:
        raise SearchError(f"unknown search provider {name!r}; choose from ddg, brave, tavily, searxng, mock")
    return PROVIDERS[name](**kw)


# ============================================================================ toolkit
@dataclass
class WebToolkit:
    """A search provider + a guarded fetcher sharing one policy/cache. The agent's internet."""

    search_provider: SearchProvider
    fetcher: WebFetcher

    @property
    def policy(self) -> DomainPolicy:
        return self.fetcher.policy

    @classmethod
    def create(
        cls,
        search: str | SearchProvider | None = None,
        *,
        policy: DomainPolicy | None = None,
        cache_dir: str | Path | None = None,
        use_cache: bool = True,
        transport: httpx.BaseTransport | None = None,
        resolver: Callable[[str], list[str]] | None = None,
        min_interval: float = 1.0,
        respect_robots: bool = True,
        timeout: float = 15.0,
        max_bytes: int = 2_000_000,
    ) -> WebToolkit:
        """Build the toolkit. ``search="mock"`` returns the bundled offline web (tests/demos)."""
        if isinstance(search, str) and search.strip().lower() == "mock":
            from .mockweb import mock_toolkit

            # never cache fixture pages: they reuse real-looking URLs and would poison the real cache
            return mock_toolkit(policy=policy, use_cache=False)
        cache = DiskCache(cache_dir, enabled=use_cache)
        rl = RateLimiter(min_interval)
        provider = search if isinstance(search, SearchProvider) else get_search_provider(
            search, transport=transport, timeout=timeout, rate_limiter=RateLimiter(min_interval), cache=cache)
        fetcher = WebFetcher(policy=policy, transport=transport, resolver=resolver, cache=cache, rate_limiter=rl,
                             timeout=timeout, max_bytes=max_bytes, respect_robots=respect_robots)
        return cls(provider, fetcher)

    def search(self, query: str, max_results: int = 8) -> list[SearchResult]:
        out = []
        for r in self.search_provider.search(query, max_results=max_results * 2):
            ok, _ = self.policy.permitted(r.url)
            if ok:
                out.append(r)
            if len(out) >= max_results:
                break
        return out

    def fetch(self, url: str, *, use_cache: bool = True) -> FetchedPage:
        return self.fetcher.fetch(url, use_cache=use_cache)

    def close(self) -> None:
        self.fetcher.close()


def ddg_query_url(query: str) -> str:
    """The equivalent human-visible DuckDuckGo URL (for logs)."""
    return f"https://duckduckgo.com/?q={quote_plus(query)}"
