"""Research-agent web tools: providers, fetch guards, priors, extraction, cache (all offline)."""

from __future__ import annotations

import json

import httpx
import pytest

from noesis.agent.mockweb import MockPage, MockWeb, mock_resolver, mock_toolkit
from noesis.agent.web import (
    BraveProvider,
    DiskCache,
    DomainPolicy,
    DuckDuckGoProvider,
    FetchError,
    RateLimiter,
    SearchError,
    SearxngProvider,
    TavilyProvider,
    WebFetcher,
    extract_main_text,
    get_search_provider,
    is_public_address,
    registrable_domain,
)

NOSLEEP = RateLimiter(0.0)

DDG_FIXTURE = """
<html><body><div id="links">
<div class="result result--ad"><h2><a class="result__a" href="https://ads.example/x">Ad</a></h2></div>
<div class="result results_links web-result"><h2 class="result__title">
  <a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.cdc.gov%2Fcaffeine%3Fa%3D1%26b%3D2&amp;rut=x">CDC &amp; caffeine</a></h2>
  <a class="result__snippet" href="#">Caffeine <b>facts</b> from the CDC.</a></div>
<div class="result results_links web-result"><h2 class="result__title">
  <a class="result__a" href="https://en.wikipedia.org/wiki/Caffeine">Caffeine - Wikipedia</a></h2>
  <a class="result__snippet">Caffeine is a stimulant.</a></div>
<div class="result results_links web-result"><h2 class="result__title">
  <a class="result__a" href="javascript:alert(1)">bad</a></h2></div>
</div></body></html>
"""


def _client_transport(handler):
    return httpx.MockTransport(handler)


# ------------------------------------------------------------------ providers
def test_ddg_parse_unwraps_redirects_and_skips_ads():
    rows = DuckDuckGoProvider.parse(DDG_FIXTURE)
    assert [r.url for r in rows][:2] == ["https://www.cdc.gov/caffeine?a=1&b=2", "https://en.wikipedia.org/wiki/Caffeine"]
    assert rows[0].title == "CDC & caffeine" and "facts" in rows[0].snippet
    assert all("ads.example" not in r.url for r in rows)


def test_ddg_search_filters_non_http_and_ranks():
    seen = {}

    def h(req: httpx.Request) -> httpx.Response:
        seen["method"], seen["body"] = req.method, req.content.decode()
        return httpx.Response(200, text=DDG_FIXTURE, headers={"content-type": "text/html"})

    p = DuckDuckGoProvider(transport=_client_transport(h), rate_limiter=NOSLEEP)
    rows = p.search("caffeine  facts", max_results=5)
    assert seen["method"] == "POST" and "q=caffeine+facts" in seen["body"]
    assert [r.rank for r in rows] == [1, 2] and all(r.provider == "ddg" and r.query == "caffeine facts" for r in rows)


def test_ddg_bot_challenge_is_reported():
    p = DuckDuckGoProvider(transport=_client_transport(lambda r: httpx.Response(202, text="<html>anomaly detected captcha</html>")),
                           rate_limiter=NOSLEEP)
    with pytest.raises(SearchError, match="challenge"):
        p.search("x")


def test_brave_tavily_searxng_parse_and_auth():
    calls = {}

    def h(req: httpx.Request) -> httpx.Response:
        host = req.url.host
        calls[host] = req
        if host == "api.search.brave.com":
            return httpx.Response(200, json={"web": {"results": [{"title": "B", "url": "https://b.org/1", "description": "<strong>x</strong> y"}]}})
        if host == "api.tavily.com":
            return httpx.Response(200, json={"results": [{"title": "T", "url": "https://t.org/1", "content": "tav", "score": 0.9}]})
        return httpx.Response(200, json={"results": [{"title": "S", "url": "https://s.org/1", "content": "sx"}, {"title": "no url"}]})

    tr = _client_transport(h)
    b = BraveProvider(api_key="k1", transport=tr, rate_limiter=NOSLEEP).search("q one")
    assert b[0].url == "https://b.org/1" and b[0].snippet == "x y"
    assert calls["api.search.brave.com"].headers["x-subscription-token"] == "k1"
    assert calls["api.search.brave.com"].url.params["q"] == "q one"
    t = TavilyProvider(api_key="k2", transport=tr, rate_limiter=NOSLEEP).search("q two")
    assert t[0].snippet == "tav" and calls["api.tavily.com"].headers["authorization"] == "Bearer k2"
    assert json.loads(calls["api.tavily.com"].content)["query"] == "q two"
    s = SearxngProvider(base_url="https://searx.local.test/", transport=tr, rate_limiter=NOSLEEP).search("q3")
    assert [r.url for r in s] == ["https://s.org/1"]
    assert calls["searx.local.test"].url.params["format"] == "json"


def test_provider_keys_required(monkeypatch):
    for var in ("BRAVE_API_KEY", "TAVILY_API_KEY", "SEARXNG_URL"):
        monkeypatch.delenv(var, raising=False)
    for name in ("brave", "tavily", "searxng"):
        with pytest.raises(SearchError):
            get_search_provider(name)
    with pytest.raises(SearchError, match="unknown"):
        get_search_provider("bing")
    monkeypatch.setenv("BRAVE_API_KEY", "abc")
    assert get_search_provider("brave").api_key == "abc"


def test_http_error_becomes_search_error():
    p = BraveProvider(api_key="k", transport=_client_transport(lambda r: httpx.Response(429)), rate_limiter=NOSLEEP)
    with pytest.raises(SearchError, match="429"):
        p.search("x")


def test_search_cache(tmp_path):
    n = {"calls": 0}

    def h(req):
        n["calls"] += 1
        return httpx.Response(200, json={"results": [{"title": "S", "url": "https://s.org/1", "content": ""}]})

    cache = DiskCache(tmp_path)
    p = SearxngProvider(base_url="https://sx.test", transport=_client_transport(h), rate_limiter=NOSLEEP, cache=cache)
    assert p.search("same")[0].url == p.search("same")[0].url == "https://s.org/1"
    assert n["calls"] == 1


# ------------------------------------------------------------------ helpers
def test_registrable_domain_and_public_ip():
    assert registrable_domain("https://www.bbc.co.uk/news") == "bbc.co.uk"
    assert registrable_domain("https://a.b.nih.gov/x") == "nih.gov"
    assert registrable_domain("http://8.8.8.8/x") == "8.8.8.8"
    assert is_public_address("8.8.8.8") and is_public_address("2606:4700::1111")
    for ip in ("127.0.0.1", "10.0.0.1", "192.168.1.1", "172.16.0.1", "169.254.169.254", "::1", "fe80::1",
               "0.0.0.0", "100.64.0.1", "::ffff:127.0.0.1", "not-an-ip"):
        assert not is_public_address(ip), ip


def test_rate_limiter_spaces_requests():
    t = {"now": 0.0}
    slept = []
    rl = RateLimiter(2.0, clock=lambda: t["now"], sleep=slept.append)
    assert rl.wait("h") == 0.0
    t["now"] = 0.5
    assert rl.wait("h") == pytest.approx(1.5) and slept == [pytest.approx(1.5)]
    assert rl.wait("other") == 0.0


def test_disk_cache_ttl(tmp_path):
    c = DiskCache(tmp_path, ttl=100)
    c.set("ns", "k", {"a": 1})
    assert c.get("ns", "k") == {"a": 1}
    c.ttl = -1
    assert c.get("ns", "k") is None
    assert DiskCache(tmp_path, enabled=False).get("ns", "k") is None


# ------------------------------------------------------------------ reliability priors
def test_domain_policy_priors_and_lists():
    pol = DomainPolicy(deny=["badsite.com"], trust={"mylab.org": 0.92})
    r_gov, _ = pol.reliability("https://www.cdc.gov/x")
    r_edu, _ = pol.reliability("https://cs.stanford.edu/x")
    r_farm, why_farm = pol.reliability("https://www.ehow.com/x")
    r_blog, _ = pol.reliability("https://someone.medium.com/x")
    r_unknown, _ = pol.reliability("https://random-site.net/x")
    r_pubmed, _ = pol.reliability("https://pubmed.ncbi.nlm.nih.gov/123")
    assert r_pubmed >= r_gov >= r_edu > r_unknown > r_blog > r_farm
    assert "content farm" in why_farm
    assert pol.reliability("https://sub.mylab.org/p")[0] == 0.92
    r_meta, why = pol.reliability("https://random-site.net/x", {"citation_journal_title": "J Foo"})
    assert r_meta == 0.82 and "J Foo" in why
    assert pol.permitted("https://x.badsite.com/a") == (False, "denied domain")
    assert pol.permitted("https://notbadsite.com/a")[0]
    allow = DomainPolicy(allow=["who.int", "https://nih.gov/"])
    assert allow.permitted("https://www.who.int/a")[0] and allow.permitted("https://www.nih.gov/a")[0]
    assert allow.permitted("https://example.com")[1] == "not in allow list"


# ------------------------------------------------------------------ extraction
def test_extract_main_text_strips_boilerplate():
    from noesis.agent.mockweb import render_page

    page = MockPage("https://x.org/a", "A Title", ["First paragraph, with commas, and enough length to count.",
                                                  "Second paragraph that is also long enough to be scored."],
                    meta={"citation_journal_title": "J Test", "citation_doi": "10.1/abc"})
    title, text, meta = extract_main_text(render_page(page))
    assert title == "A Title"
    assert "First paragraph" in text and "Second paragraph" in text
    for junk in ("window.track", "Subscribe", "Popular", "Sign up", "Hidden text", "Copyright", ".ad{"):
        assert junk not in text, junk
    assert meta["doi"] == "10.1/abc" and meta["citation_journal_title"] == "J Test"


# ------------------------------------------------------------------ fetch guards
def _fetcher(handler=None, *, web: MockWeb | None = None, resolver=mock_resolver, **kw) -> tuple[WebFetcher, list]:
    reqs: list[str] = []
    web = web or MockWeb()

    def h(req):
        reqs.append(str(req.url))
        return handler(req) if handler else web(req)

    f = WebFetcher(transport=httpx.MockTransport(h), resolver=resolver, cache=DiskCache(enabled=False),
                   rate_limiter=NOSLEEP, **kw)
    return f, reqs


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:8080/admin",
        "http://169.254.169.254/latest/meta-data/",
        "http://[::1]/x",
        "http://localhost/x",
        "http://printer.local/x",
        "https://intranet.corp-internal.com/coffee",  # resolves to 10.1.2.3
        "http://0.0.0.0/",
    ],
)
def test_ssrf_guard_blocks_private_targets(url):
    f, reqs = _fetcher()
    with pytest.raises(FetchError) as e:
        f.fetch(url)
    assert e.value.reason == "ssrf"
    assert reqs == []  # nothing was ever sent


def test_ssrf_guard_rechecks_redirects():
    def h(req):
        if req.url.path == "/robots.txt":
            return httpx.Response(404)
        return httpx.Response(302, headers={"location": "http://10.0.0.7/secret"})

    f, reqs = _fetcher(h)
    with pytest.raises(FetchError) as e:
        f.fetch("https://public.example.org/start")
    assert e.value.reason == "ssrf"
    assert not any("10.0.0.7" in r for r in reqs)


def test_redirect_followed_when_safe():
    f, _ = _fetcher()
    page = f.fetch("https://redirect.example.net/coffee")
    assert page.final_url == "https://sleep.example.com/caffeine-and-sleep" and "half-life" in page.text


def test_dns_failure_and_bad_schemes():
    def boom(host):
        raise OSError("no such host")

    f, reqs = _fetcher(resolver=boom)
    with pytest.raises(FetchError) as e:
        f.fetch("https://unknown.test/")
    assert e.value.reason == "dns"
    for url, reason in [("file:///etc/passwd", "scheme"), ("ftp://x.org/f", "scheme"),
                        ("https://user:pw@x.org/", "credentials"), ("https://x.org/setup.exe", "executable"),
                        ("https://x.org/archive.tar.gz", "executable")]:
        with pytest.raises(FetchError) as e:
            f.fetch(url)
        assert e.value.reason == reason, url
    assert reqs == []


def test_content_type_and_size_limits():
    big = "<html><body><article>" + ("<p>" + ("word, " * 50) + "</p>") * 400 + "</article></body></html>"

    def h(req):
        p = req.url.path
        if p == "/robots.txt":
            return httpx.Response(404)
        if p == "/bin":
            return httpx.Response(200, content=b"\x7fELF", headers={"content-type": "application/octet-stream"})
        if p == "/huge":
            return httpx.Response(200, content=b"x", headers={"content-type": "text/html", "content-length": str(10**9)})
        if p == "/attach":
            return httpx.Response(200, text="hi", headers={"content-type": "text/plain", "content-disposition": 'attachment; filename="run.exe"'})
        return httpx.Response(200, text=big, headers={"content-type": "text/html"})

    f, _ = _fetcher(h, max_bytes=5000)
    for path, reason in (("/bin", "content_type"), ("/huge", "too_large"), ("/attach", "executable")):
        with pytest.raises(FetchError) as e:
            f.fetch("https://pub.example.org" + path)
        assert e.value.reason == reason, path
    page = f.fetch("https://pub.example.org/big")
    assert page.truncated and len(page.text) < 6000


def test_robots_txt_semantics():
    def h(req):
        host, path = req.url.host, req.url.path
        if path == "/robots.txt":
            return {"deny.test": httpx.Response(200, text="User-agent: *\nDisallow: /private/\n"),
                    "err.test": httpx.Response(503),
                    "none.test": httpx.Response(404)}[host]
        return httpx.Response(200, text="<html><body><p>Some content that is long enough here.</p></body></html>",
                              headers={"content-type": "text/html"})

    f, reqs = _fetcher(h)
    assert f.fetch("https://deny.test/public/a").text
    with pytest.raises(FetchError) as e:
        f.fetch("https://deny.test/private/a")
    assert e.value.reason == "robots"
    assert f.fetch("https://none.test/x").text  # 4xx robots -> allowed (RFC 9309)
    with pytest.raises(FetchError) as e:
        f.fetch("https://err.test/x")  # 5xx robots -> assume disallow
    assert e.value.reason == "robots"
    assert "https://deny.test/private/a" not in reqs
    assert sum(1 for r in reqs if r == "https://deny.test/robots.txt") == 1  # robots cached per host


def test_policy_enforced_by_fetcher_and_injection_flagged(tmp_path):
    f, reqs = _fetcher(policy=DomainPolicy(deny=["wikipedia.org"]))
    with pytest.raises(FetchError) as e:
        f.fetch("https://en.wikipedia.org/wiki/Health_effects_of_coffee")
    assert e.value.reason == "policy" and reqs == []
    page = f.fetch("https://coffeefan.medium.com/why-coffee-is-dangerous")
    assert page.injection_suspect
    doc = page.as_document()
    assert doc.startswith("<document") and 'untrusted="true"' in doc


def test_fetch_cache_avoids_refetch(tmp_path):
    web = MockWeb()
    f = WebFetcher(transport=web.transport, resolver=mock_resolver, cache=DiskCache(tmp_path), rate_limiter=NOSLEEP)
    a = f.fetch("https://www.nih.gov/news-events/coffee-and-heart-health")
    n = len(web.requests)
    b = f.fetch("https://www.nih.gov/news-events/coffee-and-heart-health")
    assert b.from_cache and not a.from_cache and b.text == a.text and len(web.requests) == n


def test_toolkit_search_respects_policy():
    tk = mock_toolkit(policy=DomainPolicy(deny=["nih.gov"]))
    rows = tk.search("coffee heart health blood pressure", max_results=8)
    assert rows and all("nih.gov" not in r.url for r in rows)
    tk_brave = mock_toolkit(provider="brave")
    assert tk_brave.search("coffee mortality", 3)[0].provider == "brave"
