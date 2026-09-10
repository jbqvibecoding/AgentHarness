"""Phase C tests: SSRF vetting, fetch dedup, negative caching, recovery.

No network: every test drives the guards and caches directly with fake
scrape functions. The one address-resolution test uses loopback and
link-local literals, which resolve without leaving the machine.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import time

import pytest

from plugins.tools import _scrape_cache as sc
from plugins.tools._academic_fetch import is_garbage_content, route_url
from plugins.tools._bounded_fetch import (
    binary_content_type,
    blocked_download_url,
    non_public_url_error,
    strip_cross_origin_credentials,
)
from plugins.tools._single_flight import SingleFlightCoalescer

web_fetch_mod = importlib.import_module("plugins.tools.web_fetch")


@pytest.fixture(autouse=True)
def _clean_caches():
    sc.cache.clear()
    yield
    sc.cache.clear()


# --------------------------------------------------------------------------
# SSRF vetting
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_private_and_metadata_targets_are_refused():
    """The prompt-injection path: a URL that arrives from page content.

    web_fetch is auto-approved, so nobody sees the URL before it is
    requested. An unguarded fetch reaches cloud metadata and anything else
    on the deployment's private network.
    """
    for url in (
        "http://127.0.0.1/admin",
        "http://169.254.169.254/latest/meta-data/",
        "http://[::1]/",
    ):
        assert await non_public_url_error(url), url


@pytest.mark.asyncio
async def test_non_http_schemes_and_embedded_credentials_are_refused():
    assert "http" in await non_public_url_error("ftp://example.com/x")
    assert "credentials" in await non_public_url_error(
        "https://user:pw@example.com/",
    )


@pytest.mark.asyncio
async def test_unresolvable_host_fails_closed():
    """A name that cannot be resolved cannot be vetted, so it is refused."""
    assert await non_public_url_error("http://no-such-host-xyz.invalid/")


@pytest.mark.asyncio
async def test_guard_can_be_disabled_for_a_deliberate_intranet_fetch(monkeypatch):
    monkeypatch.setenv("AGENT_HARNESS_ALLOW_PRIVATE_FETCH", "1")
    assert await non_public_url_error("http://127.0.0.1/admin") == ""


def test_credentials_are_dropped_on_a_cross_origin_redirect():
    headers = {"Authorization": "Bearer secret", "Accept": "text/html"}
    same = strip_cross_origin_credentials(
        headers, "https://a.example/x", "https://a.example/y",
    )
    assert same["Authorization"] == "Bearer secret"
    crossed = strip_cross_origin_credentials(
        headers, "https://a.example/x", "https://evil.example/y",
    )
    assert "Authorization" not in crossed
    assert crossed["Accept"] == "text/html"


def test_archive_and_binary_targets_are_screened_before_any_request():
    assert blocked_download_url("https://x.example/data.zip") == ".zip"
    assert blocked_download_url("https://x.example/model.safetensors")
    # A query parameter must not false-positive — the match is on the path.
    assert blocked_download_url("https://x.example/page?format=zip") is None
    assert blocked_download_url("https://x.example/paper.pdf") is None
    assert binary_content_type("image/png") == "image/png"
    assert binary_content_type("text/html; charset=utf-8") is None


# --------------------------------------------------------------------------
# Single-flight coalescing
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_concurrent_callers_share_one_upstream_call():
    """Parallel researchers hitting the same URL must not each fetch it."""
    coalescer = SingleFlightCoalescer("test")
    calls = 0

    async def fetch():
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.05)
        return "body"

    results = await asyncio.gather(*[
        coalescer.run("https://x.example/a", fetch) for _ in range(5)
    ])
    assert results == ["body"] * 5
    assert calls == 1
    assert coalescer.leaders == 1 and coalescer.coalesced == 4


@pytest.mark.asyncio
async def test_a_follower_still_gets_an_answer_when_the_leader_fails():
    """Coalescing must not turn one transient failure into five."""
    coalescer = SingleFlightCoalescer("test")
    attempts = 0

    async def flaky():
        nonlocal attempts
        attempts += 1
        await asyncio.sleep(0.02)
        if attempts == 1:
            raise RuntimeError("leader failed")
        return "recovered"

    leader = asyncio.create_task(coalescer.run("k", flaky))
    await asyncio.sleep(0.005)
    follower = asyncio.create_task(coalescer.run("k", flaky))
    results = await asyncio.gather(leader, follower, return_exceptions=True)
    assert any(r == "recovered" for r in results)


# --------------------------------------------------------------------------
# Negative cache / circuit breaker
# --------------------------------------------------------------------------

def test_a_403_bans_the_url_for_later_callers():
    url = "https://blocked.example/page"
    assert sc.cache.check(url) is None
    sc.cache.record_failure(url, 403)
    entry = sc.cache.check(url)
    assert entry is not None and entry.status == 403
    assert "403" in sc.format_skip_message(url, entry)


def test_a_ban_expires():
    url = "https://blocked.example/page"
    sc.cache.record_failure(url, 403)
    future = time.time() + 86_400
    assert sc.cache.check(url, now=future) is None


def test_a_422_needs_repeated_failures_before_banning():
    """One malformed response is not a broken host."""
    url = "https://flaky.example/page"
    sc.cache.record_failure(url, 422)
    assert sc.cache.check(url) is None
    for _ in range(5):
        sc.cache.record_failure(url, 422)
    assert sc.cache.check(url) is not None


def test_success_clears_a_prior_failure():
    url = "https://recovering.example/page"
    sc.cache.record_failure(url, 429)
    assert sc.cache.check(url) is not None
    sc.cache.record_success(url)
    assert sc.cache.check(url) is None


def test_untracked_statuses_are_not_cached():
    url = "https://x.example/page"
    sc.cache.record_failure(url, 500)
    assert sc.cache.check(url) is None


# --------------------------------------------------------------------------
# Positive cache
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_a_page_fetched_once_is_not_fetched_again(monkeypatch):
    monkeypatch.setenv("SCRAPE_POSITIVE_CACHE", "1")
    cache = sc.ScrapeResultCache()
    calls = 0

    async def scrape():
        nonlocal calls
        calls += 1
        return "page body"

    first = await cache.get_or_scrape("https://x.example/p", scrape)
    second = await cache.get_or_scrape("https://x.example/p", scrape)
    assert first == second == "page body"
    assert calls == 1
    assert cache.hits == 1


@pytest.mark.asyncio
async def test_an_anti_bot_page_is_returned_but_never_cached(monkeypatch):
    """Caching a login wall would serve it to every later caller."""
    monkeypatch.setenv("SCRAPE_POSITIVE_CACHE", "1")
    cache = sc.ScrapeResultCache()
    bodies = iter(["Please enable JavaScript and cookies to continue", "real body"])

    async def scrape():
        return next(bodies)

    first = await cache.get_or_scrape(
        "https://x.example/p", scrape,
        should_cache=lambda text: bool(text) and not is_garbage_content(text),
    )
    second = await cache.get_or_scrape(
        "https://x.example/p", scrape,
        should_cache=lambda text: bool(text) and not is_garbage_content(text),
    )
    assert "JavaScript" in first
    assert second == "real body", "the garbage page must not have been cached"


def test_garbage_detection_recognises_walls_and_empty_pages():
    assert is_garbage_content("")
    assert is_garbage_content("Please enable JavaScript and cookies to continue")
    assert not is_garbage_content(
        "Full-year revenue reached $5 billion, up 12% year over year.",
    )


# --------------------------------------------------------------------------
# Academic routing
# --------------------------------------------------------------------------

def test_publisher_urls_route_to_their_own_backends():
    assert route_url("https://pmc.ncbi.nlm.nih.gov/articles/PMC123/") == "pmc"
    assert route_url("https://pubmed.ncbi.nlm.nih.gov/12345678/") == "pubmed"
    assert route_url("https://www.biorxiv.org/content/10.1101/2024.01.01") == "biorxiv"
    assert route_url("https://example.com/blog") == "jina"
    # A non-http scheme must never be routed to a fetcher.
    assert route_url("ftp://example.com/x") == "jina"


# --------------------------------------------------------------------------
# web_fetch wiring
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fetch_refuses_a_private_target_before_scraping(monkeypatch):
    called = False

    async def never(*_a, **_kw):
        nonlocal called
        called = True
        return {"success": True, "content": "x", "error": ""}

    monkeypatch.setattr(web_fetch_mod, "_scrape_url_with_jina", never)
    out = await web_fetch_mod._fetch_single("http://127.0.0.1/admin", "anything")
    assert out.startswith("[ERROR]")
    assert not called, "the scrape must not be attempted at all"


@pytest.mark.asyncio
async def test_fetch_skips_a_url_the_host_just_blocked(monkeypatch):
    monkeypatch.setenv("WEB_FETCH_SSRF_GUARD", "0")
    url = "https://blocked.example/page"
    sc.cache.record_failure(url, 403)
    called = False

    async def never(*_a, **_kw):
        nonlocal called
        called = True
        return {"success": True, "content": "x", "error": ""}

    monkeypatch.setattr(web_fetch_mod, "_scrape_url_with_jina", never)
    out = await web_fetch_mod._fetch_single(url, "anything")
    assert out.startswith("[ERROR]") and "403" in out
    assert not called


@pytest.mark.asyncio
async def test_fetch_records_a_failure_so_the_next_caller_skips(monkeypatch):
    monkeypatch.setenv("WEB_FETCH_SSRF_GUARD", "0")
    monkeypatch.setenv("WEB_FETCH_ACADEMIC", "0")
    url = "https://forbidden.example/page"

    async def refuse(*_a, **_kw):
        return {"success": False, "content": "", "error": "HTTP 403 Forbidden"}

    monkeypatch.setattr(web_fetch_mod, "_scrape_url_with_jina", refuse)
    monkeypatch.setattr(web_fetch_mod, "_scrape_url_with_python", refuse)
    out = await web_fetch_mod._fetch_single(url, "anything")
    assert out.startswith("[ERROR]")
    assert sc.cache.check(url) is not None, "the 403 must be remembered"


def test_flag_parsing():
    assert web_fetch_mod._flag("DEFINITELY_UNSET_VAR") is True
    assert web_fetch_mod._flag("DEFINITELY_UNSET_VAR", False) is False


def test_status_extraction_from_a_scrape_error():
    assert web_fetch_mod._status_of(Exception("HTTP 403 Forbidden")) == 403
    assert web_fetch_mod._status_of(Exception("429 Too Many Requests")) == 429
    assert web_fetch_mod._status_of(Exception("boom")) == 0


# --------------------------------------------------------------------------
# recover_result
# --------------------------------------------------------------------------

def test_recover_reads_the_last_attempt_of_the_current_run(tmp_path):
    """The file is append-mode, and a turn can be retried.

    Taking the last match after the last ``start`` reads the current
    attempt of the current run rather than a stale one.
    """
    _find_record = importlib.import_module(
        "plugins.tools.recover_result",
    )._find_record

    path = tmp_path / "trace.jsonl"
    rows = [
        {"t": "start"},
        {"t": "result", "turn": 2, "tool_call_id": "c1", "result": "OLD RUN"},
        {"t": "start"},
        {"t": "result", "turn": 2, "tool_call_id": "c1", "result": "first try"},
        {"t": "result", "turn": 2, "tool_call_id": "c1", "result": "retry"},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    hit = _find_record(path, 2, "c1")
    assert hit is not None and hit["result"] == "retry"


def test_recover_never_matches_a_record_without_an_id(tmp_path):
    """An absent id treated as "" would return a confidently wrong body."""
    _find_record = importlib.import_module(
        "plugins.tools.recover_result",
    )._find_record

    path = tmp_path / "trace.jsonl"
    rows = [
        {"t": "start"},
        {"t": "result", "turn": 1, "result": "no id here"},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    assert _find_record(path, 1, "") is None
    assert _find_record(path, 1, "c1") is None


def test_recover_skips_unparseable_lines(tmp_path):
    """A large record can be half-flushed when a run is killed."""
    _find_record = importlib.import_module(
        "plugins.tools.recover_result",
    )._find_record

    path = tmp_path / "trace.jsonl"
    path.write_text(
        json.dumps({"t": "start"}) + "\n"
        + '{"t": "result", "turn": 1, "tool_call_id": "c1", "result": "trun\n'
        + json.dumps({
            "t": "result", "turn": 1, "tool_call_id": "c2", "result": "intact",
        }) + "\n",
        encoding="utf-8",
    )
    assert _find_record(path, 1, "c2")["result"] == "intact"
    assert _find_record(path, 1, "c1") is None


def test_trajectory_records_the_id_recover_addresses_by():
    """Pins the contract between the writer and the reader."""
    import inspect

    from agent_harness.components.observers.trajectory import (
        TrajectoryFileObserver,
    )

    # importlib, not `from plugins.tools import recover_result`: the package
    # re-exports the decorated Tool object under that name, shadowing the
    # module.
    rr = importlib.import_module("plugins.tools.recover_result")

    source = inspect.getsource(TrajectoryFileObserver.on_tool_result)
    assert '"tool_call_id"' in source
    assert TrajectoryFileObserver.SCOPE_KEY == rr._SCOPE_KEY
