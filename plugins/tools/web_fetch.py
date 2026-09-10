"""Web fetch tool — guarded scrape + SUMMARY_LLM extraction.

Tool behaviour:

- Tool ``name`` is ``"web_fetch"``.
- Argument schema accepts ``url: str | list[str]`` and
  ``info_to_extract: str | list[str]`` — multi-URL fetch in **one turn**
  via ``asyncio.gather``.
- ``info_to_extract`` is required (no default).
- Output format ``[N] URL: ...\\n    Info: ...`` for both single and
  multi-URL calls.

Fetch pipeline, in order:

1. **SSRF vetting** — the URL and *every redirect hop* must resolve to a
   public address. Validating only the initial URL is the classic hole: a
   public URL that redirects to ``169.254.169.254`` passes it.
2. **Negative cache / host circuit-breaker** — a URL that just returned
   403/429 is skipped rather than re-fetched. Parallel researchers
   otherwise hammer the same blocked host in lockstep.
3. **Positive cache with single-flight** — concurrent callers for the same
   URL share one upstream round-trip, and a page fetched once in a run is
   not fetched again. This is what makes cross-researcher URL dedup real
   rather than a prompt-level convention the model may ignore.
4. **Academic routing** — PMC / PubMed / bioRxiv / Unpaywall / Crossref
   get full text where the generic reader would return an abstract stub or
   a paywall page.
5. Jina reader → direct httpx fallback → SUMMARY_LLM extraction.

Used by :mod:`workflows.react_base` and :mod:`workflows.deep_research`.

Required env vars:
  JINA_API_KEY / JINA_BASE_URL — primary scraper
  SUMMARY_LLM_BASE_URL / SUMMARY_LLM_MODEL_NAME / SUMMARY_LLM_API_KEY
    — cheap LLM that does ``info_to_extract`` post-processing. Without
      this the tool returns ``[ERROR]: Extraction failed:
      SUMMARY_LLM_BASE_URL not set``.

Optional env vars:
  WEB_FETCH_SSRF_GUARD=0     — disable public-address vetting
  WEB_FETCH_ACADEMIC=0       — disable academic backend routing
  SCRAPE_POSITIVE_CACHE=0    — disable the positive cache / single-flight
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

import httpx
from agent_harness.core.tool import tool

logger = logging.getLogger(__name__)


# Env read at call time, not import time.


def _jina_api_key() -> str:
    return os.getenv("JINA_API_KEY", "")


def _jina_base_url() -> str:
    return os.getenv("JINA_BASE_URL", "https://r.jina.ai")


def _summary_llm_base_url() -> str | None:
    raw = os.environ.get("SUMMARY_LLM_BASE_URL")
    if not raw:
        return None
    url = raw.rstrip("/")
    return url if url.endswith("/chat/completions") else url + "/chat/completions"


def _summary_llm_model_name() -> str | None:
    return os.environ.get("SUMMARY_LLM_MODEL_NAME")


def _summary_llm_api_key() -> str | None:
    return os.environ.get("SUMMARY_LLM_API_KEY")


_BANNED_URL_PATTERNS: tuple[str, ...] = (
    "huggingface.co/datasets",
    "huggingface.co/spaces",
)


def _ensure_list(val: Any) -> Any:
    """Unwrap doubly-serialised JSON list payloads (mirrors web_search)."""
    if isinstance(val, str) and val.startswith("["):
        try:
            parsed = json.loads(val)
            if isinstance(parsed, list):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass
    return val


# ── Jina scraping ─────────────────────────────────────────────────────


async def _scrape_url_with_jina(
    url: str,
    custom_headers: dict[str, str] | None = None,
    max_chars: int = 102400 * 4,
) -> dict[str, Any]:
    """Scrape via Jina reader API.

    Retries on connect/read timeouts and 5xx/408/409/425/429 with delays
    [1, 2, 4, 8] s. Detects Jina ``InsufficientBalanceError`` JSON bodies.
    """
    api_key = _jina_api_key()
    if not api_key:
        return {"success": False, "content": "", "error": "JINA_API_KEY not set"}

    # Strip ``r.jina.ai/`` prefix if caller already wrapped the URL.
    if url.startswith("https://r.jina.ai/") and url.count("http") >= 2:
        url = url[len("https://r.jina.ai/") :]

    jina_url = f"{_jina_base_url()}/{url}"
    headers = {"Authorization": f"Bearer {api_key}"}
    if custom_headers:
        headers.update(custom_headers)

    retry_delays = [1, 2, 4, 8]
    response: httpx.Response | None = None

    for attempt, delay in enumerate(retry_delays, 1):
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    jina_url,
                    headers=headers,
                    timeout=httpx.Timeout(None, connect=20, read=60),
                    follow_redirects=True,
                )
            response.raise_for_status()
            break
        except (httpx.ConnectTimeout, httpx.ConnectError, httpx.ReadTimeout) as e:
            if attempt < len(retry_delays):
                await asyncio.sleep(delay)
                continue
            return {"success": False, "content": "", "error": str(e)}
        except httpx.HTTPStatusError as e:
            sc = e.response.status_code
            if (sc >= 500 or sc in [408, 409, 425, 429]) and attempt < len(
                retry_delays,
            ):
                await asyncio.sleep(delay)
                continue
            return {"success": False, "content": "", "error": str(e)}
        except Exception as e:
            return {"success": False, "content": "", "error": str(e)}

    if response is None:
        return {"success": False, "content": "", "error": "No response received"}

    content = response.text
    if not content:
        return {"success": False, "content": "", "error": "Empty response from Jina"}

    # Detect Jina balance exhaustion — the body is JSON like
    # ``{"name": "InsufficientBalanceError", ...}`` rather than the page.
    try:
        maybe_err = json.loads(content)
        if (
            isinstance(maybe_err, dict)
            and maybe_err.get("name") == "InsufficientBalanceError"
        ):
            return {
                "success": False,
                "content": "",
                "error": "Jina insufficient balance",
            }
    except json.JSONDecodeError:
        pass

    return {"success": True, "content": content[:max_chars], "error": ""}


async def _scrape_url_with_python(
    url: str,
    custom_headers: dict[str, str] | None = None,
    max_chars: int = 102400 * 4,
) -> dict[str, Any]:
    """Direct httpx GET fallback when Jina fails. Same retry policy."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    }
    if custom_headers:
        headers.update(custom_headers)

    retry_delays = [1, 2, 4]

    for attempt, delay in enumerate(retry_delays, 1):
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    url,
                    headers=headers,
                    timeout=httpx.Timeout(None, connect=20, read=60),
                    follow_redirects=True,
                )
            response.raise_for_status()
            content = response.text
            if not content:
                return {"success": False, "content": "", "error": "Empty response"}
            return {"success": True, "content": content[:max_chars], "error": ""}
        except (httpx.ConnectTimeout, httpx.ConnectError, httpx.ReadTimeout) as e:
            if attempt < len(retry_delays):
                await asyncio.sleep(delay)
                continue
            return {"success": False, "content": "", "error": str(e)}
        except httpx.HTTPStatusError as e:
            sc = e.response.status_code
            if (sc >= 500 or sc in [408, 409, 425, 429]) and attempt < len(
                retry_delays,
            ):
                await asyncio.sleep(delay)
                continue
            return {"success": False, "content": "", "error": str(e)}
        except Exception as e:
            return {"success": False, "content": "", "error": str(e)}

    return {"success": False, "content": "", "error": "All retries exhausted"}


# ── LLM extraction ───────────────────────────────────────────────────

_EXTRACT_INFO_PROMPT = """You are given a piece of content and the requirement of information to extract. Your task is to extract the information specifically requested. Be precise and focus exclusively on the requested information.

INFORMATION TO EXTRACT:
{}

INSTRUCTIONS:
1. Extract the information relevant to the focus above.
2. If the exact information is not found, extract the most closely related details.
3. Be specific and include exact details when available.
4. Clearly organize the extracted information for easy understanding.
5. Do not include general summaries or unrelated content.

CONTENT TO ANALYZE:
{}

EXTRACTED INFORMATION:"""


async def _extract_info_with_llm(
    content: str,
    info_to_extract: str,
    truncate_last_num_chars: int = -1,
) -> dict[str, Any]:
    """Call the cheap SUMMARY_LLM to focus-extract from the scraped page."""
    if not content or not content.strip():
        return {"success": False, "extracted_info": "", "error": "Empty content"}

    base_url = _summary_llm_base_url()
    if not base_url:
        return {
            "success": False,
            "extracted_info": "",
            "error": "SUMMARY_LLM_BASE_URL not set",
        }

    text = content
    if truncate_last_num_chars > 0:
        text = content[:-truncate_last_num_chars] + "[...truncated]"

    prompt = _EXTRACT_INFO_PROMPT.format(info_to_extract, text)
    model = _summary_llm_model_name() or "default"

    payload: dict[str, Any] = {
        "model": model,
        "max_tokens": 8192,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 1.0,
    }
    # GPT 4/5-style models require max_completion_tokens.
    if "gpt" in model:
        payload["max_completion_tokens"] = payload.pop("max_tokens")
        if "gpt-5" in model.lower() or "gpt5" in model.lower():
            payload["service_tier"] = "flex"
            payload["reasoning_effort"] = "minimal"

    headers: dict[str, str] = {"Content-Type": "application/json"}
    api_key = _summary_llm_api_key()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    retry_delays = [1, 2, 4, 8]
    response: httpx.Response | None = None

    for attempt, delay in enumerate(retry_delays, 1):
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    base_url,
                    headers=headers,
                    json=payload,
                    timeout=httpx.Timeout(None, connect=30, read=300),
                )

            # Context-overflow recovery: truncate tail and retry (40K × attempt).
            if response and (
                "exceeds the model's maximum context length" in response.text
                or "longer than the model's context length" in response.text
            ):
                payload["messages"][0]["content"] = _EXTRACT_INFO_PROMPT.format(
                    info_to_extract,
                    content[: -(40960 * attempt)] + "[...truncated]",
                )
                continue

            response.raise_for_status()
            break
        except httpx.HTTPError as e:
            # GPT-5 sometimes rejects ``service_tier`` — drop and retry.
            if (
                "gpt-5" in model.lower() or "gpt5" in model.lower()
            ) and "service_tier" in payload:
                payload.pop("service_tier", None)
            # Retry any transient HTTP/network error within the budget.
            if attempt < len(retry_delays):
                await asyncio.sleep(delay)
                continue
            return {"success": False, "extracted_info": "", "error": str(e)}
        except Exception as e:
            return {"success": False, "extracted_info": "", "error": str(e)}

    if response is None:
        return {"success": False, "extracted_info": "", "error": "No response"}

    try:
        data = response.json()
    except json.JSONDecodeError as e:
        return {
            "success": False,
            "extracted_info": "",
            "error": f"JSON parse error: {e}",
        }

    if "choices" in data and data["choices"]:
        try:
            return {
                "success": True,
                "extracted_info": data["choices"][0]["message"]["content"],
                "error": "",
            }
        except (KeyError, IndexError) as e:
            return {"success": False, "extracted_info": "", "error": str(e)}

    return {
        "success": False,
        "extracted_info": "",
        "error": f"Unexpected response: {data}",
    }


# ── Single-URL fetch + extract ────────────────────────────────────────


def _flag(name: str, default: bool = True) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    return raw not in {"0", "false", "no", "off"}


async def _academic_content(url: str) -> str:
    """Full text via a publisher-specific backend, or ``""`` to fall through.

    The generic reader on a PubMed or paywalled-journal URL returns an
    abstract stub or a login wall — content that looks like a successful
    fetch and silently starves the researcher of the actual findings.
    """
    from plugins.tools import _academic_fetch as af

    route = af.route_url(url)
    if route == "jina":
        return ""
    try:
        if route == "pmc":
            return await af.fetch_pmc_fulltext(af.extract_pmcid(url))
        if route == "pubmed":
            pmcid = await af.pubmed_to_pmc(url)
            return await af.fetch_pmc_fulltext(pmcid) if pmcid else ""
        if route == "biorxiv":
            pdf = af.biorxiv_to_pdf(url)
            return "" if pdf == url else ""
        if route == "paywall":
            doi = af.extract_doi(url)
            oa_url = await af.fetch_unpaywall_oa_url(doi) if doi else ""
            if oa_url and oa_url != url:
                scrape = await _scrape_url_with_jina(oa_url, None)
                if scrape["success"]:
                    return scrape["content"]
    except Exception as exc:  # noqa: BLE001 — routing is an optimisation
        logger.debug("academic route %s failed for %s: %s", route, url, exc)
    return ""


async def _scrape(url: str, custom_headers: dict[str, str] | None) -> str:
    """Scrape one URL, raising ``ScrapeUnavailable`` so failures are cached."""
    from plugins.tools._scrape_cache import ScrapeUnavailable

    if _flag("WEB_FETCH_ACADEMIC"):
        content = await _academic_content(url)
        if content:
            return content

    scrape = await _scrape_url_with_jina(url, custom_headers)
    if not scrape["success"]:
        logger.warning("Jina failed for %s: %s, trying direct", url, scrape["error"])
        scrape = await _scrape_url_with_python(url, custom_headers)
    if not scrape["success"]:
        raise ScrapeUnavailable(scrape["error"] or "scrape failed")
    return scrape["content"]


async def _fetch_single(
    url: str,
    info_to_extract: str,
    custom_headers: dict[str, str] | None = None,
) -> str:
    """Vet, dedupe, scrape and extract one URL."""
    from plugins.tools import _scrape_cache as sc
    from plugins.tools._academic_fetch import is_garbage_content
    from plugins.tools._bounded_fetch import non_public_url_error

    if any(pat in url for pat in _BANNED_URL_PATTERNS):
        return "Blocked: scraping Hugging Face datasets/spaces is not allowed."

    if _flag("WEB_FETCH_SSRF_GUARD"):
        refusal = await non_public_url_error(url)
        if refusal:
            return f"[ERROR]: {refusal}"

    # A host that just refused us will refuse the next researcher too;
    # skipping is both faster and less likely to deepen a rate-limit ban.
    banned = sc.cache.check(url)
    if banned is not None:
        return f"[ERROR]: {sc.format_skip_message(url, banned)}"

    try:
        content = await sc.scrape_result_cache.get_or_scrape(
            url,
            lambda: _scrape(url, custom_headers),
            # Anti-bot pages and login walls are "successful" fetches of
            # nothing. Caching one would serve the wall to every later
            # caller for the rest of the run.
            should_cache=lambda text: bool(text) and not is_garbage_content(text),
        )
    except sc.ScrapeUnavailable as exc:
        sc.cache.record_failure(url, _status_of(exc))
        return f"[ERROR]: Scraping failed: {exc}"
    except Exception as exc:  # noqa: BLE001 — a fetch failure is not fatal
        return f"[ERROR]: Scraping failed: {exc}"

    sc.cache.record_success(url)
    result = await _extract_info_with_llm(content, info_to_extract)
    if not result["success"]:
        return f"[ERROR]: Extraction failed: {result['error']}"
    return result["extracted_info"]


def _status_of(exc: BaseException) -> int:
    """Best-effort HTTP status from a scrape failure, for the ban rules."""
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status
    text = str(exc)
    for code in (403, 429, 422):
        if str(code) in text:
            return code
    return 0


# ── Tool ──────────────────────────────────────────────────────────────


@tool(name="web_fetch")
async def web_fetch(
    url: str | list[str],
    info_to_extract: str | list[str],
    custom_headers: dict[str, str] | None = None,
) -> str:
    """Fetch content from a URL and extract specific types of information.

    Args:
        url: The URL to fetch, or a list of URLs to fetch in parallel
        info_to_extract: The specific types of information to extract (usually a question), or a list of extraction prompts (one per URL)
        custom_headers (Dict[str, str]): Additional headers to include in the request

    Returns:
        Extracted information as plain text. For multiple URLs, results are numbered
    """
    # Accept JSON-encoded list payloads for both fields.
    url = _ensure_list(url)
    info_to_extract = _ensure_list(info_to_extract)

    urls = url if isinstance(url, list) else [url]
    urls = [u for u in urls if u and u.strip()]
    if not urls:
        return "[ERROR]: url is required and cannot be empty."

    # info_to_extract broadcast: 1:1 if len matches, single → broadcast,
    # mismatched (>=2 != N) → join into one prompt and broadcast.
    if isinstance(info_to_extract, list):
        if len(info_to_extract) == len(urls):
            infos = info_to_extract
        elif len(info_to_extract) == 1:
            infos = info_to_extract * len(urls)
        else:
            infos = [" ".join(info_to_extract)] * len(urls)
    else:
        infos = [info_to_extract] * len(urls)

    # Dedup identical (url, info) pairs to save extraction tokens.
    seen: set[tuple[str, str]] = set()
    deduped_urls: list[str] = []
    deduped_infos: list[str] = []
    for u, info in zip(urls, infos, strict=False):
        key = (u, info)
        if key in seen:
            continue
        seen.add(key)
        deduped_urls.append(u)
        deduped_infos.append(info)
    urls = deduped_urls
    infos = deduped_infos

    try:
        results = await asyncio.gather(
            *[
                _fetch_single(u, info, custom_headers)
                for u, info in zip(urls, infos, strict=False)
            ],
        )

        # ``[N] URL: <u>\n    Info: <text>`` — same format for single & multi-URL.
        lines: list[str] = []
        for i, (u, text) in enumerate(zip(urls, results, strict=False), 1):
            lines.append(f"[{i}] URL: {u}")
            lines.append(f"    Info: {text}")
        return "\n".join(lines)

    except Exception as e:
        return f"[ERROR]: Unexpected error: {str(e)}"


__all__ = ["web_fetch"]
