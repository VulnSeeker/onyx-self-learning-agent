"""
Multi-engine web search for Onyx. Zero cost, no API keys, no credit cards.

Fallback chain (tried in order, first success wins):
  1. webless        — fuses DDG + Mojeek + Bing + Wikipedia via RRF (pure Python)
  2. ddgs           — DuckDuckGo with retry
  3. SearXNG HTML   — scrape a public instance's HTML output (no JSON needed)
  4. Jina Reader    — s.jina.ai keyless search

Every result passes through a relevance filter so junk domains never
reach the crawler.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Optional

import requests

from onyx.config import CONFIG, Config
from onyx.schemas import Source

log = logging.getLogger("onyx.search")


# SearXNG instances — HTML scrape mode (JSON is disabled on most publics)
_SEARXNG_INSTANCES = [
    "https://searx.be",
    "https://search.inetol.net",
    "https://searx.tiekoetter.com",
    "https://baresearch.org",
    "https://priv.au",
    "https://searxng.site",
]

_JUNK_DOMAINS = (
    "fedex.com", "ups.com", "dhl.com",
    "pinterest.", "quora.com", "facebook.com",
    "instagram.com", "tiktok.com",
    "youtube.com", "youtu.be",
    "reddit.com",
)

_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "up", "about", "into", "through", "is",
    "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "should", "could", "can", "may",
    "how", "what", "when", "where", "why", "who", "which",
}

_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 OnyxAgent/0.1"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _query_tokens(query: str) -> set[str]:
    words = re.findall(r"[a-zA-Z0-9]+", query.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 2}


def _relevance(title: str, url: str, query_tokens: set[str]) -> float:
    if not query_tokens:
        return 1.0
    haystack = (title + " " + url).lower()
    hits = sum(1 for t in query_tokens if t in haystack)
    return hits / len(query_tokens)


def _is_junk(url: str) -> bool:
    return any(bad in url for bad in _JUNK_DOMAINS)


def _run_async(coro):
    """
    Run an async coroutine from sync code, even inside a running event loop
    (Jupyter/Colab). Falls back to asyncio.run() if nothing is looping.
    """
    import asyncio

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is None:
        return asyncio.run(coro)

    # We're inside a loop (Colab/Jupyter). Try nest_asyncio.
    try:
        import nest_asyncio
        nest_asyncio.apply()
        return loop.run_until_complete(coro)
    except ImportError:
        import threading

        result = {}

        def runner():
            try:
                result["value"] = asyncio.run(coro)
            except Exception as e:
                result["error"] = e

        t = threading.Thread(target=runner)
        t.start()
        t.join()
        if "error" in result:
            raise result["error"]
        return result.get("value")


class SearchClient:
    """Multi-engine search with failover. All engines are free and keyless."""

    def __init__(self, cfg: Config = CONFIG) -> None:
        self.cfg = cfg
        self._last_call = 0.0
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": _UA})

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(self, query: str, max_results: int = 5) -> list[Source]:
        query = (query or "").strip()
        if not query:
            return []

        elapsed = time.time() - self._last_call
        if elapsed < 1.0:
            time.sleep(1.0 - elapsed)

        # Engine chain — first one that returns results wins
        engines = [
            ("webless", self._webless_search),
            ("ddgs", self._ddgs_search),
            ("searxng-html", self._searxng_html_search),
            ("jina", self._jina_search),
        ]

        raw: list[dict] = []
        chosen = "none"
        for name, fn in engines:
            try:
                raw = fn(query, max_results * 3)
                if raw:
                    chosen = name
                    log.info("engine=%s returned %d results for %s",
                             name, len(raw), query[:50])
                    break
                log.debug("engine=%s returned 0", name)
            except Exception as e:  # noqa: BLE001
                log.debug("engine=%s failed: %s", name, e)

        self._last_call = time.time()

        if not raw:
            log.warning("all search engines failed for: %s", query[:60])
            return []

        # Filter + rank
        query_tokens = _query_tokens(query)
        scored: list[tuple[float, Source]] = []
        for r in raw:
            title = r.get("title") or ""
            url = r.get("url") or r.get("href") or ""
            snippet = r.get("snippet") or r.get("body") or ""
            if not url or _is_junk(url):
                continue
            score = _relevance(title, url, query_tokens)
            scored.append((score, Source(title=title, url=url, snippet=snippet)))

        scored.sort(key=lambda x: x[0], reverse=True)
        filtered = [src for s, src in scored if s > 0][:max_results]
        if not filtered and scored:
            filtered = [src for _, src in scored[:max_results]]

        log.info("search [%s]: %d raw -> %d filtered", chosen, len(raw), len(filtered))
        return filtered

    # ------------------------------------------------------------------
    # Engine 1: webless (fuses DDG + Mojeek + Bing + Wikipedia)
    # ------------------------------------------------------------------

    def _webless_search(self, query: str, n: int) -> list[dict]:
        try:
            from webless import search
        except ImportError:
            log.debug("webless not installed")
            return []

        try:
            result = _run_async(search(query, limit=n))
            if not result or not result.hits:
                if result and result.failed:
                    log.debug("webless: all engines failed — %s", result.failed)
                return []
            log.debug(
                "webless: succeeded=%s, %d hits",
                result.succeeded, len(result.hits),
            )
            return [
                {"title": h.title, "url": h.url, "snippet": h.snippet}
                for h in result.hits
            ]
        except Exception as e:  # noqa: BLE001
            log.debug("webless failed: %s", e)
            return []

    # ------------------------------------------------------------------
    # Engine 2: ddgs (DuckDuckGo with retry)
    # ------------------------------------------------------------------

    def _ddgs_search(self, query: str, n: int) -> list[dict]:
        try:
            from ddgs import DDGS
        except ImportError:
            try:
                from duckduckgo_search import DDGS  # type: ignore
            except ImportError:
                return []

        for attempt in range(2):
            try:
                with DDGS() as client:
                    hits = list(client.text(query, max_results=n))
                if hits:
                    return hits
            except Exception as e:  # noqa: BLE001
                log.debug("ddgs attempt %d failed: %s", attempt + 1, e)
            if attempt == 0:
                time.sleep(2.0)
        return []

    # ------------------------------------------------------------------
    # Engine 3: SearXNG HTML scrape (JSON is disabled on most publics)
    # ------------------------------------------------------------------

    def _searxng_html_search(self, query: str, n: int) -> list[dict]:
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            return []

        for instance in _SEARXNG_INSTANCES:
            try:
                r = self._session.post(
                    f"{instance}/search",
                    data={"q": query},
                    timeout=15,
                )
                if r.status_code != 200:
                    continue

                soup = BeautifulSoup(r.text, "html.parser")
                out = []
                for art in soup.select("article.result")[:n]:
                    a = art.select_one("h3 a") or art.select_one("a.url")
                    if not a:
                        continue
                    title = a.get_text(strip=True)
                    url = a.get("href", "")
                    if url.startswith("/"):
                        url = instance + url
                    p = art.select_one("p.content")
                    snippet = p.get_text(strip=True) if p else ""
                    out.append({"title": title, "url": url, "snippet": snippet})

                if out:
                    log.info("searxng-html hit: %s (%d)", instance, len(out))
                    return out
            except Exception as e:  # noqa: BLE001
                log.debug("searxng-html %s failed: %s", instance, e)
        return []

    # ------------------------------------------------------------------
    # Engine 4: Jina Reader search (keyless, 20 RPM)
    # ------------------------------------------------------------------

    def _jina_search(self, query: str, n: int) -> list[dict]:
        try:
            r = self._session.get(
                "https://s.jina.ai/",
                params={"q": query},
                headers={"Accept": "application/json", "X-Respond-With": "no-content"},
                timeout=20,
            )
            if r.status_code != 200:
                log.debug("jina: HTTP %d", r.status_code)
                return []
            items = r.json().get("data", [])[:n]
            return [
                {
                    "title": it.get("title", ""),
                    "url": it.get("url", ""),
                    "snippet": it.get("description", ""),
                }
                for it in items if it.get("url")
            ]
        except Exception as e:  # noqa: BLE001
            log.debug("jina failed: %s", e)
            return []


__all__ = ["SearchClient"]
