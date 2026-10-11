"""
Web search client for Onyx.

Uses the `ddgs` package (the maintained successor to duckduckgo_search).
Adds a relevance filter so junk results (wrong domain, unrelated pages)
are dropped before they reach the crawler.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Optional

from onyx.config import CONFIG, Config
from onyx.schemas import Source

log = logging.getLogger("onyx.search")


# Words that are too common to count for relevance
_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "up", "about", "into", "through", "is",
    "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "should", "could", "can", "may",
    "how", "what", "when", "where", "why", "who", "which",
}


def _query_tokens(query: str) -> set[str]:
    """Return meaningful lowercase tokens from a query."""
    words = re.findall(r"[a-zA-Z0-9]+", query.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 2}


def _relevance(title: str, url: str, query_tokens: set[str]) -> float:
    """
    Score a result between 0.0 and 1.0.
    Fraction of query tokens that appear in the title, URL, or domain.
    """
    if not query_tokens:
        return 1.0
    haystack = (title + " " + url).lower()
    hits = sum(1 for t in query_tokens if t in haystack)
    return hits / len(query_tokens)


class SearchClient:
    """Web search via ddgs with a relevance filter."""

    def __init__(self, cfg: Config = CONFIG) -> None:
        self.cfg = cfg
        self._last_call = 0.0

    def search(self, query: str, max_results: int = 5) -> list[Source]:
        """
        Search the web. Returns filtered results as Source objects.
        Drops results whose title/URL doesn't overlap the query.
        """
        query = (query or "").strip()
        if not query:
            return []

        # Be polite to the search backend
        elapsed = time.time() - self._last_call
        if elapsed < 1.0:
            time.sleep(1.0 - elapsed)

        raw = self._raw_search(query, max_results * 3)
        self._last_call = time.time()

        if not raw:
            log.warning("search returned 0 raw results for: %s", query[:60])
            return []

        query_tokens = _query_tokens(query)
        scored: list[tuple[float, Source]] = []

        for r in raw:
            title = r.get("title") or ""
            url = r.get("href") or r.get("url") or ""
            snippet = r.get("body") or r.get("snippet") or ""

            if not url:
                continue

            # Skip common junk domains
            if any(bad in url for bad in (
                "fedex.com", "ups.com", "dhl.com",
                "pinterest.", "quora.com",
            )):
                continue

            score = _relevance(title, url, query_tokens)
            scored.append((score, Source(title=title, url=url, snippet=snippet)))

        # Sort by score, keep only results with at least 1 matching token
        scored.sort(key=lambda x: x[0], reverse=True)
        filtered = [src for s, src in scored if s > 0][:max_results]

        if not filtered and scored:
            # Nothing matched, but we have raw results — take the top ones anyway
            log.warning(
                "relevance filter dropped all results for %s; using top raw",
                query[:60],
            )
            filtered = [src for _, src in scored[:max_results]]

        log.debug("search: %d raw -> %d filtered for %s", len(raw), len(filtered), query[:60])
        return filtered

    # ------------------------------------------------------------------
    # Backends
    # ------------------------------------------------------------------

    def _raw_search(self, query: str, n: int) -> list[dict]:
        """Try ddgs first, then a plain HTTP fallback."""
        results = self._ddgs_search(query, n)
        if results:
            return results

        log.warning("ddgs returned nothing — trying fallback")
        return self._fallback_search(query, n)

    def _ddgs_search(self, query: str, n: int) -> list[dict]:
        """Use the `ddgs` package. Retries once on 'No results found'."""
        try:
            from ddgs import DDGS
        except ImportError:
            try:
                from duckduckgo_search import DDGS  # type: ignore
            except ImportError:
                log.error("neither ddgs nor duckduckgo_search is installed")
                return []

        # Try up to 2 times — first call often fails on rate limit
        for attempt in range(2):
            try:
                with DDGS() as client:
                    hits = list(client.text(query, max_results=n))
                if hits:
                    return hits
                log.debug("ddgs returned empty (attempt %d) for: %s", attempt + 1, query[:50])
            except Exception as e:  # noqa: BLE001
                log.warning("ddgs attempt %d failed: %s", attempt + 1, e)

            if attempt == 0:
                time.sleep(2.0)  # wait before retrying

        log.warning("ddgs gave up after 2 attempts: %s", query[:60])
        return []

    def _fallback_search(self, query: str, n: int) -> list[dict]:
        """HTML scrape fallback — no external deps beyond requests."""
        try:
            import requests
            from bs4 import BeautifulSoup
        except ImportError:
            return []

        url = "https://html.duckduckgo.com/html/"
        try:
            r = requests.post(
                url,
                data={"q": query},
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=15,
            )
            soup = BeautifulSoup(r.text, "html.parser")
            out = []
            for result in soup.select(".result")[:n]:
                link = result.select_one(".result__a")
                snippet = result.select_one(".result__snippet")
                if not link:
                    continue
                out.append({
                    "title": link.get_text(strip=True),
                    "href": link.get("href", ""),
                    "body": snippet.get_text(strip=True) if snippet else "",
                })
            return out
        except Exception as e:  # noqa: BLE001
            log.warning("fallback search failed: %s", e)
            return []


__all__ = ["SearchClient"]
