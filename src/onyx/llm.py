"""
Multi-provider LLM client for Onyx.

- Tries providers in priority order
- On 429 / connection error, tries the next provider
- Bumps default max_tokens so reasoning models (gpt-oss-120b) have
  room for both reasoning and output
"""

from __future__ import annotations

import json
import logging
import re
import threading
from typing import Any, Iterable

from openai import OpenAI, APIConnectionError, APIError, RateLimitError

from onyx.config import CONFIG, Config, LLM_PROVIDERS

log = logging.getLogger("onyx.llm")

# Global counter for total seconds spent waiting on rate limits.
# Reset per task by calling reset_wait_timer().
_RATE_LIMIT_WAIT_SECONDS = 0.0


def reset_wait_timer():
    global _RATE_LIMIT_WAIT_SECONDS
    _RATE_LIMIT_WAIT_SECONDS = 0.0


def get_wait_seconds() -> float:
    return _RATE_LIMIT_WAIT_SECONDS


def _record_wait(seconds: float):
    global _RATE_LIMIT_WAIT_SECONDS
    _RATE_LIMIT_WAIT_SECONDS += seconds



_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


class LLMResponseError(RuntimeError):
    pass


def parse_json_loose(raw: str) -> Any:
    if not raw or not raw.strip():
        raise LLMResponseError("empty LLM response")
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    m = _JSON_FENCE_RE.search(raw)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass
    for open_c, close_c in (("{", "}"), ("[", "]")):
        start = raw.find(open_c)
        end = raw.rfind(close_c)
        if start != -1 and end > start:
            try:
                return json.loads(raw[start:end + 1])
            except json.JSONDecodeError:
                continue
    raise LLMResponseError(f"could not parse JSON: {raw[:200]!r}")


# Default output budget. Reasoning models need headroom for hidden tokens.
DEFAULT_MAX_TOKENS = 1500


class LLMClient:
    def __init__(self, cfg: Config = CONFIG) -> None:
        self.cfg = cfg
        self._lock = threading.Lock()

        self._clients = []
        for p in LLM_PROVIDERS:
            try:
                client = OpenAI(
                    api_key=p["api_key"] or "unused",
                    base_url=p["base_url"],
                    timeout=cfg.request_timeout * 3,
                    max_retries=0,
                )
                self._clients.append({
                    "name": p["name"],
                    "model": p["model"],
                    "client": client,
                })
                log.info("provider ready: %s (%s)", p["name"], p["model"])
            except Exception as e:
                log.warning("provider %s init failed: %s", p["name"], e)

        if not self._clients:
            raise RuntimeError("No LLM providers initialized")

        self._current = 0

    def _call_one(self, provider, system, user, temperature, max_tokens):
        import time as _time
        _t0 = _time.time()
        try:
            with self._lock:
                resp = provider["client"].chat.completions.create(
                    model=provider["model"],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                )
            if not resp.choices:
                raise LLMResponseError("no choices")
            return (resp.choices[0].message.content or "").strip()
        finally:
            # Track total time spent inside provider calls
            _elapsed = _time.time() - _t0
            # If a call took >5s, assume most of it was waiting
            # (rate-limit wait or slow provider)
            if _elapsed > 5:
                _record_wait(_elapsed - 2)  # estimate 2s real work

    def chat(self, system, user, *, temperature=0.2, max_tokens=None):
        system = (
            "Never call tools or functions. Respond with plain text or JSON only.\n\n"
            + system
        )
        max_tokens = max_tokens or DEFAULT_MAX_TOKENS

        errors = []
        n = len(self._clients)

        import time as _time
        for attempt in range(n):
            idx = (self._current + attempt) % n
            provider = self._clients[idx]
            _start = _time.time()
            try:
                text = self._call_one(provider, system, user, temperature, max_tokens)
                if not text:
                    # Empty output — treat as failure, try next provider
                    log.warning("[%s] returned empty content", provider["name"])
                    errors.append(f"{provider['name']}: empty")
                    continue
                self._current = (idx + 1) % n
                return text
            except RateLimitError as e:
                log.warning("[%s] rate-limited: %s", provider["name"], str(e)[:80])
                errors.append(f"{provider['name']}: 429")
                continue
            except APIError as e:
                log.warning("[%s] API error: %s", provider["name"], str(e)[:80])
                errors.append(f"{provider['name']}: {str(e)[:60]}")
                continue
            except Exception as e:
                log.warning("[%s] failed: %s", provider["name"], str(e)[:80])
                errors.append(f"{provider['name']}: {type(e).__name__}")
                continue

        raise LLMResponseError(f"all {n} providers failed: {'; '.join(errors)}")

    def chat_json(self, system, user, *, temperature=0.1, max_tokens=None):
        system_json = (
            system.rstrip()
            + "\n\nYou MUST respond with valid JSON only. No prose, no fences."
        )
        raw = self.chat(system_json, user, temperature=temperature, max_tokens=max_tokens)
        return parse_json_loose(raw)

    def embed(self, texts):
        from chromadb.utils import embedding_functions
        ef = embedding_functions.DefaultEmbeddingFunction()
        batch = [texts] if isinstance(texts, str) else list(texts)
        if not batch:
            return []
        return [[float(x) for x in vec] for vec in ef(batch)]

    def embed_one(self, text):
        vecs = self.embed([text])
        if not vecs:
            raise LLMResponseError("embedding returned no vectors")
        return vecs[0]

    def provider_info(self):
        return [{"name": c["name"], "model": c["model"]} for c in self._clients]


__all__ = ["LLMClient", "LLMResponseError", "parse_json_loose", "reset_wait_timer", "get_wait_seconds"]
