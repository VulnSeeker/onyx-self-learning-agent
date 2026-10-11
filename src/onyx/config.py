"""
Configuration for Onyx — multi-provider LLM pool.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv(override=False)


def _env_str(key: str, default: str) -> str:
    val = os.getenv(key)
    return val.strip() if val and val.strip() else default


def _env_opt_str(key: str) -> Optional[str]:
    val = os.getenv(key)
    return val.strip() if val and val.strip() else None


def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


# ---------------------------------------------------------------------------
# Provider pool — priority order, failover on 429 / error
# ---------------------------------------------------------------------------

LLM_PROVIDERS = [
    {
        "name": "groq",
        "base_url": "https://api.groq.com/openai/v1",
        "model": "openai/gpt-oss-120b",
        "api_key": _env_str("GROQ_API_KEY", ""),
    },
    {
        "name": "cohere",
        "base_url": "https://api.cohere.com/compatibility/v1",
        "model": "command-r-plus-08-2024",
        "api_key": _env_str("COHERE_API_KEY", ""),
    },
]

# Drop providers with no key
LLM_PROVIDERS = [p for p in LLM_PROVIDERS if p["api_key"] and p["api_key"].strip()]


@dataclass(frozen=True)
class Config:
    openai_api_key: str = "unused"
    openai_base_url: str = "https://api.groq.com/openai/v1"
    llm_model: str = "llama-3.3-70b-versatile"
    embed_model: str = "text-embedding-3-small"
    tavily_api_key: Optional[str] = None
    data_dir: Path = Path("/content/onyx-data")
    skills_dir: Path = Path("/content/onyx-data/skills")
    chroma_dir: Path = Path("/content/onyx-data/chroma")
    registry_db: Path = Path("/content/onyx-data/registry.sqlite3")
    logs_dir: Path = Path("/content/onyx-data/logs")
    max_sources_per_skill: int = 12
    max_pages_per_source: int = 1
    request_timeout: int = 20
    llm_max_attempts: int = 4
    code_timeout: int = 20
    dedupe_distance: float = 0.12
    max_chars_per_page: int = 12000
    log_level: str = "INFO"
    max_skills: int = 20
    extract_delay_seconds: int = 4

    @classmethod
    def from_env(cls) -> "Config":
        data_dir = Path(_env_str("DATA_DIR", "/content/onyx-data")).expanduser().resolve()
        skills_dir = Path(_env_str("SKILLS_DIR", str(data_dir / "skills"))).expanduser().resolve()
        chroma_dir = data_dir / "chroma"
        logs_dir = data_dir / "logs"
        registry_db = data_dir / "registry.sqlite3"
        for p in (data_dir, skills_dir, chroma_dir, logs_dir):
            p.mkdir(parents=True, exist_ok=True)

        return cls(
            tavily_api_key=_env_opt_str("TAVILY_API_KEY"),
            data_dir=data_dir,
            skills_dir=skills_dir,
            chroma_dir=chroma_dir,
            registry_db=registry_db,
            logs_dir=logs_dir,
            max_sources_per_skill=_env_int("MAX_SOURCES_PER_SKILL", 12),
            max_pages_per_source=_env_int("MAX_PAGES_PER_SOURCE", 1),
            request_timeout=_env_int("REQUEST_TIMEOUT", 20),
            llm_max_attempts=_env_int("LLM_MAX_ATTEMPTS", 4),
            code_timeout=_env_int("CODE_TIMEOUT", 20),
            dedupe_distance=_env_float("DEDUPE_DISTANCE", 0.12),
            max_chars_per_page=_env_int("MAX_CHARS_PER_PAGE", 12000),
            log_level=_env_str("LOG_LEVEL", "INFO").upper(),
            max_skills=_env_int("MAX_SKILLS", 20),
            extract_delay_seconds=_env_int("EXTRACT_DELAY_SECONDS", 4),
        )

    def summary(self) -> str:
        providers = ", ".join(p["name"] for p in LLM_PROVIDERS)
        return f"providers=[{providers}] data={self.data_dir}"


CONFIG: Config = Config.from_env()

__all__ = ["Config", "CONFIG", "LLM_PROVIDERS"]
