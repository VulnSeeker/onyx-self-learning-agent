"""
Bounded-memory manager for Onyx.

Onyx accumulates skills as it learns. Without a cap, the skill registry
grows without limit and the vector DB bloats. This module enforces a
fixed maximum and gives the user control over what gets evicted.

Design:
  - MAX_SKILLS (default 20, configurable via env var)
  - When a new skill is about to be added and the registry is full,
    the MemoryManager asks a callback for the skill to evict.
  - The callback can be:
      * interactive (CLI prompt showing usage stats)
      * programmatic (return least-used, return oldest, return None)
      * absent -> raise MemoryFullError
  - Every skill records usage_count and last_used so the user has data
    for the eviction decision.

Eviction removes the skill from BOTH the SQLite registry AND the
ChromaDB vector store.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

from onyx.config import CONFIG, Config
from onyx.storage import Registry, VectorStore

log = logging.getLogger("onyx.memory")


# Default cap. Override via MAX_SKILLS env var (see config.py).
DEFAULT_MAX_SKILLS = 20


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class MemoryFullError(RuntimeError):
    """Raised when the skill cap is reached and no eviction callback is set."""
    def __init__(self, used: int, cap: int, skills: list[dict]) -> None:
        self.used = used
        self.cap = cap
        self.skills = skills
        super().__init__(
            f"Skill memory full ({used}/{cap}). "
            f"Provide an eviction callback or delete a skill manually."
        )


# Callback shape: receives the list of skill stats, returns the skill_id
# to evict, or None to abort the operation.
EvictionCallback = Callable[[list[dict]], Optional[str]]


# ---------------------------------------------------------------------------
# MemoryManager
# ---------------------------------------------------------------------------

class MemoryManager:
    """Enforces a bounded number of skills with usage-guided eviction."""

    def __init__(
        self,
        registry: Registry,
        vectors: VectorStore,
        max_skills: int = DEFAULT_MAX_SKILLS,
        cfg: Config = CONFIG,
    ) -> None:
        self.registry = registry
        self.vectors = vectors
        self.max_skills = max(1, int(max_skills))
        self.cfg = cfg

    # ------------------------------------------------------------------
    # Capacity
    # ------------------------------------------------------------------

    def capacity(self) -> dict[str, int]:
        """Return {used, cap, free}."""
        used = self.registry.count()
        return {
            "used": used,
            "cap": self.max_skills,
            "free": max(0, self.max_skills - used),
        }

    def is_full(self) -> bool:
        """True if the registry is at or above the cap."""
        return self.registry.count() >= self.max_skills

    def stats(self) -> dict:
        """Full stats for UI display: capacity + all skills with usage."""
        cap = self.capacity()
        skills = self.registry.list_with_stats()
        return {
            "used": cap["used"],
            "cap": cap["cap"],
            "free": cap["free"],
            "full": cap["used"] >= cap["cap"],
            "skills": skills,
        }

    # ------------------------------------------------------------------
    # Eviction
    # ------------------------------------------------------------------

    def evict(self, skill_id: str) -> bool:
        """
        Delete a skill from both the registry and the vector store.
        Returns True if the skill existed and was removed.
        """
        if not skill_id:
            return False

        existed = self.registry.has(skill_id)
        if not existed:
            log.warning("evict: skill '%s' not found", skill_id)
            return False

        # Drop vectors first — if it fails, registry stays intact
        try:
            self.vectors.drop(skill_id)
        except Exception as e:  # noqa: BLE001
            log.warning("evict: vector drop failed for '%s': %s", skill_id, e)

        deleted = self.registry.delete(skill_id)
        if deleted:
            log.info("Evicted skill '%s'", skill_id)
        return deleted

    def check_capacity(
        self,
        on_full: Optional[EvictionCallback] = None,
        new_skill_id: str = "",
    ) -> Optional[str]:
        """
        Called before adding a new skill.

        Returns:
            - None  if there is room (no eviction needed)
            - skill_id  if a skill must be evicted (caller should call evict())

        Raises:
            MemoryFullError if the registry is full and no on_full callback
            is provided.
        """
        if not self.is_full():
            return None

        skills = self.registry.list_with_stats()

        if on_full is None:
            cap = self.capacity()
            raise MemoryFullError(cap["used"], cap["cap"], skills)

        chosen = on_full(skills)
        if not chosen:
            log.info("Eviction cancelled by callback")
            return None

        if chosen == new_skill_id:
            log.warning("Callback tried to evict the skill being added")
            return None

        if not self.registry.has(chosen):
            log.warning("Callback chose unknown skill '%s'", chosen)
            return None

        return chosen

    # ------------------------------------------------------------------
    # Suggest
    # ------------------------------------------------------------------

    def suggest_eviction(self) -> Optional[str]:
        """
        Fallback heuristic: return the least-used skill; ties broken by
        oldest last_used. Useful when no human is available to decide.
        """
        skills = self.registry.list_with_stats()
        if not skills:
            return None

        def sort_key(s: dict):
            # unused first, then lowest count, then oldest last_used
            return (
                int(s.get("usage_count", 0)),
                s.get("last_used") or "0000-00-00T00:00:00",
            )

        return sorted(skills, key=sort_key)[0]["skill_id"]



# ---------------------------------------------------------------------------
# Eviction policies
# ---------------------------------------------------------------------------

def _policy_fifo(skills: list[dict], new_skill_id: str = "") -> str | None:
    """Evict the oldest skill (by created_at)."""
    if not skills:
        return None
    return sorted(skills, key=lambda s: s.get("created_at", ""))[0]["skill_id"]


def _policy_lru(skills: list[dict], new_skill_id: str = "") -> str | None:
    """Evict the least-recently-used skill (by last_used)."""
    if not skills:
        return None
    return sorted(
        skills,
        key=lambda s: s.get("last_used") or "0000-00-00",
    )[0]["skill_id"]


def _policy_random(skills: list[dict], new_skill_id: str = "") -> str | None:
    """Evict a random skill."""
    if not skills:
        return None
    import random
    return random.choice(skills)["skill_id"]


def _policy_user(skills: list[dict], new_skill_id: str = "") -> str | None:
    """Placeholder — user-guided policy is handled by callback."""
    return None


EVICTION_POLICIES = {
    "fifo": _policy_fifo,
    "lru": _policy_lru,
    "random": _policy_random,
    "user": _policy_user,
}


__all__ = ["MemoryManager", "MemoryFullError", "EvictionCallback", "DEFAULT_MAX_SKILLS", "EVICTION_POLICIES"]
