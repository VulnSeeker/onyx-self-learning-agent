"""
Answer quality scorer for Onyx.

LLM-as-judge with an explicit 0-3 rubric. Used by the benchmark to
measure whether bounded memory improves answer quality.

Design notes:
  - NOT called on every production task (would double API cost).
  - The benchmark harness calls it explicitly per task.
  - Every benchmark artifact includes the score, so results are
    reproducible from saved JSON.

Rubric (0-3):
  0 = wrong or empty
  1 = partial / significant gaps
  2 = correct but incomplete
  3 = correct and complete
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict

from onyx.config import CONFIG, Config
from onyx.llm import LLMClient, LLMResponseError

log = logging.getLogger("onyx.scoring")


_SCORE_SYSTEM = """You are a strict research evaluator scoring answers
produced by an autonomous agent on a 0-3 scale.

Rubric:
  0 = Wrong or empty. The answer does not address the task, or contains
      fabricated information contradicted by the context.
  1 = Partial. Some relevant content, but misses key points or has
      significant inaccuracies.
  2 = Correct but incomplete. The answer addresses the task accurately
      but omits important details.
  3 = Correct and complete. The answer fully addresses the task, is
      accurate, and is grounded in the provided context.

Rules:
  - Score the answer, not the task.
  - If the answer says "I don't know" but the context contains the
    information, score 0.
  - If the answer invents facts not in the context, cap at 1.
  - Be strict. Most answers should score 1 or 2, not 3.

Return strict JSON:
{
  "score": 0 | 1 | 2 | 3,
  "reason": "one sentence explaining the score",
  "key_point": "the single biggest strength or flaw"
}
"""


@dataclass
class ScoreResult:
    score: int
    reason: str
    key_point: str

    def to_dict(self) -> dict:
        return asdict(self)


class AnswerScorer:
    """LLM-as-judge scorer for task answers."""

    def __init__(self, llm: LLMClient, cfg: Config = CONFIG) -> None:
        self.llm = llm
        self.cfg = cfg

    def score(
        self,
        task: str,
        answer: str,
        context: str = "",
    ) -> ScoreResult:
        if not answer or not answer.strip():
            return ScoreResult(
                score=0,
                reason="empty answer",
                key_point="no content produced",
            )

        prompt = (
            f"Task:\n{task[:2000]}\n\n"
            f"Answer to score:\n{answer[:3000]}\n\n"
            f"Context provided to the agent (may be empty):\n"
            f"{context[:2000] or '(none)'}"
        )

        try:
            data = self.llm.chat_json(_SCORE_SYSTEM, prompt)
        except LLMResponseError as e:
            log.warning("scoring failed: %s", e)
            return ScoreResult(
                score=0,
                reason=f"scorer error: {e}",
                key_point="scoring failed",
            )

        if not isinstance(data, dict):
            return ScoreResult(
                score=0,
                reason="invalid scorer output (non-object)",
                key_point="bad JSON",
            )

        try:
            score = int(data.get("score", 0))
        except (ValueError, TypeError):
            score = 0

        score = max(0, min(3, score))

        return ScoreResult(
            score=score,
            reason=str(data.get("reason", ""))[:300],
            key_point=str(data.get("key_point", ""))[:200],
        )


__all__ = ["AnswerScorer", "ScoreResult"]
