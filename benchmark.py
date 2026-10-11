"""
Benchmark harness for Onyx.

Compares two conditions:
  cold  — fresh skill registry for each task (no memory reuse)
  warm  — shared registry across tasks (memory reuse)

Records per task: wall time, context chunks, judge score (0-3).
"""

from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path


# Pacing knobs — tune these for your LLM tier.
# Groq free tier (8000 TPM) needs long delays.
# Higher-TPM providers can use 0.
TASK_DELAY_SECONDS = 5         # Gemini free tier has 1M TPM
CONDITION_DELAY_SECONDS = 10   # brief pause between conditions
TASK_RETRY_ON_429 = 2          # retries per task if rate-limited
TASK_RETRY_WAIT = 20           # seconds to wait before retrying

TASKS = [
    {"id": "f1", "category": "factual",
     "prompt": "Explain how Python decorators work with a code example."},
    {"id": "f2", "category": "factual",
     "prompt": "What is the difference between a list and a tuple in Python?"},
    {"id": "f3", "category": "factual",
     "prompt": "Describe what a git branch is and why developers use them."},
    {"id": "f4", "category": "factual",
     "prompt": "How does Docker layer caching speed up rebuilds?"},
    {"id": "f5", "category": "factual",
     "prompt": "What problem does async/await solve in Python?"},
    {"id": "c1", "category": "code",
     "prompt": "Write a Python function that validates an email address using regex."},
    {"id": "c2", "category": "code",
     "prompt": "Write a Python script that lists the 5 largest files in a directory."},
    {"id": "c3", "category": "code",
     "prompt": "Write a Python decorator that logs the execution time of a function."},
    {"id": "t1", "category": "tool",
     "prompt": "What are the essential git commands for a solo developer working on a project?"},
    {"id": "t2", "category": "tool",
     "prompt": "Explain the main components of a Kubernetes cluster and what each does."},
]


HARD_TASKS = [
    {"id": "h1", "category": "synthesis",
     "prompt": "Across all the sources you have read about Docker, what are the top 3 anti-patterns developers fall into, and why?"},
    {"id": "h2", "category": "synthesis",
     "prompt": "Compare the Git branching strategies you have learned. Recommend one for a 3-person startup and justify the choice."},
    {"id": "h3", "category": "synthesis",
     "prompt": "Combine the Docker best practices from earlier sources into a single deployment checklist for a Python Flask app."},
    {"id": "h4", "category": "synthesis",
     "prompt": "Given what you know about Python asyncio and decorators, explain how to write an async-aware timing decorator."},
    {"id": "h5", "category": "synthesis",
     "prompt": "You learned Python fundamentals and Git basics earlier. How would you teach both to a beginner in a single 30-minute session?"},
]

# Extend the main task list with the harder synthesis tasks
TASKS = TASKS + HARD_TASKS


@dataclass
class TaskResult:
    task_id: str
    category: str
    prompt: str
    condition: str
    elapsed_seconds: float
    context_used: int
    score: int
    score_reason: str
    error: str = ""


@dataclass
class BenchmarkRun:
    started_at: str
    finished_at: str = ""
    cold: list = field(default_factory=list)
    warm: list = field(default_factory=list)
    summary: dict = field(default_factory=dict)


def _fresh_dir(tag: str) -> str:
    """Always-unique local dir. Fixes SQLite 'readonly DB moved' error."""
    import tempfile
    root = Path(tempfile.mkdtemp(prefix=f"onyx-bench-{tag}-"))
    return str(root)


def _reload_onyx():
    """Clear cached onyx modules so config picks up new DATA_DIR."""
    import sys
    for mod in list(sys.modules):
        if mod.startswith("onyx"):
            del sys.modules[mod]


def _run_one(ox, scorer, task, condition):
    """Run one task with retry-on-429 at the task level."""
    start = time.time()

    last_error = ""
    for attempt in range(TASK_RETRY_ON_429 + 1):
        try:
            result = ox.task(task["prompt"], auto_learn=True)
            elapsed = time.time() - start

            if not result.answer or len(result.answer.strip()) < 50:
                raise ValueError(
                    f"empty/short answer ({len(result.answer or '')} chars)"
                )

            scored = scorer.score(task["prompt"], result.answer)
            return TaskResult(
                task_id=task["id"], category=task["category"],
                prompt=task["prompt"], condition=condition,
                elapsed_seconds=round(elapsed, 2),
                context_used=result.context_used,
                score=scored.score, score_reason=scored.reason,
            )
        except Exception as e:
            err = str(e)
            last_error = f"{type(e).__name__}: {err[:200]}"
            is_429 = "429" in err or "rate_limit" in err.lower()

            if is_429 and attempt < TASK_RETRY_ON_429:
                wait = TASK_RETRY_WAIT * (attempt + 1)
                print(f"   [retry {attempt+1}/{TASK_RETRY_ON_429}] rate-limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            break

    return TaskResult(
        task_id=task["id"], category=task["category"],
        prompt=task["prompt"], condition=condition,
        elapsed_seconds=round(time.time() - start, 2),
        context_used=0, score=0, score_reason="",
        error=last_error,
    )


def run_benchmark(tasks=None):
    tasks = tasks or TASKS
    run = BenchmarkRun(started_at=datetime.now().isoformat())

    # --- COLD: fresh dir per task ---
    print("\n" + "=" * 60)
    print("COLD CONDITION (no memory reuse)")
    print("=" * 60)
    for i, task in enumerate(tasks, 1):
        print(f"\n[{i}/{len(tasks)}] {task['id']}: {task['prompt'][:60]}")
        os.environ["DATA_DIR"] = _fresh_dir(f"cold-{task['id']}")
        _reload_onyx()
        from onyx.agent import Onyx
        from onyx.scoring import AnswerScorer
        ox = Onyx()
        scorer = AnswerScorer(ox.llm)
        r = _run_one(ox, scorer, task, "cold")
        run.cold.append(r)
        print(f"   score={r.score}  time={r.elapsed_seconds}s  ctx={r.context_used}  err={bool(r.error)}")
        if i < len(tasks) and TASK_DELAY_SECONDS > 0:
            print(f"   [pacing] sleeping {TASK_DELAY_SECONDS}s before next task")
            time.sleep(TASK_DELAY_SECONDS)

    if CONDITION_DELAY_SECONDS > 0:
        print(f"\n[pacing] waiting {CONDITION_DELAY_SECONDS}s between conditions")
        time.sleep(CONDITION_DELAY_SECONDS)

    # --- WARM: shared dir across tasks ---
    print("\n" + "=" * 60)
    print("WARM CONDITION (memory reuse)")
    print("=" * 60)
    os.environ["DATA_DIR"] = _fresh_dir("warm")
    _reload_onyx()
    from onyx.agent import Onyx
    from onyx.scoring import AnswerScorer
    ox = Onyx()
    scorer = AnswerScorer(ox.llm)
    for i, task in enumerate(tasks, 1):
        print(f"\n[{i}/{len(tasks)}] {task['id']}: {task['prompt'][:60]}")
        r = _run_one(ox, scorer, task, "warm")
        run.warm.append(r)
        print(f"   score={r.score}  time={r.elapsed_seconds}s  ctx={r.context_used}  err={bool(r.error)}")
        if i < len(tasks) and TASK_DELAY_SECONDS > 0:
            print(f"   [pacing] sleeping {TASK_DELAY_SECONDS}s before next task")
            time.sleep(TASK_DELAY_SECONDS)

    run.finished_at = datetime.now().isoformat()
    run.summary = _summarize(run)
    return run


def _summarize(run):
    def agg(results):
        if not results:
            return {}
        n = len(results)
        return {
            "n": n,
            "mean_score": round(sum(r.score for r in results) / n, 3),
            "mean_time": round(sum(r.elapsed_seconds for r in results) / n, 2),
            "mean_context": round(sum(r.context_used for r in results) / n, 2),
            "errors": sum(1 for r in results if r.error),
        }
    c, w = agg(run.cold), agg(run.warm)
    return {
        "cold": c, "warm": w,
        "delta": {
            "score": round(w.get("mean_score", 0) - c.get("mean_score", 0), 3),
            "time": round(w.get("mean_time", 0) - c.get("mean_time", 0), 2),
        },
    }


def save_run(run):
    out_dir = Path("/content/onyx-benchmark/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = out_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    payload = {
        "started_at": run.started_at, "finished_at": run.finished_at,
        "cold": [asdict(r) for r in run.cold],
        "warm": [asdict(r) for r in run.warm],
        "summary": run.summary,
    }
    with open(filename, "w") as f:
        json.dump(payload, f, indent=2)
    return str(filename)


def print_table(run):
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    c, w = run.summary.get("cold", {}), run.summary.get("warm", {})
    print(f"\n{'Metric':<20} {'Cold':<12} {'Warm':<12} {'Delta'}")
    print("-" * 55)
    print(f"{'mean score':<20} {c.get('mean_score',0):<12} {w.get('mean_score',0):<12} {round(w.get('mean_score',0)-c.get('mean_score',0),3)}")
    print(f"{'mean time (s)':<20} {c.get('mean_time',0):<12} {w.get('mean_time',0):<12} {round(w.get('mean_time',0)-c.get('mean_time',0),2)}")
    print(f"{'mean context':<20} {c.get('mean_context',0):<12} {w.get('mean_context',0):<12} {round(w.get('mean_context',0)-c.get('mean_context',0),2)}")
    print(f"{'errors':<20} {c.get('errors',0):<12} {w.get('errors',0):<12}")


if __name__ == "__main__":
    run = run_benchmark()
    print_table(run)
    path = save_run(run)
    print(f"\n✅ Saved: {path}")


# ============================================================
# Multi-policy runner (for results paper)
# ============================================================

def run_policy(policy_name: str, tasks: list, repeat: int = 3) -> list:
    """
    Run all tasks under a single memory policy, repeated N times.

    Policies:
      cold   — fresh dir per task (no memory)
      warm   — user-guided eviction (default MemoryManager behavior)
      fifo   — evict oldest skill
      lru    — evict least-recently-used skill
      random — evict a random skill
    """
    results = []

    for run_idx in range(repeat):
        print(f"\n{'='*60}")
        print(f"POLICY: {policy_name}  RUN: {run_idx+1}/{repeat}")
        print(f"{'='*60}")

        # Choose data dir
        if policy_name == "cold":
            import tempfile
            data_dir = tempfile.mkdtemp(prefix=f"bench-cold-{run_idx}-")
        else:
            data_dir = f"/content/onyx-benchmark/policy-{policy_name}"
            Path(data_dir).mkdir(parents=True, exist_ok=True)

        os.environ["DATA_DIR"] = data_dir
        os.environ.pop("EXTRACT_DELAY_SECONDS", None)
        os.environ.pop("MAX_SKILLS", None)
        _reload_onyx()

        from onyx.agent import Onyx
        from onyx.scoring import AnswerScorer
        ox = Onyx()
        scorer = AnswerScorer(ox.llm)

        # Wire eviction policy into the memory manager
        if policy_name != "cold":
            try:
                from onyx.memory import EVICTION_POLICIES
                policy_fn = EVICTION_POLICIES.get(policy_name)

                if policy_fn is not None:
                    # Monkeypatch check_capacity to use the policy
                    original_check = ox.memory.check_capacity
                    def patched_check(on_full=None, new_skill_id="", _fn=policy_fn, _orig=original_check):
                        if not ox.memory.is_full():
                            return None
                        skills = ox.memory.registry.list_with_stats()
                        return _fn(skills, new_skill_id)
                    ox.memory.check_capacity = patched_check
            except Exception as e:
                print(f"  [warn] could not wire policy {policy_name}: {e}")

        for task in tasks:
            r = _run_one(ox, scorer, task, policy_name)
            results.append(r)
            print(f"  {task['id']:<5} score={r.score}  time={r.elapsed_seconds:.1f}s  ctx={r.context_used}")
            if TASK_DELAY_SECONDS > 0:
                time.sleep(TASK_DELAY_SECONDS)

    return results


def save_policy_results(policy_name: str, results: list) -> str:
    """Save a policy run's results to disk."""
    import json
    out_dir = Path("/content/onyx-benchmark/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = out_dir / f"policy-{policy_name}.json"

    payload = [
        {
            "task_id": r.task_id, "category": r.category, "prompt": r.prompt,
            "policy": r.condition, "score": r.score,
            "score_reason": r.score_reason,
            "elapsed_seconds": r.elapsed_seconds,
            "context_used": r.context_used, "error": r.error,
        }
        for r in results
    ]
    with open(filename, "w") as f:
        json.dump(payload, f, indent=2)
    return str(filename)
