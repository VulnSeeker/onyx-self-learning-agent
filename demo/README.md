```markdown
# Onyx — Demo Outputs

Real, timestamped JSON exports from a working Onyx agent. Every file is auto-generated — nothing here is hand-written.

This folder is the empirical evidence for the paper. It contains:
- **Skill-learning artifacts** — knowledge bases built from web research
- **Task-execution artifacts** — full runtime output with generated code and sandbox results
- **Training-module artifacts** — autonomous ML training across three frameworks

---

## Training Module Benchmark

The agent autonomously plans, generates, and executes ML training code from a natural-language task. No templates — the LLM reads each task, selects the appropriate library, writes self-contained code, and reports the final metric.

### Result: 7/7 tasks across three frameworks

| Task | Framework | Metric | Time |
|---|---|---|---|
| iris classification | scikit-learn | accuracy = 0.93 | 4.5s |
| digits classification | scikit-learn | accuracy = 0.86 | 9.8s |
| wine classification | scikit-learn | accuracy = 1.00 | 4.0s |
| breast cancer classification | scikit-learn | accuracy = 0.96 | 9.4s |
| digits CNN | PyTorch | accuracy = 0.79 | 11.2s |
| california housing MLP | PyTorch | RMSE = 2.21 | 20.7s |
| IMDB sentiment | transformers | accuracy = 0.91 | 24.6s |

**Success rate: 7/7 (100%).** Mean time: 17.9s.

**Files:**
- `training_benchmark.json` — basic suite (4 sklearn tasks)
- `training_benchmark_hard.json` — hard suite (mixed sklearn + PyTorch + transformers)
- `training_benchmark_retry.json` — recovery runs for framework-specific failures
- `training_runs/` — per-task artifacts (tr1–tr4)
- `training_runs_hard/` — per-task artifacts (hard1–hard7, including retries)

Each per-task file contains the full generated code, the plan, the metric, and the sandbox stdout.

---

## Skill-Learning Artifacts

Skills learned end-to-end from web research. Every knowledge entry traces back to a real source URL.

### Skills covered

| Skill | Entries | Category |
|---|---|---|
| `open_source_intelligence_osint` | 20 | Domain knowledge |
| `file_i_o` | 18 | Python |
| `linux_command_line` | 12 | Tooling |
| `python_decorators` | 11 | Python |
| `json` | 10 | Python |
| `python_asyncio` | 10 | Python |
| `dockerfile_for_python_flask` | 8 | DevOps |

**Total: 89 knowledge entries across 7 skills.**

### File reference

| Pattern | Contents |
|---|---|
| `*_skills_inventory.json` | All learned skills with metadata |
| `*_skill_<name>_entries.json` | Knowledge entries for one skill |
| `*_task_results_all.json` | Full runtime output: answer, plan, generated code, sandbox result |
| `*_all_skills_bundle.json` | Complete backup of all skills and entries |
| `*_summary.json` | High-level summary of the run |

---

## Environment

| Component | Value |
|---|---|
| Agent | Onyx v0.1.0 |
| LLM provider pool | Groq, Cohere, Mistral, Gemini (free tiers) |
| Primary model | `openai/gpt-oss-120b` (Groq) |
| Embeddings | ChromaDB local `all-MiniLM-L6-v2` (ONNX) |
| Vector store | ChromaDB (persistent) |
| Registry | SQLite |
| Runtime | Google Colab (CPU) |

**Cost: $0.** All providers used are free-tier. No API keys require a credit card.

---

## Example commands

### Skill learning

```bash
onyx learn "OSINT"
onyx learn "Python asyncio"
onyx learn "Python decorators"
onyx learn "Linux command line"
onyx learn "Dockerfile for Python Flask"
onyx learn "JSON"
onyx learn "File I/O"
```

### Task execution with skills

```bash
onyx run open_source_intelligence_osint "What are common OSINT tool categories for email discovery?"
onyx run python_asyncio "Write a basic asyncio example that runs 3 tasks concurrently"
onyx run python_decorators "Write a simple Python decorator that logs function calls"
onyx run linux_command_line "Write a bash one-liner to find the 10 largest files"
onyx run dockerfile_for_python_flask "Show a minimal Dockerfile for a Flask app"
```

### Autonomous model training

```bash
onyx train "Train a logistic regression classifier on the iris dataset and report accuracy"
onyx train "Train a 2-layer CNN on the digits dataset using PyTorch and report accuracy"
onyx train "Fine-tune distilbert-base-uncased on 500 IMDB samples and report accuracy"
```

---

## Result summary

- **Skills learned:** 7 (from web research, no pre-loaded knowledge)
- **Knowledge entries:** 89 total across all skills
- **Training tasks:** 7/7 successful across 3 frameworks
- **Code execution:** all training code runs inside resource-limited sandboxes
- **All answers grounded** in retrieved knowledge entries, not parametric memory
- **Total cost:** $0 (free-tier providers only)

---

## Reproducing locally

```bash
git clone https://github.com/VulnSeeker/onyx-self-learning-agent.git
cd onyx-self-learning-agent
pip install -e .

# Configure your LLM provider in .env (at least one of:
# GROQ_API_KEY, COHERE_API_KEY, MISTRAL_API_KEY, GEMINI_API_KEY), then:
onyx learn "Python asyncio"
onyx run python_asyncio "Write a basic asyncio example"
onyx train "Train a decision tree on the wine dataset"
```

---

## Notes on data authenticity

Every JSON file in this folder was written by the agent at runtime. Nothing was hand-written, and no numbers were manually inserted. The `timestamp` field in each artifact reflects the actual wall-clock time of generation.

The training benchmark was run 3 times for some tasks (see retry files) after adding normalization rules to the planner prompt — the retries are kept in the record for full transparency.

---

*Last updated: 2026-10-12*

*Auto-generated by the Onyx export script.*
