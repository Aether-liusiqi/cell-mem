# Cell-mem

**Brain-inspired memory system for AI Agents** — an MCP (Model Context Protocol) server
that gives AI agents persistent, multi-layered memory with consolidation, self-reflection,
generative replay, and creative hypothesis discovery.

Cell-mem models the human brain's memory architecture: four interconnected memory layers
operating at different timescales, governed by neuro-inspired consolidation and forgetting
processes. It ships as an MCP server — drop it into Claude Code, Codex CLI, or any
MCP-compatible agent host.

> **Status:** Phase 4 (feature-complete) — see [CHANGELOG.md](CHANGELOG.md) for version history.

---

## Architecture

```
MCP Server (stdio + HTTP transport)
│
├── 12 MCP Tools
│   ├── memory_save            — Store a memory
│   ├── memory_recall          — Cross-layer retrieval
│   ├── memory_status          — System health dashboard
│   ├── memory_associate       — Link two memories (graph edge)
│   ├── memory_forget          — Manual memory removal
│   ├── memory_consolidate     — Trigger consolidation cycle
│   ├── memory_verify          — Check falsifiable conditions
│   ├── memory_reflect         — Self-reflection (failure analysis + strategy eval)
│   ├── memory_replay          — Trigger generative replay (hypothesis creation)
│   ├── memory_hypothesis_feedback — Confirm/reject a creative hypothesis
│   ├── memory_creative_pool   — Inspect the hypothesis pool
│   └── memory_check_environment  — Detect environment changes → auto-verify
│
├── Memory Layers (brain-inspired)
│   ├── Working Memory    <minutes>  ~50 items, attention-based decay
│   ├── Episodic Memory   <days>     pattern-separated experience storage
│   ├── Semantic Memory   <months>   facts with falsifiable conditions
│   └── Procedural Memory <months>   skill/strategy templates with RL weighting
│
├── Consolidation Processor
│   ├── Emotional scoring (multi-dimensional: recency, frequency, valence, surprise)
│   ├── DBSCAN pattern detection
│   ├── Forgetting (low-score → cold storage archive, rescuer support)
│   └── State persistence across restarts
│
├── Reflective System (Phase 3–4)
│   ├── Effect attribution — "What went wrong and why?"
│   ├── Strategy evaluation — Success trends, better variants, redundancy
│   ├── Knowledge gap detection — Missing info? Retrieval failure?
│   └── Result processing — Update procedural weights, adjust semantic confidence
│
├── Generative Replay Engine (Phase 4)
│   ├── 5-stage algorithm: biased sampling → random walk → cross-domain pairing
│   │                      → 4-layer noise filter → creative pool management
│   ├── Creative pool: hypothesis lifecycle (pending → confirmed/rejected → promoted)
│   └── 10 noise constraints to prevent hallucinations from persisting
│
└── Storage (SQLite)
    ├── sqlite-vec vector search (384d all-MiniLM-L6-v2 embeddings)
    ├── FTS5 full-text search with OR semantics
    ├── Graph store (NetworkX-backed, spreading activation)
    └── Cold storage archive (forgotten but rescuable)
```

---

## Quick Start

### Installation

```bash
# From the repository root
pip install -e .

# With HTTP transport support
pip install -e ".[http]"

# With development tools (linting, testing)
pip install -e ".[dev]"
```

### Run as MCP Server

```bash
# stdio mode — agent launches as subprocess (no network)
python -m cell_mem.server

# HTTP mode — daemon for hook scripts, multiple agents
python -m cell_mem.server --http --port 8765

# HTTP with shared-secret authentication (recommended for production)
python -m cell_mem.server --http --port 8765 --api-key "your-secret-here"

# Preload embedding model (avoids ~30s first-request delay)
python -m cell_mem.server --preload

# With seed knowledge (pre-populate semantic memory)
python -m cell_mem.server --seed-config config/seed_knowledge.example.json
```

### MCP Client Configuration

Add to your agent's MCP configuration:

**Claude Code** (`~/.claude/claude_desktop_config.json`):
```json
{
  "mcpServers": {
    "cell-mem": {
      "command": "python",
      "args": ["-m", "cell_mem.server", "--db", "/path/to/cell_mem.db"]
    }
  }
}
```

**Codex CLI**:
```json
{
  "mcpServers": {
    "cell-mem": {
      "command": "python",
      "args": ["-m", "cell_mem.server"],
      "env": { "CELL_MEM_DB": "/path/to/cell_mem.db" }
    }
  }
}
```

---

## Memory Layers

### Working Memory (seconds–minutes)
- Capacity-limited (~50 items), attention-based decay
- Items are pushed to a "preheated zone" before aging out
- Emulates the prefrontal cortex's short-term buffer

### Episodic Memory (hours–days)
- Pattern-separated storage: 384d content embedding → 2048d projection
- Reduces interference between similar but distinct episodes
- Consolidation scoring determines retention priority

### Semantic Memory (weeks–months)
- Facts, knowledge, and rules with optional **falsifiable conditions**
- Conditions define what would make the fact outdated (e.g., "package.json version changed")
- `memory_verify` checks conditions against environment snapshots
- High-confidence + locked lifecycle → resist unlearning

### Procedural Memory (days–months)
- Skill/strategy templates triggered by **cosine similarity** to current context
- **Reinforcement learning**: success → weight × 1.05, failure → weight × 0.85
- **Explore/exploit balance**: 80% exploit (best match), 20% explore (novel picks)
- Templates with weight < 0.25 → candidates for reflection review

---

## Key Mechanisms

### Consolidation (Phase 2)
Automatic (via `should_run()`) or manual (`memory_consolidate`) cycles:
1. Score all episodes on recency, frequency, emotional valence, surprise
2. Identify low-score candidates for forgetting
3. After 3 consecutive low-score cycles → archive to cold storage (rescuable)
4. Run DBSCAN pattern clustering to detect emerging knowledge patterns

### Self-Reflection (Phases 3–4)
Four-dimensional meta-reasoning over failure events:
- **Dimension 1 — Effect Attribution**: Causal analysis of failures
- **Dimension 2 — Strategy Evaluation**: Success rate trends, variant comparison
- **Dimension 3 — Knowledge Gap Detection**: Missing facts or retrieval failures
- **Dimension 4 — Result Processing**: Update procedural weights, adjust confidences, create meta-knowledge

### Generative Replay (Phase 4)
Five-stage creative hypothesis engine inspired by hippocampal replay:
1. **Biased sampling** — pick K=3 seeds proportional to recency × emotional salience × novelty
2. **Random walk** — L=3 steps per seed, 80/20 strong/weak edge sampling
3. **Cross-domain pairing** — pair low-similarity concepts from different seeds
4. **4-layer noise filter** — contradiction check, triviality filter, dual-source verification, stability requirement
5. **Creative pool management** — 10 noise constraints, pending → confirmed → promoted lifecycle

### Falsifiable Conditions (Phase 3)
Each semantic fact can carry a `falsifiable_condition`:
```json
{
  "field": "package.json",
  "operator": "value_changed",
  "value": "react"
}
```
`memory_check_environment` compares current vs. last snapshot → auto-triggers `memory_verify` for affected facts. Or use `memory_verify` manually with a specific fact ID.

---

## Python API

```python
from cell_mem import MemorySystem

# Initialize (all layers + embedding model)
ms = MemorySystem("cell_mem.db")

# Store across layers
ms.save("User prefers dark theme", memory_type="semantic", confidence=0.9)
ms.save("Fixed the login bug with OAuth", memory_type="episodic")
ms.save("When encountering CORS errors, check server middleware first",
        memory_type="procedural", trigger_condition="CORS error debugging")

# Recall (cross-layer: semantic FTS5 + episodic embedding + procedural context)
results = ms.recall("How to debug CORS?")

# Graph associations
ms.associate(id_a, id_b, weight=0.8, relation="related_to")

# Status dashboard
status = ms.status()
# Layers: working/episodic/semantic/procedural counts + consolidation stats
# + creative pool + LLM usage + reflection history

# Self-reflection
ms.reflect(task="Fix CORS bug", outcome="Failure", dimensions="all")

# Generative replay (auto-creates hypotheses from memory graph)
ms.replay(theme_text="frontend debugging")

# Hypothesis feedback (confirmed → confidence boost; rejected → ignore_count++)
ms.record_hypothesis_feedback("hyp_abc123", confirmed=True)

# Environment change detection → auto-verify
ms.check_environment({"node_version": "18", "react_version": "19.0"})

ms.shutdown()
```

### LLM Backend Configuration

Phase 3–4 features (reflection, replay, emotional scoring) can optionally use an LLM:

```python
ms = MemorySystem(
    "cell_mem.db",
    llm_backend="openai",       # "openai" or "claude"
    llm_api_key="sk-...",       # or set OPENAI_API_KEY / ANTHROPIC_API_KEY env var
    llm_daily_limit=100,        # rate limiting (default: 100 calls/day)
)
```

Without an LLM, emotional scoring falls back to rule-based heuristics, and
reflection/replay operations return informative errors. Core save/recall/status
do not require an LLM.

---

## Project Structure

```
src/cell_mem/
├── __init__.py              # Public API exports
├── models.py                # Pydantic data models
├── memory_system.py         # Top-level facade (main API)
├── server.py                # MCP server entry point
│
├── storage/                 # SQLite + vector storage
│   ├── sqlite_store.py      # Schema, migrations, meta table
│   ├── vector_store.py      # sqlite-vec and ChromaDB backends
│   └── search.py            # FTS5 search engine
│
├── embedding/               # Embedding models
│   └── local.py             # SentenceTransformers (all-MiniLM-L6-v2)
│
├── memory/                  # Four memory layers
│   ├── working.py           # Working memory (attention decay)
│   ├── episodic.py          # Episodic memory (pattern separation)
│   ├── semantic.py          # Semantic memory (falsifiable facts)
│   └── procedural.py        # Procedural memory (RL-weighted templates)
│
├── graph/                   # Associative graph
│   ├── store.py             # NetworkX graph store
│   ├── activation.py        # Spreading activation retrieval
│   └── networkx_store.py    # NetworkX adapter
│
├── consolidation/           # Sleep-like consolidation
│   ├── scorer.py            # Multi-dimension episode scoring
│   ├── detector.py          # DBSCAN pattern detection
│   ├── emotional.py         # Emotional valence evaluation
│   └── scheduler.py         # Cycle orchestrator + forgetting
│
├── reflection/              # Meta-reasoning (Phase 3–4)
│   └── engine.py            # 4-dimension reflection engine
│
├── conditions/              # Falsifiable conditions (Phase 3)
│   └── evaluator.py         # Condition checking + environment snapshots
│
├── replay/                  # Generative replay (Phase 4)
│   ├── engine.py            # 5-stage replay algorithm
│   └── creative_pool.py     # Hypothesis lifecycle management
│
├── llm/                     # LLM abstraction
│   ├── client.py            # Base client + rate limiter
│   └── backends.py          # OpenAI + Claude backends (stdlib only)
│
└── tools/                   # MCP tool registrations
    ├── save.py              # memory_save
    ├── recall.py            # memory_recall
    ├── status.py            # memory_status
    ├── verify.py            # memory_verify
    ├── reflect.py           # memory_reflect
    ├── replay.py            # memory_replay + creative pool tools
    └── stubs.py             # memory_associate, forget, consolidate, phase wiring
```

---

## Design Principles

- **Zero new pip dependencies for core operations.** LLM calls use stdlib `urllib` only.
  Dependencies (`mcp`, `sentence-transformers`, `numpy`, `networkx`, `scikit-learn`) are
  all well-established packages.
- **SSRF protection.** All LLM API calls validate the target URL against blocked private
  IP ranges (RFC 1918, link-local, CGNAT, IPv6 private).
- **SQL-first architecture.** SQLite with WAL mode, FTS5, sqlite-vec — all data local,
  no external services required.
- **Brain-inspired, not brain-simulated.** Algorithms are inspired by neuroscience
  (pattern separation, spreading activation, hippocampal replay) but optimized for
  practical agent memory, not biological fidelity.
- **Graceful degradation.** Optional features (LLM, HTTP, ChromaDB) degrade cleanly
  when not configured. Core memory operations always work.

---

## Requirements

- Python ≥ 3.11
- SQLite ≥ 3.35 (for sqlite-vec support)
- Optional: OpenAI or Anthropic API key (for LLM-powered features)

---

## License

MIT — see [LICENSE](LICENSE) for full text.

Copyright (c) 2026 Siqi Liu
