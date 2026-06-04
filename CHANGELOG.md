# Changelog

All notable changes to Cell-mem will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.0] — 2026-06-04

### Phase 4 — Generative Replay & Full Reflection (feature-complete release)

#### Added
- **Generative Replay Engine** (`replay/engine.py`): 5-stage algorithm for creative
  hypothesis generation (biased sampling → random walk → cross-domain pairing →
  4-layer noise filter → creative pool management)
- **Creative Pool** (`replay/creative_pool.py`): Hypothesis lifecycle management
  (pending → confirmed/rejected → promoted to semantic), with optimistic locking
  to prevent concurrent update conflicts
- **Full 4-Dimension Self-Reflection** (`reflection/engine.py`): Effect attribution,
  strategy evaluation, knowledge gap detection, and result processing
- **Explore/Exploit Balance** (`memory/procedural.py`): 80/20 epsilon-greedy
  template selection in `match_by_context()`
- **Environment Snapshots** (`conditions/evaluator.py`): Automatic change detection
  and auto-verification of falsifiable conditions via `memory_check_environment`
- **12 MCP Tools**: memory_save, memory_recall, memory_status, memory_associate,
  memory_forget, memory_consolidate, memory_verify, memory_reflect,
  memory_replay, memory_hypothesis_feedback, memory_creative_pool,
  memory_check_environment

#### Security
- SSRF prevention on all LLM API calls (blocks RFC 1918, link-local, CGNAT, IPv6 private)
- LLM output validation against expected schema types
- LLM prompt injection mitigation via content truncation and sanitization
- HTTP mode authentication via `--api-key` shared-secret
- Environment size limits (max 200 keys, 10KB per value, depth ≤ 3)

### Phase 3 — LLM, Procedural Memory & Falsifiable Conditions

#### Added
- **LLM Abstraction Layer** (`llm/`): OpenAI and Claude backends using stdlib `urllib` only
- **Procedural Memory** (`memory/procedural.py`): Basal-ganglia-inspired skill template
  storage with cosine-similarity condition-triggered retrieval and RL weight updates
- **Falsifiable Conditions** (`conditions/evaluator.py`): Fact-attached conditions that
  invalidate memories when environment changes, with operators: `version_changed`,
  `file_modified`, `not_found`, `value_changed`
- **Self-Reflection Engine** (`reflection/engine.py`): Dimension 1 — effect attribution
  with causal analysis
- **Emotional Scoring** (`consolidation/emotional.py`): Multi-dimensional scoring
  (recency, frequency, valence, surprise) with optional LLM enhancement
- `memory_verify` and `memory_reflect` MCP tools

### Phase 2 — Graph, Consolidation & Forgetting

#### Added
- **Graph Store** (`graph/store.py`): NetworkX-backed associative graph with
  `memory_associate`, `memory_forget`, and `memory_consolidate` MCP tools
- **Spreading Activation** (`graph/activation.py`): BFS-based retrieval traversal
- **Consolidation Scheduler** (`consolidation/scheduler.py`): Periodic cycle with
  scoring, forget candidate tracking, cold storage archival, and state persistence
- **Consolidation Scorer** (`consolidation/scorer.py`): Multi-factor episode scoring
- **Pattern Detector** (`consolidation/detector.py`): DBSCAN clustering on embeddings
- **Search Engine** (`storage/search.py`): FTS5 full-text search with OR semantics

### Phase 1 — Core Memory System

#### Added
- **Four Memory Layers**: Working (attention decay), Episodic (pattern separation),
  Semantic (facts with confidence), Procedural (RL-weighted templates)
- **SQLite Storage** (`storage/sqlite_store.py`): Schema migrations, WAL mode,
  meta key-value store
- **Vector Storage** (`storage/vector_store.py`): sqlite-vec + ChromaDB backends
- **Local Embeddings** (`embedding/local.py`): SentenceTransformers all-MiniLM-L6-v2 (384d)
- **Pattern Separation**: 2048d projection matrix for episodic memory
- **MCP Server** (`server.py`): Dual transport (stdio + HTTP) via FastMCP
- `memory_save`, `memory_recall`, `memory_status` MCP tools
- Pydantic data models (`MemoryObject`, `MemoryType`, `StatusReport`, `LifecycleStage`)

---

[0.1.0]: https://github.com/liusiqi/cell-mem/releases/tag/v0.1.0
