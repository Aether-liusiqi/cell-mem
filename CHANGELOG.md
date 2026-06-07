# Changelog

All notable changes to Cell-mem will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.2.1] — 2026-06-07

### Added
- **Automatic Session Recording Hooks** (`hooks/`): Auto-register hook scripts into
  Codex CLI and Claude Code configs via `--hooks install`. SessionStart and PostToolUse
  events are captured by a standalone hook script (`session_hook.py`, zero cell_mem
  dependencies) and asynchronously written to episodic memory via an internal HTTP
  ingest endpoint (`ingest.py`, stdlib-only daemon server).
- **Dual-Platform Registrar** (`hooks/registrar.py`): Idempotent merge-based registration
  for both Codex (`hooks.json` + `config.toml`) and Claude Code (`settings.json`).
  `--hooks clean` removes all traces without touching unrelated hook entries.
- **`--hooks` / `--hooks-platform` / `--ingest-port`** CLI arguments on `server.py`.

### Design
- Session content is saved immediately (embedding=NULL, ~1ms) — no dependency on
  embedding model startup. EmbeddingWorker backfills vectors asynchronously.
- Hook script never blocks the agent — all failure paths silently exit 0.
- Ingest endpoint binds 127.0.0.1 only (security: internal channel, never exposed).

---

## [0.2.0] — 2026-06-06

### Added
- **User Preference Closed-Loop System** (`consolidation/preference.py`): Five-stage
  automatic pipeline — SignalDetector → Extractor → Processor → Injector → Feedback.
  Preferences auto-extracted from episodic saves, conflict-resolved, context-triggered
  on recall (max 3, ≤300 chars), and continuously refined via implicit feedback.
- **5 Preference MCP Tools** (`tools/preference.py`): extract_preferences, get_preferences,
  check_preference_conflicts, inject_preference, record_preference_feedback.
- **Async Embedding Architecture** (`embedding/worker.py`): Background EmbeddingWorker
  daemon thread decouples `memory_save` (instant, embedding=NULL) from embedding model
  inference (batched, async). FTS5 fallback when vectors unavailable.
- **ONNX Runtime Backend** (`embedding/onnx.py`): Zero-PyTorch embedding via ONNX Runtime
  (~15MB) + tokenizers (~5MB). Auto backend selection: ONNX first, PyTorch fallback.
  Same model weights, zero quality loss, ~8x faster load time.
- **17 MCP Tools** (up from 12): +5 preference pipeline tools.

### Fixed
- 28+ bugs across 5 rounds of Codex testing (see [FIXES.md](FIXES.md))
- Key fixes: thread-safe model loading, ONNX fallback state cleanup, FTS5 result
  deduplication, associate ID validation, working_memory tags_json column, ChromaDB
  empty metadata rejection, MCP handshake timeout (lazy preload).

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

[0.2.1]: https://github.com/Aether-liusiqi/cell-mem/releases/tag/v0.2.1
[0.2.0]: https://github.com/Aether-liusiqi/cell-mem/releases/tag/v0.2.0
[0.1.0]: https://github.com/Aether-liusiqi/cell-mem/releases/tag/v0.1.0
