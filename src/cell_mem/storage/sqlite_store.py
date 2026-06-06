"""SQLite storage engine with WAL mode, schema management, and sqlite-vec support.

This module is the foundation of Cell-mem's persistence layer. All memory layers
write through this single store. WAL mode enables concurrent read/write without
locking — critical for Phase 2's background consolidation processor.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 6

_SCHEMA_DDL = """
-- ============================================================
-- Meta: system configuration, projection matrix, schema version
-- ============================================================
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value BLOB NOT NULL
);

-- ============================================================
-- Working Memory: current session items (up to ~50)
-- ============================================================
CREATE TABLE IF NOT EXISTS working_memory (
    id               TEXT PRIMARY KEY,
    content          TEXT    NOT NULL,
    attention_score  REAL    NOT NULL DEFAULT 1.0,
    base_priority    REAL    NOT NULL DEFAULT 1.0,
    relevance        REAL    NOT NULL DEFAULT 1.0,
    last_accessed_at TEXT    NOT NULL,
    was_referenced   INTEGER NOT NULL DEFAULT 0,
    task_completed   INTEGER NOT NULL DEFAULT 0,
    session_id       TEXT    NOT NULL,
    created_at       TEXT    NOT NULL,
    tags_json        TEXT    NOT NULL DEFAULT '[]',
    metadata_json    TEXT    NOT NULL DEFAULT '{}'
);

-- ============================================================
-- Episodic Memory: interaction records
-- ============================================================
CREATE TABLE IF NOT EXISTS episodic_memory (
    id                  TEXT PRIMARY KEY,
    content             TEXT    NOT NULL,
    embedding           BLOB,
    projection_vector   BLOB,
    confidence          REAL    NOT NULL DEFAULT 0.0,
    valence             REAL    NOT NULL DEFAULT 0.0,
    consolidation_score REAL    NOT NULL DEFAULT 0.0,
    was_in_wm           INTEGER NOT NULL DEFAULT 0,
    session_id          TEXT,
    task_id             TEXT,
    created_at          TEXT    NOT NULL,
    event_at            TEXT,
    valid_until         TEXT,
    invalidated_at      TEXT,
    tags_json           TEXT    NOT NULL DEFAULT '[]',
    source_refs_json    TEXT    NOT NULL DEFAULT '[]',
    metadata_json       TEXT    NOT NULL DEFAULT '{}'
);

CREATE VIRTUAL TABLE IF NOT EXISTS episodic_fts USING fts5(
    content,
    content='episodic_memory',
    content_rowid='rowid'
);

CREATE TRIGGER IF NOT EXISTS episodic_ai AFTER INSERT ON episodic_memory BEGIN
    INSERT INTO episodic_fts(rowid, content) VALUES (new.rowid, new.content);
END;

CREATE TRIGGER IF NOT EXISTS episodic_ad AFTER DELETE ON episodic_memory BEGIN
    INSERT INTO episodic_fts(episodic_fts, rowid, content) VALUES ('delete', old.rowid, old.content);
END;

CREATE TRIGGER IF NOT EXISTS episodic_au AFTER UPDATE ON episodic_memory BEGIN
    INSERT INTO episodic_fts(episodic_fts, rowid, content) VALUES ('delete', old.rowid, old.content);
    INSERT INTO episodic_fts(rowid, content) VALUES (new.rowid, new.content);
END;

-- ============================================================
-- Semantic Memory: abstract knowledge
-- ============================================================
CREATE TABLE IF NOT EXISTS semantic_memory (
    id                      TEXT PRIMARY KEY,
    content                 TEXT    NOT NULL,
    embedding               BLOB,
    confidence              REAL    NOT NULL DEFAULT 0.0,
    lifecycle               TEXT    NOT NULL DEFAULT 'plastic',
    falsifiable_condition   TEXT,
    invalidated_at          TEXT,
    source_refs_json        TEXT    NOT NULL DEFAULT '[]',
    created_at              TEXT    NOT NULL,
    updated_at              TEXT    NOT NULL,
    tags_json               TEXT    NOT NULL DEFAULT '[]',
    metadata_json           TEXT    NOT NULL DEFAULT '{}'
);

CREATE VIRTUAL TABLE IF NOT EXISTS semantic_fts USING fts5(
    content,
    content='semantic_memory',
    content_rowid='rowid'
);

CREATE TRIGGER IF NOT EXISTS semantic_ai AFTER INSERT ON semantic_memory BEGIN
    INSERT INTO semantic_fts(rowid, content) VALUES (new.rowid, new.content);
END;

CREATE TRIGGER IF NOT EXISTS semantic_ad AFTER DELETE ON semantic_memory BEGIN
    INSERT INTO semantic_fts(semantic_fts, rowid, content) VALUES ('delete', old.rowid, old.content);
END;

CREATE TRIGGER IF NOT EXISTS semantic_au AFTER UPDATE ON semantic_memory BEGIN
    INSERT INTO semantic_fts(semantic_fts, rowid, content) VALUES ('delete', old.rowid, old.content);
    INSERT INTO semantic_fts(rowid, content) VALUES (new.rowid, new.content);
END;

-- ============================================================
-- Graph: association edges between memory nodes (Phase 2a)
-- ============================================================
CREATE TABLE IF NOT EXISTS graph_nodes (
    id         TEXT PRIMARY KEY,
    node_type  TEXT NOT NULL CHECK(node_type IN ('memory_concept','entity','session','project')),
    label      TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS graph_edges (
    source_id         TEXT NOT NULL,
    target_id         TEXT NOT NULL,
    weight            REAL NOT NULL CHECK(weight >= -1.0 AND weight <= 1.0),
    relation_type     TEXT NOT NULL CHECK(relation_type IN ('associated_with','causes','contradicts','is_a','part_of')),
    created_at        TEXT NOT NULL,
    last_traversed_at TEXT,
    PRIMARY KEY (source_id, target_id, relation_type)
);

CREATE INDEX IF NOT EXISTS idx_graph_edges_source ON graph_edges(source_id);
CREATE INDEX IF NOT EXISTS idx_graph_edges_target ON graph_edges(target_id);

-- ============================================================
-- Cold Storage: archived/forgotten episodic memories (Phase 2b)
-- ============================================================
CREATE TABLE IF NOT EXISTS cold_storage (
    id              TEXT PRIMARY KEY,
    original_id     TEXT NOT NULL,
    content         TEXT NOT NULL,
    summary         TEXT,
    original_type   TEXT NOT NULL DEFAULT 'episodic',
    compressed_at   TEXT NOT NULL,
    retrieval_count INTEGER NOT NULL DEFAULT 0,
    score_at_archive REAL NOT NULL DEFAULT 0.0
);

-- ============================================================
-- Procedural Memory: task templates / skills (Phase 3)
-- ============================================================
CREATE TABLE IF NOT EXISTS procedural_memory (
    id                  TEXT PRIMARY KEY,
    template_content    TEXT    NOT NULL,
    trigger_condition   TEXT,
    condition_embedding BLOB,
    activation_weight   REAL    NOT NULL DEFAULT 0.5,
    success_count       INTEGER NOT NULL DEFAULT 0,
    failure_count       INTEGER NOT NULL DEFAULT 0,
    last_triggered_at   TEXT,
    last_outcome_at     TEXT,
    lifecycle           TEXT    NOT NULL DEFAULT 'plastic',
    task_type           TEXT,
    source_episode_ids  TEXT    NOT NULL DEFAULT '[]',
    created_at          TEXT    NOT NULL,
    updated_at          TEXT    NOT NULL,
    tags_json           TEXT    NOT NULL DEFAULT '[]',
    metadata_json       TEXT    NOT NULL DEFAULT '{}'
);

CREATE VIRTUAL TABLE IF NOT EXISTS procedural_fts USING fts5(
    template_content, trigger_condition,
    content='procedural_memory',
    content_rowid='rowid'
);

CREATE TRIGGER IF NOT EXISTS procedural_ai AFTER INSERT ON procedural_memory BEGIN
    INSERT INTO procedural_fts(rowid, template_content, trigger_condition)
    VALUES (new.rowid, new.template_content, new.trigger_condition);
END;

CREATE TRIGGER IF NOT EXISTS procedural_ad AFTER DELETE ON procedural_memory BEGIN
    INSERT INTO procedural_fts(procedural_fts, rowid, template_content, trigger_condition)
    VALUES ('delete', old.rowid, old.template_content, old.trigger_condition);
END;

CREATE TRIGGER IF NOT EXISTS procedural_au AFTER UPDATE ON procedural_memory BEGIN
    INSERT INTO procedural_fts(procedural_fts, rowid, template_content, trigger_condition)
    VALUES ('delete', old.rowid, old.template_content, old.trigger_condition);
    INSERT INTO procedural_fts(rowid, template_content, trigger_condition)
    VALUES (new.rowid, new.template_content, new.trigger_condition);
END;

-- ============================================================
-- Creative Pool: generative replay hypotheses (Phase 4)
-- ============================================================
CREATE TABLE IF NOT EXISTS creative_pool (
    id               TEXT PRIMARY KEY,
    hypothesis_text  TEXT    NOT NULL,
    source_seed_ids  TEXT    NOT NULL,
    source_node_ids  TEXT    NOT NULL,
    confidence       REAL    NOT NULL DEFAULT 0.2,
    status           TEXT    NOT NULL DEFAULT 'pending',
    push_count       INTEGER NOT NULL DEFAULT 0,
    ignore_count     INTEGER NOT NULL DEFAULT 0,
    topic_tags       TEXT    NOT NULL DEFAULT '[]',
    last_pushed_at   TEXT,
    created_at       TEXT    NOT NULL,
    seed_content     TEXT    NOT NULL DEFAULT '',
    concept_pair     TEXT    NOT NULL DEFAULT '',
    stability_count  INTEGER NOT NULL DEFAULT 0,
    metadata_json    TEXT    NOT NULL DEFAULT '{}'
);

-- ============================================================
-- Environment Snapshots: auto-falsifiable tracking (Phase 4)
-- ============================================================
CREATE TABLE IF NOT EXISTS environment_snapshots (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_json    TEXT    NOT NULL,
    captured_at      TEXT    NOT NULL
);

-- ============================================================
-- Preference Candidates: user preference extraction pipeline
-- ============================================================
CREATE TABLE IF NOT EXISTS preference_candidates (
    id                    TEXT PRIMARY KEY,
    preference_text       TEXT    NOT NULL,
    preference_type       TEXT    NOT NULL DEFAULT 'general',
    confidence            REAL    NOT NULL DEFAULT 0.3,
    source_episode_ids    TEXT    NOT NULL DEFAULT '[]',
    signal_strength       REAL    NOT NULL DEFAULT 0.0,
    status                TEXT    NOT NULL DEFAULT 'pending',
    conflict_with         TEXT,
    falsifiable_condition TEXT,
    trigger_context       TEXT,
    lifecycle             TEXT    NOT NULL DEFAULT 'plastic',
    push_count            INTEGER NOT NULL DEFAULT 0,
    ignore_count          INTEGER NOT NULL DEFAULT 0,
    last_pushed_at        TEXT,
    created_at            TEXT    NOT NULL,
    updated_at            TEXT    NOT NULL,
    metadata_json         TEXT    NOT NULL DEFAULT '{}'
);

CREATE VIRTUAL TABLE IF NOT EXISTS preference_fts USING fts5(
    preference_text, trigger_context,
    content='preference_candidates',
    content_rowid='rowid'
);

CREATE TRIGGER IF NOT EXISTS preference_ai AFTER INSERT ON preference_candidates BEGIN
    INSERT INTO preference_fts(rowid, preference_text, trigger_context)
    VALUES (new.rowid, new.preference_text, new.trigger_context);
END;

CREATE TRIGGER IF NOT EXISTS preference_ad AFTER DELETE ON preference_candidates BEGIN
    INSERT INTO preference_fts(preference_fts, rowid, preference_text, trigger_context)
    VALUES ('delete', old.rowid, old.preference_text, old.trigger_context);
END;

CREATE TRIGGER IF NOT EXISTS preference_au AFTER UPDATE ON preference_candidates BEGIN
    INSERT INTO preference_fts(preference_fts, rowid, preference_text, trigger_context)
    VALUES ('delete', old.rowid, old.preference_text, old.trigger_context);
    INSERT INTO preference_fts(rowid, preference_text, trigger_context)
    VALUES (new.rowid, new.preference_text, new.trigger_context);
END;
"""


class SqliteStore:
    """SQLite storage backend with WAL mode and sqlite-vec support.

    Thread-safe via per-thread connection cache. All memory layers share
    a single SqliteStore instance pointing to one database file.

    Usage:
        store = SqliteStore("cell_mem.db")
        store.initialize_schema()
        conn = store.get_connection()
        conn.execute("INSERT INTO ...")
    """

    def __init__(self, db_path: str = "cell_mem.db"):
        if db_path == ":memory:":
            self._db_path = ":memory:"
        else:
            self._db_path = str(Path(db_path).resolve())
        self._vec_available: Optional[bool] = None
        self._local = threading.local()
        logger.info("SqliteStore opened: %s", self._db_path)

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def get_connection(self) -> sqlite3.Connection:
        """Return thread-local connection. Creates + configures on first access."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(self._db_path, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute("PRAGMA foreign_keys=ON;")
            conn.execute("PRAGMA busy_timeout=5000;")
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
            logger.debug("Created thread-local connection for %s", self._db_path)
        return self._local.conn

    @property
    def is_wal(self) -> bool:
        conn = self.get_connection()
        row = conn.execute("PRAGMA journal_mode;").fetchone()
        return row[0].upper() == "WAL"

    @property
    def vec_available(self) -> bool:
        """Whether sqlite-vec extension was loaded successfully."""
        if self._vec_available is None:
            self._vec_available = self._load_vec_extension()
        return self._vec_available

    # ------------------------------------------------------------------
    # Schema management
    # ------------------------------------------------------------------

    def initialize_schema(self) -> None:
        """Create all tables, indexes, and triggers if they don't exist.
        Idempotent — safe to call on every startup.
        """
        conn = self.get_connection()
        conn.executescript(_SCHEMA_DDL)
        conn.commit()

        # --- Schema migrations for existing databases ---
        # v6→v7: working_memory missing tags_json column
        try:
            conn.execute(
                "ALTER TABLE working_memory ADD COLUMN tags_json TEXT NOT NULL DEFAULT '[]'"
            )
        except Exception:
            pass  # Column already exists or table doesn't exist yet

        # Record schema version
        self._set_meta("schema_version", str(SCHEMA_VERSION).encode("utf-8"))
        logger.info("Schema initialized (version %d)", SCHEMA_VERSION)

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def execute(
        self, sql: str, params: tuple = ()
    ) -> sqlite3.Cursor:
        """Execute a parameterized SQL statement. Returns the cursor."""
        conn = self.get_connection()
        return conn.execute(sql, params)

    def execute_many(
        self, sql: str, params_list: List[tuple]
    ) -> sqlite3.Cursor:
        """Execute a parameterized SQL with many parameter sets."""
        conn = self.get_connection()
        return conn.executemany(sql, params_list)

    def commit(self) -> None:
        self.get_connection().commit()

    def fetchone(self, sql: str, params: tuple = ()) -> Optional[sqlite3.Row]:
        return self.execute(sql, params).fetchone()

    def fetchall(self, sql: str, params: tuple = ()) -> List[sqlite3.Row]:
        return self.execute(sql, params).fetchall()

    def row_count(self, table_name: str) -> int:
        """Return the number of rows in a table."""
        row = self.fetchone(f"SELECT COUNT(*) AS cnt FROM {table_name}")
        return row["cnt"] if row else 0

    # ------------------------------------------------------------------
    # Meta table helpers
    # ------------------------------------------------------------------

    def set_meta(self, key: str, value: bytes) -> None:
        self.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            (key, value),
        )
        self.commit()

    def get_meta(self, key: str) -> Optional[bytes]:
        row = self.fetchone("SELECT value FROM meta WHERE key = ?", (key,))
        return row[0] if row else None

    # ------------------------------------------------------------------
    # sqlite-vec extension
    # ------------------------------------------------------------------

    def _load_vec_extension(self) -> bool:
        """Load sqlite-vec extension. Returns True on success, False on failure.

        On Windows, sqlite-vec may fail to load (Errno 22 / EINVAL) when the
        sqlite3 library bundled with the Python interpreter lacks extension
        support or was compiled with different flags. Common fixes:
        1. Use official Python from python.org (not MS Store or embedded).
        2. Ensure sqlite3.dll in Python directory is from the same build.
        3. Verify SQLite version >= 3.41: python -c "import sqlite3; print(sqlite3.sqlite_version)"
        4. On failure, the system falls back to ChromaDB for vector search.
        """
        import sqlite_vec

        conn = self.get_connection()
        try:
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
            logger.info("sqlite-vec extension loaded successfully")
            return True
        except (OSError, AttributeError) as exc:
            err_msg = str(exc)
            if "Errno 22" in err_msg or "error 22" in err_msg.lower():
                logger.warning(
                    "sqlite-vec failed to load (Errno 22/EINVAL — likely sqlite3 "
                    "library incompatibility). Vector search will use ChromaDB backend. "
                    "To fix: reinstall Python from python.org or "
                    "pip install pysqlite3-binary && pip install sqlite-vec --force-reinstall. "
                    "Error details: %s", err_msg
                )
            else:
                logger.warning("Failed to load sqlite-vec extension: %s", exc)
            return False
        except Exception as exc:
            logger.warning("Failed to load sqlite-vec extension: %s", exc)
            return False

    # Legacy alias for compatibility
    _set_meta = set_meta
    _get_meta = get_meta

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the thread-local connection if open."""
        if hasattr(self._local, "conn") and self._local.conn is not None:
            self._local.conn.close()
            self._local.conn = None
            logger.debug("Closed thread-local connection")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
