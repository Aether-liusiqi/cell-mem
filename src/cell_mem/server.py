"""Cell-mem MCP Server entry point.

Dual transport: stdio (default, for agent launches) and HTTP (--http flag, for hook scripts).
Both point to the same MemorySystem instance.

Usage:
    # stdio mode (agent launches MCP Server as subprocess)
    python -m cell_mem.server

    # HTTP mode (daemon for hook scripts)
    python -m cell_mem.server --http

    # With custom DB path
    python -m cell_mem.server --db /path/to/cell_mem.db

    # Preload embedding model (reduces first-request latency)
    python -m cell_mem.server --preload

    # Register session recording hooks (Codex CLI / Claude Code)
    python -m cell_mem.server --hooks install

    # Remove hooks
    python -m cell_mem.server --hooks clean
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from cell_mem.memory_system import DEFAULT_DB_PATH, MemorySystem

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,  # stderr to avoid contaminating stdio transport
)
logger = logging.getLogger("cell_mem")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cell-mem: Brain-inspired memory system MCP Server"
    )
    parser.add_argument(
        "--http",
        action="store_true",
        help="Run with HTTP transport instead of stdio",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="HTTP host (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="HTTP port (default: 8765)",
    )
    parser.add_argument(
        "--db",
        default=os.environ.get("CELL_MEM_DB", DEFAULT_DB_PATH),
        help=f"SQLite database path (default: $CELL_MEM_DB or {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--seed-config",
        default=None,
        help="Path to seed knowledge JSON config file",
    )
    parser.add_argument(
        "--preload",
        action="store_true",
        help="Preload the embedding model at startup (avoids first-request delay)",
    )
    parser.add_argument(
        "--vector-backend",
        default="sqlite-vec",
        choices=["sqlite-vec", "chromadb"],
        help="Vector storage backend (default: sqlite-vec)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Optional shared secret for HTTP mode authentication. "
             "When set, clients must pass the key via the Authorization header. "
             "Strongly recommended for non-localhost deployments.",
    )
    parser.add_argument(
        "--llm-backend",
        default="openai",
        help="LLM backend (default: openai)",
    )
    parser.add_argument(
        "--llm-api-key",
        default=None,
        help="LLM API key (falls back to CELL_MEM_LLM_API_KEY env)",
    )
    parser.add_argument(
        "--llm-model",
        default="gpt-4o-mini",
        help="LLM model name (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--llm-base-url",
        default="https://api.openai.com/v1",
        help="LLM API base URL (default: https://api.openai.com/v1)",
    )
    parser.add_argument(
        "--hooks",
        choices=["install", "clean"],
        default=None,
        help="Auto-register or remove session recording hooks "
             "(installs hook scripts into Codex CLI / Claude Code config)",
    )
    parser.add_argument(
        "--hooks-platform",
        choices=["auto", "codex", "claude", "both"],
        default="auto",
        help="Target platform for --hooks (default: auto-detect)",
    )
    parser.add_argument(
        "--ingest-port",
        type=int,
        default=8766,
        help="Internal ingest endpoint port for hook scripts (default: 8766)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------


def register_all_tools(mcp, memory_system: MemorySystem) -> None:
    """Register all MCP tools."""
    from cell_mem.tools.save import register_save_tool
    from cell_mem.tools.recall import register_recall_tool
    from cell_mem.tools.status import register_status_tool
    from cell_mem.tools.stubs import register_associate_tool, register_phase2b_tools, register_phase3_tools, register_phase4_tools
    from cell_mem.tools.preference import register_preference_tools

    register_save_tool(mcp, memory_system)
    register_recall_tool(mcp, memory_system)
    register_status_tool(mcp, memory_system)
    register_associate_tool(mcp, memory_system)
    register_phase2b_tools(mcp, memory_system)  # memory_forget + memory_consolidate
    register_phase3_tools(mcp, memory_system)   # memory_verify + memory_reflect
    register_phase4_tools(mcp, memory_system)   # memory_replay + creative pool + environment
    register_preference_tools(mcp, memory_system)  # 5 preference pipeline tools

    logger.info("MCP tools: memory_save, memory_recall, memory_status, "
                "memory_associate, memory_forget, memory_consolidate, "
                "memory_verify, memory_reflect, "
                "memory_replay, memory_hypothesis_feedback, "
                "memory_creative_pool, memory_check_environment, "
                "memory_extract_preferences, memory_get_preferences, "
                "memory_check_preference_conflicts, memory_inject_preference, "
                "memory_record_preference_feedback")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    # Resolve DB path (relative to CWD)
    db_path = str(Path(args.db).resolve())
    logger.info("DB path: %s", db_path)

    # Warn if HTTP mode without authentication
    if args.http and not args.api_key:
        logger.warning(
            "⚠ HTTP mode enabled without --api-key. "
            "Any process on this machine can read/write ALL memories. "
            "Set --api-key to enable shared-secret authentication."
        )

    # Initialize MemorySystem
    ms = MemorySystem(
        db_path=db_path,
        seed_config_path=args.seed_config,
        vector_backend=args.vector_backend,
        api_key=args.api_key,
        llm_backend=args.llm_backend,
        llm_api_key=args.llm_api_key,
        llm_model=args.llm_model,
        llm_base_url=args.llm_base_url,
    )

    # Preload embedding model only when --preload flag is explicitly passed.
    # In stdio mode without --preload, model loads lazily on first tool call.
    # This prevents MCP handshake timeout from 27s cold-start model loading.
    if args.preload:
        logger.info("Preloading embedding model...")
        ms.embed_model.ensure_loaded()
        logger.info("Embedding model ready (dim=%d)", ms.embed_model.DIM)

    # ------------------------------------------------------------------
    # Hook registration (--hooks install / --hooks clean)
    # ------------------------------------------------------------------
    from cell_mem.hooks.registrar import install as hooks_install, uninstall as hooks_uninstall
    from cell_mem.hooks.ingest import IngestServer

    if args.hooks == "install":
        result = hooks_install(platform=args.hooks_platform, ingest_port=args.ingest_port)
        logger.info("Hook install result: %s", result)
        # Start internal ingest endpoint so hook scripts can reach us
        IngestServer(ms, port=args.ingest_port).start()
        logger.info(
            "Tip: For best results, run cell-mem as a daemon before opening "
            "your agent session:  cell-mem --http --preload &"
        )
    elif args.hooks == "clean":
        result = hooks_uninstall(platform=args.hooks_platform)
        logger.info("Hook clean result: %s", result)
        logger.info("Hooks cleaned. Use --hooks install to re-register.")
        ms.shutdown()
        return

    # ------------------------------------------------------------------
    # FastMCP server
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP(
        name="Cell-mem",
        instructions=(
            "Brain-inspired memory system for AI Agents. "
            "Four memory layers (working, episodic, semantic, procedural) "
            "with consolidation processor, LLM emotional scoring, "
            "falsifiable conditions, self-reflection, generative replay, "
            "and creative pool management. "
            "Cell-mem provides 17 MCP tools: save, recall, status, associate, "
            "forget, consolidate, verify, reflect, replay, hypothesis_feedback, "
            "creative_pool, check_environment, extract_preferences, get_preferences, "
            "check_preference_conflicts, inject_preference, record_preference_feedback."
        ),
        host=args.host,
        port=args.port,
    )

    # Register tools
    register_all_tools(mcp, ms)

    # EmbeddingWorker auto-started during MemorySystem.__init__()

    # Run with chosen transport
    transport = "streamable-http" if args.http else "stdio"
    logger.info("Starting Cell-mem MCP Server (transport=%s)", transport)

    try:
        mcp.run(transport=transport)
    except KeyboardInterrupt:
        logger.info("Received interrupt, shutting down...")
    finally:
        ms.shutdown()
        logger.info("Cell-mem Server stopped")


if __name__ == "__main__":
    main()
