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
"""

from __future__ import annotations

import argparse
import logging
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
        default=DEFAULT_DB_PATH,
        help=f"SQLite database path (default: {DEFAULT_DB_PATH})",
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
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------


def register_all_tools(mcp, memory_system: MemorySystem) -> None:
    """Register all MCP tools (Phase 1 + Phase 2a + Phase 2b)."""
    from cell_mem.tools.save import register_save_tool
    from cell_mem.tools.recall import register_recall_tool
    from cell_mem.tools.status import register_status_tool
    from cell_mem.tools.stubs import register_associate_tool, register_phase2b_tools, register_phase3_tools, register_phase4_tools

    register_save_tool(mcp, memory_system)
    register_recall_tool(mcp, memory_system)
    register_status_tool(mcp, memory_system)
    register_associate_tool(mcp, memory_system)
    register_phase2b_tools(mcp, memory_system)  # memory_forget + memory_consolidate
    register_phase3_tools(mcp, memory_system)   # memory_verify + memory_reflect
    register_phase4_tools(mcp, memory_system)   # memory_replay + creative pool + environment

    logger.info("MCP tools: memory_save, memory_recall, memory_status, "
                "memory_associate, memory_forget, memory_consolidate, "
                "memory_verify, memory_reflect, "
                "memory_replay, memory_hypothesis_feedback, "
                "memory_creative_pool, memory_check_environment")


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
    )

    # Preload embedding model: always for HTTP, or if --preload flag set.
    # For stdio mode without --preload, the first request pays the 30s cost.
    if args.preload or not args.http:
        logger.info("Preloading embedding model...")
        ms.embed_model.ensure_loaded()
        logger.info("Embedding model ready (dim=%d, %.1fs startup)",
                     ms.embed_model.DIM, 0.0)

    # Create FastMCP server
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP(
        name="Cell-mem",
        instructions=(
            "Brain-inspired memory system for AI Agents. "
            "Four memory layers (working, episodic, semantic, procedural) "
            "with consolidation processor, LLM emotional scoring, "
            "falsifiable conditions, self-reflection, generative replay, "
            "and creative pool management. "
            "Phase 4 provides 12 MCP tools: save, recall, status, associate, "
            "forget, consolidate, verify, reflect, replay, hypothesis_feedback, "
            "creative_pool, check_environment."
        ),
        host=args.host,
        port=args.port,
    )

    # Register tools
    register_all_tools(mcp, ms)

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
