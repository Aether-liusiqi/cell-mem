"""Auto-register cell-mem session hooks in Codex CLI and Claude Code.

Provides:
- ``install()`` — copy hook script + register in platform config
- ``uninstall()`` — remove hook script + config entries
- ``detect_platforms()`` — discover which platforms are installed

All operations are **idempotent** — they merge with existing config,
never overwrite unrelated hook entries.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_HOOK_SCRIPT_NAME = "cell_mem_session_hook.py"

# Path to the template hook script (in the installed package)
_HOOK_TEMPLATE = Path(__file__).resolve().parent / "session_hook.py"


# ---------------------------------------------------------------------------
# Platform detection
# ---------------------------------------------------------------------------


def detect_platforms() -> List[str]:
    """Detect which agent platforms are installed on this machine.

    Returns:
        List of platform names: ``["codex"]``, ``["claude"]``, ``["codex", "claude"]``,
        or ``["codex"]`` as fallback when neither is detectable.
    """
    platforms: List[str] = []
    home = Path.home()

    if (home / ".codex" / "config.toml").exists():
        platforms.append("codex")
    if (home / ".claude" / "settings.json").exists():
        platforms.append("claude")

    if not platforms:
        # Default: assume Codex (it's the primary target)
        platforms.append("codex")
        logger.info("No platform detected, defaulting to codex")

    return platforms


def _codex_dir() -> Path:
    """Codex config directory."""
    codex_home = os.environ.get("CODEX_HOME", "")
    if codex_home:
        return Path(codex_home)
    return Path.home() / ".codex"


def _claude_dir() -> Path:
    """Claude config directory."""
    return Path.home() / ".claude"


# ---------------------------------------------------------------------------
# Hook script installation
# ---------------------------------------------------------------------------


def _install_hook_script(target_dir: Path, ingest_port: int) -> Path:
    """Copy the hook script template to *target_dir*, replacing the port placeholder.

    Returns the path to the installed script.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    dest = target_dir / _HOOK_SCRIPT_NAME

    # Read template (it's the same script — we just inject the port via env,
    # so a direct copy is sufficient)
    src = _HOOK_TEMPLATE
    if not src.exists():
        # Fallback: when running from source, the template is relative to this file
        src = Path(__file__).resolve().parent / "session_hook.py"

    shutil.copy2(str(src), str(dest))
    logger.info("Hook script installed: %s", dest)
    return dest


def _compute_hash(script_path: Path) -> str:
    """Compute SHA-256 hash of the hook script (used by Codex for trust verification)."""
    sha = hashlib.sha256()
    with open(script_path, "rb") as f:
        while chunk := f.read(8192):
            sha.update(chunk)
    return f"sha256:{sha.hexdigest()}"


# ---------------------------------------------------------------------------
# Codex registration
# ---------------------------------------------------------------------------


def _register_codex(ingest_port: int) -> bool:
    """Register hooks in Codex CLI config.

    Writes:
    - ``~/.codex/hooks/cell_mem_session_hook.py``  (hook script)
    - ``~/.codex/hooks.json``                       (hook entries, merged)
    - ``~/.codex/config.toml``                      (trust hash + hooks=true, merged)
    """
    codex = _codex_dir()
    hooks_dir = codex / "hooks"

    # 1. Install hook script
    script_path = _install_hook_script(hooks_dir, ingest_port)
    trusted_hash = _compute_hash(script_path)

    # 2. Update hooks.json
    hooks_json_path = codex / "hooks.json"
    hooks_json = _read_json(hooks_json_path)

    # Cell-mem hook entries (supported events)
    cell_mem_entries = [
        {
            "matcher": "SessionStart",
            "command": ["python", str(script_path)],
            "env": {"CELL_MEM_INGEST_PORT": str(ingest_port)},
        },
        {
            "matcher": "PostToolUse",
            "command": ["python", str(script_path)],
            "env": {"CELL_MEM_INGEST_PORT": str(ingest_port)},
        },
    ]

    # Merge: Codex uses "post_tool_use" as the event key
    existing = hooks_json.get("post_tool_use", [])

    # Remove any previous cell_mem entries (by command path match)
    existing = [
        e
        for e in existing
        if not any(
            "cell_mem_session_hook" in str(arg)
            for arg in (e.get("command", []))
        )
    ]

    hooks_json["post_tool_use"] = existing + cell_mem_entries
    _write_json(hooks_json_path, hooks_json)
    logger.info("Codex hooks.json updated (%d hook entries)", len(hooks_json["post_tool_use"]))

    # 3. Update config.toml
    config_toml_path = codex / "config.toml"
    _update_codex_config(config_toml_path, script_path, trusted_hash)

    logger.info("Codex hooks registered successfully")
    return True


def _update_codex_config(config_path: Path, script_path: Path, trusted_hash: str) -> None:
    """Ensure hooks=true and set trusted_hash in Codex config.toml."""
    if not config_path.exists():
        logger.warning("Codex config.toml not found at %s", config_path)
        return

    lines = _read_lines(config_path)
    new_lines: List[str] = []
    hooks_enabled = False
    state_key = f"{config_path.parent / 'hooks.json'}:post_tool_use:0:0"
    hash_set = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("hooks ="):
            new_lines.append("hooks = true\n")
            hooks_enabled = True
        elif f"trusted_hash = " in stripped and state_key in stripped:
            # Will be rewritten below
            continue
        else:
            new_lines.append(line)

    # Ensure hooks=true
    if not hooks_enabled:
        new_lines.append("\nhooks = true\n")

    # Add trusted_hash under [hooks.state]
    if "[hooks.state]" not in "".join(new_lines):
        new_lines.append("\n[hooks.state]\n")
    new_lines.append(f'[hooks.state.\'{str(script_path)}\']\n')
    new_lines.append(f"trusted_hash = \"{trusted_hash}\"\n")

    _write_lines(config_path, new_lines)
    logger.info("Codex config.toml updated (trusted_hash=%s)", trusted_hash[:20] + "...")


# ---------------------------------------------------------------------------
# Claude Code registration
# ---------------------------------------------------------------------------


def _register_claude(ingest_port: int) -> bool:
    """Register hooks in Claude Code settings.

    Writes:
    - ``~/.claude/hooks/cell_mem_session_hook.py``  (hook script)
    - ``~/.claude/settings.json``                    (hooks section, merged)
    """
    claude = _claude_dir()
    hooks_dir = claude / "hooks"

    # 1. Install hook script
    script_path = _install_hook_script(hooks_dir, ingest_port)

    # 2. Update settings.json
    settings_path = claude / "settings.json"
    settings = _read_json(settings_path)

    if "hooks" not in settings:
        settings["hooks"] = {}

    # Claude Code hook structure for SessionStart and PostToolUse
    cell_mem_command = f"python {script_path}"

    for event in ("SessionStart", "PostToolUse"):
        if event not in settings["hooks"]:
            settings["hooks"][event] = []

        # Remove any existing cell_mem entries for this event
        settings["hooks"][event] = [
            h
            for h in settings["hooks"][event]
            if "cell_mem_session_hook" not in str(h)
        ]

        settings["hooks"][event].append({
            "matcher": "",
            "hooks": [{
                "type": "command",
                "command": cell_mem_command,
            }],
        })

    _write_json(settings_path, settings)
    logger.info("Claude Code settings.json updated")

    return True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def install(
    platform: str = "auto",
    ingest_port: int = 8766,
) -> dict:
    """Install cell-mem session recording hooks.

    Args:
        platform: ``"auto"`` (detect), ``"codex"``, ``"claude"``, or ``"both"``.
        ingest_port: Port the cell-mem ingest server is listening on.

    Returns:
        Dict with ``{"installed": [...], "skipped": [...], "errors": [...]}``.
    """
    result = {"installed": [], "skipped": [], "errors": []}

    # Resolve target platforms
    if platform == "auto":
        targets = detect_platforms()
    elif platform == "both":
        targets = ["codex", "claude"]
    else:
        targets = [platform]

    logger.info("Installing hooks for platforms: %s (ingest_port=%d)", targets, ingest_port)

    for p in targets:
        try:
            if p == "codex":
                if _register_codex(ingest_port):
                    result["installed"].append("codex")
            elif p == "claude":
                if _register_claude(ingest_port):
                    result["installed"].append("claude")
            else:
                result["errors"].append(f"unknown platform: {p}")
        except Exception as exc:
            logger.error("Failed to register %s hooks: %s", p, exc)
            result["errors"].append(f"{p}: {exc}")

    return result


def uninstall(platform: str = "auto") -> dict:
    """Remove cell-mem hooks from platform configs.

    Args:
        platform: ``"auto"`` (detect), ``"codex"``, ``"claude"``, or ``"both"``.

    Returns:
        Dict with ``{"removed": [...], "errors": [...]}``.
    """
    result = {"removed": [], "errors": []}

    if platform == "auto":
        targets = detect_platforms()
    elif platform == "both":
        targets = ["codex", "claude"]
    else:
        targets = [platform]

    for p in targets:
        try:
            if p == "codex":
                _unregister_codex()
                result["removed"].append("codex")
            elif p == "claude":
                _unregister_claude()
                result["removed"].append("claude")
            else:
                result["errors"].append(f"unknown platform: {p}")
        except Exception as exc:
            logger.error("Failed to unregister %s hooks: %s", p, exc)
            result["errors"].append(f"{p}: {exc}")

    return result


def _unregister_codex() -> None:
    """Remove cell_mem entries from Codex config."""
    codex = _codex_dir()

    # 1. Remove from hooks.json
    hooks_json_path = codex / "hooks.json"
    if hooks_json_path.exists():
        hooks = _read_json(hooks_json_path)
        hooks["post_tool_use"] = [
            e
            for e in hooks.get("post_tool_use", [])
            if not any(
                "cell_mem_session_hook" in str(arg)
                for arg in (e.get("command", []))
            )
        ]
        _write_json(hooks_json_path, hooks)
        logger.info("Removed cell_mem entries from Codex hooks.json")

    # 2. Remove from config.toml [hooks.state]
    config_path = codex / "config.toml"
    if config_path.exists():
        lines = _read_lines(config_path)
        new_lines = []
        skip_next = False
        for line in lines:
            stripped = line.strip()
            if "cell_mem_session_hook" in stripped:
                skip_next = True
                continue
            if skip_next and stripped.startswith("trusted_hash"):
                skip_next = False
                continue
            skip_next = False
            new_lines.append(line)
        _write_lines(config_path, new_lines)
        logger.info("Removed cell_mem entries from Codex config.toml")

    # 3. Remove hook script
    script_path = codex / "hooks" / _HOOK_SCRIPT_NAME
    _safe_remove(script_path)


def _unregister_claude() -> None:
    """Remove cell_mem entries from Claude Code config."""
    claude = _claude_dir()

    # 1. Remove from settings.json
    settings_path = claude / "settings.json"
    if settings_path.exists():
        settings = _read_json(settings_path)
        hooks = settings.get("hooks", {})
        for event in ("SessionStart", "PostToolUse"):
            if event in hooks:
                hooks[event] = [
                    h
                    for h in hooks[event]
                    if "cell_mem_session_hook" not in str(h)
                ]
        settings["hooks"] = hooks
        _write_json(settings_path, settings)
        logger.info("Removed cell_mem entries from Claude settings.json")

    # 2. Remove hook script
    script_path = claude / "hooks" / _HOOK_SCRIPT_NAME
    _safe_remove(script_path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> dict:
    """Read JSON file, return {} if missing or invalid."""
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to read %s: %s", path, exc)
        return {}


def _write_json(path: Path, data: dict) -> None:
    """Write JSON file with pretty formatting."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _read_lines(path: Path) -> List[str]:
    """Read file as list of lines. Returns [] if missing."""
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return f.readlines()


def _write_lines(path: Path, lines: List[str]) -> None:
    """Write lines to file."""
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def _safe_remove(path: Path) -> None:
    """Remove a file, silently succeed if missing."""
    try:
        path.unlink(missing_ok=True)
        logger.info("Removed: %s", path)
    except OSError as exc:
        logger.warning("Failed to remove %s: %s", path, exc)
