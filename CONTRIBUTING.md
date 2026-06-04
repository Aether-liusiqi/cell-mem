# Contributing to Cell-mem

Thanks for your interest in contributing! Cell-mem is a brain-inspired memory system
for AI agents. Whether you're fixing a bug, adding a feature, or improving documentation,
this guide will help you get started.

## Getting Started

### Prerequisites
- Python 3.11+
- Git

### Setup

```bash
# Clone the repository
git clone https://github.com/Aether-liusiqi/cell-mem.git
cd cell-mem

# Install with dev dependencies
pip install -e ".[dev]"

# Verify everything works
pytest tests/ -v
```

## Development Workflow

### Project Structure

```
src/cell_mem/     — Production code (what ships)
tests/            — Test suite
config/           — Example configuration files
```

### Before You Start

1. **Open an issue first** for significant changes — discuss the approach before writing code.
2. **Check existing tests** — your feature or fix should have corresponding test coverage.
3. **Read the architecture** — the [README](README.md) explains the memory layers and data flow.

### Code Style

- Follow [PEP 8](https://peps.python.org/pep-0008/) with a 100-character line limit
- Use type hints for all function signatures (`from __future__ import annotations`)
- Match the existing code style — docstrings, naming, import ordering
- No new pip dependencies without prior discussion (see Rule 7 in CLAUDE.md)

### Testing

```bash
# Run the full test suite
pytest tests/ -v

# Run a specific test file
pytest tests/test_procedural_memory.py -v

# Run with coverage
pytest tests/ -v --cov=cell_mem --cov-report=term-missing
```

- All new features must include tests
- Bug fixes should include a regression test
- Target: tests pass on Python 3.11+

### Commit Messages

- Use present tense: "Add feature" not "Added feature"
- Keep the first line under 72 characters
- Reference issue numbers: "Fix #42: Handle empty FTS5 query results"

## Design Principles

When contributing, keep these principles in mind:

1. **Zero new pip dependencies.** Core operations use stdlib only. If you need a
   new package, it must be justified and discussed in an issue first.
2. **Brain-inspired, not brain-simulated.** Algorithms are inspired by neuroscience
   but optimized for practical agent memory.
3. **Graceful degradation.** Optional features (LLM, HTTP, ChromaDB) degrade cleanly
   when not configured.
4. **SQL-first.** All persistent state lives in SQLite. No external databases.
5. **Backward compatibility.** Schema migrations must be additive. Existing MCP
   tool signatures are stable.

## Pull Request Process

1. Fork the repository and create a feature branch.
2. Add tests that cover your changes.
3. Ensure the full test suite passes (`pytest tests/ -v`).
4. Update documentation if your changes affect the public API.
5. Open a PR against the `main` branch with a clear description.

## Security

If you discover a security vulnerability, please **do not** open a public issue.
Email the maintainer directly. See [SECURITY.md](SECURITY.md) for the full policy.

## License

By contributing, you agree that your contributions will be licensed under the
MIT License that covers this project.
