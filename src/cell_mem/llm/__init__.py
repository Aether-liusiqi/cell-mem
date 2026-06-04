"""Cell-mem LLM abstraction layer.

Provides:
- LLMClient ABC: abstract interface for LLM backends
- OpenAABackend: OpenAI-compatible chat completions
- ClaudeBackend: Anthropic Claude Messages API
- RateLimiter: daily call cap with persistent state
- LLMError: unified error for all LLM failures
"""

from cell_mem.llm.client import LLMClient, LLMError, RateLimiter
from cell_mem.llm.backends import OpenAIBackend, ClaudeBackend

__all__ = [
    "LLMClient",
    "LLMError",
    "RateLimiter",
    "OpenAIBackend",
    "ClaudeBackend",
]
