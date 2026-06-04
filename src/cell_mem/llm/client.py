"""LLMClient abstraction and RateLimiter.

Phase 3 introduces optional LLM dependency. All LLM calls go through
this abstraction layer, with daily rate limiting and persistent state.
When no LLM is configured, all LLM-dependent features degrade gracefully
to their Phase 2 deterministic equivalents.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Raised when an LLM API call fails after exhausting retries."""


class LLMClient(ABC):
    """Abstract LLM client — all backends implement this.

    Usage:
        client = OpenAIBackend(api_key="...", model="gpt-4o-mini")
        result = client.generate("What is the valence of this text?")
        # result = {"valence": 0.8, "confidence": 0.9}
    """

    @abstractmethod
    def generate(self, prompt: str, schema: dict | None = None) -> dict:
        """Send a prompt and return a parsed JSON dict.

        Args:
            prompt: The prompt text.
            schema: Optional JSON schema description. When provided, the prompt
                    is augmented to instruct the LLM to return JSON matching
                    the schema. No client-side validation is performed.

        Returns:
            Parsed JSON dict from the LLM response.

        Raises:
            LLMError: On timeout, network failure, or exhausted retries.
        """
        ...

    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        """Generate an embedding vector for the given text.

        Not all backends support embeddings (Claude raises NotImplementedError).
        Cell-mem uses local sentence-transformers for memory embeddings;
        this method is for LLM-based embedding scenarios.
        """
        ...


class RateLimiter:
    """Configurable daily call cap with persistent state.

    State is persisted to the meta table under keys:
    - llm_daily_count: number of calls made today
    - llm_daily_date: ISO date string, resets count when date changes

    Thread-safe via the store's connection locking.
    """

    def __init__(
        self,
        daily_limit: int = 100,
        store: Optional["SqliteStore"] = None,  # noqa: F821
    ):
        self._daily_limit = daily_limit
        self._store = store
        self._count = 0
        self._date: Optional[str] = None
        self._restore()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def allow(self) -> bool:
        """Check if a call is allowed under the daily limit."""
        self._check_date_rollover()
        return self._count < self._daily_limit

    def record(self) -> None:
        """Record a successful LLM call."""
        self._check_date_rollover()
        self._count += 1
        self._persist()

    @property
    def remaining(self) -> int:
        """Number of calls remaining today."""
        self._check_date_rollover()
        return max(0, self._daily_limit - self._count)

    @property
    def count(self) -> int:
        self._check_date_rollover()
        return self._count

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _check_date_rollover(self) -> None:
        """Reset count if the date has changed."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self._date != today:
            self._count = 0
            self._date = today

    def _persist(self) -> None:
        """Write current state to meta table."""
        if self._store is None:
            return
        try:
            self._store.set_meta("llm_daily_count", str(self._count).encode("utf-8"))
            self._store.set_meta("llm_daily_date", (self._date or "").encode("utf-8"))
        except Exception:
            logger.debug("Failed to persist rate limiter state", exc_info=True)

    def _restore(self) -> None:
        """Restore state from meta table (survives server restart)."""
        if self._store is None:
            return
        try:
            raw_date = self._store.get_meta("llm_daily_date")
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            if raw_date:
                stored_date = raw_date.decode("utf-8")
                if stored_date == today:
                    raw_count = self._store.get_meta("llm_daily_count")
                    if raw_count:
                        self._count = int(raw_count.decode("utf-8"))
                        self._date = stored_date
                else:
                    # New day — reset
                    self._count = 0
                    self._date = today
            else:
                self._date = today
        except Exception:
            logger.debug("Failed to restore rate limiter state", exc_info=True)
            self._date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
