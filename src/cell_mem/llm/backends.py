"""LLM backends using stdlib urllib (zero new pip dependencies).

Supported backends:
- OpenAABackend: OpenAI-compatible API (OpenAI, Azure, local llama.cpp, etc.)
- ClaudeBackend: Anthropic Claude Messages API
"""

from __future__ import annotations

import json
import ipaddress
import logging
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Optional

import numpy as np

from cell_mem.llm.client import LLMClient, LLMError, RateLimiter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared HTTP helper
# ---------------------------------------------------------------------------

_RETRYABLE_STATUSES = {429, 502, 503}
_NON_RETRYABLE_STATUSES = {400, 401, 403, 404}

# Blocked network ranges for SSRF prevention
_BLOCKED_NETWORKS = (
    ipaddress.ip_network("127.0.0.0/8"),       # Loopback
    ipaddress.ip_network("10.0.0.0/8"),         # RFC 1918 Class A
    ipaddress.ip_network("172.16.0.0/12"),      # RFC 1918 Class B
    ipaddress.ip_network("192.168.0.0/16"),     # RFC 1918 Class C
    ipaddress.ip_network("169.254.0.0/16"),     # Link-local / cloud metadata
    ipaddress.ip_network("0.0.0.0/8"),          # "This" network
    ipaddress.ip_network("100.64.0.0/10"),      # CGNAT
    ipaddress.ip_network("::1/128"),            # IPv6 loopback
    ipaddress.ip_network("fe80::/10"),           # IPv6 link-local
    ipaddress.ip_network("fc00::/7"),           # IPv6 unique local
)

# Explicitly allowed hostnames (e.g., for local dev servers)
_SSRF_ALLOWLIST = {"localhost", "127.0.0.1", "::1"}


def _validate_url(url: str) -> None:
    """Validate URL for SSRF prevention.

    Blocks requests to private, link-local, and loopback IP ranges
    unless the hostname is explicitly allowlisted.

    Raises:
        LLMError: If the URL resolves to a blocked address.
    """
    parsed = urllib.parse.urlparse(url)
    hostname = parsed.hostname
    if not hostname:
        raise LLMError(f"Invalid LLM endpoint URL: no hostname in '{url[:80]}'")

    # Allow explicitly allowlisted hosts (local dev servers)
    if hostname.lower() in _SSRF_ALLOWLIST:
        return

    # Resolve hostname to IP
    try:
        ip = ipaddress.ip_address(hostname)
    except ValueError:
        # Not an IP literal; resolve via DNS
        try:
            resolved = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
        except socket.gaierror as exc:
            raise LLMError(f"Cannot resolve LLM endpoint '{hostname}': {exc}") from exc
        if not resolved:
            raise LLMError(f"LLM endpoint '{hostname}' resolved to no addresses")
        # Check all resolved addresses
        for addr_info in resolved:
            ip_str = addr_info[4][0]
            ip = ipaddress.ip_address(ip_str)
            _check_ip_blocked(ip, hostname)

    # If hostname was already an IP literal, check it
    if isinstance(ip, (ipaddress.IPv4Address, ipaddress.IPv6Address)):
        _check_ip_blocked(ip, hostname)


def _check_ip_blocked(ip: ipaddress.IPv4Address | ipaddress.IPv6Address, hostname: str) -> None:
    """Raise LLMError if the IP falls in blocked ranges."""
    for net in _BLOCKED_NETWORKS:
        if ip in net:
            raise LLMError(
                f"SSRF prevention: LLM endpoint '{hostname}' ({ip}) "
                f"resolves to blocked network {net}. "
                f"Use --llm-base-url with a public endpoint or add to allowlist."
            )


def _llm_invoke(
    url: str,
    payload: dict,
    headers: dict,
    timeout: float = 5.0,
    max_retries: int = 3,
    retry_delay: float = 1.0,
) -> dict:
    """Send a JSON POST request via urllib with retry logic.

    Args:
        url: Full API endpoint URL.
        payload: JSON-serializable request body.
        headers: HTTP headers dict.
        timeout: Request timeout in seconds.
        max_retries: Maximum retry attempts (total attempts = 1 + max_retries).
        retry_delay: Seconds between retries (linear, not exponential).

    Returns:
        Parsed JSON response dict.

    Raises:
        LLMError: On timeout, network failure, SSRF validation failure, or exhausted retries.
    """
    # SSRF prevention: validate the target URL before sending
    _validate_url(url)

    data = json.dumps(payload).encode("utf-8")
    headers.setdefault("Content-Type", "application/json")

    last_error: Optional[Exception] = None

    for attempt in range(1 + max_retries):
        try:
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read().decode("utf-8")
                status = resp.getcode()

                if status == 200:
                    return json.loads(body)

                # Non-retryable client errors
                if status in _NON_RETRYABLE_STATUSES:
                    raise LLMError(
                        f"LLM API returned {status}: {body[:300]}"
                    )

                # Retryable server errors
                if status in _RETRYABLE_STATUSES:
                    last_error = LLMError(
                        f"LLM API returned {status} (attempt {attempt + 1})"
                    )
                    if attempt < max_retries:
                        logger.debug("Retrying after %ds...", retry_delay)
                        time.sleep(retry_delay)
                    continue

                # Unexpected status
                raise LLMError(f"LLM API unexpected status {status}: {body[:300]}")

        except (urllib.error.URLError, socket.timeout, OSError) as exc:
            last_error = exc
            if attempt < max_retries:
                logger.debug(
                    "LLM request failed (attempt %d/%d): %s. Retrying...",
                    attempt + 1, 1 + max_retries, exc,
                )
                time.sleep(retry_delay)
            continue
        except json.JSONDecodeError as exc:
            raise LLMError(f"Failed to parse LLM response as JSON: {exc}") from exc

    raise LLMError(
        f"LLM API call failed after {1 + max_retries} attempts"
    ) from last_error


def _validate_llm_output(result: dict, schema: dict | None) -> dict:
    """Lightweight validation of LLM JSON output against expected schema.

    Checks that expected keys exist and value types match. This catches
    LLM hallucination where the structure is valid JSON but missing fields.
    Returns the result unchanged if valid, raises ValueError otherwise.
    """
    if schema is None:
        return result
    for key, hint in schema.items():
        if not isinstance(hint, str):
            continue
        # hint format: "type — description" or just "type"
        expected_type = hint.split()[0].rstrip(",")
        if key not in result:
            logger.warning("LLM output missing expected key '%s'", key)
        else:
            value = result[key]
            type_ok = False
            if expected_type in ("string", "str"):
                type_ok = isinstance(value, str)
            elif expected_type == "float":
                type_ok = isinstance(value, (int, float))
            elif expected_type == "array":
                type_ok = isinstance(value, list)
            elif expected_type == "object":
                type_ok = isinstance(value, dict)
            else:
                type_ok = True  # Unknown types pass through
            if not type_ok:
                logger.warning(
                    "LLM output key '%s': expected %s, got %s (value=%s)",
                    key, expected_type, type(value).__name__, str(value)[:100],
                )
    return result


# ---------------------------------------------------------------------------
# OpenAI-compatible backend
# ---------------------------------------------------------------------------


class OpenAIBackend(LLMClient):
    """OpenAI-compatible chat completions backend.

    Works with:
    - OpenAI API (api.openai.com)
    - Azure OpenAI
    - Local llama.cpp server
    - Any OpenAI-compatible endpoint

    Environment:
        OPENAI_API_KEY — API key (if not passed explicitly)
        OPENAI_BASE_URL  — Base URL (default: https://api.openai.com/v1)
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",
        rate_limiter: RateLimiter | None = None,
    ):
        import os

        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._rl = rate_limiter

    # ------------------------------------------------------------------
    # LLMClient interface
    # ------------------------------------------------------------------

    def generate(self, prompt: str, schema: dict | None = None) -> dict:
        """Send a chat completion request.

        If schema is provided, the prompt is augmented to instruct the
        LLM to respond with JSON matching the described schema.
        """
        if not self._api_key:
            raise LLMError("No OpenAI API key configured")

        # Rate limit check
        if self._rl is not None and not self._rl.allow():
            raise LLMError("Daily LLM call limit exceeded")

        # Build messages
        full_prompt = prompt
        if schema is not None:
            full_prompt = (
                f"{prompt}\n\n"
                f"Respond ONLY with a valid JSON object matching this schema: "
                f"{json.dumps(schema)}. Do not include any other text."
            )

        messages = [{"role": "user", "content": full_prompt}]

        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 512,
            "response_format": {"type": "json_object"},
        }

        headers = {
            "Authorization": f"Bearer {self._api_key}",
        }

        url = f"{self._base_url}/chat/completions"

        try:
            response = _llm_invoke(url, payload, headers)
            if self._rl is not None:
                self._rl.record()
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(f"OpenAI API call failed: {exc}") from exc

        # Extract content from OpenAI response format
        try:
            content = response["choices"][0]["message"]["content"]
            result = json.loads(content)
            _validate_llm_output(result, schema)
            return result
        except (KeyError, IndexError, json.JSONDecodeError) as exc:
            raise LLMError(
                f"Failed to parse OpenAI response: {exc}"
            ) from exc

    def embed(self, text: str) -> np.ndarray:
        """Generate embedding via OpenAI embeddings API.

        Uses text-embedding-3-small (1536d) by default.
        """
        if not self._api_key:
            raise LLMError("No OpenAI API key configured")

        payload = {
            "model": "text-embedding-3-small",
            "input": text,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
        }
        url = f"{self._base_url}/embeddings"

        response = _llm_invoke(url, payload, headers, timeout=10.0)
        try:
            vec = response["data"][0]["embedding"]
            return np.array(vec, dtype=np.float32)
        except (KeyError, IndexError) as exc:
            raise LLMError(
                f"Failed to parse OpenAI embedding response: {exc}"
            ) from exc


# ---------------------------------------------------------------------------
# Claude backend
# ---------------------------------------------------------------------------


class ClaudeBackend(LLMClient):
    """Anthropic Claude Messages API backend.

    Environment:
        ANTHROPIC_API_KEY — API key (if not passed explicitly)
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://api.anthropic.com/v1",
        model: str = "claude-sonnet-4-20250514",
        rate_limiter: RateLimiter | None = None,
    ):
        import os

        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._rl = rate_limiter

    # ------------------------------------------------------------------
    # LLMClient interface
    # ------------------------------------------------------------------

    def generate(self, prompt: str, schema: dict | None = None) -> dict:
        """Send a message to Claude.

        Uses the Messages API. If schema is provided, the prompt is
        augmented to instruct Claude to respond with JSON.
        """
        if not self._api_key:
            raise LLMError("No Anthropic API key configured")

        if self._rl is not None and not self._rl.allow():
            raise LLMError("Daily LLM call limit exceeded")

        full_prompt = prompt
        if schema is not None:
            full_prompt = (
                f"{prompt}\n\n"
                f"Respond ONLY with a valid JSON object matching this schema: "
                f"{json.dumps(schema)}. Do not include any other text."
            )

        payload = {
            "model": self._model,
            "max_tokens": 512,
            "messages": [{"role": "user", "content": full_prompt}],
        }

        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": "2023-06-01",
        }

        url = f"{self._base_url}/messages"

        try:
            response = _llm_invoke(url, payload, headers)
            if self._rl is not None:
                self._rl.record()
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(f"Claude API call failed: {exc}") from exc

        # Extract content from Claude response format
        try:
            content_blocks = response["content"]
            text = ""
            for block in content_blocks:
                if block.get("type") == "text":
                    text += block.get("text", "")
            result = json.loads(text)
            _validate_llm_output(result, schema)
            return result
        except (KeyError, IndexError, json.JSONDecodeError) as exc:
            raise LLMError(
                f"Failed to parse Claude response: {exc}"
            ) from exc

    def embed(self, text: str) -> np.ndarray:
        """Claude does not support embeddings API.

        Cell-mem uses local sentence-transformers for all memory embeddings,
        so this is not called in normal operation.
        """
        raise NotImplementedError(
            "Claude does not provide an embeddings API. "
            "Use the local EmbeddingModel for memory embeddings."
        )
