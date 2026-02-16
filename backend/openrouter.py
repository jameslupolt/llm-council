"""
OpenRouter API client for making LLM requests.

This version is designed to be more robust than the upstream minimal client:

- Reuses a single AsyncClient (connection pooling, fewer TLS handshakes).
- Defaults to trust_env=False and http2=False to avoid common EndOfStream issues.
- Adds light retry + connection limits.
- Prints *useful* error diagnostics (HTTP status/body, underlying causes).
- Adds a small concurrency cap for parallel model fan-out.

Drop-in compatible with the upstream backend:
- query_model(model, messages, timeout=...) -> {content, reasoning_details} | None
- query_models_parallel(models, messages) -> {model: response|None}
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, List, Optional

import httpx

from .config import OPENROUTER_API_KEY, OPENROUTER_API_URL

# -----------------------------
# Tuning knobs (env-overridable)
# -----------------------------

# Default request timeout (seconds) if caller doesn't override.
_DEFAULT_TIMEOUT = float(os.getenv("OPENROUTER_TIMEOUT", "120.0"))

# Disable proxy env vars by default (common source of flaky EndOfStream/EOF).
_TRUST_ENV = os.getenv("OPENROUTER_TRUST_ENV", "0").strip().lower() in ("1", "true", "yes", "on")

# Disable HTTP/2 by default (some networks/proxies behave badly with it).
_HTTP2 = os.getenv("OPENROUTER_HTTP2", "0").strip().lower() in ("1", "true", "yes", "on")

# Limit parallel fan-out to avoid too many simultaneous connections.
_MAX_PARALLEL = int(os.getenv("OPENROUTER_MAX_PARALLEL", "8"))

# Transport retries: retries on connect errors, some read errors, etc (depends on httpx version).
_RETRIES = int(os.getenv("OPENROUTER_RETRIES", "2"))

# Optional: if set, force a max_tokens in payload. If unset, we omit it and let provider defaults apply.
# If you do set it, keep it >= 16 to avoid some provider minimums.
_ENV_MAX_TOKENS = os.getenv("OPENROUTER_MAX_TOKENS", "").strip()

# Optional OpenRouter metadata headers (nice to have, not required).
_ENV_SITE_URL = os.getenv("OPENROUTER_SITE_URL", "").strip()
_ENV_APP_TITLE = os.getenv("OPENROUTER_APP_TITLE", "llm-council").strip()

# Connection pool limits (keep conservative; OpenRouter is one host, many requests).
_LIMITS = httpx.Limits(max_connections=20, max_keepalive_connections=10, keepalive_expiry=30.0)

# -----------------------------
# Client lifecycle
# -----------------------------

_client: Optional[httpx.AsyncClient] = None
_client_lock = asyncio.Lock()


def _build_transport() -> httpx.AsyncHTTPTransport:
    """
    httpx.AsyncHTTPTransport signature differs slightly across versions.
    We try to enable retries if supported, otherwise fall back.
    """
    try:
        return httpx.AsyncHTTPTransport(retries=_RETRIES)
    except TypeError:
        return httpx.AsyncHTTPTransport()


async def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is not None:
        return _client

    async with _client_lock:
        if _client is not None:
            return _client

        transport = _build_transport()
        _client = httpx.AsyncClient(
            transport=transport,
            limits=_LIMITS,
            timeout=None,          # we pass per-request timeouts
            follow_redirects=True,
            trust_env=_TRUST_ENV,
            http2=_HTTP2,
        )
        return _client


async def aclose() -> None:
    """Optional: call on FastAPI shutdown to close sockets cleanly."""
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


# -----------------------------
# Helpers
# -----------------------------

def _headers() -> Dict[str, str]:
    if not OPENROUTER_API_KEY:
        # Fail fast with a clear error instead of mysterious connection issues.
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    h: Dict[str, str] = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    # Optional metadata headers (harmless if omitted)
    if _ENV_SITE_URL:
        h["HTTP-Referer"] = _ENV_SITE_URL
    if _ENV_APP_TITLE:
        h["X-Title"] = _ENV_APP_TITLE
    return h


def _coerce_max_tokens(max_tokens: Optional[int]) -> Optional[int]:
    """
    Some providers enforce minimum output tokens; if you set a max, keep it >= 16.
    If neither env nor argument provides a value, return None (omit from payload).
    """
    value: Optional[int] = None

    if max_tokens is not None:
        value = int(max_tokens)
    elif _ENV_MAX_TOKENS:
        try:
            value = int(_ENV_MAX_TOKENS)
        except ValueError:
            value = None

    if value is None:
        return None

    # Guardrail: providers that have a min may reject tiny values.
    if value < 16:
        value = 16

    return value


def _truncate(s: str, n: int = 800) -> str:
    s = s or ""
    return s if len(s) <= n else s[:n] + "…"


# -----------------------------
# Public API
# -----------------------------

async def query_model(
    model: str,
    messages: List[Dict[str, str]],
    timeout: float = _DEFAULT_TIMEOUT,
    max_tokens: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """
    Query a single model via OpenRouter API.

    Args:
        model: OpenRouter model identifier (e.g., "openai/gpt-5.2")
        messages: List of message dicts with 'role' and 'content'
        timeout: Request timeout in seconds
        max_tokens: Optional max output tokens (if None, uses env OPENROUTER_MAX_TOKENS if set)

    Returns:
        Dict with 'content' and optional 'reasoning_details', or None if failed.
    """
    client = await _get_client()

    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
    }

    mt = _coerce_max_tokens(max_tokens)
    if mt is not None:
        payload["max_tokens"] = mt

    # Per-request timeout
    req_timeout = httpx.Timeout(timeout)

    try:
        resp = await client.post(
            OPENROUTER_API_URL,
            headers=_headers(),
            json=payload,
            timeout=req_timeout,
        )
        resp.raise_for_status()

        data = resp.json()
        # OpenRouter shape: {"choices":[{"message":{"content":...,"reasoning_details":...}}], ...}
        message = data["choices"][0]["message"]
        return {
            "content": message.get("content"),
            "reasoning_details": message.get("reasoning_details"),
        }

    except httpx.HTTPStatusError as e:
        # HTTP error *with* response details
        status = e.response.status_code
        body = _truncate(e.response.text, 1200)
        print(f"Error querying model {model}: HTTP {status} body={body}")
        return None

    except (httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError) as e:
        # Transport-level errors (EndOfStream/BrokenResourceError often bubble up here)
        cause = getattr(e, "__cause__", None)
        ctx = getattr(e, "__context__", None)
        print(f"Error querying model {model}: {type(e).__name__} {repr(e)}")
        if cause is not None:
            print(f"  cause: {type(cause).__name__} {repr(cause)}")
        if ctx is not None:
            print(f"  context: {type(ctx).__name__} {repr(ctx)}")
        return None

    except httpx.TimeoutException as e:
        print(f"Error querying model {model}: Timeout {repr(e)}")
        return None

    except Exception as e:
        # Catch-all (keep it informative)
        cause = getattr(e, "__cause__", None)
        ctx = getattr(e, "__context__", None)
        print(f"Error querying model {model}: {type(e).__name__} {repr(e)}")
        if cause is not None:
            print(f"  cause: {type(cause).__name__} {repr(cause)}")
        if ctx is not None:
            print(f"  context: {type(ctx).__name__} {repr(ctx)}")
        return None


async def query_models_parallel(
    models: List[str],
    messages: List[Dict[str, str]],
) -> Dict[str, Optional[Dict[str, Any]]]:
    """
    Query multiple models in parallel (bounded).

    Returns:
        Dict mapping model identifier to response dict (or None if failed)
    """
    sem = asyncio.Semaphore(max(1, _MAX_PARALLEL))

    async def _run_one(m: str) -> Optional[Dict[str, Any]]:
        await sem.acquire()
        try:
            return await query_model(m, messages, timeout=_DEFAULT_TIMEOUT)
        finally:
            sem.release()

    tasks = [_run_one(m) for m in models]
    responses = await asyncio.gather(*tasks)

    return {m: r for m, r in zip(models, responses)}

