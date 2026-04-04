"""Shared LLM pacing and retry helpers for strict provider quotas."""

from __future__ import annotations

import asyncio
import random
import re
import threading
import time
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

from core.config import config

_RATE_LIMIT_PHRASES = (
    "too many requests",
    "too many bad requests",
    "rate limit",
    "rate-limit",
    "rate_limit",
    "requests per minute",
    "request limit",
    "retry after",
    "rate exceeded",
    "quota exceeded",
    "resource exhausted",
)


class GlobalRateLimiter:
    """Global, cross-event-loop request pacing for all provider calls."""

    _lock = threading.Lock()
    _next_request_ts = 0.0

    @classmethod
    def reserve_delay(cls, max_requests_per_minute: int) -> float:
        """Reserve the next outbound request slot and return how long to wait."""
        min_interval = 60.0 / max(max_requests_per_minute, 1)
        with cls._lock:
            now = time.monotonic()
            scheduled = max(now, cls._next_request_ts)
            cls._next_request_ts = scheduled + min_interval
        return max(0.0, scheduled - now)

    @classmethod
    async def wait_for_slot(cls, max_requests_per_minute: int) -> None:
        """Sleep until the next globally allowed request slot."""
        delay = cls.reserve_delay(max_requests_per_minute)
        if delay > 0:
            await asyncio.sleep(delay)

    @classmethod
    def defer(cls, delay_seconds: float) -> None:
        """Push future requests back when the provider asks us to slow down."""
        if delay_seconds <= 0:
            return
        with cls._lock:
            cls._next_request_ts = max(cls._next_request_ts, time.monotonic() + delay_seconds)


def _get_error_headers(err: Exception) -> dict[str, str]:
    response = getattr(err, "response", None)
    headers = getattr(response, "headers", None) or getattr(err, "headers", None)
    if not headers or not hasattr(headers, "items"):
        return {}
    return {str(key).lower(): str(value) for key, value in headers.items()}


def _parse_retry_after(raw_value: str) -> float | None:
    value = raw_value.strip()
    if not value:
        return None

    try:
        return max(0.0, float(value))
    except ValueError:
        pass

    try:
        retry_at = parsedate_to_datetime(value)
    except (TypeError, ValueError, IndexError):
        return None

    if retry_at.tzinfo is None:
        retry_at = retry_at.replace(tzinfo=timezone.utc)

    now = datetime.now(timezone.utc)
    return max(0.0, (retry_at - now).total_seconds())


def extract_retry_after_seconds(err: Exception) -> float | None:
    """Best-effort retry-after parsing from headers or provider error text."""
    headers = _get_error_headers(err)

    retry_after_ms = headers.get("retry-after-ms")
    if retry_after_ms:
        try:
            return max(0.0, float(retry_after_ms) / 1000.0)
        except ValueError:
            pass

    retry_after = headers.get("retry-after")
    if retry_after:
        parsed = _parse_retry_after(retry_after)
        if parsed is not None:
            return parsed

    message = str(err).lower()
    match = re.search(r"retry after\s+(\d+(?:\.\d+)?)\s*(?:s|sec|secs|second|seconds)?", message)
    if match:
        return float(match.group(1))

    return None


def is_rate_limited_error(err: Exception) -> bool:
    """Return True when an exception looks like a provider quota response."""
    status_code = getattr(err, "status_code", None)
    if status_code == 429:
        return True

    headers = _get_error_headers(err)
    if "retry-after" in headers or "retry-after-ms" in headers:
        return True

    message = str(err).lower()
    if any(phrase in message for phrase in _RATE_LIMIT_PHRASES):
        return True

    return re.search(r"\b\d+(?:\.\d+)?\s*rpm\b", message) is not None


def completion_token_limit_kwargs(
    max_tokens: int,
    *,
    base_url: str | None = None,
    model: str | None = None,
) -> dict[str, int]:
    """Return provider-compatible token limit kwargs for chat completions.

    OpenAI GPT-5 family models reject `max_tokens` and require
    `max_completion_tokens`.
    """
    normalized_base_url = (base_url or "").strip().lower()
    normalized_model = (model or "").strip().lower()
    token_cap = max(1, int(max_tokens))

    if "api.openai.com" in normalized_base_url or normalized_model.startswith("gpt-5"):
        return {"max_completion_tokens": token_cap}

    return {"max_tokens": token_cap}


async def run_with_rate_limit(
    request_factory: Callable[[], Awaitable[Any]],
    *,
    max_requests_per_minute: int | None = None,
    max_retries: int | None = None,
) -> Any:
    """Run an async provider request with global pacing and quota retries."""
    rpm = max_requests_per_minute or config.MAX_REQUESTS_PER_MINUTE
    retries = config.LLM_MAX_RETRIES if max_retries is None else max_retries
    last_error: Exception | None = None

    for attempt in range(retries + 1):
        await GlobalRateLimiter.wait_for_slot(rpm)
        try:
            return await request_factory()
        except Exception as err:  # noqa: BLE001
            last_error = err
            if not is_rate_limited_error(err):
                raise
            if attempt >= retries:
                break

            backoff = min(
                config.LLM_BACKOFF_BASE_SECONDS * (2**attempt),
                config.LLM_BACKOFF_MAX_SECONDS,
            )
            retry_after = extract_retry_after_seconds(err) or 0.0
            delay = max(backoff, retry_after)
            GlobalRateLimiter.defer(delay)
            await asyncio.sleep(delay + random.uniform(0.0, 0.25))

    if last_error is not None:
        raise last_error
    raise RuntimeError("LLM request failed without an exception")
