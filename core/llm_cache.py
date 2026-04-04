"""Persistent cache for deterministic LLM requests."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from hashlib import sha256
from pathlib import Path
from typing import Any


class PersistentLLMCache:
    """Small SQLite-backed cache for semantic LLM request/response reuse."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS llm_cache (
                    cache_key TEXT PRIMARY KEY,
                    value_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=30.0)

    def get(self, cache_key: str) -> dict[str, Any] | None:
        with self._lock:
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT value_json FROM llm_cache WHERE cache_key = ?",
                    (cache_key,),
                ).fetchone()
        if row is None:
            return None
        return json.loads(row[0])

    def set(self, cache_key: str, value: dict[str, Any]) -> None:
        payload = json.dumps(value, sort_keys=True)
        with self._lock:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT OR REPLACE INTO llm_cache (cache_key, value_json, created_at)
                    VALUES (?, ?, ?)
                    """,
                    (cache_key, payload, time.time()),
                )


_CACHE_INSTANCES: dict[str, PersistentLLMCache] = {}
_CACHE_LOCK = threading.Lock()


def get_cache(path: str | Path) -> PersistentLLMCache:
    """Return a process-wide cache instance for the given path."""
    normalized = str(Path(path))
    with _CACHE_LOCK:
        if normalized not in _CACHE_INSTANCES:
            _CACHE_INSTANCES[normalized] = PersistentLLMCache(normalized)
        return _CACHE_INSTANCES[normalized]


def build_cache_key(
    *,
    base_url: str,
    model: str,
    provider_profile: str,
    execution_mode: str,
    prompt_version: str,
    purpose: str,
    payload: dict[str, Any],
) -> str:
    """Build a stable semantic cache key."""
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    digest = sha256(serialized.encode("utf-8")).hexdigest()
    return "::".join(
        [
            purpose,
            base_url.strip().lower(),
            model.strip().lower(),
            provider_profile,
            execution_mode,
            prompt_version,
            digest,
        ]
    )
