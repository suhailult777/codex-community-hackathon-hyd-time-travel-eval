"""Execution planning, telemetry, and cache-aware request helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from core.config import config
from core.llm_cache import PersistentLLMCache, get_cache
from core.llm_utils import run_with_rate_limit
from core.provider_capabilities import ProviderCapabilities, resolve_provider_capabilities


@dataclass
class ExecutionStats:
    """Per-run telemetry for provider scheduling and cache use."""

    cache_hits: int = 0
    cache_misses: int = 0
    scheduled_provider_calls: int = 0


@dataclass
class ExecutionContext:
    """Shared execution context for one TTE run."""

    provider_name: str
    provider_label: str
    api_key: str
    base_url: str
    agent_model: str
    generator_model: str
    agent_capabilities: ProviderCapabilities
    generator_capabilities: ProviderCapabilities
    requested_mode: str
    execution_mode: str
    effective_rpm: int
    cache: PersistentLLMCache
    stats: ExecutionStats = field(default_factory=ExecutionStats)
    prompt_version: str = "tte-opt-v1"


def compute_effective_rpm(
    *,
    configured_rpm: int,
    agent_capabilities: ProviderCapabilities,
    generator_capabilities: ProviderCapabilities,
) -> int:
    """Compute a conservative effective RPM for all provider calls."""
    factor = min(
        max(0.1, config.RATE_LIMIT_SAFETY_FACTOR),
        agent_capabilities.preferred_effective_rpm_factor,
        generator_capabilities.preferred_effective_rpm_factor,
    )
    return max(1, int(math.floor(max(1, configured_rpm) * factor)))


def choose_execution_mode(
    *,
    requested_mode: str,
    logical_calls: int,
    effective_rpm: int,
    use_llm_agent: bool,
    agent_capabilities: ProviderCapabilities,
) -> str:
    """Resolve the actual execution mode for this run."""
    normalized = (requested_mode or "auto").strip().lower()
    if normalized == "standard":
        return "standard"
    if normalized == "turbo":
        return "turbo" if use_llm_agent and agent_capabilities.supports_batched_agent_prompt else "standard"
    if use_llm_agent and logical_calls > effective_rpm and agent_capabilities.supports_batched_agent_prompt:
        return "turbo"
    return "standard"


def build_execution_context(
    *,
    logical_calls: int,
    use_llm_agent: bool,
    provider: str | None = None,
) -> ExecutionContext:
    """Build the shared execution context from config and planner inputs."""
    settings = config.get_provider_settings(provider)
    agent_capabilities = resolve_provider_capabilities(
        base_url=settings.base_url,
        model=settings.agent_model,
        provider_profile=config.PROVIDER_PROFILE,
    )
    generator_capabilities = resolve_provider_capabilities(
        base_url=settings.base_url,
        model=settings.generator_model,
        provider_profile=config.PROVIDER_PROFILE,
    )
    effective_rpm = compute_effective_rpm(
        configured_rpm=config.MAX_REQUESTS_PER_MINUTE,
        agent_capabilities=agent_capabilities,
        generator_capabilities=generator_capabilities,
    )
    execution_mode = choose_execution_mode(
        requested_mode=config.EXECUTION_MODE,
        logical_calls=logical_calls,
        effective_rpm=effective_rpm,
        use_llm_agent=use_llm_agent,
        agent_capabilities=agent_capabilities,
    )
    return ExecutionContext(
        provider_name=settings.name,
        provider_label=settings.label,
        api_key=settings.api_key,
        base_url=settings.base_url,
        agent_model=settings.agent_model,
        generator_model=settings.generator_model,
        agent_capabilities=agent_capabilities,
        generator_capabilities=generator_capabilities,
        requested_mode=config.EXECUTION_MODE,
        execution_mode=execution_mode,
        effective_rpm=effective_rpm,
        cache=get_cache(config.CACHE_PATH),
    )


async def execute_with_cache(
    *,
    execution_context: ExecutionContext | None,
    cache_key: str | None,
    request_factory: Callable[[], Awaitable[Any]],
    parser: Callable[[Any], dict[str, Any]],
) -> tuple[dict[str, Any], bool]:
    """Run a request with semantic cache reuse when context is available."""
    if execution_context is None or not cache_key:
        response = await request_factory()
        return parser(response), False

    cached = execution_context.cache.get(cache_key)
    if cached is not None:
        execution_context.stats.cache_hits += 1
        return cached, True

    execution_context.stats.cache_misses += 1
    execution_context.stats.scheduled_provider_calls += 1
    response = await run_with_rate_limit(
        request_factory,
        max_requests_per_minute=execution_context.effective_rpm,
    )
    payload = parser(response)
    execution_context.cache.set(cache_key, payload)
    return payload, False
