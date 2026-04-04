"""Helpers for estimating evaluation cost and choosing fast UI defaults."""

from __future__ import annotations

from dataclasses import dataclass

from core.provider_capabilities import resolve_provider_capabilities


@dataclass(frozen=True)
class RuntimeEstimate:
    logical_calls: int
    scheduled_provider_calls: int
    estimated_min_seconds: int
    execution_mode: str
    effective_rpm: int
    provider_profile: str

    @property
    def total_api_calls(self) -> int:
        """Backward-compatible alias for logical call count."""
        return self.logical_calls


def estimate_total_api_calls(
    *,
    n_branches: int,
    max_steps: int,
    use_llm: bool,
    use_llm_agent: bool,
    use_llm_judge: bool = False,
) -> int:
    """Estimate outbound provider calls for a single evaluation run."""
    calls = 0
    if use_llm:
        calls += 1  # scenario generation
    if use_llm_agent:
        calls += max(0, n_branches) * max(0, max_steps)
    if use_llm_judge:
        calls += max(0, n_branches)
    return calls


def estimate_min_runtime_seconds(total_api_calls: int, rpm: int) -> int:
    """Estimate the minimum wall-clock runtime implied by the RPM budget."""
    if total_api_calls <= 0:
        return 0
    safe_rpm = max(1, rpm)
    seconds = int(round((total_api_calls * 60.0) / safe_rpm))
    return max(1, seconds)


def estimate_runtime(
    *,
    n_branches: int,
    max_steps: int,
    use_llm: bool,
    use_llm_agent: bool,
    rpm: int,
    use_llm_judge: bool = False,
    execution_mode: str = "auto",
    provider_profile: str = "auto",
    base_url: str = "",
    agent_model: str = "",
    generator_model: str = "",
    rate_limit_safety_factor: float = 0.8,
) -> RuntimeEstimate:
    """Return a combined call-count and runtime estimate."""
    logical_calls = estimate_total_api_calls(
        n_branches=n_branches,
        max_steps=max_steps,
        use_llm=use_llm,
        use_llm_agent=use_llm_agent,
        use_llm_judge=use_llm_judge,
    )
    agent_capabilities = resolve_provider_capabilities(
        base_url=base_url,
        model=agent_model,
        provider_profile=provider_profile,
    )
    generator_capabilities = resolve_provider_capabilities(
        base_url=base_url,
        model=generator_model,
        provider_profile=provider_profile,
    )
    effective_factor = min(
        max(0.1, rate_limit_safety_factor),
        agent_capabilities.preferred_effective_rpm_factor,
        generator_capabilities.preferred_effective_rpm_factor,
    )
    effective_rpm = max(1, int(rpm * effective_factor))

    normalized_mode = execution_mode.strip().lower()
    if normalized_mode == "turbo" and not agent_capabilities.supports_batched_agent_prompt:
        resolved_mode = "standard"
    elif normalized_mode == "auto":
        resolved_mode = (
            "turbo"
            if use_llm_agent and logical_calls > effective_rpm and agent_capabilities.supports_batched_agent_prompt
            else "standard"
        )
    else:
        resolved_mode = normalized_mode or "standard"

    scheduled_calls = logical_calls
    if resolved_mode == "turbo" and use_llm_agent and agent_capabilities.supports_batched_agent_prompt:
        scheduled_calls = 0
        if use_llm:
            scheduled_calls += 1
        scheduled_calls += max_steps
        if use_llm_judge:
            scheduled_calls += 1 if generator_capabilities.supports_batched_judge_prompt else max(0, n_branches)

    return RuntimeEstimate(
        logical_calls=logical_calls,
        scheduled_provider_calls=scheduled_calls,
        estimated_min_seconds=estimate_min_runtime_seconds(scheduled_calls, effective_rpm),
        execution_mode=resolved_mode,
        effective_rpm=effective_rpm,
        provider_profile=agent_capabilities.provider_profile,
    )


def recommend_live_profile(rpm: int) -> tuple[int, int]:
    """Choose UI defaults that stay responsive under the available RPM budget."""
    safe_rpm = max(1, rpm)

    if safe_rpm >= 40:
        return 3, 6
    if safe_rpm >= 25:
        return 3, 5
    if safe_rpm >= 15:
        return 2, 5
    return 2, 4


def format_duration(seconds: int) -> str:
    """Render a compact human-readable duration."""
    if seconds < 60:
        return f"{seconds}s"
    minutes, rem = divmod(seconds, 60)
    if rem == 0:
        return f"{minutes}m"
    return f"{minutes}m {rem}s"
