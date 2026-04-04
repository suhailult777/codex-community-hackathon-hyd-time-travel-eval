"""Helpers for estimating evaluation cost and choosing fast UI defaults."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeEstimate:
    total_api_calls: int
    estimated_min_seconds: int


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
) -> RuntimeEstimate:
    """Return a combined call-count and runtime estimate."""
    total_calls = estimate_total_api_calls(
        n_branches=n_branches,
        max_steps=max_steps,
        use_llm=use_llm,
        use_llm_agent=use_llm_agent,
        use_llm_judge=use_llm_judge,
    )
    return RuntimeEstimate(
        total_api_calls=total_calls,
        estimated_min_seconds=estimate_min_runtime_seconds(total_calls, rpm),
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
