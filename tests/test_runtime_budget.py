from core.runtime_budget import (
    estimate_min_runtime_seconds,
    estimate_runtime,
    estimate_total_api_calls,
    format_duration,
    recommend_live_profile,
)


def test_estimate_total_api_calls_live_mode():
    assert estimate_total_api_calls(
        n_branches=3,
        max_steps=6,
        use_llm=True,
        use_llm_agent=True,
    ) == 19


def test_estimate_total_api_calls_demo_mode():
    assert estimate_total_api_calls(
        n_branches=3,
        max_steps=6,
        use_llm=False,
        use_llm_agent=False,
    ) == 0


def test_estimate_total_api_calls_with_llm_judge():
    assert estimate_total_api_calls(
        n_branches=3,
        max_steps=6,
        use_llm=True,
        use_llm_agent=True,
        use_llm_judge=True,
    ) == 22


def test_estimate_runtime_switches_to_turbo_for_batched_models():
    estimate = estimate_runtime(
        n_branches=7,
        max_steps=12,
        use_llm=True,
        use_llm_agent=True,
        use_llm_judge=True,
        rpm=40,
        execution_mode="auto",
        provider_profile="auto",
        base_url="https://integrate.api.nvidia.com/v1",
        agent_model="moonshotai/kimi-k2.5",
        generator_model="moonshotai/kimi-k2.5",
        rate_limit_safety_factor=0.8,
    )
    assert estimate.logical_calls == 92
    assert estimate.scheduled_provider_calls == 14
    assert estimate.execution_mode == "turbo"
    assert estimate.effective_rpm == 32
    assert estimate.estimated_min_seconds == 26


def test_estimate_runtime_stays_standard_for_conservative_profile():
    estimate = estimate_runtime(
        n_branches=3,
        max_steps=6,
        use_llm=True,
        use_llm_agent=True,
        use_llm_judge=True,
        rpm=40,
        execution_mode="auto",
        provider_profile="generic",
        base_url="https://example.com/v1",
        agent_model="unknown-model",
        generator_model="unknown-model",
        rate_limit_safety_factor=0.8,
    )
    assert estimate.logical_calls == 22
    assert estimate.scheduled_provider_calls == 22
    assert estimate.execution_mode == "standard"
    assert estimate.effective_rpm == 26


def test_estimate_runtime_supports_openai_batched_models():
    estimate = estimate_runtime(
        n_branches=7,
        max_steps=12,
        use_llm=True,
        use_llm_agent=True,
        use_llm_judge=True,
        rpm=40,
        execution_mode="auto",
        provider_profile="auto",
        base_url="https://api.openai.com/v1",
        agent_model="gpt-5.4-nano",
        generator_model="gpt-5.4-nano",
        rate_limit_safety_factor=0.8,
    )
    assert estimate.logical_calls == 92
    assert estimate.scheduled_provider_calls == 14
    assert estimate.execution_mode == "turbo"
    assert estimate.provider_profile == "batched_json"
    assert estimate.effective_rpm == 32


def test_recommend_live_profile_prefers_faster_defaults_for_40_rpm():
    assert recommend_live_profile(40) == (3, 6)
    assert recommend_live_profile(20) == (2, 5)


def test_format_duration():
    assert format_duration(28) == "28s"
    assert format_duration(60) == "1m"
    assert format_duration(91) == "1m 31s"


def test_min_runtime_never_negative():
    assert estimate_min_runtime_seconds(0, 40) == 0
    assert estimate_min_runtime_seconds(1, 40) == 2
