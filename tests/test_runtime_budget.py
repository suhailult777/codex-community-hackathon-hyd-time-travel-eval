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


def test_estimate_runtime_uses_rpm_budget():
    estimate = estimate_runtime(
        n_branches=3,
        max_steps=6,
        use_llm=True,
        use_llm_agent=True,
        rpm=40,
    )
    assert estimate.total_api_calls == 19
    assert estimate.estimated_min_seconds == 28


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
