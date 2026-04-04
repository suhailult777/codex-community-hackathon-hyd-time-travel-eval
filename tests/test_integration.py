"""Integration smoke test — runs full pipeline in rules-only mode (no API calls)."""

import asyncio

import pytest

from main import run_tte


@pytest.mark.asyncio
async def test_full_pipeline_rules_only():
    """End-to-end test using rule-based branches and mock agent (zero API calls)."""
    result = await run_tte(
        task="Deploy version 2 of the frontend",
        n_branches=3,
        max_steps=5,
        use_llm=False,
        use_llm_agent=False,
    )

    # Basic structural checks
    assert result.base_task == "Deploy version 2 of the frontend"
    assert len(result.branches) == 3
    assert 0.0 <= result.robustness_score <= 1.0
    assert 0.0 <= result.success_rate <= 1.0
    assert 0.0 <= result.stability_score <= 1.0

    # At least one branch should be baseline
    baselines = [b for b in result.branches if b.branch.is_baseline]
    assert len(baselines) == 1

    # Each branch should have steps
    for trace in result.branches:
        assert len(trace.steps) > 0
        assert trace.final_state is not None

    # JSON serialization should work
    json_str = result.model_dump_json()
    assert len(json_str) > 100
    assert result.total_api_calls == 0
