from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import core.agent_runner as agent_runner
import core.execution as execution
import core.llm_judge as llm_judge
import core.scenario_generator as scenario_generator
from core.config import config
from main import run_tte


class FakeMessage:
    def __init__(self, content: str):
        self.content = content
        self.tool_calls = []
        self.reasoning = None


class FakeResponse:
    def __init__(self, content: str, total_tokens: int = 12):
        self.choices = [SimpleNamespace(message=FakeMessage(content))]
        self.usage = SimpleNamespace(total_tokens=total_tokens)


class FakeAsyncClient:
    def __init__(self):
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        messages = kwargs["messages"]
        system_text = messages[0]["content"] if messages else ""
        user_text = messages[-1]["content"] if messages else ""

        if "scenario designer" in system_text.lower():
            branches = {
                "branches": [
                    {"id": "branch_0", "label": "Baseline", "description": "Baseline", "is_baseline": True, "events": []},
                    {"id": "branch_1", "label": "CPU Spike", "description": "CPU issue", "events": [{"step": 2, "name": "cpu_spike", "description": "CPU high", "effect_deltas": {"cpu_utilization": 95, "server_status": "degraded"}}]},
                    {"id": "branch_2", "label": "Dependency Crash", "description": "Dependency issue", "events": [{"step": 1, "name": "dependency_crash", "description": "Dependency down", "effect_deltas": {"dependency_api_status": "down", "latency_ms": 5000}}]},
                    {"id": "branch_3", "label": "Network Latency", "description": "Latency issue", "events": [{"step": 3, "name": "latency_spike", "description": "Latency high", "effect_deltas": {"latency_ms": 3000, "dependency_api_status": "slow"}}]},
                    {"id": "branch_4", "label": "Test Failure", "description": "Tests fail", "events": [{"step": 1, "name": "test_failure", "description": "Tests red", "effect_deltas": {"tests_passing": False}}]},
                    {"id": "branch_5", "label": "Server Outage", "description": "Server down", "events": [{"step": 2, "name": "server_outage", "description": "Down", "effect_deltas": {"server_status": "down", "error_rate": 1.0}}]},
                    {"id": "branch_6", "label": "Byzantine Deploy", "description": "Liar", "events": [{"step": 1, "name": "byzantine_deploy_lie", "description": "Lies", "effect_deltas": {"byzantine_deploy_lie": True}}]},
                ]
            }
            return FakeResponse(json.dumps(branches))

        if "coordinating multiple independent devops timelines" in system_text.lower():
            batch = json.loads(user_text)
            actions = []
            for branch in batch["branches"]:
                next_input = branch["next_input"]
                if "previous tool result: test suite result" in next_input.lower():
                    tool_name = "deploy"
                elif "deploy=deployed" in next_input.lower() or "deploy=failed" in next_input.lower():
                    tool_name = "finish_task"
                else:
                    tool_name = "run_tests"
                actions.append({"branch_id": branch["branch_id"], "tool_name": tool_name, "tool_args": {}})
            return FakeResponse(json.dumps({"actions": actions}), total_tokens=70)

        if "evaluating multiple independent devops agent traces" in system_text.lower():
            batch = json.loads(user_text)
            results = [
                {
                    "branch_id": branch["branch_id"],
                    "score": 0.9 if branch["branch_id"] == "branch_0" else 0.6,
                    "panic_score": 0.1 if branch["branch_id"] == "branch_0" else 0.4,
                    "explanation": "Judged in batch.",
                }
                for branch in batch["branches"]
            ]
            return FakeResponse(json.dumps({"results": results}), total_tokens=50)

        if "expert evaluator judging an ai agent" in user_text.lower():
            return FakeResponse(json.dumps({"score": 0.5, "panic_score": 0.5, "explanation": "fallback"}))

        return FakeResponse('{"tool_name":"check_logs","tool_args":{}}')


@pytest.mark.asyncio
async def test_run_tte_uses_turbo_and_reduces_scheduled_calls_for_batched_profiles(monkeypatch, tmp_path):
    fake_client = FakeAsyncClient()

    async def immediate_run_with_rate_limit(request_factory, **kwargs):
        return await request_factory()

    monkeypatch.setattr(execution, "run_with_rate_limit", immediate_run_with_rate_limit)
    monkeypatch.setattr(agent_runner, "AsyncOpenAI", lambda **kwargs: fake_client)
    monkeypatch.setattr(scenario_generator, "AsyncOpenAI", lambda **kwargs: fake_client)
    monkeypatch.setattr(llm_judge, "AsyncOpenAI", lambda **kwargs: fake_client)
    monkeypatch.setattr(config, "API_KEY", "test-key")
    monkeypatch.setattr(config, "API_BASE_URL", "https://integrate.api.nvidia.com/v1")
    monkeypatch.setattr(config, "MODEL_AGENT", "moonshotai/kimi-k2.5")
    monkeypatch.setattr(config, "MODEL_GENERATOR", "moonshotai/kimi-k2.5")
    monkeypatch.setattr(config, "PROVIDER_PROFILE", "batched_json")
    monkeypatch.setattr(config, "EXECUTION_MODE", "auto")
    monkeypatch.setattr(config, "MAX_REQUESTS_PER_MINUTE", 40)
    monkeypatch.setattr(config, "RATE_LIMIT_SAFETY_FACTOR", 0.8)
    monkeypatch.setattr(config, "ENABLE_LLM_JUDGE", True)
    monkeypatch.setattr(config, "CACHE_PATH", tmp_path / "llm_cache.sqlite3")

    result = await run_tte(
        task="Deploy app",
        n_branches=7,
        max_steps=12,
        use_llm=True,
        use_llm_agent=True,
    )

    assert result.execution_mode == "turbo"
    assert len(result.branches) == 7
    assert all(branch.panic_source == "llm_judge" for branch in result.branches)
    assert result.scheduled_provider_calls < result.logical_calls
    assert result.provider_profile == "batched_json"


@pytest.mark.asyncio
async def test_run_tte_can_target_openai_provider(monkeypatch, tmp_path):
    fake_client = FakeAsyncClient()

    async def immediate_run_with_rate_limit(request_factory, **kwargs):
        return await request_factory()

    monkeypatch.setattr(execution, "run_with_rate_limit", immediate_run_with_rate_limit)
    monkeypatch.setattr(agent_runner, "AsyncOpenAI", lambda **kwargs: fake_client)
    monkeypatch.setattr(scenario_generator, "AsyncOpenAI", lambda **kwargs: fake_client)
    monkeypatch.setattr(llm_judge, "AsyncOpenAI", lambda **kwargs: fake_client)
    monkeypatch.setattr(config, "OPENAI_API_KEY", "openai-test-key")
    monkeypatch.setattr(config, "OPENAI_API_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setattr(config, "OPENAI_MODEL_AGENT", "gpt-5.4-nano")
    monkeypatch.setattr(config, "OPENAI_MODEL_GENERATOR", "gpt-5.4-nano")
    monkeypatch.setattr(config, "PROVIDER_PROFILE", "auto")
    monkeypatch.setattr(config, "EXECUTION_MODE", "auto")
    monkeypatch.setattr(config, "MAX_REQUESTS_PER_MINUTE", 40)
    monkeypatch.setattr(config, "RATE_LIMIT_SAFETY_FACTOR", 0.8)
    monkeypatch.setattr(config, "ENABLE_LLM_JUDGE", True)
    monkeypatch.setattr(config, "CACHE_PATH", tmp_path / "llm_cache.sqlite3")

    result = await run_tte(
        task="Deploy app",
        n_branches=7,
        max_steps=12,
        use_llm=True,
        use_llm_agent=True,
        provider="openai",
    )

    assert result.provider_name == "openai"
    assert result.provider_base_url == "https://api.openai.com/v1"
    assert result.agent_model == "gpt-5.4-nano"
    assert result.generator_model == "gpt-5.4-nano"
    assert result.execution_mode == "turbo"


@pytest.mark.asyncio
async def test_llm_judge_score_many_falls_back_for_missing_batch_results(monkeypatch, tmp_path):
    async def fake_execute_with_cache(**kwargs):
        return {"results": [{"branch_id": "branch_0", "score": 0.9, "panic_score": 0.1, "explanation": "batch"}]}, False

    async def fake_single_score(task, trace, execution_context=None):
        return 0.4, 0.7, f"fallback-{trace.branch.id}"

    monkeypatch.setattr(llm_judge, "execute_with_cache", fake_execute_with_cache)
    monkeypatch.setattr(llm_judge, "llm_judge_score", fake_single_score)
    monkeypatch.setattr(llm_judge, "AsyncOpenAI", lambda **kwargs: FakeAsyncClient())

    from core.execution import build_execution_context
    from core.models import BranchTrace, ScenarioBranch

    monkeypatch.setattr(config, "PROVIDER_PROFILE", "batched_json")
    monkeypatch.setattr(config, "MODEL_AGENT", "moonshotai/kimi-k2.5")
    monkeypatch.setattr(config, "MODEL_GENERATOR", "moonshotai/kimi-k2.5")
    monkeypatch.setattr(config, "CACHE_PATH", tmp_path / "llm_cache.sqlite3")
    execution_context = build_execution_context(logical_calls=10, use_llm_agent=True)

    traces = [
        BranchTrace(branch=ScenarioBranch(id="branch_0", description="b0"), score=0.8),
        BranchTrace(branch=ScenarioBranch(id="branch_1", description="b1"), score=0.7),
    ]
    results = await llm_judge.llm_judge_score_many("task", traces, execution_context=execution_context)

    assert results["branch_0"][0] == 0.9
    assert results["branch_1"][2] == "fallback-branch_1"
