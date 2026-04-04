from __future__ import annotations

from types import SimpleNamespace

import pytest

import core.agent_runner as agent_runner
import core.llm_utils as llm_utils
import core.scenario_generator as scenario_generator
from core.config import config


class FakeAPIError(Exception):
    def __init__(self, message: str, *, status_code: int | None = None, headers: dict | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.response = SimpleNamespace(headers=headers or {})


class FakeToolCall:
    def __init__(self, name: str, arguments: str = "{}"):
        self.id = "tool-1"
        self.function = SimpleNamespace(name=name, arguments=arguments)


class FakeMessage:
    def __init__(self, tool_name: str, arguments: str = "{}"):
        self.content = None
        self.tool_calls = [FakeToolCall(tool_name, arguments)]

    def model_dump(self):
        tool_call = self.tool_calls[0]
        return {
            "role": "assistant",
            "content": self.content,
            "tool_calls": [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    },
                }
            ],
        }


class FakeCompletionResponse:
    def __init__(self, *, tool_name: str = "restart_service", arguments: str = "{}"):
        self.usage = SimpleNamespace(total_tokens=21)
        self.choices = [SimpleNamespace(message=FakeMessage(tool_name, arguments))]


class FakeJSONMessage:
    def __init__(self, content: str):
        self.content = content
        self.tool_calls = []
        self.reasoning = "Keep the system stable."

    def model_dump(self):
        return {
            "role": "assistant",
            "content": self.content,
        }


class FakeJSONCompletionResponse:
    def __init__(self, content: str):
        self.usage = SimpleNamespace(total_tokens=13)
        self.choices = [SimpleNamespace(message=FakeJSONMessage(content))]


def test_is_rate_limited_error_handles_provider_specific_messages():
    assert llm_utils.is_rate_limited_error(
        FakeAPIError("Too many bad requests", status_code=400)
    )
    assert llm_utils.is_rate_limited_error(
        FakeAPIError("Request limit reached, retry after 3 seconds", status_code=400)
    )
    assert not llm_utils.is_rate_limited_error(
        FakeAPIError("Bad request: malformed JSON body", status_code=400)
    )


@pytest.mark.asyncio
async def test_run_with_rate_limit_retries_and_defers_global_queue(monkeypatch):
    waits: list[int] = []
    defers: list[float] = []
    sleeps: list[float] = []

    async def fake_wait_for_slot(cls, rpm: int):
        waits.append(rpm)

    def fake_defer(cls, delay: float):
        defers.append(delay)

    async def fake_sleep(delay: float):
        sleeps.append(delay)

    monkeypatch.setattr(llm_utils.GlobalRateLimiter, "wait_for_slot", classmethod(fake_wait_for_slot))
    monkeypatch.setattr(llm_utils.GlobalRateLimiter, "defer", classmethod(fake_defer))
    monkeypatch.setattr(llm_utils.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(llm_utils.random, "uniform", lambda _a, _b: 0.0)
    monkeypatch.setattr(config, "MAX_REQUESTS_PER_MINUTE", 40)
    monkeypatch.setattr(config, "LLM_MAX_RETRIES", 2)
    monkeypatch.setattr(config, "LLM_BACKOFF_BASE_SECONDS", 1.5)
    monkeypatch.setattr(config, "LLM_BACKOFF_MAX_SECONDS", 20.0)

    attempts = {"count": 0}

    async def flaky_request():
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise FakeAPIError(
                "Too many bad requests",
                status_code=400,
                headers={"Retry-After": "2"},
            )
        return "ok"

    result = await llm_utils.run_with_rate_limit(flaky_request)

    assert result == "ok"
    assert attempts["count"] == 2
    assert waits == [40, 40]
    assert defers == [2.0]
    assert sleeps == [2.0]


@pytest.mark.asyncio
async def test_agent_runner_recovers_from_too_many_bad_requests(monkeypatch):
    async def fake_wait_for_slot(cls, rpm: int):
        return None

    def fake_defer(cls, delay: float):
        return None

    async def fake_sleep(delay: float):
        return None

    class FakeClient:
        def __init__(self):
            self.calls = 0
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        async def create(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise FakeAPIError("Too many bad requests", status_code=400)
            return FakeCompletionResponse(tool_name="restart_service")

    fake_client = FakeClient()

    monkeypatch.setattr(llm_utils.GlobalRateLimiter, "wait_for_slot", classmethod(fake_wait_for_slot))
    monkeypatch.setattr(llm_utils.GlobalRateLimiter, "defer", classmethod(fake_defer))
    monkeypatch.setattr(llm_utils.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(llm_utils.random, "uniform", lambda _a, _b: 0.0)
    monkeypatch.setattr(config, "LLM_MAX_RETRIES", 1)
    monkeypatch.setattr(agent_runner, "AsyncOpenAI", lambda **kwargs: fake_client)

    runner = agent_runner.AgentRunner(max_steps=3)
    action = await runner.step("server=degraded")

    assert action.tool_name == "restart_service"
    assert fake_client.calls == 2
    assert runner.total_tokens == 21


@pytest.mark.asyncio
async def test_agent_runner_parses_json_action_payload(monkeypatch):
    class FakeClient:
        def __init__(self):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        async def create(self, **kwargs):
            return FakeJSONCompletionResponse('{"tool_name":"restart_service","tool_args":{}}')

    fake_client = FakeClient()

    monkeypatch.setattr(agent_runner, "AsyncOpenAI", lambda **kwargs: fake_client)

    runner = agent_runner.AgentRunner(max_steps=3)
    action = await runner.step("server=degraded")

    assert action.tool_name == "restart_service"
    assert action.tool_args == {}
    assert runner.total_tokens == 13


@pytest.mark.asyncio
async def test_generate_branches_llm_uses_shared_rate_limiter(monkeypatch):
    calls = {"count": 0}

    async def fake_run_with_rate_limit(request_factory, **kwargs):
        calls["count"] += 1
        return await request_factory()

    class FakeClient:
        def __init__(self):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        async def create(self, **kwargs):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content="""
                            {
                              "branches": [
                                {
                                  "id": "branch_0",
                                  "label": "Baseline",
                                  "description": "Happy path",
                                  "is_baseline": true,
                                  "events": []
                                },
                                {
                                  "id": "branch_1",
                                  "label": "CPU Spike",
                                  "description": "CPU climbs during deploy",
                                  "is_baseline": false,
                                  "events": [
                                    {
                                      "step": 1,
                                      "name": "cpu_spike",
                                      "description": "CPU jumps",
                                      "effect_deltas": {"cpu_utilization": 95}
                                    }
                                  ]
                                }
                              ]
                            }
                            """
                        )
                    )
                ]
            )

    fake_client = FakeClient()

    monkeypatch.setattr(scenario_generator, "run_with_rate_limit", fake_run_with_rate_limit)
    monkeypatch.setattr(scenario_generator, "AsyncOpenAI", lambda **kwargs: fake_client)

    branches = await scenario_generator.generate_branches_llm(
        "Deploy the frontend",
        {"server_status": "healthy", "cpu_utilization": 40},
        n_branches=2,
    )

    assert calls["count"] == 1
    assert len(branches) == 2
    assert branches[0].is_baseline is True
    assert branches[1].events[0].name == "cpu_spike"
