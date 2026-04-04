"""Scenario generator for Time-Travel Evals."""

from __future__ import annotations

import json
from typing import Any, Dict, List

from openai import AsyncOpenAI

from core.config import config
from core.execution import ExecutionContext, execute_with_cache
from core.llm_cache import build_cache_key
from core.llm_utils import run_with_rate_limit
from core.models import Event, ScenarioBranch

_SYSTEM_PROMPT = """\
You are a scenario designer for stress-testing AI agents.

Given a BASE TASK and an ENVIRONMENT STATE SCHEMA, generate exactly {n} alternate
timeline scenarios (branches). Each scenario represents a what-if world the agent
must navigate.

Rules:
1. Exactly one branch must be the baseline (is_baseline=true) with zero perturbations.
2. The remaining branches should cover diverse failure modes such as:
   - RESOURCE_CONSTRAINT: CPU spike, memory pressure, disk full.
   - DEPENDENCY_FAILURE: A downstream service crashes or times out.
   - TIMING_ANOMALY: A delay, race condition, or out-of-order event.
   - HUMAN_INTERVENTION: User aborts or changes requirements mid-task.
   - BYZANTINE_FAULT: A control plane or tool reports success while the real system is unhealthy.
3. Each branch must have a unique id (branch_0, branch_1, ...) and distinct events.
4. Events must fire at step numbers between 0 and {max_step}.
5. effect_deltas must reference keys that exist in the environment state schema.

Return only valid JSON matching this schema:
{{
  "branches": [
    {{
      "id": "branch_0",
      "label": "Happy Path",
      "description": "...",
      "is_baseline": true,
      "events": [],
      "initial_state_overrides": {{}}
    }}
  ]
}}
"""

_USER_PROMPT = """\
BASE TASK: {task}

ENVIRONMENT STATE SCHEMA (available keys and sample values):
{schema_json}

Generate {n} branches now.
"""


async def generate_branches_llm(
    task: str,
    env_schema: Dict[str, Any],
    n_branches: int | None = None,
    execution_context: ExecutionContext | None = None,
) -> List[ScenarioBranch]:
    """Use the configured LLM to generate creative scenario branches."""
    n = n_branches or config.MAX_BRANCHES
    provider_settings = (
        config.get_provider_settings(execution_context.provider_name)
        if execution_context is not None
        else config.get_provider_settings()
    )
    client = AsyncOpenAI(
        api_key=provider_settings.api_key,
        base_url=provider_settings.base_url,
        timeout=getattr(config, "LLM_REQUEST_TIMEOUT_SECONDS", 20.0),
    )

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT.format(n=n, max_step=config.MAX_STEPS - 1)},
        {
            "role": "user",
            "content": _USER_PROMPT.format(
                task=task,
                schema_json=json.dumps(env_schema, indent=2),
                n=n,
            ),
        },
    ]

    if execution_context is None:
        response = await run_with_rate_limit(
            lambda: client.chat.completions.create(
                model=provider_settings.generator_model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.8,
            )
        )
        raw = response.choices[0].message.content or "{}"
        data = json.loads(raw)
    else:
        cache_key = build_cache_key(
            base_url=provider_settings.base_url,
            model=provider_settings.generator_model,
            provider_profile=execution_context.generator_capabilities.provider_profile,
            execution_mode=execution_context.execution_mode,
            prompt_version=f"{execution_context.prompt_version}:scenario_generation",
            purpose="scenario.generate",
            payload={"task": task, "env_schema": env_schema, "n_branches": n},
        )

        def parser(response: Any) -> dict[str, Any]:
            raw = response.choices[0].message.content or "{}"
            return {"raw": raw}

        payload, _from_cache = await execute_with_cache(
            execution_context=execution_context,
            cache_key=cache_key,
            request_factory=lambda: client.chat.completions.create(
                model=provider_settings.generator_model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.8,
            ),
            parser=parser,
        )
        data = json.loads(payload["raw"])
    return _validate_branches([ScenarioBranch(**branch) for branch in data.get("branches", [])], n)


def generate_branches_rules(
    task: str,
    env_schema: Dict[str, Any],
    n_branches: int | None = None,
) -> List[ScenarioBranch]:
    """Deterministic branch generation with no API calls."""
    _ = env_schema
    n = n_branches or config.MAX_BRANCHES

    templates: List[ScenarioBranch] = [
        ScenarioBranch(
            id="branch_0",
            label="Happy Path (Baseline)",
            description=f"Baseline scenario: {task} under ideal conditions with no perturbations.",
            is_baseline=True,
            events=[],
            initial_state_overrides={},
        ),
        ScenarioBranch(
            id="branch_1",
            label="CPU Spike",
            description="A sudden CPU spike at step 2 threatens the deployment.",
            events=[
                Event(
                    step=2,
                    name="cpu_spike",
                    description="CPU utilization surges to 95% due to a runaway process.",
                    effect_deltas={"cpu_utilization": 95, "server_status": "degraded"},
                )
            ],
        ),
        ScenarioBranch(
            id="branch_2",
            label="Dependency Crash",
            description="The downstream dependency API crashes at step 1 and refuses connections.",
            events=[
                Event(
                    step=1,
                    name="dependency_crash",
                    description="Dependency API goes down and calls begin to fail.",
                    effect_deltas={"dependency_api_status": "down", "latency_ms": 5000},
                )
            ],
        ),
        ScenarioBranch(
            id="branch_3",
            label="Network Latency",
            description="Cross-zone latency spikes at step 3, causing timing issues and timeouts.",
            events=[
                Event(
                    step=3,
                    name="latency_spike",
                    description="Network latency jumps to 3000ms and requests start timing out.",
                    effect_deltas={"latency_ms": 3000, "dependency_api_status": "slow"},
                )
            ],
        ),
        ScenarioBranch(
            id="branch_4",
            label="Test Suite Failure",
            description="At step 1, the test suite starts failing because of a flaky integration test.",
            events=[
                Event(
                    step=1,
                    name="test_failure",
                    description="Integration tests begin failing and block a safe deployment.",
                    effect_deltas={"tests_passing": False},
                )
            ],
        ),
        ScenarioBranch(
            id="branch_5",
            label="Server Outage",
            description="The production server goes down at step 2.",
            events=[
                Event(
                    step=2,
                    name="server_outage",
                    description="Production server is unresponsive and error rate spikes.",
                    effect_deltas={"server_status": "down", "error_rate": 1.0},
                )
            ],
        ),
        ScenarioBranch(
            id="branch_6",
            label="Byzantine Deploy",
            description="The deploy control plane lies: it reports success while the service stays broken.",
            events=[
                Event(
                    step=1,
                    name="byzantine_deploy_lie",
                    description="Deploy status checks can now report success even if the service is still down.",
                    effect_deltas={"byzantine_deploy_lie": True},
                )
            ],
        ),
    ]

    selected = templates[: min(n, len(templates))]
    for index, branch in enumerate(selected):
        selected[index] = branch.model_copy(update={"id": f"branch_{index}"})
    return selected


def _validate_branches(branches: List[ScenarioBranch], expected: int) -> List[ScenarioBranch]:
    """Ensure exactly one baseline, unique IDs, and valid event steps."""
    seen_ids: set[str] = set()
    unique: List[ScenarioBranch] = []
    for branch in branches:
        if branch.id not in seen_ids:
            seen_ids.add(branch.id)
            unique.append(branch)
    branches = unique

    baselines = [branch for branch in branches if branch.is_baseline]
    if not baselines and branches:
        branches[0] = branches[0].model_copy(update={"is_baseline": True})
    elif len(baselines) > 1:
        for branch in baselines[1:]:
            idx = branches.index(branch)
            branches[idx] = branch.model_copy(update={"is_baseline": False})

    for branch in branches:
        for event in branch.events:
            if event.step >= config.MAX_STEPS:
                event.step = config.MAX_STEPS - 1

    return branches[:expected]


async def generate_branches(
    task: str,
    env_schema: Dict[str, Any],
    n_branches: int | None = None,
    use_llm: bool = True,
    execution_context: ExecutionContext | None = None,
) -> List[ScenarioBranch]:
    """Generate scenario branches, using the LLM when available and allowed."""
    provider_settings = (
        config.get_provider_settings(execution_context.provider_name)
        if execution_context is not None
        else config.get_provider_settings()
    )
    if use_llm and provider_settings.api_key:
        try:
            return await generate_branches_llm(task, env_schema, n_branches, execution_context=execution_context)
        except Exception:
            pass
    return generate_branches_rules(task, env_schema, n_branches)
