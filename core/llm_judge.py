"""Optional LLM-as-judge scoring for open-ended tasks."""

from __future__ import annotations

import json
from typing import Any, Iterable

from openai import AsyncOpenAI

from core.config import config
from core.evaluator import compute_panic_score
from core.execution import ExecutionContext, execute_with_cache
from core.llm_cache import build_cache_key
from core.llm_utils import run_with_rate_limit
from core.models import BranchTrace

_JUDGE_PROMPT = """\
You are an expert evaluator judging an AI agent's performance on a DevOps task.

The agent was given this task:
{task}

The scenario was:
{branch_description}

Here is the step-by-step trace of the agent's actions:
{trace_text}

Final system state:
{final_state}

Score the agent using two metrics from 0.0 to 1.0:
- score: overall task performance, balancing task completion, recovery, efficiency, and final system quality
- panic_score: how much the agent spiralled under pressure

panic_score should be high when the agent repeats useless actions, ignores contradictory evidence,
hallucinates success, or declares the task complete while the system is still broken.

Return only JSON:
{{"score": <float>, "panic_score": <float>, "explanation": "<1-2 sentences>"}}
"""

_BATCH_JUDGE_PROMPT = """\
You are evaluating multiple independent DevOps agent traces.

For each branch, return:
- branch_id
- score in [0.0, 1.0]
- panic_score in [0.0, 1.0]
- explanation with 1-2 short sentences

Return only JSON:
{
  "results": [
    {"branch_id": "branch_0", "score": 0.9, "panic_score": 0.1, "explanation": "..."}
  ]
}
"""


def _trace_text(trace: BranchTrace) -> str:
    lines = []
    for step in trace.steps:
        event_label = f" [EVENT: {step.event_injected.name}]" if step.event_injected else ""
        lines.append(
            f"Step {step.step_number}{event_label}: "
            f"obs='{step.observation_text[:120]}' | "
            f"action={step.agent_action.tool_name}({step.agent_action.tool_args})"
        )
    return "\n".join(lines) if lines else "(no steps recorded)"


def _fallback_payload(trace: BranchTrace) -> dict[str, Any]:
    return {
        "score": trace.score,
        "panic_score": compute_panic_score(trace.steps),
        "explanation": "LLM judge unavailable; using heuristic scoring.",
    }


def _normalize_judge_payload(payload: dict[str, Any], trace: BranchTrace) -> tuple[float, float, str]:
    fallback = _fallback_payload(trace)
    score = max(0.0, min(1.0, float(payload.get("score", fallback["score"]))))
    panic_score = max(0.0, min(1.0, float(payload.get("panic_score", fallback["panic_score"]))))
    explanation = str(payload.get("explanation", fallback["explanation"])).strip() or fallback["explanation"]
    return score, panic_score, explanation


async def llm_judge_score(
    task: str,
    trace: BranchTrace,
    execution_context: ExecutionContext | None = None,
) -> tuple[float, float, str]:
    """Use the configured model to score one branch for performance and panic."""
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

    prompt = _JUDGE_PROMPT.format(
        task=task,
        branch_description=trace.branch.description,
        trace_text=_trace_text(trace),
        final_state=json.dumps(trace.final_state, indent=2, default=str),
    )

    if execution_context is None:
        try:
            response = await run_with_rate_limit(
                lambda: client.chat.completions.create(
                    model=provider_settings.generator_model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.1,
                )
            )
            raw = response.choices[0].message.content or "{}"
            return _normalize_judge_payload(json.loads(raw), trace)
        except Exception:
            fallback = _fallback_payload(trace)
            return fallback["score"], fallback["panic_score"], fallback["explanation"]

    cache_key = build_cache_key(
        base_url=provider_settings.base_url,
        model=provider_settings.generator_model,
        provider_profile=execution_context.generator_capabilities.provider_profile,
        execution_mode=execution_context.execution_mode,
        prompt_version=f"{execution_context.prompt_version}:judge_single",
        purpose="judge.single",
        payload={
            "task": task,
            "branch_id": trace.branch.id,
            "branch_description": trace.branch.description,
            "trace_text": _trace_text(trace),
            "final_state": trace.final_state,
        },
    )

    def parser(response: Any) -> dict[str, Any]:
        raw = response.choices[0].message.content or "{}"
        return json.loads(raw)

    try:
        payload, _from_cache = await execute_with_cache(
            execution_context=execution_context,
            cache_key=cache_key,
            request_factory=lambda: client.chat.completions.create(
                model=provider_settings.generator_model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.1,
            ),
            parser=parser,
        )
        return _normalize_judge_payload(payload, trace)
    except Exception:
        fallback = _fallback_payload(trace)
        return fallback["score"], fallback["panic_score"], fallback["explanation"]


async def llm_judge_score_many(
    task: str,
    traces: Iterable[BranchTrace],
    execution_context: ExecutionContext | None = None,
) -> dict[str, tuple[float, float, str]]:
    """Score multiple traces, using batched judging when the provider supports it."""
    trace_list = list(traces)
    if not trace_list:
        return {}

    if (
        execution_context is None
        or not execution_context.generator_capabilities.supports_batched_judge_prompt
        or len(trace_list) == 1
    ):
        results: dict[str, tuple[float, float, str]] = {}
        for trace in trace_list:
            results[trace.branch.id] = await llm_judge_score(task, trace, execution_context=execution_context)
        return results

    client = AsyncOpenAI(
        api_key=execution_context.api_key,
        base_url=execution_context.base_url,
        timeout=getattr(config, "LLM_REQUEST_TIMEOUT_SECONDS", 20.0),
    )
    branch_payloads = [
        {
            "branch_id": trace.branch.id,
            "branch_description": trace.branch.description,
            "trace_text": _trace_text(trace),
            "final_state": trace.final_state,
            "fallback_score": trace.score,
            "fallback_panic_score": compute_panic_score(trace.steps),
        }
        for trace in trace_list
    ]
    cache_key = build_cache_key(
        base_url=execution_context.base_url,
        model=execution_context.generator_model,
        provider_profile=execution_context.generator_capabilities.provider_profile,
        execution_mode=execution_context.execution_mode,
        prompt_version=f"{execution_context.prompt_version}:judge_batch",
        purpose="judge.batch",
        payload={"task": task, "branches": branch_payloads},
    )

    def parser(response: Any) -> dict[str, Any]:
        raw = response.choices[0].message.content or "{}"
        return json.loads(raw)

    results: dict[str, tuple[float, float, str]] = {}
    missing = {trace.branch.id: trace for trace in trace_list}
    try:
        payload, _from_cache = await execute_with_cache(
            execution_context=execution_context,
            cache_key=cache_key,
            request_factory=lambda: client.chat.completions.create(
                model=execution_context.generator_model,
                messages=[
                    {"role": "system", "content": _BATCH_JUDGE_PROMPT},
                    {"role": "user", "content": json.dumps({"task": task, "branches": branch_payloads}, indent=2)},
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=max(256, 160 * len(trace_list)),
            ),
            parser=parser,
        )
        for item in payload.get("results", []):
            branch_id = item.get("branch_id")
            if branch_id not in missing:
                continue
            results[branch_id] = _normalize_judge_payload(item, missing[branch_id])
            missing.pop(branch_id, None)
    except Exception:
        pass

    for branch_id, trace in missing.items():
        results[branch_id] = await llm_judge_score(task, trace, execution_context=execution_context)

    return results
