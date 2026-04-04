"""TTE main orchestrator."""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from core.config import config
from core.evaluator import compute_branch_score, compute_eval_result, evaluate_branch_success
from core.execution import ExecutionContext, build_execution_context
from core.logger import (
    _safe_console_text,
    _should_emit_console_output,
    print_branch_result,
    print_eval_summary,
    print_header,
    print_step,
)
from core.models import AgentAction, BranchTrace, EvalResult, ScenarioBranch, StepRecord
from core.runtime_budget import estimate_total_api_calls
from core.scenario_generator import generate_branches
from core.simulator import DEFAULT_DEVOPS_STATE, WorldSimulator

DATA_DIR = Path(__file__).parent / "data"


def load_test_cases() -> List[Dict[str, Any]]:
    path = DATA_DIR / "test_cases.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return []


def _first_divergence_step(branch: ScenarioBranch, max_steps: int) -> int:
    if branch.initial_state_overrides:
        return 0
    if branch.events:
        return min(event.step for event in branch.events)
    return max_steps


def _clone_steps(steps: List[StepRecord], count: int) -> List[StepRecord]:
    return [step.model_copy(deep=True) for step in steps[:count]]


def _build_warm_start(
    branch: ScenarioBranch,
    *,
    baseline_trace: BranchTrace,
    simulator_snapshots: List[Dict[str, Any]],
    agent_snapshots: List[Dict[str, Any]],
    max_steps: int,
) -> dict[str, Any] | None:
    if branch.is_baseline or branch.initial_state_overrides:
        return None

    prefix_len = min(_first_divergence_step(branch, max_steps), len(baseline_trace.steps))
    if prefix_len <= 0:
        return None

    completed = prefix_len >= len(baseline_trace.steps)
    return {
        "steps": _clone_steps(baseline_trace.steps, prefix_len),
        "simulator_snapshot": copy.deepcopy(simulator_snapshots[prefix_len]),
        "agent_snapshot": copy.deepcopy(agent_snapshots[prefix_len]) if agent_snapshots else None,
        "completed": completed,
    }


async def _execute_branch(
    branch: ScenarioBranch,
    task: str,
    env_state: Dict[str, Any],
    max_steps: int,
    *,
    use_llm_agent: bool,
    execution_context: ExecutionContext | None,
    warm_start: dict[str, Any] | None = None,
    capture_snapshots: bool = False,
) -> tuple[BranchTrace, List[Dict[str, Any]], List[Dict[str, Any]]]:
    simulator = WorldSimulator(default_state=env_state, branch=branch)
    if use_llm_agent:
        from core.agent_runner import AgentRunner

        agent = AgentRunner(max_steps=max_steps, execution_context=execution_context)
    else:
        agent = None

    steps: List[StepRecord] = []
    simulator_snapshots: List[Dict[str, Any]] = [simulator.snapshot()] if capture_snapshots else []
    agent_snapshots: List[Dict[str, Any]] = [agent.snapshot()] if capture_snapshots and agent else []

    if warm_start:
        simulator.restore(warm_start["simulator_snapshot"])
        if agent and warm_start.get("agent_snapshot") is not None:
            agent.restore(warm_start["agent_snapshot"])
        steps = list(warm_start.get("steps", []))

        if warm_start.get("completed"):
            final_state = simulator.get_final_state()
            trace = BranchTrace(
                branch=branch,
                steps=steps,
                final_state=final_state,
                success=evaluate_branch_success(final_state),
                score=compute_branch_score(final_state, steps),
                api_calls_used=len(steps) if agent else 0,
                total_tokens_used=agent.total_tokens if agent else 0,
            )
            return trace, simulator_snapshots, agent_snapshots

        if capture_snapshots:
            simulator_snapshots = [copy.deepcopy(warm_start["simulator_snapshot"])]
            if agent and warm_start.get("agent_snapshot") is not None:
                agent_snapshots = [copy.deepcopy(warm_start["agent_snapshot"])]

    for step_number in range(simulator.current_step, max_steps):
        observation = simulator.tick()
        state_before = simulator.get_final_state()

        if agent:
            action = await agent.step(observation)
            action_result = simulator.apply_action(action)
            agent.feed_tool_result(action_result)
        else:
            if step_number == 0:
                action = AgentAction(tool_name="run_tests", tool_args={})
            elif step_number == 1:
                action = AgentAction(tool_name="deploy", tool_args={})
            else:
                action = AgentAction(tool_name="finish_task", tool_args={})
            simulator.apply_action(action)

        steps.append(
            StepRecord(
                step_number=step_number,
                world_state_before=state_before,
                event_injected=simulator.last_event,
                observation_text=observation,
                agent_action=action,
                world_state_after=simulator.get_final_state(),
            )
        )

        if capture_snapshots:
            simulator_snapshots.append(simulator.snapshot())
            if agent:
                agent_snapshots.append(agent.snapshot())

        if action.tool_name == "finish_task" or simulator.is_terminal():
            break

    final_state = simulator.get_final_state()
    trace = BranchTrace(
        branch=branch,
        steps=steps,
        final_state=final_state,
        success=evaluate_branch_success(final_state),
        score=compute_branch_score(final_state, steps),
        api_calls_used=len(steps) if agent else 0,
        total_tokens_used=agent.total_tokens if agent else 0,
    )
    return trace, simulator_snapshots, agent_snapshots


async def _run_branches_standard(
    branches: List[ScenarioBranch],
    task: str,
    env_state: Dict[str, Any],
    max_steps: int,
    *,
    use_llm_agent: bool,
    execution_context: ExecutionContext | None,
) -> List[BranchTrace]:
    if not branches:
        return []

    baseline_index = next((index for index, branch in enumerate(branches) if branch.is_baseline), 0)
    baseline_branch = branches[baseline_index]
    baseline_trace, simulator_snapshots, agent_snapshots = await _execute_branch(
        baseline_branch,
        task,
        env_state,
        max_steps,
        use_llm_agent=use_llm_agent,
        execution_context=execution_context,
        capture_snapshots=use_llm_agent,
    )

    traces_by_id = {baseline_branch.id: baseline_trace}
    remaining_branches = [branch for branch in branches if branch.id != baseline_branch.id]
    remaining_results = await asyncio.gather(
        *[
            _execute_branch(
                branch,
                task,
                env_state,
                max_steps,
                use_llm_agent=use_llm_agent,
                execution_context=execution_context,
                warm_start=(
                    _build_warm_start(
                        branch,
                        baseline_trace=baseline_trace,
                        simulator_snapshots=simulator_snapshots,
                        agent_snapshots=agent_snapshots,
                        max_steps=max_steps,
                    )
                    if use_llm_agent
                    else None
                ),
            )
            for branch in remaining_branches
        ]
    )

    for trace, _sim_snapshots, _agent_snapshots in remaining_results:
        traces_by_id[trace.branch.id] = trace

    return [traces_by_id[branch.id] for branch in branches]


async def _run_branches_turbo(
    branches: List[ScenarioBranch],
    env_state: Dict[str, Any],
    max_steps: int,
    *,
    execution_context: ExecutionContext,
) -> List[BranchTrace]:
    from core.agent_runner import AgentRunner, run_batched_step

    simulators = {branch.id: WorldSimulator(default_state=env_state, branch=branch) for branch in branches}
    agents = {
        branch.id: AgentRunner(max_steps=max_steps, execution_context=execution_context)
        for branch in branches
    }
    traces: dict[str, List[StepRecord]] = {branch.id: [] for branch in branches}
    active_branch_ids = {branch.id for branch in branches}

    for _ in range(max_steps):
        branch_items: List[dict[str, Any]] = []
        observations: dict[str, str] = {}
        states_before: dict[str, dict[str, Any]] = {}
        active_order: List[str] = []

        for branch in branches:
            branch_id = branch.id
            if branch_id not in active_branch_ids:
                continue
            simulator = simulators[branch_id]
            if simulator.is_terminal():
                active_branch_ids.discard(branch_id)
                continue

            observation = simulator.tick()
            observations[branch_id] = observation
            states_before[branch_id] = simulator.get_final_state()
            branch_items.append(agents[branch_id].build_batch_item(branch_id, observation))
            active_order.append(branch_id)

        if not branch_items:
            break

        batched_payloads: dict[str, dict[str, Any]] = {}
        try:
            batched_payloads = await run_batched_step(
                branch_items=branch_items,
                execution_context=execution_context,
            )
        except Exception:
            batched_payloads = {}

        for item in branch_items:
            branch_id = item["branch_id"]
            runner = agents[branch_id]
            simulator = simulators[branch_id]

            if branch_id in batched_payloads:
                payload = batched_payloads[branch_id]
                action = AgentRunner._payload_to_action(payload)
                runner.commit_batched_action(
                    item["user_message"],
                    action,
                    usage_total_tokens=int(payload.get("usage_total_tokens", 0)),
                )
            else:
                action = await runner.step(observations[branch_id])

            action_result = simulator.apply_action(action)
            runner.feed_tool_result(action_result)

            traces[branch_id].append(
                StepRecord(
                    step_number=len(traces[branch_id]),
                    world_state_before=states_before[branch_id],
                    event_injected=simulator.last_event,
                    observation_text=observations[branch_id],
                    agent_action=action,
                    world_state_after=simulator.get_final_state(),
                )
            )

            if action.tool_name == "finish_task" or simulator.is_terminal():
                active_branch_ids.discard(branch_id)

    result_traces: List[BranchTrace] = []
    for branch in branches:
        simulator = simulators[branch.id]
        runner = agents[branch.id]
        branch_steps = traces[branch.id]
        final_state = simulator.get_final_state()
        result_traces.append(
            BranchTrace(
                branch=branch,
                steps=branch_steps,
                final_state=final_state,
                success=evaluate_branch_success(final_state),
                score=compute_branch_score(final_state, branch_steps),
                api_calls_used=len(branch_steps),
                total_tokens_used=runner.total_tokens,
            )
        )
    return result_traces


async def _apply_llm_judging(
    task: str,
    traces: List[BranchTrace],
    execution_context: ExecutionContext | None,
) -> None:
    if not traces:
        return

    from core.llm_judge import llm_judge_score_many

    results = await llm_judge_score_many(task, traces, execution_context=execution_context)
    for trace in traces:
        score, panic_score, explanation = results[trace.branch.id]
        trace.score = score
        trace.panic_score = panic_score
        trace.panic_source = "llm_judge"
        trace.judge_explanation = explanation


def _compute_logical_calls(*, traces: List[BranchTrace], use_llm: bool, use_llm_agent: bool) -> int:
    logical_calls = 1 if use_llm else 0
    if use_llm_agent:
        logical_calls += sum(len(trace.steps) for trace in traces)
        if config.ENABLE_LLM_JUDGE:
            logical_calls += len(traces)
    return logical_calls


async def run_tte(
    task: str,
    env_state: Dict[str, Any] | None = None,
    n_branches: int | None = None,
    max_steps: int | None = None,
    use_llm: bool = True,
    use_llm_agent: bool = True,
    provider: str | None = None,
) -> EvalResult:
    """Run the full TTE pipeline end to end."""
    env = env_state or DEFAULT_DEVOPS_STATE.copy()
    n = n_branches or config.MAX_BRANCHES
    steps = max_steps or config.MAX_STEPS
    logical_call_estimate = estimate_total_api_calls(
        n_branches=n,
        max_steps=steps,
        use_llm=use_llm,
        use_llm_agent=use_llm_agent,
        use_llm_judge=use_llm_agent and config.ENABLE_LLM_JUDGE,
    )
    execution_context = build_execution_context(
        logical_calls=logical_call_estimate,
        use_llm_agent=use_llm_agent,
        provider=provider,
    )

    print_header(f"Time-Travel Evals - Task: {task}")

    print_header("Phase 1: Generating Scenario Branches")
    branches = await generate_branches(
        task,
        env,
        n_branches=n,
        use_llm=use_llm,
        execution_context=execution_context if use_llm else None,
    )
    if _should_emit_console_output():
        for branch in branches:
            label = branch.label or branch.description[:60]
            prefix = "BASE" if branch.is_baseline else "ALT"
            print(f"  {prefix} {_safe_console_text(branch.id)}: {_safe_console_text(label)}")

    print_header("Phase 2: Running Agent on Each Branch")
    if use_llm_agent and execution_context.execution_mode == "turbo":
        traces = await _run_branches_turbo(
            branches,
            env,
            steps,
            execution_context=execution_context,
        )
    else:
        traces = await _run_branches_standard(
            branches,
            task,
            env,
            steps,
            use_llm_agent=use_llm_agent,
            execution_context=execution_context if use_llm_agent else None,
        )

    if use_llm_agent and config.ENABLE_LLM_JUDGE:
        await _apply_llm_judging(task, traces, execution_context if use_llm else None)

    for trace in traces:
        print_branch_result(trace.branch.id, trace.success, trace.score)
        for step in trace.steps:
            print_step(
                trace.branch.id,
                step.step_number,
                step.observation_text[:80],
                step.agent_action.tool_name,
            )

    print_header("Phase 3: Computing Evaluation Metrics")
    result = compute_eval_result(task, traces)
    result.logical_calls = _compute_logical_calls(traces=traces, use_llm=use_llm, use_llm_agent=use_llm_agent)
    result.total_api_calls = result.logical_calls
    result.scheduled_provider_calls = execution_context.stats.scheduled_provider_calls
    result.cache_hits = execution_context.stats.cache_hits
    result.cache_misses = execution_context.stats.cache_misses
    result.execution_mode = execution_context.execution_mode
    result.provider_name = execution_context.provider_name
    result.provider_profile = execution_context.agent_capabilities.provider_profile
    result.effective_rpm = execution_context.effective_rpm
    result.provider_base_url = execution_context.base_url
    result.agent_model = execution_context.agent_model
    result.generator_model = execution_context.generator_model
    print_eval_summary(result.robustness_score, result.success_rate, result.stability_score)
    return result


def main() -> EvalResult:
    parser = argparse.ArgumentParser(description="Time-Travel Evals (TTE)")
    parser.add_argument("--task", type=str, default="Deploy version 2 of the frontend application to production")
    parser.add_argument("--branches", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--provider", type=str, choices=config.provider_options(), default=None)
    parser.add_argument("--demo", action="store_true", help="Rule-based branches, real LLM agent")
    parser.add_argument("--rules-only", action="store_true", help="No provider calls at all")
    parser.add_argument("--output", type=str, default=None, help="Save results JSON to this path")
    args = parser.parse_args()

    use_llm = not args.demo and not args.rules_only
    use_llm_agent = not args.rules_only

    provider_settings = config.get_provider_settings(args.provider)
    if use_llm_agent and not config.validate(args.provider):
        print(
            f"{provider_settings.api_key_env_var} is not set for {provider_settings.label}. "
            "Use --rules-only for offline mode."
        )
        sys.exit(1)

    result = asyncio.run(
        run_tte(
            task=args.task,
            n_branches=args.branches,
            max_steps=args.steps,
            use_llm=use_llm,
            use_llm_agent=use_llm_agent,
            provider=args.provider,
        )
    )

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        print(f"\nResults saved to {_safe_console_text(str(output_path))}")

    return result


if __name__ == "__main__":
    main()
