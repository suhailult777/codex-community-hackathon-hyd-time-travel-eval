"""TTE main orchestrator."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from core.config import config
from core.evaluator import compute_branch_score, compute_eval_result, evaluate_branch_success
from core.logger import (
    _safe_console_text,
    _should_emit_console_output,
    print_branch_result,
    print_eval_summary,
    print_header,
    print_step,
)
from core.models import AgentAction, BranchTrace, EvalResult, ScenarioBranch, StepRecord
from core.scenario_generator import generate_branches
from core.simulator import DEFAULT_DEVOPS_STATE, WorldSimulator

DATA_DIR = Path(__file__).parent / "data"


def load_test_cases() -> List[Dict[str, Any]]:
    path = DATA_DIR / "test_cases.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return []


async def run_branch(
    branch: ScenarioBranch,
    task: str,
    env_state: Dict[str, Any],
    max_steps: int,
    use_llm_agent: bool = True,
) -> BranchTrace:
    """Execute one branch simulation."""
    simulator = WorldSimulator(default_state=env_state, branch=branch)
    steps: List[StepRecord] = []

    if use_llm_agent:
        from core.agent_runner import AgentRunner

        agent = AgentRunner(max_steps=max_steps)
    else:
        agent = None

    for step_number in range(max_steps):
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

    if agent and config.ENABLE_LLM_JUDGE:
        from core.llm_judge import llm_judge_score

        judged_score, judged_panic, explanation = await llm_judge_score(task, trace)
        trace.score = judged_score
        trace.panic_score = judged_panic
        trace.panic_source = "llm_judge"
        trace.judge_explanation = explanation
        trace.api_calls_used += 1

    return trace


async def run_tte(
    task: str,
    env_state: Dict[str, Any] | None = None,
    n_branches: int | None = None,
    max_steps: int | None = None,
    use_llm: bool = True,
    use_llm_agent: bool = True,
) -> EvalResult:
    """Run the full TTE pipeline end to end."""
    env = env_state or DEFAULT_DEVOPS_STATE.copy()
    n = n_branches or config.MAX_BRANCHES
    steps = max_steps or config.MAX_STEPS

    print_header(f"Time-Travel Evals - Task: {task}")

    print_header("Phase 1: Generating Scenario Branches")
    branches = await generate_branches(task, env, n_branches=n, use_llm=use_llm)
    if _should_emit_console_output():
        for branch in branches:
            label = branch.label or branch.description[:60]
            prefix = "BASE" if branch.is_baseline else "ALT"
            print(f"  {prefix} {_safe_console_text(branch.id)}: {_safe_console_text(label)}")

    print_header("Phase 2: Running Agent on Each Branch")
    traces: List[BranchTrace] = await asyncio.gather(
        *[run_branch(branch, task, env, steps, use_llm_agent=use_llm_agent) for branch in branches]
    )

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
    if use_llm:
        result.total_api_calls += 1
    print_eval_summary(result.robustness_score, result.success_rate, result.stability_score)
    return result


def main() -> EvalResult:
    parser = argparse.ArgumentParser(description="Time-Travel Evals (TTE)")
    parser.add_argument("--task", type=str, default="Deploy version 2 of the frontend application to production")
    parser.add_argument("--branches", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--demo", action="store_true", help="Rule-based branches, real LLM agent")
    parser.add_argument("--rules-only", action="store_true", help="No provider calls at all")
    parser.add_argument("--output", type=str, default=None, help="Save results JSON to this path")
    args = parser.parse_args()

    use_llm = not args.demo and not args.rules_only
    use_llm_agent = not args.rules_only

    if use_llm_agent and not config.validate():
        print("NVIDIA_API_KEY (or KIMI/OPENAI_API_KEY) not set. Use --rules-only for offline mode.")
        sys.exit(1)

    result = asyncio.run(
        run_tte(
            task=args.task,
            n_branches=args.branches,
            max_steps=args.steps,
            use_llm=use_llm,
            use_llm_agent=use_llm_agent,
        )
    )

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        print(f"\nResults saved to {_safe_console_text(str(output_path))}")

    return result


if __name__ == "__main__":
    main()
