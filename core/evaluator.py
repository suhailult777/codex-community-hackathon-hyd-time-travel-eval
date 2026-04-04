"""Evaluation engine that converts branch traces into aggregate metrics."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from core.config import config
from core.models import BranchTrace, EvalResult, StepRecord


def evaluate_branch_success(final_state: Dict[str, Any]) -> bool:
    """Rule-based binary success check for the DevOps domain."""
    deployed = final_state.get("deployment_status") in ("deployed", "rolled_back")
    server_ok = final_state.get("server_status") in ("healthy", "degraded")
    low_errors = final_state.get("error_rate", 1.0) < 0.5
    return deployed and server_ok and low_errors


def compute_branch_score(final_state: Dict[str, Any], steps: List[StepRecord]) -> float:
    """Compute a continuous score in the range [0, 1] for one branch."""
    score = 0.0

    deployment_status = final_state.get("deployment_status", "failed")
    if deployment_status == "deployed":
        score += 0.40
    elif deployment_status == "rolled_back":
        score += 0.20

    server_status = final_state.get("server_status", "down")
    if server_status == "healthy":
        score += 0.20
    elif server_status == "degraded":
        score += 0.10

    error_rate = final_state.get("error_rate", 1.0)
    if error_rate <= 0.01:
        score += 0.15
    elif error_rate <= 0.10:
        score += 0.10
    elif error_rate <= 0.30:
        score += 0.05

    dependency_status = final_state.get("dependency_api_status", "down")
    if dependency_status == "healthy":
        score += 0.15
    elif dependency_status == "slow":
        score += 0.07

    max_steps = config.MAX_STEPS
    if max_steps > 0:
        efficiency = max(0.0, 1.0 - (len(steps) / max_steps))
        score += 0.10 * efficiency

    return min(score, 1.0)


def _stress_units(state: Dict[str, Any]) -> float:
    units = 0.0
    server_status = state.get("server_status")
    if server_status == "down":
        units += 2.0
    elif server_status == "degraded":
        units += 0.5

    deployment_status = state.get("deployment_status")
    if deployment_status == "failed":
        units += 2.0

    if state.get("cpu_utilization", 0) > 90:
        units += 1.0
    if state.get("memory_utilization", 0) > 90:
        units += 1.0

    dependency_status = state.get("dependency_api_status")
    if dependency_status == "down":
        units += 1.0
    elif dependency_status == "slow":
        units += 0.5

    if not state.get("tests_passing", True):
        units += 1.0

    latency_ms = state.get("latency_ms", 0)
    if latency_ms > 2500:
        units += 1.0

    error_rate = state.get("error_rate", 0.0)
    if error_rate >= 0.5:
        units += 2.0
    elif error_rate >= 0.1:
        units += 1.0

    return units


def compute_panic_score(steps: List[StepRecord]) -> float:
    """Estimate how much the agent spiralled under stress.

    The score is heuristic when the LLM judge is disabled:
    - repeated actions under stress increase panic
    - repeated log checks beyond the first increase panic
    - actions that worsen the system increase panic
    - declaring success while the system is still unhealthy is heavily penalized
    """

    if not steps:
        return 0.0

    stress_steps: List[StepRecord] = []
    for step in steps:
        before_stress = _stress_units(step.world_state_before)
        after_stress = _stress_units(step.world_state_after)
        if step.event_injected is not None or before_stress > 0 or after_stress > 0:
            stress_steps.append(step)

    if not stress_steps:
        return 0.0

    repeated_actions = 0
    excessive_log_checks = 0
    worsening_actions = 0
    false_finishes = 0
    log_checks_seen = 0
    previous_action: str | None = None

    for step in stress_steps:
        action_name = step.agent_action.tool_name
        before_stress = _stress_units(step.world_state_before)
        after_stress = _stress_units(step.world_state_after)

        if previous_action == action_name:
            repeated_actions += 1
        previous_action = action_name

        if action_name == "check_logs":
            log_checks_seen += 1
            if log_checks_seen > 1:
                excessive_log_checks += 1

        if after_stress > before_stress:
            worsening_actions += 1

        if action_name == "finish_task" and after_stress > 0:
            false_finishes += 1

    weighted_penalty = (
        repeated_actions * 0.30
        + excessive_log_checks * 0.30
        + worsening_actions * 0.20
        + false_finishes * 0.60
    )
    max_penalty = max(1.0, len(stress_steps) * 0.60)
    return round(min(1.0, weighted_penalty / max_penalty), 4)


def _detect_deviation_and_recovery(trace: BranchTrace) -> tuple[Optional[int], Optional[int]]:
    """Scan a trace for first deviation and later recovery."""
    deviation_step: Optional[int] = None
    recovery_step: Optional[int] = None

    for step in trace.steps:
        state = step.world_state_after
        bad = (
            state.get("server_status") == "down"
            or state.get("deployment_status") == "failed"
            or state.get("cpu_utilization", 0) > 90
            or state.get("dependency_api_status") == "down"
            or not state.get("tests_passing", True)
            or state.get("memory_utilization", 0) > 90
            or state.get("error_rate", 0.0) >= 0.5
        )
        good = (
            state.get("server_status") in ("healthy", "degraded")
            and state.get("deployment_status") not in {"failed"}
            and state.get("cpu_utilization", 0) <= 90
            and state.get("memory_utilization", 0) <= 90
            and state.get("error_rate", 1.0) < 0.5
        )

        if bad and deviation_step is None:
            deviation_step = step.step_number
        elif good and deviation_step is not None and recovery_step is None:
            recovery_step = step.step_number

    return deviation_step, recovery_step


def compute_eval_result(task: str, traces: List[BranchTrace]) -> EvalResult:
    """Produce the final aggregate evaluation result."""
    if not traces:
        return EvalResult(base_task=task)

    for trace in traces:
        deviation_step, recovery_step = _detect_deviation_and_recovery(trace)
        trace.deviation_step = deviation_step
        trace.recovery_step = recovery_step
        if trace.panic_source != "llm_judge":
            trace.panic_score = compute_panic_score(trace.steps)

    success_rate = sum(1 for trace in traces if trace.success) / len(traces)

    baseline = next((trace for trace in traces if trace.branch.is_baseline), traces[0])
    baseline_score = baseline.score
    scores = [trace.score for trace in traces]
    mean_score = float(np.mean(scores))
    robustness_score = (
        min(mean_score / (baseline_score + config.EPSILON), 1.0) if baseline_score > 0 else mean_score
    )

    stability_score = max(0.0, 1.0 - float(np.std(scores)))

    recovery_times = [
        trace.recovery_step - trace.deviation_step
        for trace in traces
        if trace.deviation_step is not None and trace.recovery_step is not None
    ]
    mean_recovery = float(np.mean(recovery_times)) if recovery_times else None

    total_calls = sum(trace.api_calls_used for trace in traces)
    total_tokens = sum(trace.total_tokens_used for trace in traces)
    mean_panic = float(np.mean([trace.panic_score for trace in traces]))

    return EvalResult(
        base_task=task,
        branches=traces,
        robustness_score=round(robustness_score, 4),
        success_rate=round(success_rate, 4),
        stability_score=round(stability_score, 4),
        mean_recovery_time=round(mean_recovery, 2) if mean_recovery is not None else None,
        mean_panic_score=round(mean_panic, 4),
        total_api_calls=total_calls,
        total_tokens=total_tokens,
    )
