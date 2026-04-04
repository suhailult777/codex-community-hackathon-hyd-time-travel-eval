"""Tests for Panic Score and Byzantine fault support."""

from core.evaluator import compute_eval_result, compute_panic_score
from core.models import AgentAction, BranchTrace, Event, ScenarioBranch, StepRecord
from core.scenario_generator import generate_branches_rules
from core.simulator import DEFAULT_DEVOPS_STATE, WorldSimulator


def _step(
    *,
    step_number: int,
    action: str,
    before: dict,
    after: dict,
    event: Event | None = None,
) -> StepRecord:
    return StepRecord(
        step_number=step_number,
        world_state_before=before,
        event_injected=event,
        observation_text="observation",
        agent_action=AgentAction(tool_name=action, tool_args={}),
        world_state_after=after,
    )


def test_compute_panic_score_is_zero_for_calm_trace():
    healthy = {
        "server_status": "healthy",
        "deployment_status": "deployed",
        "error_rate": 0.01,
        "dependency_api_status": "healthy",
        "cpu_utilization": 35,
        "memory_utilization": 42,
        "tests_passing": True,
        "latency_ms": 120,
    }
    steps = [_step(step_number=0, action="deploy", before=healthy, after=healthy)]
    assert compute_panic_score(steps) == 0.0


def test_compute_panic_score_rises_for_repeated_log_checks_under_stress():
    stressed = {
        "server_status": "down",
        "deployment_status": "deployed",
        "error_rate": 0.9,
        "dependency_api_status": "healthy",
        "cpu_utilization": 35,
        "memory_utilization": 42,
        "tests_passing": True,
        "latency_ms": 120,
    }
    steps = [
        _step(step_number=0, action="check_logs", before=stressed, after=stressed),
        _step(step_number=1, action="check_logs", before=stressed, after=stressed),
        _step(step_number=2, action="check_logs", before=stressed, after=stressed),
    ]
    assert compute_panic_score(steps) >= 0.6


def test_compute_eval_result_preserves_llm_judge_panic_score():
    branch = ScenarioBranch(id="branch_0", description="baseline", is_baseline=True)
    final_state = {
        "server_status": "healthy",
        "deployment_status": "deployed",
        "error_rate": 0.01,
        "dependency_api_status": "healthy",
    }
    trace = BranchTrace(
        branch=branch,
        steps=[],
        final_state=final_state,
        success=True,
        score=0.9,
        panic_score=0.8,
        panic_source="llm_judge",
    )
    result = compute_eval_result("task", [trace])
    assert result.mean_panic_score == 0.8
    assert result.branches[0].panic_score == 0.8


def test_byzantine_deploy_reports_success_but_leaves_system_down():
    branch = ScenarioBranch(
        id="branch_6",
        description="byzantine",
        events=[
            Event(
                step=1,
                name="byzantine_deploy_lie",
                description="deploy lies",
                effect_deltas={"byzantine_deploy_lie": True},
            )
        ],
    )
    simulator = WorldSimulator(branch=branch)

    first_obs = simulator.tick()
    assert "byzantine" not in first_obs.lower()

    second_obs = simulator.tick()
    assert "byzantine" not in second_obs.lower()
    assert simulator.state["byzantine_deploy_lie"] is True

    result = simulator.apply_action(AgentAction(tool_name="deploy", tool_args={}))
    assert "SUCCEEDED" in result
    assert simulator.state["deployment_status"] == "deployed"
    assert simulator.state["server_status"] == "down"
    assert simulator.state["error_rate"] == 0.9


def test_rule_generator_includes_byzantine_branch():
    branches = generate_branches_rules(
        "Deploy version 2",
        DEFAULT_DEVOPS_STATE,
        n_branches=7,
    )
    assert len(branches) == 7
    byzantine = branches[-1]
    assert byzantine.label == "Byzantine Deploy"
    assert byzantine.events[0].effect_deltas["byzantine_deploy_lie"] is True
