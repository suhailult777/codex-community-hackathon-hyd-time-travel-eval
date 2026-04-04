"""Tests for the interactive multiverse DAG builder."""

from core.models import AgentAction, BranchTrace, EvalResult, Event, ScenarioBranch, StepRecord
from ui.components.tree_view import build_multiverse_dag, build_multiverse_dot


def _step(
    step_number: int,
    tool_name: str,
    *,
    event: Event | None = None,
    server_status: str = "healthy",
    deployment_status: str = "pending",
    error_rate: float = 0.01,
) -> StepRecord:
    state = {
        "server_status": server_status,
        "deployment_status": deployment_status,
        "cpu_utilization": 35,
        "memory_utilization": 42,
        "error_rate": error_rate,
    }
    return StepRecord(
        step_number=step_number,
        world_state_before=state,
        event_injected=event,
        observation_text=f"obs {step_number}",
        agent_action=AgentAction(tool_name=tool_name, tool_args={}),
        world_state_after=state,
    )


def _sample_result() -> EvalResult:
    baseline = BranchTrace(
        branch=ScenarioBranch(id="branch_0", label="Happy Path", description="baseline", is_baseline=True),
        steps=[
            _step(0, "run_tests", deployment_status="pending"),
            _step(1, "deploy", deployment_status="pending"),
            _step(2, "finish_task", deployment_status="deployed"),
        ],
        final_state={"server_status": "healthy", "deployment_status": "deployed", "error_rate": 0.01},
        success=True,
        score=0.96,
        panic_score=0.0,
    )
    byzantine_event = Event(
        step=1,
        name="byzantine_deploy_lie",
        description="deploy lies",
        effect_deltas={"byzantine_deploy_lie": True},
    )
    byzantine = BranchTrace(
        branch=ScenarioBranch(id="branch_6", label="Byzantine Deploy", description="liar branch"),
        steps=[
            _step(0, "run_tests", deployment_status="pending"),
            _step(1, "deploy", event=byzantine_event, deployment_status="deployed", server_status="down", error_rate=0.9),
            _step(2, "finish_task", deployment_status="deployed", server_status="down", error_rate=0.9),
        ],
        final_state={"server_status": "down", "deployment_status": "deployed", "error_rate": 0.9},
        success=False,
        score=0.61,
        panic_score=0.67,
        deviation_step=1,
    )
    return EvalResult(base_task="Deploy app", branches=[baseline, byzantine], mean_panic_score=0.335)


def test_build_multiverse_dag_creates_shared_baseline_and_branch_edges():
    graph = build_multiverse_dag(_sample_result())
    node_ids = {node["id"] for node in graph["nodes"]}
    edges = {(link["source"], link["target"]) for link in graph["links"]}

    assert "task_root" in node_ids
    assert "baseline_step_1" in node_ids
    assert "branch_6_step_1" in node_ids
    assert ("task_root", "baseline_step_0") in edges
    assert ("baseline_step_1", "branch_6_step_1") in edges


def test_build_multiverse_dag_marks_byzantine_branch_as_high_panic():
    graph = build_multiverse_dag(_sample_result())
    node = next(node for node in graph["nodes"] if node["id"] == "branch_6_step_1")
    category = graph["categories"][node["category"]]["name"]

    assert category == "High Panic"
    assert "Byzantine Deploy" in node["name"]
    assert "byzantine_deploy_lie" in node["tooltip"]["formatter"]


def test_build_multiverse_dot_contains_branch_labels_and_edges():
    dot = build_multiverse_dot(_sample_result())

    assert '"task_root" -> "baseline_step_0"' in dot
    assert '"baseline_step_1" -> "branch_6_step_1"' in dot
    assert "Byzantine Deploy" in dot
