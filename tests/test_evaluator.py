"""Tests for core/evaluator.py — metric computation correctness."""

import pytest

from core.evaluator import compute_branch_score, compute_eval_result, evaluate_branch_success
from core.models import AgentAction, BranchTrace, Event, ScenarioBranch, StepRecord


def _make_trace(branch_id: str, is_baseline: bool, final_state: dict, n_steps: int = 3) -> BranchTrace:
    branch = ScenarioBranch(id=branch_id, description=f"Branch {branch_id}", is_baseline=is_baseline)
    steps = [
        StepRecord(
            step_number=i,
            world_state_before=final_state,
            observation_text="obs",
            agent_action=AgentAction(tool_name="check_logs", tool_args={}),
            world_state_after=final_state,
        )
        for i in range(n_steps)
    ]
    success = evaluate_branch_success(final_state)
    score = compute_branch_score(final_state, steps)
    return BranchTrace(
        branch=branch, steps=steps, final_state=final_state, success=success, score=score
    )


class TestEvaluateBranchSuccess:
    def test_deployed_and_healthy(self):
        state = {"deployment_status": "deployed", "server_status": "healthy", "error_rate": 0.01}
        assert evaluate_branch_success(state) is True

    def test_failed_deployment(self):
        state = {"deployment_status": "failed", "server_status": "healthy", "error_rate": 0.01}
        assert evaluate_branch_success(state) is False

    def test_server_down(self):
        state = {"deployment_status": "deployed", "server_status": "down", "error_rate": 0.01}
        assert evaluate_branch_success(state) is False

    def test_high_error_rate(self):
        state = {"deployment_status": "deployed", "server_status": "healthy", "error_rate": 0.9}
        assert evaluate_branch_success(state) is False

    def test_rolled_back_is_success(self):
        state = {"deployment_status": "rolled_back", "server_status": "healthy", "error_rate": 0.01}
        assert evaluate_branch_success(state) is True


class TestComputeBranchScore:
    def test_perfect_score(self):
        state = {
            "deployment_status": "deployed",
            "server_status": "healthy",
            "error_rate": 0.01,
            "dependency_api_status": "healthy",
        }
        score = compute_branch_score(state, [])
        assert score > 0.8

    def test_failed_score_is_low(self):
        state = {
            "deployment_status": "failed",
            "server_status": "down",
            "error_rate": 0.9,
            "dependency_api_status": "down",
        }
        score = compute_branch_score(state, [])
        assert score < 0.2


class TestComputeEvalResult:
    def test_all_succeed_rs_near_one(self):
        good = {"deployment_status": "deployed", "server_status": "healthy", "error_rate": 0.01, "dependency_api_status": "healthy"}
        traces = [
            _make_trace("b0", True, good),
            _make_trace("b1", False, good),
            _make_trace("b2", False, good),
        ]
        result = compute_eval_result("test task", traces)
        assert result.success_rate == 1.0
        assert result.robustness_score >= 0.95

    def test_one_fails_rs_drops(self):
        good = {"deployment_status": "deployed", "server_status": "healthy", "error_rate": 0.01, "dependency_api_status": "healthy"}
        bad = {"deployment_status": "failed", "server_status": "down", "error_rate": 0.9, "dependency_api_status": "down"}
        traces = [
            _make_trace("b0", True, good),
            _make_trace("b1", False, good),
            _make_trace("b2", False, bad),
        ]
        result = compute_eval_result("test task", traces)
        assert result.success_rate < 1.0
        assert result.robustness_score < 1.0

    def test_empty_traces(self):
        result = compute_eval_result("test task", [])
        assert result.robustness_score == 0.0
        assert result.success_rate == 0.0

    def test_stability_all_same(self):
        good = {"deployment_status": "deployed", "server_status": "healthy", "error_rate": 0.01, "dependency_api_status": "healthy"}
        traces = [
            _make_trace("b0", True, good),
            _make_trace("b1", False, good),
        ]
        result = compute_eval_result("task", traces)
        assert result.stability_score >= 0.95  # very low variance

    def test_baseline_zero_score_no_crash(self):
        """RS should not crash when baseline score is 0."""
        bad = {"deployment_status": "failed", "server_status": "down", "error_rate": 0.9, "dependency_api_status": "down"}
        traces = [
            _make_trace("b0", True, bad),
            _make_trace("b1", False, bad),
        ]
        result = compute_eval_result("task", traces)
        assert result.robustness_score >= 0.0  # should not be NaN or crash
