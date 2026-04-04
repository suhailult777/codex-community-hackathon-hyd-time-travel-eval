"""Tests for core/simulator.py — state mutation and event injection."""

import pytest

from core.models import AgentAction, Event, ScenarioBranch
from core.simulator import DEFAULT_DEVOPS_STATE, WorldSimulator


@pytest.fixture
def baseline_branch():
    return ScenarioBranch(id="b0", description="baseline", is_baseline=True, events=[])


@pytest.fixture
def event_branch():
    return ScenarioBranch(
        id="b1",
        description="cpu spike at step 2",
        events=[
            Event(
                step=2,
                name="cpu_spike",
                description="CPU surges to 95%",
                effect_deltas={"cpu_utilization": 95, "server_status": "degraded"},
            )
        ],
    )


class TestWorldSimulator:
    def test_initial_state_matches_default(self, baseline_branch):
        sim = WorldSimulator(branch=baseline_branch)
        assert sim.state["server_status"] == "healthy"
        assert sim.state["deployment_status"] == "pending"

    def test_tick_no_event_returns_nominal(self, baseline_branch):
        sim = WorldSimulator(branch=baseline_branch)
        obs = sim.tick()
        assert "nominal" in obs.lower() or "server=" in obs
        assert sim.last_event is None

    def test_tick_applies_event_at_correct_step(self, event_branch):
        sim = WorldSimulator(branch=event_branch)
        # Step 0 and 1 — no event
        sim.tick()
        assert sim.state["cpu_utilization"] == 35
        sim.tick()
        assert sim.state["cpu_utilization"] == 35
        # Step 2 — event fires
        obs = sim.tick()
        assert sim.state["cpu_utilization"] == 95
        assert sim.state["server_status"] == "degraded"
        assert "cpu_spike" in obs

    def test_apply_action_deploy_success(self, baseline_branch):
        sim = WorldSimulator(branch=baseline_branch)
        action = AgentAction(tool_name="deploy", tool_args={})
        result = sim.apply_action(action)
        assert "SUCCEEDED" in result
        assert sim.state["deployment_status"] == "deployed"

    def test_apply_action_deploy_fails_when_server_down(self, baseline_branch):
        sim = WorldSimulator(branch=baseline_branch)
        sim.state["server_status"] = "down"
        action = AgentAction(tool_name="deploy", tool_args={})
        result = sim.apply_action(action)
        assert "FAILED" in result
        assert sim.state["deployment_status"] == "failed"

    def test_apply_action_rollback(self, baseline_branch):
        sim = WorldSimulator(branch=baseline_branch)
        action = AgentAction(tool_name="rollback", tool_args={})
        result = sim.apply_action(action)
        assert "SUCCEEDED" in result
        assert sim.state["deployment_status"] == "rolled_back"

    def test_apply_action_unknown_tool(self, baseline_branch):
        sim = WorldSimulator(branch=baseline_branch)
        state_before = sim.state.copy()
        action = AgentAction(tool_name="hack_the_planet", tool_args={})
        result = sim.apply_action(action)
        assert "Unknown" in result
        # State unchanged (except logs)
        for key in state_before:
            if key != "logs":
                assert sim.state[key] == state_before[key]

    def test_finish_task_sets_terminal(self, baseline_branch):
        sim = WorldSimulator(branch=baseline_branch)
        assert not sim.is_terminal()
        action = AgentAction(tool_name="finish_task", tool_args={})
        sim.apply_action(action)
        assert sim.is_terminal()

    def test_restart_service_heals_server(self, baseline_branch):
        sim = WorldSimulator(branch=baseline_branch)
        sim.state["server_status"] = "down"
        sim.state["cpu_utilization"] = 99
        action = AgentAction(tool_name="restart_service", tool_args={})
        sim.apply_action(action)
        assert sim.state["server_status"] == "healthy"
        assert sim.state["cpu_utilization"] == 35

    def test_scale_resources(self, baseline_branch):
        sim = WorldSimulator(branch=baseline_branch)
        sim.state["cpu_utilization"] = 90
        action = AgentAction(tool_name="scale_resources", tool_args={})
        sim.apply_action(action)
        assert sim.state["cpu_utilization"] < 90

    def test_initial_state_overrides(self):
        branch = ScenarioBranch(
            id="b_custom",
            description="custom start",
            initial_state_overrides={"server_status": "degraded", "cpu_utilization": 80},
        )
        sim = WorldSimulator(branch=branch)
        assert sim.state["server_status"] == "degraded"
        assert sim.state["cpu_utilization"] == 80
