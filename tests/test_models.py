"""Tests for core/models.py — Pydantic schema validation."""

import pytest
from pydantic import ValidationError

from core.models import (
    AgentAction,
    BranchTrace,
    EvalResult,
    Event,
    ScenarioBranch,
    StepRecord,
)


class TestEvent:
    def test_valid_event(self):
        ev = Event(step=2, name="cpu_spike", description="CPU surges", effect_deltas={"cpu": 95})
        assert ev.step == 2
        assert ev.name == "cpu_spike"
        assert ev.effect_deltas == {"cpu": 95}

    def test_event_missing_fields(self):
        with pytest.raises(ValidationError):
            Event(step=2)  # missing name and description

    def test_event_negative_step(self):
        with pytest.raises(ValidationError):
            Event(step=-1, name="x", description="y")


class TestScenarioBranch:
    def test_valid_branch(self):
        b = ScenarioBranch(id="b0", description="baseline", is_baseline=True)
        assert b.is_baseline is True
        assert b.events == []

    def test_branch_with_events(self):
        ev = Event(step=1, name="crash", description="server crash", effect_deltas={"server": "down"})
        b = ScenarioBranch(id="b1", description="crash scenario", events=[ev])
        assert len(b.events) == 1

    def test_roundtrip_serialization(self):
        ev = Event(step=3, name="lag", description="net lag", effect_deltas={"latency": 3000})
        b = ScenarioBranch(id="b2", description="lag test", events=[ev], is_baseline=False)
        json_str = b.model_dump_json()
        b2 = ScenarioBranch.model_validate_json(json_str)
        assert b2.id == b.id
        assert b2.events[0].name == "lag"


class TestAgentAction:
    def test_valid_action(self):
        a = AgentAction(tool_name="deploy", tool_args={})
        assert a.tool_name == "deploy"

    def test_action_with_reasoning(self):
        a = AgentAction(tool_name="rollback", tool_args={}, reasoning="Server is down")
        assert a.reasoning == "Server is down"


class TestBranchTrace:
    def test_score_bounds(self):
        b = ScenarioBranch(id="b0", description="test")
        t = BranchTrace(branch=b, score=0.85, success=True)
        assert 0.0 <= t.score <= 1.0

    def test_score_out_of_bounds(self):
        b = ScenarioBranch(id="b0", description="test")
        with pytest.raises(ValidationError):
            BranchTrace(branch=b, score=1.5)


class TestEvalResult:
    def test_empty_result(self):
        r = EvalResult(base_task="test task")
        assert r.robustness_score == 0.0
        assert r.branches == []
