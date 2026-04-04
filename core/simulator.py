"""Discrete-event world simulator for TTE."""

from __future__ import annotations

import copy
import random
from typing import Any, Callable, Dict, List, Optional

from core.models import AgentAction, Event, ScenarioBranch

DEFAULT_DEVOPS_STATE: Dict[str, Any] = {
    "server_status": "healthy",  # healthy | degraded | down
    "cpu_utilization": 35,  # 0-100
    "memory_utilization": 42,  # 0-100
    "deployment_status": "pending",  # pending | in_progress | deployed | failed | rolled_back
    "tests_passing": True,
    "dependency_api_status": "healthy",  # healthy | slow | down
    "latency_ms": 120,
    "error_rate": 0.01,
    "rollback_available": True,
    "byzantine_deploy_lie": False,
    "logs": [],
}


class WorldSimulator:
    """Runs a single branch simulation from start to terminal state."""

    def __init__(
        self,
        default_state: Dict[str, Any] | None = None,
        branch: ScenarioBranch | None = None,
    ) -> None:
        base = copy.deepcopy(default_state or DEFAULT_DEVOPS_STATE)
        if branch:
            base.update(branch.initial_state_overrides)
        self.state: Dict[str, Any] = base
        self.branch = branch
        self.current_step = 0
        self.last_event: Optional[Event] = None
        self._terminal = False
        self._event_index: Dict[int, List[Event]] = {}
        if branch:
            for event in branch.events:
                self._event_index.setdefault(event.step, []).append(event)

    def tick(self) -> str:
        """Advance one step, inject events, and return the latest observation."""
        self.last_event = None
        observations: List[str] = []

        for event in self._event_index.get(self.current_step, []):
            self.last_event = event
            for key, value in event.effect_deltas.items():
                if key in self.state:
                    self.state[key] = value

            # Byzantine faults are hidden from the agent's direct observation.
            if not event.name.startswith("byzantine_"):
                observations.append(f"EVENT [{event.name}]: {event.description}")

            self.state.setdefault("logs", []).append(
                f"[step {self.current_step}] Event: {event.name} - {event.description}"
            )

        if not observations:
            observations.append(f"System nominal at step {self.current_step}.")

        observations.append(self._summarise_state())
        self.current_step += 1
        return "  |  ".join(observations)

    def apply_action(self, action: AgentAction) -> str:
        """Interpret the agent's tool call and mutate the world state."""
        handler = _TOOL_HANDLERS.get(action.tool_name, _handle_unknown)
        result = handler(self.state, action.tool_args)
        self.state.setdefault("logs", []).append(
            f"[step {self.current_step}] Agent: {action.tool_name}({action.tool_args}) -> {result}"
        )
        if action.tool_name == "finish_task":
            self._terminal = True
        return result

    def get_observation(self) -> str:
        return self._summarise_state()

    def get_final_state(self) -> Dict[str, Any]:
        return copy.deepcopy(self.state)

    def is_terminal(self) -> bool:
        return self._terminal

    def snapshot(self) -> Dict[str, Any]:
        """Capture a restorable simulator snapshot."""
        return {
            "state": copy.deepcopy(self.state),
            "current_step": self.current_step,
            "last_event": self.last_event.model_copy(deep=True) if self.last_event else None,
            "terminal": self._terminal,
        }

    def restore(self, snapshot: Dict[str, Any]) -> None:
        """Restore the simulator to a previously captured snapshot."""
        self.state = copy.deepcopy(snapshot["state"])
        self.current_step = int(snapshot["current_step"])
        self.last_event = snapshot["last_event"].model_copy(deep=True) if snapshot.get("last_event") else None
        self._terminal = bool(snapshot["terminal"])

    def _summarise_state(self) -> str:
        state = self.state
        parts = [
            f"server={state.get('server_status', '?')}",
            f"cpu={state.get('cpu_utilization', '?')}%",
            f"mem={state.get('memory_utilization', '?')}%",
            f"deploy={state.get('deployment_status', '?')}",
            f"tests={'pass' if state.get('tests_passing') else 'fail'}",
            f"dep_api={state.get('dependency_api_status', '?')}",
            f"latency={state.get('latency_ms', '?')}ms",
            f"err_rate={state.get('error_rate', '?')}",
        ]
        return " | ".join(parts)


def _handle_run_tests(state: Dict[str, Any], args: Dict[str, Any]) -> str:
    passing = state.get("tests_passing", True)
    return f"Test suite result: {'ALL PASSING' if passing else 'FAILURES DETECTED'}"


def _handle_deploy(state: Dict[str, Any], args: Dict[str, Any]) -> str:
    if state.get("server_status") == "down":
        state["deployment_status"] = "failed"
        return "Deployment FAILED - server is down."
    if not state.get("tests_passing", True):
        state["deployment_status"] = "failed"
        return "Deployment FAILED - test suite is failing."
    if state.get("cpu_utilization", 0) > 90:
        state["deployment_status"] = "failed"
        return "Deployment FAILED - CPU critically high."
    if state.get("dependency_api_status") == "down":
        state["deployment_status"] = "failed"
        return "Deployment FAILED - dependency API is unreachable."

    if state.get("byzantine_deploy_lie"):
        state["deployment_status"] = "deployed"
        state["server_status"] = "down"
        state["error_rate"] = 0.9
        state.setdefault("logs", []).append(
            "Health probe mismatch: deploy control plane reported success while the service stayed down."
        )
        return "Deployment SUCCEEDED - control plane greenlit the release."

    state["deployment_status"] = "deployed"
    state["error_rate"] = min(state.get("error_rate", 0.01), 0.01)
    return "Deployment SUCCEEDED - code is live in production."


def _handle_rollback(state: Dict[str, Any], args: Dict[str, Any]) -> str:
    if not state.get("rollback_available", False):
        return "Rollback UNAVAILABLE - no previous version to revert to."
    state["deployment_status"] = "rolled_back"
    state["error_rate"] = max(state.get("error_rate", 0.0) * 0.1, 0.01)
    state["tests_passing"] = True
    return "Rollback SUCCEEDED - reverted to previous stable version."


def _handle_restart_service(state: Dict[str, Any], args: Dict[str, Any]) -> str:
    state["server_status"] = "healthy"
    state["cpu_utilization"] = 35
    state["memory_utilization"] = 42
    state["latency_ms"] = 120
    state["error_rate"] = 0.01
    return "Service restarted - server healthy and resources normalised."


def _handle_check_logs(state: Dict[str, Any], args: Dict[str, Any]) -> str:
    logs = state.get("logs", [])
    recent = logs[-5:] if logs else ["No log entries yet."]
    return "Recent logs:\n" + "\n".join(f"  - {line}" for line in recent)


def _handle_scale_resources(state: Dict[str, Any], args: Dict[str, Any]) -> str:
    state["cpu_utilization"] = max(int(state.get("cpu_utilization", 50) * 0.5), 10)
    state["memory_utilization"] = max(int(state.get("memory_utilization", 50) * 0.6), 15)
    return f"Resources scaled - CPU now {state['cpu_utilization']}%, MEM now {state['memory_utilization']}%."


def _handle_retry_connection(state: Dict[str, Any], args: Dict[str, Any]) -> str:
    current = state.get("dependency_api_status", "healthy")
    if current == "healthy":
        return "Dependency API is already healthy - no retry needed."
    chance = 0.7 if current == "slow" else 0.3
    if random.random() < chance:
        state["dependency_api_status"] = "healthy"
        state["latency_ms"] = 120
        return "Connection retry SUCCEEDED - dependency API restored."
    return f"Connection retry FAILED - dependency API still {current}."


def _handle_finish_task(state: Dict[str, Any], args: Dict[str, Any]) -> str:
    return "Agent has declared the task complete."


def _handle_unknown(state: Dict[str, Any], args: Dict[str, Any]) -> str:
    return "Unknown tool - no state change applied."


_TOOL_HANDLERS: Dict[str, Callable[[Dict[str, Any], Dict[str, Any]], str]] = {
    "run_tests": _handle_run_tests,
    "deploy": _handle_deploy,
    "rollback": _handle_rollback,
    "restart_service": _handle_restart_service,
    "check_logs": _handle_check_logs,
    "scale_resources": _handle_scale_resources,
    "retry_connection": _handle_retry_connection,
    "finish_task": _handle_finish_task,
}
