"""Pydantic v2 data models shared across all TTE modules."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Event(BaseModel):
    """A single timed perturbation injected into a simulation branch."""

    step: int = Field(..., ge=0, description="Simulation step at which this event fires.")
    name: str = Field(..., description="Short machine-readable event identifier.")
    description: str = Field(..., description="Human-readable description of what happens.")
    effect_deltas: Dict[str, Any] = Field(
        default_factory=dict,
        description="Key-value overrides applied to the world state when this event fires.",
    )


class ScenarioBranch(BaseModel):
    """One alternate timeline produced by the scenario generator."""

    id: str = Field(..., description="Unique branch identifier, e.g. branch_0.")
    label: str = Field(default="", description="Short human-readable label.")
    description: str = Field(..., description="Narrative description of the scenario.")
    is_baseline: bool = Field(default=False, description="True for the unperturbed happy-path branch.")
    events: List[Event] = Field(default_factory=list, description="Timed events for this branch.")
    initial_state_overrides: Dict[str, Any] = Field(
        default_factory=dict,
        description="State keys that differ from the default at simulation start.",
    )
    expected_outcome: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional expected final-state fragment for validation.",
    )


class ToolDefinition(BaseModel):
    """Schema for a tool the agent is allowed to call."""

    name: str
    description: str
    parameters: Dict[str, Any] = Field(default_factory=dict)


class AgentAction(BaseModel):
    """A single action the agent chose at one simulation step."""

    tool_name: str = Field(..., description="Name of the tool invoked.")
    tool_args: Dict[str, Any] = Field(default_factory=dict, description="Arguments passed to the tool.")
    reasoning: Optional[str] = Field(default=None, description="Optional chain-of-thought snippet.")
    raw_llm_response: Optional[str] = Field(default=None, description="Full LLM output for debugging.")


class StepRecord(BaseModel):
    """Complete record of one simulation tick."""

    step_number: int
    world_state_before: Dict[str, Any]
    event_injected: Optional[Event] = None
    observation_text: str
    agent_action: AgentAction
    world_state_after: Dict[str, Any]


class BranchTrace(BaseModel):
    """Full execution trace for a single branch."""

    branch: ScenarioBranch
    steps: List[StepRecord] = Field(default_factory=list)
    final_state: Dict[str, Any] = Field(default_factory=dict)
    success: bool = False
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    panic_score: float = Field(default=0.0, ge=0.0, le=1.0)
    panic_source: str = Field(default="heuristic", description="heuristic or llm_judge")
    judge_explanation: Optional[str] = None
    deviation_step: Optional[int] = Field(
        default=None,
        description="First step where agent behaviour diverged from baseline.",
    )
    recovery_step: Optional[int] = Field(
        default=None,
        description="Step at which the agent recovered after deviation.",
    )
    api_calls_used: int = 0
    total_tokens_used: int = 0


class EvalResult(BaseModel):
    """Aggregated evaluation output across all branches."""

    base_task: str
    branches: List[BranchTrace] = Field(default_factory=list)
    robustness_score: float = Field(default=0.0, ge=0.0, le=1.0)
    success_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    stability_score: float = Field(default=0.0, ge=0.0, le=1.0)
    mean_recovery_time: Optional[float] = None
    mean_panic_score: float = Field(default=0.0, ge=0.0, le=1.0)
    provider_name: str = ""
    provider_profile: str = "generic"
    execution_mode: str = "standard"
    effective_rpm: int = 0
    logical_calls: int = 0
    scheduled_provider_calls: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    provider_base_url: str = ""
    agent_model: str = ""
    generator_model: str = ""
    total_api_calls: int = 0
    total_tokens: int = 0
