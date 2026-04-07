from __future__ import annotations

from typing import Annotated, Literal

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, Field


class NodeObservation(BaseModel):
    node_id: str = Field(..., description="City identifier.")
    population: int = Field(..., gt=0, description="Total city population.")
    known_infection_rate: float = Field(
        ..., ge=0.0, le=1.0, description="Reported infected fraction for this city."
    )
    economic_health: float = Field(
        ..., ge=0.0, le=1.0, description="Observable local economic health score."
    )
    is_quarantined: bool = Field(
        ..., description="Whether cross-city travel is currently blocked."
    )


class QuarantineIntervention(BaseModel):
    kind: Literal["quarantine"] = "quarantine"
    node_id: str = Field(..., description="City to place under quarantine.")


class VaccinateIntervention(BaseModel):
    kind: Literal["vaccinate"] = "vaccinate"
    node_id: str = Field(..., description="City to vaccinate.")
    amount: float = Field(
        ...,
        gt=0.0,
        description="Requested number of people to vaccinate this step.",
    )


class LiftQuarantineIntervention(BaseModel):
    kind: Literal["lift_quarantine"] = "lift_quarantine"
    node_id: str = Field(..., description="City to release from quarantine.")


Intervention = Annotated[
    QuarantineIntervention | VaccinateIntervention | LiftQuarantineIntervention,
    Field(discriminator="kind"),
]


class EpidemicAction(Action):
    interventions: list[Intervention] = Field(
        default_factory=list,
        max_length=3,
        description=(
            "Up to three public-health interventions for the current timestep. "
            "Use an empty list to monitor without intervening."
        ),
    )


class EpidemicObservation(Observation):
    benchmark: str = Field(..., description="Benchmark identifier.")
    task_name: str = Field(..., description="Active task identifier.")
    difficulty: str = Field(..., description="Task difficulty label.")
    goal: str = Field(..., description="Natural-language objective for the task.")
    step_count: int = Field(..., ge=0, description="Current environment step.")
    max_steps: int = Field(..., gt=0, description="Episode horizon.")
    reporting_lag_steps: int = Field(
        ..., ge=0, description="Delay applied to reported infections."
    )
    vaccine_budget: float = Field(
        ..., ge=0.0, description="Remaining vaccine budget for the episode."
    )
    global_economic_score: float = Field(
        ..., ge=0.0, le=1.0, description="Population-weighted global economy score."
    )
    reported_total_infection_rate: float = Field(
        ..., ge=0.0, le=1.0, description="Reported global infection fraction."
    )
    nodes: list[NodeObservation] = Field(
        default_factory=list, description="Per-city reported health indicators."
    )
    alerts: list[str] = Field(
        default_factory=list, description="Short operational alerts for the agent."
    )


class NodeInternalState(BaseModel):
    node_id: str
    population: int = Field(..., gt=0)
    susceptible: float = Field(..., ge=0.0)
    infected: float = Field(..., ge=0.0)
    recovered: float = Field(..., ge=0.0)
    actual_infection_rate: float = Field(..., ge=0.0, le=1.0)
    reported_infection_rate: float = Field(..., ge=0.0, le=1.0)
    economic_health: float = Field(..., ge=0.0, le=1.0)
    local_transmission_rate: float = Field(..., ge=0.0)
    recovery_rate: float = Field(..., ge=0.0, le=1.0)
    quarantine_economic_drain: float = Field(..., ge=0.0)
    economic_recovery_rate: float = Field(..., ge=0.0)
    disease_economic_drag: float = Field(..., ge=0.0)
    quarantined: bool = False
    quarantine_streak: int = Field(default=0, ge=0)


class EdgeInternalState(BaseModel):
    source: str
    target: str
    transmission_rate: float = Field(..., ge=0.0)
    active: bool = Field(
        ..., description="Whether travel-based transmission is active this step."
    )


class StepHistoryEntry(BaseModel):
    step: int = Field(..., ge=1)
    action_descriptions: list[str] = Field(default_factory=list)
    reward: float
    actual_total_infection_rate: float = Field(..., ge=0.0, le=1.0)
    reported_total_infection_rate: float = Field(..., ge=0.0, le=1.0)
    global_economic_score: float = Field(..., ge=0.0, le=1.0)
    vaccine_budget: float = Field(..., ge=0.0)
    severe_node_ids: list[str] = Field(default_factory=list)
    collapsed_node_ids: list[str] = Field(default_factory=list)
    unnecessary_quarantine_node_ids: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class EpidemicState(State):
    benchmark: str
    task_name: str
    difficulty: str
    goal: str
    max_steps: int = Field(..., gt=0)
    initial_vaccine_budget: float = Field(..., ge=0.0)
    vaccine_budget: float = Field(..., ge=0.0)
    reporting_lag_steps: int = Field(..., ge=0)
    global_economic_score: float = Field(..., ge=0.0, le=1.0)
    actual_total_infection_rate: float = Field(..., ge=0.0, le=1.0)
    reported_total_infection_rate: float = Field(..., ge=0.0, le=1.0)
    peak_infection_rate: float = Field(..., ge=0.0, le=1.0)
    min_global_economic_score: float = Field(..., ge=0.0, le=1.0)
    invalid_action_count: int = Field(default=0, ge=0)
    severe_infection_events: int = Field(default=0, ge=0)
    collapsed_quarantine_events: int = Field(default=0, ge=0)
    unnecessary_quarantine_steps: int = Field(default=0, ge=0)
    nodes: list[NodeInternalState] = Field(default_factory=list)
    edges: list[EdgeInternalState] = Field(default_factory=list)
    history: list[StepHistoryEntry] = Field(default_factory=list)


class TaskEvaluation(BaseModel):
    task_name: str
    difficulty: str
    score: float = Field(..., ge=0.0, le=1.0)
    success: bool
    goal: str
    summary: str
    metrics: dict[str, float] = Field(default_factory=dict)


def describe_intervention(intervention: Intervention) -> str:
    if isinstance(intervention, QuarantineIntervention):
        return f"quarantine({intervention.node_id})"
    if isinstance(intervention, LiftQuarantineIntervention):
        return f"lift_quarantine({intervention.node_id})"
    return f"vaccinate({intervention.node_id},{intervention.amount:.1f})"


def format_action(action: EpidemicAction) -> str:
    if not action.interventions:
        return "noop"
    return "|".join(
        describe_intervention(intervention) for intervention in action.interventions
    )
