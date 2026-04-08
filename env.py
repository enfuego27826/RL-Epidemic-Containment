from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from openenv.core.env_server.interfaces import Environment

from engine import GraphEpidemicEngine
from models import (
    EdgeInternalState,
    EpidemicAction,
    EpidemicObservation,
    EpidemicState,
    Intervention,
    NodeInternalState,
    NodeObservation,
    StepHistoryEntry,
    describe_intervention,
)
from tasks import BENCHMARK_NAME, TaskDefinition, TaskEvaluation, get_task_definition, grade_task


@dataclass(frozen=True)
class SanitizedActionResult:
    action: EpidemicAction
    action_descriptions: list[str]
    dropped_action_descriptions: list[str]
    errors: list[str]
    raw_intervention_count: int
    invalid_action_count_delta: int = 0

    @property
    def blocked(self) -> bool:
        return self.raw_intervention_count > 0 and not self.action.interventions


class EpidemicContainmentStrategyEnv:
    def __init__(self, task_name: str = "easy_localized_outbreak", seed: int | None = None):
        self.task: TaskDefinition = get_task_definition(task_name)
        self.engine = GraphEpidemicEngine(self.task.config)
        self._seed = seed
        self._done = False
        self._history: list[StepHistoryEntry] = []
        self._last_evaluation: TaskEvaluation | None = None
        self.reset(seed=seed)

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        task_name: str | None = None,
    ) -> EpidemicObservation:
        if task_name and task_name != self.task.name:
            self.task = get_task_definition(task_name)
            self.engine = GraphEpidemicEngine(self.task.config)
        if seed is not None:
            self._seed = seed
        self.engine.reset(seed=self._seed, episode_id=episode_id)
        self._done = False
        self._history = []
        self._last_evaluation = grade_task(self.task.name, self.state())
        return self._build_observation(reward=0.0, done=False)

    def step(
        self,
        action: EpidemicAction | dict[str, Any],
    ) -> tuple[EpidemicObservation, float, bool, dict[str, Any]]:
        if self._done:
            observation = self._build_observation(reward=0.0, done=True)
            return observation, 0.0, True, {"error": "Episode already completed."}

        sanitized_action = self.sanitize_action(action)
        previous_metrics = self._metrics_snapshot()
        action_descriptions = list(sanitized_action.action_descriptions)
        errors = list(sanitized_action.errors)
        self.engine.invalid_action_count += sanitized_action.invalid_action_count_delta

        for intervention in sanitized_action.action.interventions:
            self._apply_intervention(intervention, errors)

        engine_result = self.engine.step()
        reward_components = self._compute_reward_components(
            previous_metrics, engine_result, len(errors)
        )
        reward = (
            reward_components["reward_health"]
            + reward_components["reward_economy"]
            + reward_components["reward_control"]
            + reward_components["reward_penalty"]
        )
        self._done = bool(engine_result["done"])

        history_entry = StepHistoryEntry(
            step=self.engine.step_count,
            action_descriptions=action_descriptions,
            reward=reward,
            actual_total_infection_rate=engine_result["actual_total_infection_rate"],
            reported_total_infection_rate=engine_result["reported_total_infection_rate"],
            global_economic_score=engine_result["global_economic_score"],
            vaccine_budget=self.engine.vaccine_budget,
            severe_node_ids=list(engine_result["severe_node_ids"]),
            collapsed_node_ids=list(engine_result["collapsed_node_ids"]),
            unnecessary_quarantine_node_ids=list(
                engine_result["unnecessary_quarantine_node_ids"]
            ),
            errors=errors,
        )
        self._history.append(history_entry)
        current_state = self.state()
        self._last_evaluation = grade_task(self.task.name, current_state)
        observation = self._build_observation(reward=reward, done=self._done)
        info = {
            "action_descriptions": action_descriptions,
            "dropped_action_descriptions": list(
                sanitized_action.dropped_action_descriptions
            ),
            "sanitized_action": sanitized_action.action.model_dump(),
            "action_blocked": sanitized_action.blocked,
            "errors": errors,
            "task_score": self._last_evaluation.score,
            "task_success": self._last_evaluation.success,
            "metrics": self._last_evaluation.metrics,
            "reward_health": reward_components["reward_health"],
            "reward_economy": reward_components["reward_economy"],
            "reward_control": reward_components["reward_control"],
            "reward_penalty": reward_components["reward_penalty"],
        }
        return observation, reward, self._done, info

    def state(self) -> EpidemicState:
        reported_rates = self.engine.reported_node_infection_rates()
        node_states = [
            NodeInternalState(
                node_id=node.node_id,
                population=node.population,
                susceptible=round(node.susceptible, 3),
                infected=round(node.infected, 3),
                recovered=round(node.recovered, 3),
                actual_infection_rate=round(node.actual_infection_rate, 6),
                reported_infection_rate=round(reported_rates[node.node_id], 6),
                economic_health=round(node.economic_health, 6),
                local_transmission_rate=node.local_transmission_rate,
                recovery_rate=node.recovery_rate,
                quarantine_economic_drain=node.quarantine_economic_drain,
                economic_recovery_rate=node.economic_recovery_rate,
                disease_economic_drag=node.disease_economic_drag,
                quarantined=node.quarantined,
                quarantine_streak=node.quarantine_streak,
            )
            for node in self.engine.nodes.values()
        ]
        edge_states = [
            EdgeInternalState(**edge_state) for edge_state in self.engine.active_edges()
        ]
        return EpidemicState(
            episode_id=self.engine.episode_id,
            step_count=self.engine.step_count,
            benchmark=BENCHMARK_NAME,
            task_name=self.task.name,
            difficulty=self.task.difficulty,
            goal=self.task.goal,
            max_steps=self.task.config.max_steps,
            initial_vaccine_budget=self.engine.initial_vaccine_budget,
            vaccine_budget=round(self.engine.vaccine_budget, 4),
            reporting_lag_steps=self.task.config.reporting_lag_steps,
            global_economic_score=round(self.engine.global_economic_score(), 6),
            actual_total_infection_rate=round(self.engine.actual_total_infection_rate(), 6),
            reported_total_infection_rate=round(
                self.engine.reported_total_infection_rate(), 6
            ),
            peak_infection_rate=round(self.engine.peak_infection_rate, 6),
            min_global_economic_score=round(self.engine.min_global_economic_score, 6),
            invalid_action_count=self.engine.invalid_action_count,
            severe_infection_events=self.engine.severe_infection_events,
            collapsed_quarantine_events=self.engine.collapsed_quarantine_events,
            unnecessary_quarantine_steps=self.engine.unnecessary_quarantine_steps,
            nodes=node_states,
            edges=edge_states,
            history=self._history,
        )

    def latest_evaluation(self) -> TaskEvaluation:
        return self._last_evaluation or grade_task(self.task.name, self.state())

    def sanitize_action(self, action: EpidemicAction | dict[str, Any]) -> SanitizedActionResult:
        from models import LiftQuarantineIntervention, QuarantineIntervention

        normalized_action = (
            action if isinstance(action, EpidemicAction) else EpidemicAction.model_validate(action)
        )
        action_descriptions: list[str] = []
        dropped_action_descriptions: list[str] = []
        errors: list[str] = []
        sanitized_interventions: list[Intervention] = []
        used_node_ids: set[str] = set()
        invalid_action_count_delta = 0

        for intervention in normalized_action.interventions[: self.task.config.max_interventions_per_step]:
            description = describe_intervention(intervention)
            node = self.engine.nodes.get(intervention.node_id)

            if node is None:
                dropped_action_descriptions.append(description)
                errors.append(f"Dropped {description}; unknown node '{intervention.node_id}'.")
                invalid_action_count_delta += 1
                continue

            if intervention.node_id in used_node_ids:
                dropped_action_descriptions.append(description)
                errors.append(
                    f"Dropped {description}; only one intervention per node is allowed per step."
                )
                continue

            if isinstance(intervention, QuarantineIntervention) and node.quarantined:
                dropped_action_descriptions.append(description)
                errors.append(f"Dropped {description}; {intervention.node_id} is already quarantined.")
                continue

            if isinstance(intervention, LiftQuarantineIntervention) and not node.quarantined:
                dropped_action_descriptions.append(description)
                errors.append(f"Dropped {description}; {intervention.node_id} is not quarantined.")
                continue

            sanitized_interventions.append(intervention)
            action_descriptions.append(description)
            used_node_ids.add(intervention.node_id)

        return SanitizedActionResult(
            action=EpidemicAction(interventions=sanitized_interventions),
            action_descriptions=action_descriptions,
            dropped_action_descriptions=dropped_action_descriptions,
            errors=errors,
            raw_intervention_count=len(normalized_action.interventions),
            invalid_action_count_delta=invalid_action_count_delta,
        )

    def _apply_intervention(self, intervention: Intervention, errors: list[str]) -> None:
        from models import LiftQuarantineIntervention, QuarantineIntervention

        if isinstance(intervention, QuarantineIntervention):
            error = self.engine.quarantine(intervention.node_id)
            if error:
                errors.append(error)
            return

        if isinstance(intervention, LiftQuarantineIntervention):
            error = self.engine.lift_quarantine(intervention.node_id)
            if error:
                errors.append(error)
            return

        vaccinated, error = self.engine.vaccinate(intervention.node_id, intervention.amount)
        if error:
            errors.append(error)
        elif vaccinated < intervention.amount:
            errors.append(
                f"{intervention.node_id} vaccination clipped to {vaccinated:.1f} due to budget or susceptibility."
            )

    def _build_observation(self, reward: float, done: bool) -> EpidemicObservation:
        reported_rates = self.engine.reported_node_infection_rates()
        node_observations = [
            NodeObservation(
                node_id=node.node_id,
                population=node.population,
                known_infection_rate=round(reported_rates[node.node_id], 6),
                economic_health=round(node.economic_health, 6),
                is_quarantined=node.quarantined,
            )
            for node in self.engine.nodes.values()
        ]
        alerts = []
        hotspots = [
            node.node_id
            for node in sorted(
                node_observations,
                key=lambda item: item.known_infection_rate,
                reverse=True,
            )[:3]
            if node.known_infection_rate >= 0.10
        ]
        if hotspots:
            alerts.append(f"hotspots={','.join(hotspots)}")
        if self.engine.vaccine_budget <= 0.15 * self.engine.initial_vaccine_budget:
            alerts.append("vaccine_budget_low")
        if self.engine.global_economic_score() < 0.50:
            alerts.append("economy_fragile")
        if self.task.config.reporting_lag_steps > 0:
            alerts.append(f"reporting_lag={self.task.config.reporting_lag_steps}")

        return EpidemicObservation(
            benchmark=BENCHMARK_NAME,
            task_name=self.task.name,
            difficulty=self.task.difficulty,
            goal=self.task.goal,
            step_count=self.engine.step_count,
            max_steps=self.task.config.max_steps,
            reporting_lag_steps=self.task.config.reporting_lag_steps,
            vaccine_budget=round(self.engine.vaccine_budget, 4),
            global_economic_score=round(self.engine.global_economic_score(), 6),
            reported_total_infection_rate=round(
                self.engine.reported_total_infection_rate(), 6
            ),
            nodes=node_observations,
            alerts=alerts,
            reward=reward,
            done=done,
        )

    def _metrics_snapshot(self) -> dict[str, Any]:
        return {
            "actual_total_infection_rate": self.engine.actual_total_infection_rate(),
            "global_economic_score": self.engine.global_economic_score(),
        }

    def _compute_reward_components(
        self,
        previous_metrics: dict[str, Any],
        current_metrics: dict[str, Any],
        error_count: int,
    ) -> dict[str, float]:
        infection_delta = (
            previous_metrics["actual_total_infection_rate"]
            - current_metrics["actual_total_infection_rate"]
        )
        economy_delta = (
            current_metrics["global_economic_score"]
            - previous_metrics["global_economic_score"]
        )
        reward_health = 8.0 * infection_delta
        reward_economy = 2.5 * economy_delta
        reward_control = 0.0
        if (
            current_metrics["actual_total_infection_rate"]
            <= previous_metrics["actual_total_infection_rate"] + 1e-6
            and current_metrics["global_economic_score"] > 0.50
        ):
            reward_control = 1.0
        reward_penalty = (
            -1.25 * len(current_metrics["severe_node_ids"])
            - 1.00 * len(current_metrics["collapsed_node_ids"])
            - 0.35 * len(current_metrics["unnecessary_quarantine_node_ids"])
            - 0.25 * error_count
        )
        return {
            "reward_health": round(reward_health, 6),
            "reward_economy": round(reward_economy, 6),
            "reward_control": round(reward_control, 6),
            "reward_penalty": round(reward_penalty, 6),
        }

class OpenEnvEpidemicContainmentEnv(
    Environment[EpidemicAction, EpidemicObservation, EpidemicState]
):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, task_name: str = "easy_localized_outbreak", seed: int | None = None):
        super().__init__()
        self.env = EpidemicContainmentStrategyEnv(task_name=task_name, seed=seed)

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        task_name: str | None = None,
        **_: Any,
    ) -> EpidemicObservation:
        return self.env.reset(seed=seed, episode_id=episode_id, task_name=task_name)

    def step(
        self,
        action: EpidemicAction,
        timeout_s: float | None = None,
        **_: Any,
    ) -> EpidemicObservation:
        del timeout_s
        observation, _, _, _ = self.env.step(action)
        return observation

    @property
    def state(self) -> EpidemicState:
        return self.env.state()
