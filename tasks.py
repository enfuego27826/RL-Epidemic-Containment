from __future__ import annotations

from dataclasses import dataclass
from statistics import mean

from engine import EdgeConfig, NodeConfig, TaskConfig
from models import (
    EpidemicAction,
    EpidemicObservation,
    EpidemicState,
    LiftQuarantineIntervention,
    QuarantineIntervention,
    TaskEvaluation,
    VaccinateIntervention,
)


BENCHMARK_NAME = "epidemic_containment_strategy"


@dataclass(frozen=True)
class TaskDefinition:
    config: TaskConfig

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def difficulty(self) -> str:
        return self.config.difficulty

    @property
    def goal(self) -> str:
        return self.config.goal

    def route_descriptions(self) -> list[str]:
        return [f"{edge.source}<->{edge.target}" for edge in self.config.edges]

    def prompt_context(self) -> str:
        node_lines = [
            f"- {node.node_id}: population={node.population}, initial_infected={node.initial_infected:.0f}"
            for node in self.config.nodes
        ]
        route_lines = [f"- {route}" for route in self.route_descriptions()]
        return "\n".join(
            [
                f"Task: {self.config.name}",
                f"Difficulty: {self.config.difficulty}",
                f"Goal: {self.config.goal}",
                f"Description: {self.config.description}",
                f"Max steps: {self.config.max_steps}",
                f"Initial vaccine budget: {self.config.vaccine_budget:.1f}",
                f"Reporting lag: {self.config.reporting_lag_steps} step(s)",
                "Cities:",
                *node_lines,
                "Travel routes:",
                *route_lines,
            ]
        )


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _hard_score_components(state: EpidemicState) -> tuple[float, float, float]:
    infection_series = [entry.actual_total_infection_rate for entry in state.history]
    average_infection = (
        mean(infection_series) if infection_series else state.actual_total_infection_rate
    )
    containment = _clamp(1.0 - average_infection / 0.45)
    economy = _clamp(
        0.5 * state.global_economic_score + 0.5 * state.min_global_economic_score
    )
    quarantine_efficiency = _clamp(
        1.0 - state.unnecessary_quarantine_steps / max(1.0, state.max_steps * 2.0)
    )
    return containment, economy, quarantine_efficiency


def grade_easy_localized_outbreak(state: EpidemicState) -> TaskEvaluation:
    peak = state.peak_infection_rate
    threshold = 0.20
    score = (
        1.0
        if peak <= threshold
        else _clamp(1.0 - (peak - threshold) / 0.20)
    )
    success = peak <= threshold and state.step_count >= state.max_steps
    return TaskEvaluation(
        task_name=state.task_name,
        difficulty=state.difficulty,
        score=score,
        success=success,
        goal=state.goal,
        summary=(
            "Contained the localized outbreak."
            if success
            else "Peak infection rose above the 20% containment target."
        ),
        metrics={
            "peak_infection_rate": round(peak, 4),
            "final_infection_rate": round(state.actual_total_infection_rate, 4),
            "min_global_economic_score": round(state.min_global_economic_score, 4),
        },
    )


def grade_medium_multi_center_spread(state: EpidemicState) -> TaskEvaluation:
    infection_threshold = 0.30
    economic_threshold = 0.60
    containment = (
        1.0
        if state.peak_infection_rate <= infection_threshold
        else _clamp(
            1.0
            - (state.peak_infection_rate - infection_threshold)
            / (1.0 - infection_threshold)
        )
    )
    economy = (
        1.0
        if state.min_global_economic_score >= economic_threshold
        else _clamp(state.min_global_economic_score / economic_threshold)
    )
    score = _clamp(0.7 * containment + 0.3 * economy)
    success = (
        state.peak_infection_rate <= infection_threshold
        and state.min_global_economic_score >= economic_threshold
        and state.step_count >= state.max_steps
    )
    return TaskEvaluation(
        task_name=state.task_name,
        difficulty=state.difficulty,
        score=score,
        success=success,
        goal=state.goal,
        summary=(
            "Maintained both containment and economic resilience."
            if success
            else "Either infection pressure or economic damage exceeded the medium-task target."
        ),
        metrics={
            "containment_component": round(containment, 4),
            "economy_component": round(economy, 4),
            "peak_infection_rate": round(state.peak_infection_rate, 4),
            "min_global_economic_score": round(state.min_global_economic_score, 4),
        },
    )


def grade_hard_asymptomatic_high_density(state: EpidemicState) -> TaskEvaluation:
    containment, economy, quarantine_efficiency = _hard_score_components(state)
    score = _clamp((containment * economy * quarantine_efficiency) ** (1.0 / 3.0))
    success = score >= 0.75 and state.step_count >= state.max_steps
    return TaskEvaluation(
        task_name=state.task_name,
        difficulty=state.difficulty,
        score=score,
        success=success,
        goal=state.goal,
        summary=(
            "Balanced outbreak control, economy, and restraint under incomplete information."
            if success
            else "The policy lost too much ground on at least one hard-task objective."
        ),
        metrics={
            "containment_component": round(containment, 4),
            "economy_component": round(economy, 4),
            "quarantine_efficiency_component": round(quarantine_efficiency, 4),
            "average_infection_rate": round(
                mean(entry.actual_total_infection_rate for entry in state.history)
                if state.history
                else state.actual_total_infection_rate,
                4,
            ),
        },
    )


def _easy_task() -> TaskDefinition:
    nodes = (
        NodeConfig("city_0", 1100, 150.0, 0.96, 0.30, 0.08, 0.10, 0.030, 0.050),
        NodeConfig("city_1", 900, 0.0, 0.97, 0.26, 0.09, 0.09, 0.028, 0.046),
        NodeConfig("city_2", 1000, 0.0, 0.98, 0.27, 0.09, 0.09, 0.028, 0.045),
        NodeConfig("city_3", 950, 0.0, 0.99, 0.25, 0.09, 0.09, 0.026, 0.042),
        NodeConfig("city_4", 1050, 0.0, 0.97, 0.26, 0.09, 0.09, 0.028, 0.044),
    )
    edges = (
        EdgeConfig("city_0", "city_1", 0.070),
        EdgeConfig("city_0", "city_2", 0.085),
        EdgeConfig("city_1", "city_2", 0.055),
        EdgeConfig("city_1", "city_3", 0.045),
        EdgeConfig("city_2", "city_4", 0.055),
        EdgeConfig("city_3", "city_4", 0.040),
    )
    return TaskDefinition(
        TaskConfig(
            name="easy_localized_outbreak",
            benchmark=BENCHMARK_NAME,
            difficulty="easy",
            description="A single infected city sits near the edge of a small travel graph.",
            goal="Keep total infected population below 20% for 10 steps.",
            max_steps=10,
            vaccine_budget=1050.0,
            vaccine_cost_per_person=1.0,
            reporting_lag_steps=0,
            max_interventions_per_step=3,
            nodes=nodes,
            edges=edges,
            easy_infection_threshold=0.20,
        )
    )


def _medium_task() -> TaskDefinition:
    nodes = (
        NodeConfig("city_0", 1200, 132.0, 0.95, 0.32, 0.07, 0.09, 0.020, 0.060),
        NodeConfig("city_1", 900, 0.0, 0.96, 0.29, 0.08, 0.09, 0.020, 0.056),
        NodeConfig("city_2", 1050, 126.0, 0.94, 0.31, 0.07, 0.09, 0.019, 0.058),
        NodeConfig("city_3", 980, 0.0, 0.95, 0.30, 0.08, 0.09, 0.019, 0.056),
        NodeConfig("city_4", 1120, 0.0, 0.96, 0.31, 0.07, 0.09, 0.018, 0.057),
        NodeConfig("city_5", 1020, 138.0, 0.94, 0.34, 0.07, 0.09, 0.018, 0.061),
        NodeConfig("city_6", 890, 0.0, 0.97, 0.28, 0.08, 0.09, 0.020, 0.052),
        NodeConfig("city_7", 970, 0.0, 0.95, 0.30, 0.08, 0.09, 0.019, 0.054),
        NodeConfig("city_8", 1080, 0.0, 0.96, 0.31, 0.07, 0.09, 0.018, 0.057),
        NodeConfig("city_9", 930, 0.0, 0.97, 0.29, 0.08, 0.09, 0.020, 0.053),
    )
    edges = (
        EdgeConfig("city_0", "city_1", 0.070),
        EdgeConfig("city_0", "city_2", 0.085),
        EdgeConfig("city_1", "city_3", 0.060),
        EdgeConfig("city_2", "city_3", 0.072),
        EdgeConfig("city_2", "city_4", 0.078),
        EdgeConfig("city_4", "city_5", 0.090),
        EdgeConfig("city_5", "city_6", 0.070),
        EdgeConfig("city_5", "city_7", 0.080),
        EdgeConfig("city_6", "city_8", 0.068),
        EdgeConfig("city_7", "city_8", 0.072),
        EdgeConfig("city_8", "city_9", 0.060),
        EdgeConfig("city_3", "city_7", 0.055),
        EdgeConfig("city_1", "city_6", 0.050),
    )
    return TaskDefinition(
        TaskConfig(
            name="medium_multi_center_spread",
            benchmark=BENCHMARK_NAME,
            difficulty="medium",
            description="Three seeded outbreaks can spill across two dense commuter corridors.",
            goal="Keep infections below 30% and total economic health above 60%.",
            max_steps=15,
            vaccine_budget=760.0,
            vaccine_cost_per_person=1.0,
            reporting_lag_steps=0,
            max_interventions_per_step=3,
            nodes=nodes,
            edges=edges,
            medium_infection_threshold=0.30,
            medium_economic_threshold=0.60,
        )
    )


def _hard_task() -> TaskDefinition:
    nodes = (
        NodeConfig("city_0", 1500, 135.0, 0.95, 0.38, 0.06, 0.12, 0.014, 0.062),
        NodeConfig("city_1", 1300, 0.0, 0.96, 0.36, 0.06, 0.11, 0.014, 0.060),
        NodeConfig("city_2", 1250, 0.0, 0.95, 0.37, 0.06, 0.11, 0.014, 0.061),
        NodeConfig("city_3", 1400, 0.0, 0.95, 0.36, 0.06, 0.11, 0.014, 0.060),
        NodeConfig("city_4", 1180, 110.0, 0.94, 0.38, 0.06, 0.12, 0.014, 0.062),
        NodeConfig("city_5", 1320, 0.0, 0.96, 0.37, 0.06, 0.12, 0.014, 0.061),
        NodeConfig("city_6", 1450, 0.0, 0.95, 0.40, 0.06, 0.12, 0.013, 0.064),
        NodeConfig("city_7", 1200, 96.0, 0.94, 0.39, 0.06, 0.12, 0.013, 0.063),
        NodeConfig("city_8", 1370, 0.0, 0.95, 0.37, 0.06, 0.11, 0.014, 0.061),
        NodeConfig("city_9", 1280, 0.0, 0.96, 0.36, 0.06, 0.11, 0.014, 0.059),
        NodeConfig("city_10", 1600, 125.0, 0.95, 0.42, 0.06, 0.12, 0.012, 0.066),
        NodeConfig("city_11", 1420, 0.0, 0.95, 0.39, 0.06, 0.12, 0.013, 0.063),
        NodeConfig("city_12", 1260, 0.0, 0.96, 0.37, 0.06, 0.11, 0.014, 0.061),
        NodeConfig("city_13", 1350, 0.0, 0.95, 0.39, 0.06, 0.11, 0.013, 0.062),
        NodeConfig("city_14", 1500, 0.0, 0.95, 0.40, 0.06, 0.12, 0.013, 0.064),
        NodeConfig("city_15", 1220, 0.0, 0.96, 0.36, 0.06, 0.11, 0.014, 0.059),
        NodeConfig("city_16", 1310, 0.0, 0.95, 0.37, 0.06, 0.11, 0.014, 0.061),
        NodeConfig("city_17", 1470, 108.0, 0.94, 0.40, 0.06, 0.12, 0.013, 0.064),
        NodeConfig("city_18", 1290, 0.0, 0.96, 0.36, 0.06, 0.11, 0.014, 0.059),
        NodeConfig("city_19", 1380, 0.0, 0.95, 0.39, 0.06, 0.11, 0.013, 0.062),
    )
    edges = (
        EdgeConfig("city_0", "city_1", 0.090),
        EdgeConfig("city_1", "city_2", 0.085),
        EdgeConfig("city_2", "city_3", 0.090),
        EdgeConfig("city_3", "city_4", 0.085),
        EdgeConfig("city_4", "city_0", 0.095),
        EdgeConfig("city_5", "city_6", 0.092),
        EdgeConfig("city_6", "city_7", 0.105),
        EdgeConfig("city_7", "city_8", 0.090),
        EdgeConfig("city_8", "city_9", 0.088),
        EdgeConfig("city_9", "city_5", 0.092),
        EdgeConfig("city_10", "city_11", 0.110),
        EdgeConfig("city_11", "city_12", 0.098),
        EdgeConfig("city_12", "city_13", 0.094),
        EdgeConfig("city_13", "city_14", 0.100),
        EdgeConfig("city_14", "city_10", 0.112),
        EdgeConfig("city_15", "city_16", 0.088),
        EdgeConfig("city_16", "city_17", 0.108),
        EdgeConfig("city_17", "city_18", 0.098),
        EdgeConfig("city_18", "city_19", 0.090),
        EdgeConfig("city_19", "city_15", 0.092),
        EdgeConfig("city_2", "city_7", 0.088),
        EdgeConfig("city_4", "city_10", 0.094),
        EdgeConfig("city_8", "city_12", 0.088),
        EdgeConfig("city_9", "city_15", 0.084),
        EdgeConfig("city_13", "city_17", 0.090),
        EdgeConfig("city_1", "city_18", 0.082),
        EdgeConfig("city_6", "city_14", 0.096),
        EdgeConfig("city_11", "city_19", 0.088),
    )
    return TaskDefinition(
        TaskConfig(
            name="hard_asymptomatic_high_density",
            benchmark=BENCHMARK_NAME,
            difficulty="hard",
            description=(
                "Dense metropolitan clusters are tied together by high-throughput routes, "
                "and reported infections lag actual infections by one step."
            ),
            goal=(
                "Maximize the final score over 20 steps while avoiding unnecessary quarantines "
                "that severely damage the economy."
            ),
            max_steps=20,
            vaccine_budget=620.0,
            vaccine_cost_per_person=1.0,
            reporting_lag_steps=1,
            max_interventions_per_step=3,
            nodes=nodes,
            edges=edges,
        )
    )


TASKS: dict[str, TaskDefinition] = {
    "easy_localized_outbreak": _easy_task(),
    "medium_multi_center_spread": _medium_task(),
    "hard_asymptomatic_high_density": _hard_task(),
}


def get_task_definition(task_name: str) -> TaskDefinition:
    try:
        return TASKS[task_name]
    except KeyError as exc:
        known = ", ".join(sorted(TASKS))
        raise ValueError(
            f"Unknown task '{task_name}'. Expected one of: {known}."
        ) from exc


def grade_task(task_name: str, state: EpidemicState) -> TaskEvaluation:
    if task_name == "easy_localized_outbreak":
        return grade_easy_localized_outbreak(state)
    if task_name == "medium_multi_center_spread":
        return grade_medium_multi_center_spread(state)
    if task_name == "hard_asymptomatic_high_density":
        return grade_hard_asymptomatic_high_density(state)
    raise ValueError(f"No grader is registered for task '{task_name}'.")


def baseline_policy(task_name: str, observation: EpidemicObservation) -> EpidemicAction:
    _ = get_task_definition(task_name)
    max_actions = 2 if task_name == "hard_asymptomatic_high_density" else 3
    interventions = []
    nodes = sorted(
        observation.nodes, key=lambda node: node.known_infection_rate, reverse=True
    )
    used_node_ids: set[str] = set()
    planned_vaccine_spend = 0.0

    release_threshold = {
        "easy_localized_outbreak": 0.05,
        "medium_multi_center_spread": 0.06,
        "hard_asymptomatic_high_density": 0.07,
    }[task_name]
    critical_release_economy = {
        "easy_localized_outbreak": 0.48,
        "medium_multi_center_spread": 0.54,
        "hard_asymptomatic_high_density": 0.58,
    }[task_name]

    if observation.global_economic_score < 0.70:
        releasable = [
            node
            for node in observation.nodes
            if node.is_quarantined
            and node.known_infection_rate < release_threshold
        ]
        for node in sorted(
            releasable,
            key=lambda item: (item.known_infection_rate, item.economic_health),
        )[:1]:
            interventions.append(LiftQuarantineIntervention(node_id=node.node_id))
            used_node_ids.add(node.node_id)
    else:
        fragile_quarantine = [
            node
            for node in observation.nodes
            if node.is_quarantined
            and node.economic_health <= critical_release_economy
            and node.known_infection_rate < release_threshold * 0.6
        ]
        for node in sorted(
            fragile_quarantine,
            key=lambda item: (item.economic_health, item.known_infection_rate),
        )[:1]:
            interventions.append(LiftQuarantineIntervention(node_id=node.node_id))
            used_node_ids.add(node.node_id)

    vaccination_threshold = {
        "easy_localized_outbreak": 0.05,
        "medium_multi_center_spread": 0.08,
        "hard_asymptomatic_high_density": 0.10,
    }[task_name]

    for node in nodes:
        if len(interventions) >= max_actions:
            break
        if node.node_id in used_node_ids:
            continue
        if node.known_infection_rate < vaccination_threshold:
            continue
        available_budget = max(0.0, observation.vaccine_budget - planned_vaccine_spend)
        if available_budget < 25.0:
            break
        target_amount = {
            "easy_localized_outbreak": max(90.0, node.population * node.known_infection_rate * 0.60),
            "medium_multi_center_spread": max(95.0, node.population * node.known_infection_rate * 0.60),
            "hard_asymptomatic_high_density": max(70.0, node.population * node.known_infection_rate * 0.32),
        }[task_name]
        spend = round(min(available_budget, target_amount), 1)
        if spend < 25.0:
            continue
        interventions.append(
            VaccinateIntervention(
                node_id=node.node_id,
                amount=spend,
            )
        )
        used_node_ids.add(node.node_id)
        planned_vaccine_spend += spend

    quarantine_threshold = {
        "easy_localized_outbreak": 0.32,
        "medium_multi_center_spread": 0.50,
        "hard_asymptomatic_high_density": 0.45,
    }[task_name]
    economy_guardrail = {
        "easy_localized_outbreak": 0.62,
        "medium_multi_center_spread": 0.80,
        "hard_asymptomatic_high_density": 0.76,
    }[task_name]
    node_economy_floor = {
        "easy_localized_outbreak": 0.46,
        "medium_multi_center_spread": 0.58,
        "hard_asymptomatic_high_density": 0.64,
    }[task_name]
    emergency_quarantine_threshold = {
        "easy_localized_outbreak": 0.48,
        "medium_multi_center_spread": 0.62,
        "hard_asymptomatic_high_density": 0.58,
    }[task_name]

    for node in nodes:
        if len(interventions) >= max_actions:
            break
        if node.node_id in used_node_ids:
            continue
        if (
            observation.global_economic_score >= economy_guardrail
            and node.known_infection_rate >= quarantine_threshold
            and not node.is_quarantined
            and (
                node.economic_health >= node_economy_floor
                or node.known_infection_rate >= emergency_quarantine_threshold
            )
        ):
            interventions.append(QuarantineIntervention(node_id=node.node_id))
            used_node_ids.add(node.node_id)

    if (
        not interventions
        and observation.global_economic_score > 0.68
        and observation.vaccine_budget >= 40.0
        and nodes
        and nodes[0].known_infection_rate >= max(0.02, vaccination_threshold * 0.5)
    ):
        node = nodes[0]
        interventions.append(
            VaccinateIntervention(
                node_id=node.node_id,
                amount=round(min(observation.vaccine_budget, 40.0), 1),
            )
        )

    return EpidemicAction(interventions=interventions[:max_actions])
