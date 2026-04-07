from __future__ import annotations

from dataclasses import dataclass
from math import exp
from random import Random
from typing import Any
from uuid import uuid4


@dataclass(frozen=True)
class NodeConfig:
    node_id: str
    population: int
    initial_infected: float
    initial_economic_health: float
    local_transmission_rate: float
    recovery_rate: float
    quarantine_economic_drain: float
    economic_recovery_rate: float
    disease_economic_drag: float


@dataclass(frozen=True)
class EdgeConfig:
    source: str
    target: str
    transmission_rate: float


@dataclass(frozen=True)
class TaskConfig:
    name: str
    benchmark: str
    difficulty: str
    description: str
    goal: str
    max_steps: int
    vaccine_budget: float
    vaccine_cost_per_person: float
    reporting_lag_steps: int
    max_interventions_per_step: int
    nodes: tuple[NodeConfig, ...]
    edges: tuple[EdgeConfig, ...]
    easy_infection_threshold: float | None = None
    medium_infection_threshold: float | None = None
    medium_economic_threshold: float | None = None
    unnecessary_infection_threshold: float = 0.05
    unnecessary_neighbor_exposure_threshold: float = 0.03


@dataclass
class RuntimeNode:
    node_id: str
    population: int
    susceptible: float
    infected: float
    recovered: float
    economic_health: float
    local_transmission_rate: float
    recovery_rate: float
    quarantine_economic_drain: float
    economic_recovery_rate: float
    disease_economic_drag: float
    quarantined: bool = False
    quarantine_streak: int = 0

    @property
    def actual_infection_rate(self) -> float:
        if self.population <= 0:
            return 0.0
        return max(0.0, min(1.0, self.infected / self.population))


class GraphEpidemicEngine:
    def __init__(self, task_config: TaskConfig):
        self.task_config = task_config
        self._node_order = [node.node_id for node in task_config.nodes]
        self._incoming_edges: dict[str, list[EdgeConfig]] = {
            node_id: [] for node_id in self._node_order
        }
        for edge in task_config.edges:
            self._incoming_edges[edge.target].append(edge)
            self._incoming_edges[edge.source].append(
                EdgeConfig(edge.target, edge.source, edge.transmission_rate)
            )

        self.reset()

    def reset(self, seed: int | None = None, episode_id: str | None = None) -> None:
        self._rng = Random(seed if seed is not None else 0)
        self.episode_id = episode_id or str(uuid4())
        self.step_count = 0
        self.vaccine_budget = float(self.task_config.vaccine_budget)
        self.initial_vaccine_budget = float(self.task_config.vaccine_budget)
        self.nodes: dict[str, RuntimeNode] = {}
        for config in self.task_config.nodes:
            infected = min(float(config.initial_infected), float(config.population))
            susceptible = max(0.0, float(config.population) - infected)
            self.nodes[config.node_id] = RuntimeNode(
                node_id=config.node_id,
                population=config.population,
                susceptible=susceptible,
                infected=infected,
                recovered=0.0,
                economic_health=max(0.0, min(1.0, config.initial_economic_health)),
                local_transmission_rate=config.local_transmission_rate,
                recovery_rate=config.recovery_rate,
                quarantine_economic_drain=config.quarantine_economic_drain,
                economic_recovery_rate=config.economic_recovery_rate,
                disease_economic_drag=config.disease_economic_drag,
            )

        initial_rates = self.actual_node_infection_rates()
        self._actual_rate_history: list[dict[str, float]] = [initial_rates]
        self.peak_infection_rate = self.actual_total_infection_rate()
        self.min_global_economic_score = self.global_economic_score()
        self.invalid_action_count = 0
        self.severe_infection_events = 0
        self.collapsed_quarantine_events = 0
        self.unnecessary_quarantine_steps = 0

    def quarantine(self, node_id: str) -> str | None:
        node = self.nodes.get(node_id)
        if node is None:
            self.invalid_action_count += 1
            return f"Unknown node '{node_id}'."
        if node.quarantined:
            return f"{node_id} is already quarantined."
        node.quarantined = True
        node.quarantine_streak = 0
        return None

    def lift_quarantine(self, node_id: str) -> str | None:
        node = self.nodes.get(node_id)
        if node is None:
            self.invalid_action_count += 1
            return f"Unknown node '{node_id}'."
        if not node.quarantined:
            return f"{node_id} is not quarantined."
        node.quarantined = False
        node.quarantine_streak = 0
        return None

    def vaccinate(self, node_id: str, amount: float) -> tuple[float, str | None]:
        node = self.nodes.get(node_id)
        if node is None:
            self.invalid_action_count += 1
            return 0.0, f"Unknown node '{node_id}'."
        if amount <= 0.0:
            self.invalid_action_count += 1
            return 0.0, "Vaccination amount must be positive."

        affordable = self.vaccine_budget / self.task_config.vaccine_cost_per_person
        vaccinated = min(amount, node.susceptible, affordable)
        if vaccinated <= 0.0:
            return 0.0, f"No vaccination capacity remains for {node_id}."

        node.susceptible -= vaccinated
        node.recovered += vaccinated
        self.vaccine_budget = max(
            0.0,
            self.vaccine_budget - vaccinated * self.task_config.vaccine_cost_per_person,
        )
        return vaccinated, None

    def neighbor_exposure(self, node_id: str) -> float:
        exposure = 0.0
        for edge in self._incoming_edges[node_id]:
            source = self.nodes[edge.source]
            exposure += edge.transmission_rate * source.actual_infection_rate
        return exposure

    def actual_node_infection_rates(self) -> dict[str, float]:
        return {
            node_id: node.actual_infection_rate for node_id, node in self.nodes.items()
        }

    def reported_node_infection_rates(self) -> dict[str, float]:
        lag = self.task_config.reporting_lag_steps
        index = max(0, len(self._actual_rate_history) - 1 - lag)
        return dict(self._actual_rate_history[index])

    def actual_total_infection_rate(self) -> float:
        total_population = sum(node.population for node in self.nodes.values())
        if total_population <= 0:
            return 0.0
        total_infected = sum(node.infected for node in self.nodes.values())
        return max(0.0, min(1.0, total_infected / total_population))

    def reported_total_infection_rate(self) -> float:
        reported_rates = self.reported_node_infection_rates()
        total_population = sum(node.population for node in self.nodes.values())
        weighted = sum(
            self.nodes[node_id].population * rate
            for node_id, rate in reported_rates.items()
        )
        return (
            0.0
            if total_population <= 0
            else max(0.0, min(1.0, weighted / total_population))
        )

    def global_economic_score(self) -> float:
        total_population = sum(node.population for node in self.nodes.values())
        if total_population <= 0:
            return 0.0
        weighted_score = sum(
            node.population * node.economic_health for node in self.nodes.values()
        )
        return max(0.0, min(1.0, weighted_score / total_population))

    def active_edges(self) -> list[dict[str, Any]]:
        edges: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for edge in self.task_config.edges:
            pair = tuple(sorted((edge.source, edge.target)))
            if pair in seen:
                continue
            seen.add(pair)
            active = (
                not self.nodes[edge.source].quarantined
                and not self.nodes[edge.target].quarantined
            )
            edges.append(
                {
                    "source": edge.source,
                    "target": edge.target,
                    "transmission_rate": edge.transmission_rate,
                    "active": active,
                }
            )
        return edges

    def step(self) -> dict[str, Any]:
        unnecessary_quarantine_node_ids = [
            node.node_id
            for node in self.nodes.values()
            if node.quarantined
            and node.actual_infection_rate < self.task_config.unnecessary_infection_threshold
            and self.neighbor_exposure(node.node_id)
            < self.task_config.unnecessary_neighbor_exposure_threshold
        ]
        self.unnecessary_quarantine_steps += len(unnecessary_quarantine_node_ids)

        infection_updates: dict[str, tuple[float, float]] = {}
        for node_id, node in self.nodes.items():
            local_force = node.local_transmission_rate * node.actual_infection_rate
            travel_force = 0.0
            if not node.quarantined:
                for edge in self._incoming_edges[node_id]:
                    source = self.nodes[edge.source]
                    if source.quarantined:
                        continue
                    travel_force += edge.transmission_rate * source.actual_infection_rate

            infection_probability = 1.0 - exp(-(local_force + travel_force))
            new_infections = min(node.susceptible, node.susceptible * infection_probability)
            recoveries = min(node.infected, node.infected * node.recovery_rate)
            infection_updates[node_id] = (new_infections, recoveries)

        for node_id, (new_infections, recoveries) in infection_updates.items():
            node = self.nodes[node_id]
            node.susceptible = max(0.0, node.susceptible - new_infections)
            node.infected = max(0.0, node.infected + new_infections - recoveries)
            node.recovered = min(float(node.population), node.recovered + recoveries)

        severe_node_ids: list[str] = []
        collapsed_node_ids: list[str] = []
        for node in self.nodes.values():
            infection_drag = node.disease_economic_drag * node.actual_infection_rate
            if node.quarantined:
                node.quarantine_streak += 1
                node.economic_health = max(
                    0.0,
                    node.economic_health - node.quarantine_economic_drain - infection_drag,
                )
            else:
                node.quarantine_streak = 0
                node.economic_health = min(
                    1.0,
                    max(
                        0.0,
                        node.economic_health
                        - infection_drag
                        + node.economic_recovery_rate * (1.0 - node.economic_health),
                    ),
                )

            if node.actual_infection_rate >= 0.80:
                severe_node_ids.append(node.node_id)
            if node.quarantined and node.economic_health <= 0.0:
                collapsed_node_ids.append(node.node_id)

        self.severe_infection_events += len(severe_node_ids)
        self.collapsed_quarantine_events += len(collapsed_node_ids)
        self.step_count += 1

        actual_rates = self.actual_node_infection_rates()
        self._actual_rate_history.append(actual_rates)
        actual_total = self.actual_total_infection_rate()
        global_economy = self.global_economic_score()
        self.peak_infection_rate = max(self.peak_infection_rate, actual_total)
        self.min_global_economic_score = min(
            self.min_global_economic_score, global_economy
        )

        return {
            "step_count": self.step_count,
            "actual_total_infection_rate": actual_total,
            "reported_total_infection_rate": self.reported_total_infection_rate(),
            "global_economic_score": global_economy,
            "severe_node_ids": severe_node_ids,
            "collapsed_node_ids": collapsed_node_ids,
            "unnecessary_quarantine_node_ids": unnecessary_quarantine_node_ids,
            "done": self.step_count >= self.task_config.max_steps,
        }
