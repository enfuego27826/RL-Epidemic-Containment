"""OpenEnv adapter that bridges EpidemicContainmentStrategyEnv to an RL training interface.

The adapter normalises raw Pydantic observations into flat tensors that a
policy network can consume, and maps integer/float action tensors back into
``EpidemicAction`` objects accepted by the environment.

Observation vector layout (per step)
-------------------------------------
For each node i  (up to ``max_nodes`` slots, padded with zeros):

  [0]  known_infection_rate   float in [0, 1]
  [1]  economic_health        float in [0, 1]
  [2]  is_quarantined         0.0 or 1.0
  [3]  population_fraction    node_population / sum_populations

Global scalars appended after all node slots:

  [-4] vaccine_budget_fraction        vaccine_budget / initial_vaccine_budget
  [-3] global_economic_score          float in [0, 1]
  [-2] reported_total_infection_rate  float in [0, 1]
  [-1] step_fraction                  step_count / max_steps

Total obs dim = max_nodes * NODE_FEATURE_DIM + GLOBAL_FEATURE_DIM

Action encoding (v0 — discrete-only baseline)
---------------------------------------------
The action is represented as an integer array of shape (max_nodes,) where
each element encodes the intervention for node i:

  0 → no-op
  1 → quarantine
  2 → lift_quarantine
  3 → vaccinate (fixed amount = VACCINATE_AMOUNT_FRACTION * vaccine_budget)

TODO (Phase 2): replace with hybrid action space:
  - discrete head:  per-node {no-op, quarantine, lift, vaccinate-candidate}
  - continuous head: vaccine allocation vector (nonneg, budget-projected)

TODO (Phase 3): add graph adjacency tensor output for ST-GNN encoder.
TODO (Phase 4): expose observation history buffer for belief-state module.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path fix: allow importing env/tasks/models from the repo root whether this
# module is invoked from the project root or from within src/.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from env import EpidemicContainmentStrategyEnv  # noqa: E402  (after path fix)
from models import EpidemicAction  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NODE_FEATURE_DIM: int = 4
GLOBAL_FEATURE_DIM: int = 4

# Fraction of current vaccine_budget used for a single vaccinate action.
VACCINATE_AMOUNT_FRACTION: float = 0.25

# Discrete action codes
ACTION_NOOP: int = 0
ACTION_QUARANTINE: int = 1
ACTION_LIFT: int = 2
ACTION_VACCINATE: int = 3
NUM_DISCRETE_ACTIONS: int = 4


class OpenEnvAdapter:
    """Thin adapter wrapping :class:`EpidemicContainmentStrategyEnv`.

    Provides a Gym-like interface returning normalised tensors (plain Python
    lists of floats) rather than Pydantic models, making it easy to plug into
    any PyTorch training loop.

    Parameters
    ----------
    task_name:
        One of ``"easy_localized_outbreak"``, ``"medium_multi_center_spread"``,
        or ``"hard_asymptomatic_high_density"``.
    seed:
        Random seed forwarded to the underlying environment.
    max_nodes:
        Maximum number of nodes expected.  Observations are padded to this
        length so the tensor shape is always fixed.
    """

    def __init__(
        self,
        task_name: str = "easy_localized_outbreak",
        seed: int | None = 42,
        max_nodes: int = 20,
    ) -> None:
        self.task_name = task_name
        self.seed = seed
        self.max_nodes = max_nodes

        self._env = EpidemicContainmentStrategyEnv(task_name=task_name, seed=seed)
        self._node_ids: list[str] = []
        self._total_population: float = 1.0
        self._initial_vaccine_budget: float = 1.0

        # obs_dim is fixed after the first reset()
        self.obs_dim: int = max_nodes * NODE_FEATURE_DIM + GLOBAL_FEATURE_DIM

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None) -> tuple[list[float], dict[str, Any]]:
        """Reset the episode and return the initial observation tensor.

        Parameters
        ----------
        seed:
            If provided, overrides the adapter-level seed for this episode.

        Returns
        -------
        obs:
            Flat observation vector of length ``self.obs_dim``.
        info:
            Dict with ``node_ids``, ``num_nodes``, ``obs_dim`` for downstream
            use.
        """
        obs_model = self._env.reset(seed=seed if seed is not None else self.seed)
        self._node_ids = [n.node_id for n in obs_model.nodes]
        self._total_population = float(
            sum(n.population for n in obs_model.nodes) or 1.0
        )
        self._initial_vaccine_budget = float(
            self._env.engine.initial_vaccine_budget or 1.0
        )

        obs_tensor = self._obs_to_tensor(obs_model)
        info: dict[str, Any] = {
            "node_ids": self._node_ids,
            "num_nodes": len(self._node_ids),
            "obs_dim": self.obs_dim,
        }
        return obs_tensor, info

    def step(
        self,
        action: list[int],
    ) -> tuple[list[float], float, bool, dict[str, Any]]:
        """Apply an action and return the next transition.

        Parameters
        ----------
        action:
            Integer list of length ``max_nodes``.  Entries beyond
            ``num_nodes`` are ignored.  Each integer is one of
            ``ACTION_NOOP``, ``ACTION_QUARANTINE``, ``ACTION_LIFT``,
            ``ACTION_VACCINATE``.

        Returns
        -------
        obs_tensor:
            Flat observation vector of length ``self.obs_dim``.
        reward:
            Scalar float reward.
        done:
            Episode termination flag.
        info:
            Dict forwarded from the underlying env plus ``"node_ids"``.
        """
        epidemic_action = self._action_to_epidemic(action)
        obs_model, reward, done, info = self._env.step(epidemic_action)
        obs_tensor = self._obs_to_tensor(obs_model)
        info["node_ids"] = self._node_ids
        return obs_tensor, float(reward), bool(done), info

    @property
    def num_nodes(self) -> int:
        """Number of active nodes in the current episode."""
        return len(self._node_ids)

    # ------------------------------------------------------------------
    # Observation normalisation
    # ------------------------------------------------------------------

    def _obs_to_tensor(self, obs_model: Any) -> list[float]:
        """Convert a Pydantic ``EpidemicObservation`` to a flat float list.

        Node features are stored in node-major order (all features for node 0,
        then node 1, …).  Slots beyond ``num_nodes`` are zero-padded.

        TODO (Phase 3): also return an adjacency matrix / edge-index tensor
        for the ST-GNN encoder.
        """
        tensor: list[float] = []

        node_map = {n.node_id: n for n in obs_model.nodes}

        for i in range(self.max_nodes):
            if i < len(self._node_ids):
                node = node_map[self._node_ids[i]]
                pop_frac = node.population / self._total_population
                tensor += [
                    float(node.known_infection_rate),
                    float(node.economic_health),
                    1.0 if node.is_quarantined else 0.0,
                    float(pop_frac),
                ]
            else:
                # padding
                tensor += [0.0, 0.0, 0.0, 0.0]

        # Global scalars
        vax_frac = float(obs_model.vaccine_budget) / max(
            self._initial_vaccine_budget, 1e-8
        )
        step_frac = float(obs_model.step_count) / max(float(obs_model.max_steps), 1.0)
        tensor += [
            min(vax_frac, 1.0),
            float(obs_model.global_economic_score),
            float(obs_model.reported_total_infection_rate),
            step_frac,
        ]

        assert len(tensor) == self.obs_dim, (
            f"Obs tensor length {len(tensor)} != expected {self.obs_dim}"
        )
        return tensor

    # ------------------------------------------------------------------
    # Action mapping
    # ------------------------------------------------------------------

    def _action_to_epidemic(self, action: list[int]) -> EpidemicAction:
        """Convert a discrete per-node action array to ``EpidemicAction``.

        Parameters
        ----------
        action:
            Integer list of length ``max_nodes``.

        Returns
        -------
        EpidemicAction with a (possibly empty) interventions list.

        TODO (Phase 2): extend to handle hybrid (discrete + continuous) by
        accepting a dict with ``"discrete"`` and ``"continuous"`` keys.
        TODO (Phase 2): generate action masks for invalid action pruning.
        """
        interventions: list[dict[str, Any]] = []
        vax_budget = float(self._env.engine.vaccine_budget)
        vax_amount = VACCINATE_AMOUNT_FRACTION * vax_budget

        for i, act in enumerate(action[: self.num_nodes]):
            node_id = self._node_ids[i]
            if act == ACTION_QUARANTINE:
                interventions.append({"kind": "quarantine", "node_id": node_id})
            elif act == ACTION_LIFT:
                interventions.append({"kind": "lift_quarantine", "node_id": node_id})
            elif act == ACTION_VACCINATE and vax_amount > 0:
                interventions.append(
                    {"kind": "vaccinate", "node_id": node_id, "amount": vax_amount}
                )
            # ACTION_NOOP → no entry

        # EpidemicAction enforces a hard cap of 3 interventions per step.
        MAX_INTERVENTIONS = 3
        return EpidemicAction.model_validate({"interventions": interventions[:MAX_INTERVENTIONS]})
