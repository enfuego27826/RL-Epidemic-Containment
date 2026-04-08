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

Action encoding (v1 — Phase 2 hybrid)
--------------------------------------
A dict with two keys is accepted:

  {
    "discrete":   list[int]   # per-node {0,1,2,3}
    "continuous": list[float] # per-node vaccine allocation, sum <= budget
  }

When the ``"discrete"`` key for node i is ACTION_VACCINATE (3), the
corresponding ``continuous[i]`` amount is injected as the vaccinate amount.
Nodes with discrete != 3 do not consume vaccine budget.

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

# Fraction of current vaccine_budget used for a single vaccinate action (v0 baseline).
VACCINATE_AMOUNT_FRACTION: float = 0.25

# Discrete action codes
ACTION_NOOP: int = 0
ACTION_QUARANTINE: int = 1
ACTION_LIFT: int = 2
ACTION_VACCINATE: int = 3
NUM_DISCRETE_ACTIONS: int = 4

# Minimum amount treated as a meaningful vaccine dose (continuous head)
_MIN_VACCINE_AMOUNT: float = 1e-4
# Mirror reward coefficients currently defined in /home/runner/work/RL-Epidemic-Containment/RL-Epidemic-Containment/env.py
# (_compute_reward). Keep these in sync with env.py if reward weights change.
#   reward += 8.0 * infection_delta + 2.5 * economy_delta + ...
# These are exposed here for per-step diagnostics / MORL decomposition logging.
HEALTH_REWARD_COEFFICIENT: float = 8.0
ECONOMY_REWARD_COEFFICIENT: float = 2.5


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

        # Phase 2: invalid-action diagnostics accumulated per episode.
        # _invalid_action_count: total invalid node-decisions across all steps.
        # _total_decisions: total node-decisions checked (= steps * num_nodes),
        #   used as denominator so invalid_action_rate is truly in [0, 1].
        self._invalid_action_count: int = 0
        self._total_steps: int = 0
        self._total_decisions: int = 0

        # Per-action-type invalid counts for diagnostics.
        # Keys match action name strings: "quarantine", "lift", "vaccinate".
        self._invalid_by_type: dict[str, int] = {
            "quarantine": 0,
            "lift": 0,
            "vaccinate": 0,
        }
        self._prev_global_economic_score: float | None = None
        self._prev_actual_infection_rate: float | None = None

    # ------------------------------------------------------------------
    # OpenEnv conformance assertion
    # ------------------------------------------------------------------

    @staticmethod
    def assert_openenv_conformance() -> None:
        """Assert that the OpenEnv integration is active and well-formed.

        Raises
        ------
        ImportError
            If the ``env`` module (providing ``EpidemicContainmentStrategyEnv``)
            cannot be imported, indicating openenv-core is not installed or
            the env adapter is missing.
        AssertionError
            If the ``EpidemicContainmentStrategyEnv`` class is not exported
            from the env module (unexpected but defensive check).
        """
        import env as _env_module
        assert hasattr(_env_module, "EpidemicContainmentStrategyEnv"), (
            "OpenEnv conformance check failed: EpidemicContainmentStrategyEnv not "
            "exported from env module.  Ensure openenv-core is installed and the "
            "env.py adapter is up-to-date."
        )

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

        # Reset per-episode diagnostics
        self._invalid_action_count = 0
        self._total_steps = 0
        self._total_decisions = 0
        self._invalid_by_type = {"quarantine": 0, "lift": 0, "vaccinate": 0}
        self._prev_global_economic_score = float(self._env.engine.global_economic_score())
        self._prev_actual_infection_rate = float(self._env.engine.actual_total_infection_rate())

        obs_tensor = self._obs_to_tensor(obs_model)
        info: dict[str, Any] = {
            "node_ids": self._node_ids,
            "num_nodes": len(self._node_ids),
            "obs_dim": self.obs_dim,
        }
        return obs_tensor, info

    def step(
        self,
        action: list[int] | dict[str, Any],
    ) -> tuple[list[float], float, bool, dict[str, Any]]:
        """Apply an action and return the next transition.

        Parameters
        ----------
        action:
            Either:

            * **Phase 1 (discrete-only)**: integer list of length ``max_nodes``.
              Each element is one of ``ACTION_NOOP``, ``ACTION_QUARANTINE``,
              ``ACTION_LIFT``, ``ACTION_VACCINATE``.

            * **Phase 2 (hybrid)**: dict with keys:
              - ``"discrete"``:   list[int] of length ``max_nodes``
              - ``"continuous"``: list[float] of length ``max_nodes``,
                non-negative vaccine allocations.

        Returns
        -------
        obs_tensor:
            Flat observation vector of length ``self.obs_dim``.
        reward:
            Scalar float reward.
        done:
            Episode termination flag.
        info:
            Dict forwarded from the underlying env plus ``"node_ids"`` and
            Phase 2 diagnostics (``"invalid_action_count"``, ``"invalid_action_rate"``).
            Also includes ``reward_health`` and ``reward_economy`` terms
            reconstructed from infection/economy deltas; these are partial
            decomposition terms and do not necessarily sum to scalar reward.
        """
        epidemic_action = self._action_to_epidemic(action)
        prev_global_economic_score = (
            float(self._prev_global_economic_score)
            if self._prev_global_economic_score is not None
            else float(self._env.engine.global_economic_score())
        )
        prev_actual_infection_rate = (
            float(self._prev_actual_infection_rate)
            if self._prev_actual_infection_rate is not None
            else float(self._env.engine.actual_total_infection_rate())
        )
        obs_model, reward, done, info = self._env.step(epidemic_action)
        obs_tensor = self._obs_to_tensor(obs_model)
        curr_global_economic_score = float(self._env.engine.global_economic_score())
        curr_actual_infection_rate = float(self._env.engine.actual_total_infection_rate())
        economy_delta = curr_global_economic_score - prev_global_economic_score
        infection_delta = prev_actual_infection_rate - curr_actual_infection_rate
        self._total_steps += 1
        self._total_decisions += max(self.num_nodes, 1)
        info["node_ids"] = self._node_ids
        info["global_economic_score"] = curr_global_economic_score
        info["actual_total_infection_rate"] = curr_actual_infection_rate
        # Decomposed reward diagnostics (matches env.py coefficient terms for
        # infection/economy deltas). Full scalar reward also includes bonuses
        # and penalties in env.py, so these fields are partial decomposition.
        info["reward_health"] = HEALTH_REWARD_COEFFICIENT * infection_delta
        info["reward_economy"] = ECONOMY_REWARD_COEFFICIENT * economy_delta
        info["economy_score"] = curr_global_economic_score
        info["invalid_action_count"] = self._invalid_action_count
        info["invalid_action_rate"] = (
            self._invalid_action_count / self._total_decisions
            if self._total_decisions > 0 else 0.0
        )
        # Per-action-type invalid counts (cumulative for this episode)
        info["invalid_by_type"] = dict(self._invalid_by_type)
        self._prev_global_economic_score = curr_global_economic_score
        self._prev_actual_infection_rate = curr_actual_infection_rate
        return obs_tensor, float(reward), bool(done), info

    @property
    def num_nodes(self) -> int:
        """Number of active nodes in the current episode."""
        return len(self._node_ids)

    @property
    def vaccine_budget(self) -> float:
        """Current remaining vaccine budget from the underlying environment."""
        return float(self._env.engine.vaccine_budget)

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

    def _action_to_epidemic(self, action: list[int] | dict[str, Any]) -> EpidemicAction:
        """Convert a discrete or hybrid action to ``EpidemicAction``.

        Supports two calling conventions:

        Phase 1 (discrete-only):
            ``action`` is a list[int] of length ``max_nodes``.

        Phase 2 (hybrid):
            ``action`` is a dict with:
              - ``"discrete"``:   list[int] per-node action codes
              - ``"continuous"``: list[float] per-node vaccine allocations

        Invalid discrete choices (e.g., lifting a non-quarantined node) are
        detected here and counted in ``self._invalid_action_count`` as a
        defensive measure; the masking layer should prevent them before
        reaching this point.

        Returns
        -------
        EpidemicAction with a (possibly empty) interventions list.
        """
        interventions: list[dict[str, Any]] = []
        vax_budget = float(self._env.engine.vaccine_budget)

        # --- Unpack action -----------------------------------------------
        if isinstance(action, dict):
            discrete_acts: list[int] = list(action["discrete"])
            continuous_alloc: list[float] = list(action.get("continuous", []))
            # Pad/clip continuous to num_nodes length
            while len(continuous_alloc) < self.num_nodes:
                continuous_alloc.append(0.0)
        else:
            # Phase 1 fallback: list of ints
            discrete_acts = list(action)
            continuous_alloc = [VACCINATE_AMOUNT_FRACTION * vax_budget] * self.num_nodes

        # --- Build quarantine state for validation -----------------------
        # Read from the current observation (lazy rebuild from env state)
        quarantined: set[str] = set()
        try:
            state = self._env.state()
            for node_state in state.nodes:
                if getattr(node_state, "is_quarantined", False):
                    quarantined.add(node_state.node_id)
        except Exception:
            pass  # best-effort; don't crash on state access

        # --- Map actions -------------------------------------------------
        for i, act in enumerate(discrete_acts[: self.num_nodes]):
            node_id = self._node_ids[i]
            is_q = node_id in quarantined

            if act == ACTION_QUARANTINE:
                if is_q:
                    # Already quarantined — invalid; count and skip
                    self._invalid_action_count += 1
                    self._invalid_by_type["quarantine"] += 1
                    continue
                interventions.append({"kind": "quarantine", "node_id": node_id})

            elif act == ACTION_LIFT:
                if not is_q:
                    # Not quarantined — invalid; count and skip
                    self._invalid_action_count += 1
                    self._invalid_by_type["lift"] += 1
                    continue
                interventions.append({"kind": "lift_quarantine", "node_id": node_id})

            elif act == ACTION_VACCINATE:
                amount = float(continuous_alloc[i]) if i < len(continuous_alloc) else 0.0
                amount = max(amount, 0.0)  # clamp negative
                if amount > _MIN_VACCINE_AMOUNT and vax_budget > 0:
                    # Cap amount to remaining budget
                    amount = min(amount, vax_budget)
                    interventions.append(
                        {"kind": "vaccinate", "node_id": node_id, "amount": amount}
                    )
            # ACTION_NOOP → no entry

        # EpidemicAction enforces a hard cap of 3 interventions per step.
        MAX_INTERVENTIONS = 3
        return EpidemicAction.model_validate({"interventions": interventions[:MAX_INTERVENTIONS]})
