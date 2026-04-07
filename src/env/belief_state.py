"""Phase 4 — Delayed Observation / Belief State.

This module provides lag-aware observation handling for the epidemic
containment environment.  Under a DOMDP (Delayed Observation MDP), the
agent receives observations that are ``lag`` steps behind reality.

Architecture
------------
1. **ObservationBuffer** — circular buffer that stores the last
   ``max_history`` raw observation tensors and corresponding actions.

2. **BeliefStateBuilder** — combines the buffered observations and actions
   into a single belief-state vector suitable for consumption by the
   actor-critic.  Two modes are supported:

   - ``"concat"`` (default):  concatenate the last ``history_len``
     observations.  Fast and simple; output dim = obs_dim * history_len.
   - ``"mean"``  : mean-pool the buffered observations.  Output dim = obs_dim.

3. **LagAwareAdapter** — thin wrapper around ``OpenEnvAdapter`` that:
   - maintains a ``ObservationBuffer``,
   - returns belief-state observations rather than raw observations,
   - exposes the lag setting through the config.

Usage
-----
The adapter is configured via the ``lag`` section of the experiment config:

    lag:
      steps: 2            # reporting lag in env steps
      history_len: 4      # number of past obs to include in belief state
      mode: concat        # "concat" | "mean"
      pad_with_zeros: true  # fill missing history entries with zeros

The ``LagAwareAdapter`` can be used as a drop-in replacement for
``OpenEnvAdapter`` in training/evaluation scripts.
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# ObservationBuffer
# ---------------------------------------------------------------------------

class ObservationBuffer:
    """Fixed-size circular buffer for observation + action history.

    Parameters
    ----------
    obs_dim:
        Length of each observation vector.
    action_keys:
        Keys expected in the action dict (for hybrid actions).
        Set to None for scalar actions.
    max_history:
        Maximum number of steps to retain.
    """

    def __init__(
        self,
        obs_dim: int,
        max_history: int = 8,
        action_keys: list[str] | None = None,
    ) -> None:
        self.obs_dim = obs_dim
        self.max_history = max_history
        self.action_keys = action_keys or []
        self._obs_buf: list[list[float]] = []
        self._act_buf: list[Any] = []

    def reset(self, initial_obs: list[float]) -> None:
        """Reset buffer for a new episode, pre-filling with initial obs."""
        self._obs_buf = [list(initial_obs)] * self.max_history
        self._act_buf = [None] * self.max_history

    def push(self, obs: list[float], action: Any = None) -> None:
        """Push a new observation (and optionally the action taken) into the buffer."""
        self._obs_buf.append(list(obs))
        self._act_buf.append(action)
        if len(self._obs_buf) > self.max_history:
            self._obs_buf.pop(0)
            self._act_buf.pop(0)

    def get_history(self, history_len: int) -> list[list[float]]:
        """Return the last ``history_len`` observations (oldest first).

        If fewer observations are available, the oldest slots are returned
        (pre-filled with the initial observation at reset).
        """
        buf = self._obs_buf[-history_len:]
        # Pad front with zeros if buffer is still filling
        while len(buf) < history_len:
            buf = [[0.0] * self.obs_dim] + buf
        return buf

    def get_lagged_obs(self, lag: int) -> list[float]:
        """Return the observation from ``lag`` steps ago.

        Parameters
        ----------
        lag:
            Number of steps to look back (0 = current obs).

        Returns
        -------
        Observation vector.  If lag exceeds buffer, returns the oldest
        available observation.
        """
        if lag <= 0:
            return list(self._obs_buf[-1])
        idx = max(0, len(self._obs_buf) - 1 - lag)
        return list(self._obs_buf[idx])

    @property
    def latest_obs(self) -> list[float]:
        """The most recently pushed observation."""
        return list(self._obs_buf[-1]) if self._obs_buf else [0.0] * self.obs_dim


# ---------------------------------------------------------------------------
# BeliefStateBuilder
# ---------------------------------------------------------------------------

class BeliefStateBuilder:
    """Constructs a belief-state vector from the observation history buffer.

    Parameters
    ----------
    obs_dim:
        Dimension of each individual observation.
    history_len:
        Number of past steps to include.
    mode:
        ``"concat"`` — concatenate last history_len obs (output: obs_dim * history_len).
        ``"mean"``   — mean-pool over history (output: obs_dim).
    """

    def __init__(
        self,
        obs_dim: int,
        history_len: int = 4,
        mode: str = "concat",
    ) -> None:
        if mode not in ("concat", "mean"):
            raise ValueError(f"BeliefStateBuilder mode must be 'concat' or 'mean', got {mode!r}")
        self.obs_dim = obs_dim
        self.history_len = history_len
        self.mode = mode

        if mode == "concat":
            self.belief_dim = obs_dim * history_len
        else:
            self.belief_dim = obs_dim

    def build(self, buffer: ObservationBuffer) -> list[float]:
        """Build the belief state from the buffer.

        Parameters
        ----------
        buffer:
            An ``ObservationBuffer`` that has been reset and pushed to.

        Returns
        -------
        belief:
            Flat belief-state vector of length ``self.belief_dim``.
        """
        history = buffer.get_history(self.history_len)
        if self.mode == "concat":
            belief: list[float] = []
            for obs in history:
                belief.extend(obs)
            return belief
        else:
            # Mean pool
            n = len(history)
            result = [0.0] * self.obs_dim
            for obs in history:
                for k, v in enumerate(obs):
                    result[k] += v
            return [v / n for v in result]


# ---------------------------------------------------------------------------
# LagAwareAdapter
# ---------------------------------------------------------------------------

class LagAwareAdapter:
    """Drop-in replacement for OpenEnvAdapter with lag-aware belief state.

    Parameters
    ----------
    task_name:
        Task name forwarded to ``OpenEnvAdapter``.
    seed:
        Random seed.
    max_nodes:
        Maximum node slots.
    lag_steps:
        Reporting lag (in env steps).  0 = no lag.
    history_len:
        Number of past observations to include in the belief state.
    mode:
        Belief state construction mode: ``"concat"`` or ``"mean"``.
    """

    def __init__(
        self,
        task_name: str = "easy_localized_outbreak",
        seed: int | None = 42,
        max_nodes: int = 20,
        lag_steps: int = 2,
        history_len: int = 4,
        mode: str = "concat",
    ) -> None:
        import sys
        from pathlib import Path

        _REPO_ROOT = Path(__file__).resolve().parents[2]
        if str(_REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(_REPO_ROOT))

        from src.env.openenv_adapter import OpenEnvAdapter

        self._adapter = OpenEnvAdapter(
            task_name=task_name,
            seed=seed,
            max_nodes=max_nodes,
        )
        self.lag_steps = max(0, lag_steps)
        self.history_len = history_len
        self.mode = mode
        self.obs_dim = self._adapter.obs_dim  # raw obs dim
        self.max_nodes = max_nodes

        self._buffer = ObservationBuffer(
            obs_dim=self.obs_dim,
            max_history=max(history_len + lag_steps + 2, 8),
        )
        self._belief_builder = BeliefStateBuilder(
            obs_dim=self.obs_dim,
            history_len=history_len,
            mode=mode,
        )

        # Belief state dim exposed for policy construction
        self.belief_dim = self._belief_builder.belief_dim

    # ------------------------------------------------------------------
    # Gym-like interface
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None) -> tuple[list[float], dict[str, Any]]:
        """Reset episode and return initial belief state."""
        raw_obs, info = self._adapter.reset(seed=seed)
        self._buffer.reset(raw_obs)
        belief = self._belief_builder.build(self._buffer)
        info["belief_dim"] = self.belief_dim
        info["lag_steps"] = self.lag_steps
        return belief, info

    def step(
        self,
        action: list[int] | dict[str, Any],
    ) -> tuple[list[float], float, bool, dict[str, Any]]:
        """Step the environment and return lagged belief state."""
        raw_obs, reward, done, info = self._adapter.step(action)
        self._buffer.push(raw_obs, action)

        # Return the lagged observation as belief
        if self.lag_steps > 0:
            lagged_raw = self._buffer.get_lagged_obs(self.lag_steps)
            # Build belief from lagged position
            # Use internal buffer up to lag position
            lagged_history = self._buffer.get_history(self.history_len)
            # Shift back by lag: use older entries
            lag_buf = ObservationBuffer(obs_dim=self.obs_dim, max_history=self.history_len)
            lag_buf.reset(lagged_history[0])
            for obs in lagged_history[:-self.lag_steps] if self.lag_steps < self.history_len else [lagged_history[0]]:
                lag_buf.push(obs)
            belief = self._belief_builder.build(lag_buf)
        else:
            belief = self._belief_builder.build(self._buffer)

        info["lag_steps"] = self.lag_steps
        info["belief_dim"] = self.belief_dim
        return belief, reward, done, info

    @property
    def num_nodes(self) -> int:
        return self._adapter.num_nodes

    @property
    def vaccine_budget(self) -> float:
        return self._adapter.vaccine_budget
