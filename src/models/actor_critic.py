"""Minimal MLP actor-critic for the Phase 1 PPO baseline.

Architecture
------------
Shared trunk:   obs → Linear(hidden) → ReLU → Linear(hidden) → ReLU
Actor head:     trunk → Linear(action_dim)  → raw logits per node-action
Critic head:    trunk → Linear(1)            → state value

The actor treats each node independently (per-node discrete action), which
is a simplification that will be replaced by the hybrid head in Phase 2.

TODO (Phase 2): add hybrid head with separate discrete/continuous branches.
TODO (Phase 3): replace MLP trunk with ST-GNN encoder.
"""

from __future__ import annotations

import math
from typing import Sequence


def _relu(x: float) -> float:
    return max(0.0, x)


def _softmax(logits: list[float]) -> list[float]:
    max_l = max(logits)
    exps = [math.exp(l - max_l) for l in logits]
    total = sum(exps)
    return [e / total for e in exps]


def _linear(
    x: list[float],
    weight: list[list[float]],
    bias: list[float],
) -> list[float]:
    """Dense layer: y = x @ W^T + b (no external deps)."""
    out_dim = len(bias)
    out = list(bias)
    for j in range(out_dim):
        for i, xi in enumerate(x):
            out[j] += xi * weight[j][i]
    return out


class MLP:
    """Simple multi-layer perceptron using only the standard library.

    Parameters
    ----------
    in_dim:
        Input feature dimension.
    hidden_dims:
        Sequence of hidden layer widths.
    out_dim:
        Output dimension.
    seed:
        Random seed for weight initialisation (Xavier uniform-style).
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dims: Sequence[int],
        out_dim: int,
        seed: int = 0,
    ) -> None:
        import random

        rng = random.Random(seed)
        self._layers: list[tuple[list[list[float]], list[float]]] = []
        dims = [in_dim, *list(hidden_dims), out_dim]
        for in_d, out_d in zip(dims[:-1], dims[1:]):
            limit = math.sqrt(6.0 / (in_d + out_d))
            W = [[rng.uniform(-limit, limit) for _ in range(in_d)] for _ in range(out_d)]
            b = [0.0] * out_d
            self._layers.append((W, b))

    def forward(self, x: list[float]) -> list[float]:
        """Forward pass; applies ReLU after all but the last layer."""
        h = list(x)
        for idx, (W, b) in enumerate(self._layers):
            h = _linear(h, W, b)
            if idx < len(self._layers) - 1:
                h = [_relu(v) for v in h]
        return h


class ActorCritic:
    """Shared-trunk MLP actor-critic network.

    Parameters
    ----------
    obs_dim:
        Dimension of the flat observation vector.
    action_dim:
        Number of discrete actions per node.
    max_nodes:
        Maximum nodes (sets action head output = max_nodes * action_dim).
    hidden_dims:
        Widths of shared hidden layers.
    seed:
        Weight initialisation seed.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        max_nodes: int,
        hidden_dims: Sequence[int] = (128, 128),
        seed: int = 0,
    ) -> None:
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.max_nodes = max_nodes

        # Shared trunk
        self._trunk = MLP(obs_dim, hidden_dims, hidden_dims[-1], seed=seed)

        # Actor: outputs logits for (max_nodes * action_dim) actions
        actor_out_dim = max_nodes * action_dim
        self._actor_head = MLP(hidden_dims[-1], [], actor_out_dim, seed=seed + 1)

        # Critic: outputs a single state value
        self._critic_head = MLP(hidden_dims[-1], [], 1, seed=seed + 2)

    def forward(
        self,
        obs: list[float],
    ) -> tuple[list[list[float]], float]:
        """Compute per-node action logits and state value.

        Parameters
        ----------
        obs:
            Flat observation vector of length ``obs_dim``.

        Returns
        -------
        action_logits:
            Shape (max_nodes, action_dim) — logits for each node's discrete
            action distribution.
        value:
            Scalar state-value estimate.
        """
        h = self._trunk.forward(obs)
        flat_logits = self._actor_head.forward(h)
        value = self._critic_head.forward(h)[0]

        # Reshape flat logits to (max_nodes, action_dim)
        action_logits = [
            flat_logits[i * self.action_dim : (i + 1) * self.action_dim]
            for i in range(self.max_nodes)
        ]
        return action_logits, value

    def act(
        self,
        obs: list[float],
        deterministic: bool = False,
    ) -> tuple[list[int], list[float], float]:
        """Sample (or greedily select) a per-node action.

        Parameters
        ----------
        obs:
            Flat observation vector.
        deterministic:
            If True, choose the argmax per node; otherwise sample.

        Returns
        -------
        actions:
            Integer list of length ``max_nodes``.
        log_probs:
            Per-node log-probabilities of the chosen actions.
        value:
            Critic estimate V(obs).
        """
        import math
        import random

        action_logits, value = self.forward(obs)
        actions: list[int] = []
        log_probs: list[float] = []

        for logits in action_logits:
            probs = _softmax(logits)
            if deterministic:
                a = max(range(len(probs)), key=lambda i: probs[i])
            else:
                r = random.random()
                cumsum = 0.0
                a = len(probs) - 1
                for idx, p in enumerate(probs):
                    cumsum += p
                    if r < cumsum:
                        a = idx
                        break
            actions.append(a)
            log_probs.append(math.log(max(probs[a], 1e-8)))

        return actions, log_probs, value
