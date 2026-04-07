"""MLP actor-critic for the Phase 1 PPO baseline (PyTorch implementation).

Architecture
------------
Shared trunk:   obs → Linear(hidden) → ReLU → … → Linear(hidden) → ReLU
Actor head:     trunk → Linear(max_nodes * action_dim)  → per-node logits
Critic head:    trunk → Linear(1)                        → state value

Both ``ActorCritic`` and ``HybridActorCritic`` are ``torch.nn.Module``
subclasses so that ``torch.optim`` can track all parameters and
``loss.backward()`` propagates gradients correctly.

Phase 1 (``ActorCritic``):
    ``forward(obs_tensor)`` returns tensors; ``act()`` wraps it for Python
    callers that expect plain Python lists.

Phase 2 (``HybridActorCritic``):
    ``_forward_tensor(obs_tensor)`` returns tensors for gradient computation.
    ``forward(obs_list)`` wraps it, converting back to Python lists for
    backward-compatibility with ``HybridActionDist`` and the smoke tests.

TODO (Phase 3): replace MLP trunk with ST-GNN encoder.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Kept for backward-compatibility (used by hybrid_action.py and tests)
# ---------------------------------------------------------------------------

def _softmax(logits: list[float]) -> list[float]:
    """Numerically stable softmax over a 1-D list (stdlib-only)."""
    max_l = max(logits)
    exps = [math.exp(l - max_l) for l in logits]
    total = sum(exps)
    return [e / total for e in exps]


# ---------------------------------------------------------------------------
# Internal helper: build a shared MLP trunk
# ---------------------------------------------------------------------------

def _build_mlp(
    in_dim: int,
    hidden_dims: Sequence[int],
    out_dim: int,
) -> nn.Sequential:
    """Build a fully-connected MLP with ReLU after every layer except the last.

    Matches the original pure-Python MLP architecture:
        dims = [in_dim, *hidden_dims, out_dim]
        For each consecutive pair apply Linear; ReLU on all but the last.
    """
    all_dims = [in_dim, *list(hidden_dims), out_dim]
    layers: list[nn.Module] = []
    for i, (d_in, d_out) in enumerate(zip(all_dims[:-1], all_dims[1:])):
        layers.append(nn.Linear(d_in, d_out))
        if i < len(all_dims) - 2:          # ReLU after every layer except the last
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def _init_weights(module: nn.Module, seed: int = 0) -> None:
    """Xavier-uniform init for all ``nn.Linear`` layers."""
    torch.manual_seed(seed)
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)


# ---------------------------------------------------------------------------
# Phase 1: Discrete-only actor-critic
# ---------------------------------------------------------------------------

class ActorCritic(nn.Module):
    """Shared-trunk MLP actor-critic network (PyTorch).

    Parameters
    ----------
    obs_dim:
        Dimension of the flat observation vector.
    action_dim:
        Number of discrete actions per node.
    max_nodes:
        Maximum nodes (action head outputs ``max_nodes * action_dim`` logits).
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
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.max_nodes = max_nodes

        # Shared trunk: obs → [hidden] → hidden_dims[-1] (no final ReLU)
        self.trunk = _build_mlp(obs_dim, list(hidden_dims), hidden_dims[-1])

        # Actor head: one linear layer on top of the trunk
        self.actor_head = nn.Linear(hidden_dims[-1], max_nodes * action_dim)

        # Critic head: single scalar value
        self.critic_head = nn.Linear(hidden_dims[-1], 1)

        _init_weights(self, seed)

    def forward(
        self,
        obs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute per-node action logits and state value.

        Parameters
        ----------
        obs:
            Float tensor of shape ``(obs_dim,)`` or ``(batch, obs_dim)``.

        Returns
        -------
        action_logits:
            Shape ``(..., max_nodes, action_dim)``.
        value:
            Shape ``(..., 1)``.
        """
        h = self.trunk(obs)
        flat_logits = self.actor_head(h)
        value = self.critic_head(h)
        # Reshape to (..., max_nodes, action_dim)
        action_logits = flat_logits.view(
            *flat_logits.shape[:-1], self.max_nodes, self.action_dim
        )
        return action_logits, value

    def act(
        self,
        obs: list[float],
        deterministic: bool = False,
    ) -> tuple[list[int], list[float], float]:
        """Sample (or greedily select) a per-node action without gradients.

        Parameters
        ----------
        obs:
            Flat observation vector (Python list).
        deterministic:
            If True, choose the argmax per node; otherwise sample.

        Returns
        -------
        actions:
            Integer list of length ``max_nodes``.
        log_probs:
            Per-node log-probabilities of the chosen actions.
        value:
            Scalar critic estimate V(obs).
        """
        obs_t = torch.tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            action_logits, value_t = self.forward(obs_t)  # (max_nodes, action_dim), (1,)

        actions: list[int] = []
        log_probs: list[float] = []

        for node_logits in action_logits:          # iterate over max_nodes
            probs = torch.softmax(node_logits, dim=-1)
            if deterministic:
                a = int(probs.argmax().item())
            else:
                a = int(torch.multinomial(probs, 1).item())
            actions.append(a)
            log_probs.append(math.log(max(probs[a].item(), 1e-8)))

        return actions, log_probs, value_t.item()


# ---------------------------------------------------------------------------
# Phase 2: Hybrid actor-critic (discrete + continuous heads)
# ---------------------------------------------------------------------------

class HybridActorCritic(nn.Module):
    """MLP actor-critic with separate discrete and continuous actor heads.

    The continuous head outputs one raw logit per node; ``HybridActionDist``
    applies softplus + budget projection during rollout collection.

    Parameters
    ----------
    obs_dim:
        Dimension of the flat observation vector.
    action_dim:
        Number of discrete actions per node (default 4).
    max_nodes:
        Maximum nodes.
    hidden_dims:
        Widths of shared hidden layers.
    seed:
        Weight initialisation seed.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int = 4,
        max_nodes: int = 20,
        hidden_dims: Sequence[int] = (128, 128),
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.max_nodes = max_nodes

        # Shared trunk
        self.trunk = _build_mlp(obs_dim, list(hidden_dims), hidden_dims[-1])

        # Discrete actor head: (max_nodes * action_dim) logits
        self.discrete_head = nn.Linear(hidden_dims[-1], max_nodes * action_dim)

        # Continuous actor head: max_nodes raw logits
        self.continuous_head = nn.Linear(hidden_dims[-1], max_nodes)

        # Critic head
        self.critic_head = nn.Linear(hidden_dims[-1], 1)

        _init_weights(self, seed)

    # ------------------------------------------------------------------
    # Tensor forward (used for gradient computation in update_policy)
    # ------------------------------------------------------------------

    def _forward_tensor(
        self,
        obs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Differentiable forward pass returning raw tensors.

        Parameters
        ----------
        obs:
            Float tensor of shape ``(obs_dim,)`` or ``(batch, obs_dim)``.

        Returns
        -------
        discrete_logits:
            Shape ``(..., max_nodes, action_dim)``.
        continuous_logits:
            Shape ``(..., max_nodes)``.
        value:
            Shape ``(..., 1)``.
        """
        h = self.trunk(obs)
        flat_disc = self.discrete_head(h)
        cont = self.continuous_head(h)
        value = self.critic_head(h)
        disc_logits = flat_disc.view(
            *flat_disc.shape[:-1], self.max_nodes, self.action_dim
        )
        return disc_logits, cont, value

    # ------------------------------------------------------------------
    # Python-list forward (backward-compat for HybridActionDist + tests)
    # ------------------------------------------------------------------

    def forward(
        self,
        obs: list[float],
    ) -> tuple[list[list[float]], list[float], float]:
        """Compute discrete logits, continuous logits, and state value.

        Returns Python lists so that ``HybridActionDist`` and the Phase 2
        smoke tests work without modification.

        Parameters
        ----------
        obs:
            Flat observation vector (Python list).

        Returns
        -------
        discrete_logits:
            Shape (max_nodes, action_dim).
        continuous_logits:
            Shape (max_nodes,).
        value:
            Scalar critic estimate.
        """
        obs_t = torch.tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            disc_t, cont_t, val_t = self._forward_tensor(obs_t)
        return disc_t.tolist(), cont_t.tolist(), val_t.item()

    def act(
        self,
        obs: list[float],
        mask: list[list[bool]] | None = None,
        vaccine_budget: float = 1.0,
        deterministic: bool = False,
        seed: int | None = None,
    ) -> tuple[dict[str, list], float, float, float]:
        """Sample (or greedily select) a hybrid action.

        Parameters
        ----------
        obs:
            Flat observation vector.
        mask:
            Per-node action validity mask (max_nodes, action_dim).
            If None, all actions are valid.
        vaccine_budget:
            Current vaccine budget for budget projection.
        deterministic:
            If True, use ``mode()``; otherwise ``sample()``.
        seed:
            Optional RNG seed forwarded to ``HybridActionDist``.

        Returns
        -------
        action_dict:
            ``{"discrete": list[int], "continuous": list[float]}``.
        log_prob_discrete:
            Scalar combined discrete log-probability.
        log_prob_continuous:
            Scalar combined continuous log-probability.
        value:
            Critic estimate V(obs).
        """
        from src.models.hybrid_action import HybridActionDist

        discrete_logits, cont_logits, value = self.forward(obs)
        dist = HybridActionDist(
            discrete_logits=discrete_logits,
            continuous_logits=cont_logits,
            mask=mask,
            vaccine_budget=vaccine_budget,
            seed=seed,
        )

        if deterministic:
            sample = dist.mode()
        else:
            sample = dist.sample()

        action_dict = {
            "discrete": sample.discrete,
            "continuous": sample.continuous,
        }
        return action_dict, sample.log_prob_discrete, sample.log_prob_continuous, value
