"""PPO baseline training loop for the epidemic containment environment.

Training loop sketch
--------------------
1. ``_collect_rollout()`` — run the policy for ``n_steps``, storing
   transitions in a ``RolloutBuffer``.
2. ``compute_advantages()`` — compute GAE-λ advantages and discounted returns.
3. ``update_policy()`` — run ``n_epochs`` passes over mini-batches:
   - compute PPO clipped-surrogate + value + entropy losses as PyTorch tensors,
   - call ``loss.backward()`` + ``optimizer.step()`` to update parameters.
4. Repeat until ``total_timesteps`` is reached.

Phase 1 (baseline): per-node Categorical actor, shared MLP trunk.
Phase 2 (hybrid):   discrete PPO objective; continuous head receives indirect
                    gradients through the shared trunk.  Full continuous policy
                    gradient support is a TODO (Phase 3 PyTorch port).

Diagnostic metrics logged every ``log_interval`` updates:
    policy_loss, value_loss, entropy, clip_frac, approx_kl
"""

from __future__ import annotations

import logging
import math
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rollout buffer
# ---------------------------------------------------------------------------

@dataclass
class RolloutBuffer:
    """Stores one rollout of (obs, action, log_prob, reward, done, value).

    Phase 2 additions:
    - ``actions_hybrid``: list of action dicts (discrete + continuous)
    - ``log_probs_discrete``: per-step combined discrete log-prob (scalar)
    - ``log_probs_continuous``: per-step combined continuous log-prob (scalar)

    All fields are plain Python lists to avoid external dependencies.
    """

    obs: list[list[float]] = field(default_factory=list)
    # Phase 1: list[int] per step; Phase 2: list[dict] per step
    actions: list[Any] = field(default_factory=list)
    log_probs: list[Any] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    dones: list[bool] = field(default_factory=list)
    values: list[float] = field(default_factory=list)

    # Phase 2 fields (empty in phase 1, populated in phase 2)
    log_probs_discrete: list[float] = field(default_factory=list)
    log_probs_continuous: list[float] = field(default_factory=list)

    # Filled by compute_advantages()
    returns: list[float] = field(default_factory=list)
    advantages: list[float] = field(default_factory=list)

    def clear(self) -> None:
        """Reset the buffer for the next rollout."""
        for lst in (
            self.obs, self.actions, self.log_probs,
            self.rewards, self.dones, self.values,
            self.log_probs_discrete, self.log_probs_continuous,
            self.returns, self.advantages,
        ):
            lst.clear()

    def __len__(self) -> int:
        return len(self.rewards)


# ---------------------------------------------------------------------------
# PPO baseline
# ---------------------------------------------------------------------------

class PPOBaseline:
    """Minimal PPO training loop.

    Parameters
    ----------
    config:
        Dictionary of hyperparameters (typically loaded from ``baseline.yaml``).
        Required keys: ``env``, ``ppo``, ``model``, ``logging``.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self._setup_logging()

        env_cfg = config["env"]
        ppo_cfg = config["ppo"]
        model_cfg = config.get("model", {})
        hybrid_cfg = config.get("hybrid", {})

        self.task_name: str = env_cfg["task_name"]
        self.seed: int = int(env_cfg.get("seed", 42))
        self.max_nodes: int = int(env_cfg.get("max_nodes", 20))

        self.total_timesteps: int = int(ppo_cfg["total_timesteps"])
        self.n_steps: int = int(ppo_cfg["n_steps"])
        self.n_epochs: int = int(ppo_cfg["n_epochs"])
        self.batch_size: int = int(ppo_cfg["batch_size"])
        self.gamma: float = float(ppo_cfg["gamma"])
        self.gae_lambda: float = float(ppo_cfg["gae_lambda"])
        self.clip_eps: float = float(ppo_cfg["clip_eps"])
        self.entropy_coef: float = float(ppo_cfg["entropy_coef"])
        self.value_loss_coef: float = float(ppo_cfg["value_loss_coef"])
        self.lr: float = float(ppo_cfg["lr"])
        self.max_grad_norm: float = float(ppo_cfg.get("max_grad_norm", 0.5))

        self.log_interval: int = int(config.get("logging", {}).get("log_interval", 10))
        self.checkpoint_dir: str = config.get("logging", {}).get(
            "checkpoint_dir", "checkpoints/baseline"
        )
        self.checkpoint_interval: int = int(
            config.get("logging", {}).get("checkpoint_interval", 50)
        )

        hidden_dims: list[int] = list(model_cfg.get("hidden_dims", [128, 128]))

        # Phase 2: hybrid mode flag
        self.hybrid_mode: bool = bool(hybrid_cfg.get("enabled", False))
        self.entropy_coef_discrete: float = float(
            hybrid_cfg.get("entropy_coef_discrete", self.entropy_coef)
        )
        self.entropy_coef_continuous: float = float(
            hybrid_cfg.get("entropy_coef_continuous", self.entropy_coef * 0.1)
        )

        # Lazy import of adapter — keeps module importable even without env deps
        from src.env.openenv_adapter import (
            NUM_DISCRETE_ACTIONS,
            OpenEnvAdapter,
        )

        self._adapter = OpenEnvAdapter(
            task_name=self.task_name,
            seed=self.seed,
            max_nodes=self.max_nodes,
        )
        obs_tensor, _ = self._adapter.reset()
        obs_dim = len(obs_tensor)

        if self.hybrid_mode:
            from src.models.actor_critic import HybridActorCritic

            self._policy = HybridActorCritic(
                obs_dim=obs_dim,
                action_dim=NUM_DISCRETE_ACTIONS,
                max_nodes=self.max_nodes,
                hidden_dims=hidden_dims,
                seed=self.seed,
            )
        else:
            from src.models.actor_critic import ActorCritic

            self._policy = ActorCritic(
                obs_dim=obs_dim,
                action_dim=NUM_DISCRETE_ACTIONS,
                max_nodes=self.max_nodes,
                hidden_dims=hidden_dims,
                seed=self.seed,
            )

        self._buffer = RolloutBuffer()
        self._global_step: int = 0
        self._n_updates: int = 0

        # Adam optimizer — initialised after the policy is constructed above
        self._optimizer = torch.optim.Adam(self._policy.parameters(), lr=self.lr)

    def train(self) -> None:
        """Run the full PPO training loop until ``total_timesteps``."""
        self._set_seeds(self.seed)
        logger.info(
            "Starting PPO baseline | task=%s seed=%d total_timesteps=%d",
            self.task_name, self.seed, self.total_timesteps,
        )

        obs, _ = self._adapter.reset(seed=self.seed)

        t_start = time.time()

        while self._global_step < self.total_timesteps:
            # ── Collect rollout ──────────────────────────────────────────
            obs = self._collect_rollout(obs)

            # ── Compute advantages ───────────────────────────────────────
            self.compute_advantages()

            # ── Optimise ────────────────────────────────────────────────
            metrics = self.update_policy()
            self._n_updates += 1

            # ── Logging ─────────────────────────────────────────────────
            if self._n_updates % self.log_interval == 0:
                elapsed = time.time() - t_start
                fps = self._global_step / max(elapsed, 1e-8)
                logger.info(
                    "update=%d  step=%d  fps=%.0f  "
                    "policy_loss=%.4f  value_loss=%.4f  entropy=%.4f  "
                    "clip_frac=%.3f  approx_kl=%.4f",
                    self._n_updates,
                    self._global_step,
                    fps,
                    metrics["policy_loss"],
                    metrics["value_loss"],
                    metrics["entropy"],
                    metrics.get("clip_frac", 0.0),
                    metrics.get("approx_kl", 0.0),
                )

            # ── Checkpoint ──────────────────────────────────────────────
            if self._n_updates % self.checkpoint_interval == 0:
                self._save_checkpoint()

        logger.info("Training complete. Total steps: %d", self._global_step)
        self._save_checkpoint(final=True)

    # ------------------------------------------------------------------
    # Rollout collection
    # ------------------------------------------------------------------

    def _collect_rollout(self, obs: list[float]) -> list[float]:
        """Run the current policy for ``n_steps`` and fill the buffer.

        In Phase 2 hybrid mode, uses ``HybridActorCritic.act()`` to produce
        both discrete and continuous actions, builds an action mask from the
        current observation, and stores combined log-probs.

        Parameters
        ----------
        obs:
            Current observation tensor (carries over episode boundaries).

        Returns
        -------
        obs:
            Observation at the end of the rollout (for bootstrapping).
        """
        self._buffer.clear()

        for _ in range(self.n_steps):
            if self.hybrid_mode:
                # Build action mask from current observation
                from src.env.action_masking import build_mask

                mask = build_mask(
                    obs_tensor=obs,
                    max_nodes=self.max_nodes,
                    num_active_nodes=self._adapter.num_nodes,
                )
                vax_budget = self._adapter.vaccine_budget
                action_dict, lp_disc, lp_cont, value = self._policy.act(
                    obs,
                    mask=mask,
                    vaccine_budget=vax_budget,
                    deterministic=False,
                )
                combined_lp = lp_disc + lp_cont

                next_obs, reward, done, _info = self._adapter.step(action_dict)
                self._global_step += 1

                self._buffer.obs.append(list(obs))
                self._buffer.actions.append(action_dict)
                self._buffer.log_probs.append(combined_lp)
                self._buffer.log_probs_discrete.append(lp_disc)
                self._buffer.log_probs_continuous.append(lp_cont)
                self._buffer.rewards.append(reward)
                self._buffer.dones.append(done)
                self._buffer.values.append(value)
            else:
                actions, log_probs, value = self._policy.act(obs)
                next_obs, reward, done, _info = self._adapter.step(actions)
                self._global_step += 1

                self._buffer.obs.append(list(obs))
                self._buffer.actions.append(list(actions))
                self._buffer.log_probs.append(list(log_probs))
                self._buffer.rewards.append(reward)
                self._buffer.dones.append(done)
                self._buffer.values.append(value)

            obs = next_obs
            if done:
                obs, _ = self._adapter.reset()

        return obs

    # ------------------------------------------------------------------
    # Advantage computation (GAE)
    # ------------------------------------------------------------------

    def compute_advantages(self, next_obs: list[float] | None = None) -> None:
        """Compute GAE-lambda advantages and discounted returns.

        Stores results in ``self._buffer.advantages`` and
        ``self._buffer.returns``.

        Parameters
        ----------
        next_obs:
            Observation *after* the last rollout step used for value
            bootstrapping.  If ``None``, bootstraps with 0 (treats last
            step as terminal).

        TODO: pass ``next_obs`` from ``_collect_rollout`` for proper
        bootstrapping when episodes don't end at rollout boundaries.
        """
        T = len(self._buffer)
        advantages = [0.0] * T
        returns = [0.0] * T

        next_value = 0.0  # bootstrap: 0 if terminal or not provided
        next_advantage = 0.0

        for t in reversed(range(T)):
            mask = 0.0 if self._buffer.dones[t] else 1.0
            delta = (
                self._buffer.rewards[t]
                + self.gamma * next_value * mask
                - self._buffer.values[t]
            )
            next_advantage = delta + self.gamma * self.gae_lambda * next_advantage * mask
            advantages[t] = next_advantage
            returns[t] = advantages[t] + self._buffer.values[t]
            next_value = self._buffer.values[t]

        # Normalise advantages
        adv_mean = sum(advantages) / max(len(advantages), 1)
        adv_std = math.sqrt(
            sum((a - adv_mean) ** 2 for a in advantages) / max(len(advantages), 1)
        )
        self._buffer.advantages = [
            (a - adv_mean) / (adv_std + 1e-8) for a in advantages
        ]
        self._buffer.returns = returns

    # ------------------------------------------------------------------
    # Optimisation step (PyTorch)
    # ------------------------------------------------------------------

    def update_policy(self) -> dict[str, float]:
        """Run ``n_epochs`` passes over the rollout buffer with real gradient updates.

        For Phase 1 (discrete-only):
            Per-node Categorical distributions; PPO ratio computed per
            (step, node) pair so the clipping is applied correctly.

        For Phase 2 (hybrid):
            PPO ratio based on the discrete head only (full continuous
            policy-gradient support is a TODO for the Phase 3 PyTorch port).
            The continuous head receives indirect gradients through the shared
            trunk via both the discrete policy loss and the value loss.

        Returns
        -------
        metrics:
            Dict with ``policy_loss``, ``value_loss``, ``entropy``,
            ``clip_frac``, ``approx_kl``.
        """
        T = len(self._buffer)
        indices = list(range(T))

        # ── Convert rollout buffer to tensors ─────────────────────────────
        obs_t = torch.tensor(self._buffer.obs, dtype=torch.float32)          # (T, obs_dim)
        adv_t = torch.tensor(self._buffer.advantages, dtype=torch.float32)   # (T,)
        ret_t = torch.tensor(self._buffer.returns, dtype=torch.float32)      # (T,)

        if self.hybrid_mode:
            # Phase 2: discrete actions extracted from action dicts;
            # old log-probs use discrete-only component for stable ratio.
            actions_t = torch.tensor(
                [a["discrete"] for a in self._buffer.actions], dtype=torch.long
            )  # (T, max_nodes)
            old_lp_t = torch.tensor(
                self._buffer.log_probs_discrete, dtype=torch.float32
            )  # (T,) — combined discrete log-prob per step
        else:
            # Phase 1: list[int] actions, list[float] per-node log-probs
            actions_t = torch.tensor(
                self._buffer.actions, dtype=torch.long
            )  # (T, max_nodes)
            old_lp_t = torch.tensor(
                self._buffer.log_probs, dtype=torch.float32
            )  # (T, max_nodes)

        # ── Diagnostic accumulators ────────────────────────────────────────
        total_policy_loss = 0.0
        total_value_loss  = 0.0
        total_entropy     = 0.0
        total_clip_frac   = 0.0
        total_approx_kl   = 0.0
        n_batches = 0

        clip_eps = self.clip_eps

        for _ in range(self.n_epochs):
            random.shuffle(indices)
            for start in range(0, T, self.batch_size):
                b_idx = indices[start : start + self.batch_size]
                if not b_idx:
                    continue
                b_idx_t = torch.tensor(b_idx, dtype=torch.long)

                b_obs     = obs_t[b_idx_t]      # (B, obs_dim)
                b_adv     = adv_t[b_idx_t]      # (B,)
                b_ret     = ret_t[b_idx_t]      # (B,)
                b_actions = actions_t[b_idx_t]  # (B, max_nodes)

                self._optimizer.zero_grad()

                if self.hybrid_mode:
                    # ── Phase 2: discrete-head PPO ─────────────────────────
                    disc_logits_t, _cont_logits_t, values_t = \
                        self._policy._forward_tensor(b_obs)
                    # (B, max_nodes, action_dim), (B, max_nodes), (B, 1)

                    log_probs_t = torch.log_softmax(disc_logits_t, dim=-1)  # (B, max_nodes, action_dim)
                    new_lp_node = log_probs_t.gather(
                        -1, b_actions.unsqueeze(-1)
                    ).squeeze(-1)               # (B, max_nodes)
                    new_lp = new_lp_node.sum(dim=-1)  # (B,) combined discrete log-prob

                    b_old_lp = old_lp_t[b_idx_t]     # (B,) combined discrete log-prob
                    ratio    = torch.exp(new_lp - b_old_lp)  # (B,)

                    surr1 = ratio * b_adv
                    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * b_adv
                    policy_loss = -torch.min(surr1, surr2).mean()

                    probs_t = torch.softmax(disc_logits_t, dim=-1)
                    entropy = -(probs_t * log_probs_t).sum(dim=-1).mean()

                    with torch.no_grad():
                        clip_frac  = ((ratio - 1.0).abs() > clip_eps).float().mean().item()
                        approx_kl  = (b_old_lp - new_lp).mean().item()

                else:
                    # ── Phase 1: per-node Categorical PPO ─────────────────
                    logits_t, values_t = self._policy.forward(b_obs)
                    # (B, max_nodes, action_dim), (B, 1)

                    log_probs_t = torch.log_softmax(logits_t, dim=-1)    # (B, max_nodes, action_dim)
                    new_lp_node = log_probs_t.gather(
                        -1, b_actions.unsqueeze(-1)
                    ).squeeze(-1)                                          # (B, max_nodes)

                    b_old_lp   = old_lp_t[b_idx_t]                        # (B, max_nodes)
                    ratio_node = torch.exp(new_lp_node - b_old_lp)        # (B, max_nodes)

                    # Expand advantage for per-node comparison
                    adv_exp  = b_adv.unsqueeze(-1)                         # (B, 1)
                    surr1    = ratio_node * adv_exp
                    surr2    = torch.clamp(ratio_node, 1.0 - clip_eps, 1.0 + clip_eps) * adv_exp
                    policy_loss = -torch.min(surr1, surr2).mean()

                    probs_t = torch.softmax(logits_t, dim=-1)
                    entropy = -(probs_t * log_probs_t).sum(dim=-1).mean()

                    with torch.no_grad():
                        clip_frac = ((ratio_node - 1.0).abs() > clip_eps).float().mean().item()
                        approx_kl = (b_old_lp - new_lp_node).mean().item()

                # ── Value loss (shared between both modes) ─────────────────
                value_loss = 0.5 * (values_t.squeeze(-1) - b_ret).pow(2).mean()

                # ── Combined loss and parameter update ─────────────────────
                loss = (
                    policy_loss
                    + self.value_loss_coef * value_loss
                    - self.entropy_coef * entropy
                )
                loss.backward()
                nn.utils.clip_grad_norm_(self._policy.parameters(), self.max_grad_norm)
                self._optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss  += value_loss.item()
                total_entropy     += entropy.item()
                total_clip_frac   += clip_frac
                total_approx_kl   += approx_kl
                n_batches         += 1

        n_batches = max(n_batches, 1)
        return {
            "policy_loss": total_policy_loss / n_batches,
            "value_loss":  total_value_loss  / n_batches,
            "entropy":     total_entropy     / n_batches,
            "clip_frac":   total_clip_frac   / n_batches,
            "approx_kl":   total_approx_kl   / n_batches,
        }

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _save_checkpoint(self, final: bool = False) -> None:
        """Persist a checkpoint: metadata text file + PyTorch weights file."""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        tag = "final" if final else f"step_{self._global_step}"

        # Human-readable metadata
        meta_path = os.path.join(self.checkpoint_dir, f"checkpoint_{tag}.txt")
        with open(meta_path, "w") as f:
            f.write(f"global_step={self._global_step}\n")
            f.write(f"n_updates={self._n_updates}\n")
            f.write(f"task={self.task_name}\n")
            f.write(f"seed={self.seed}\n")

        # PyTorch weights (policy + optimizer state)
        weights_path = os.path.join(self.checkpoint_dir, f"checkpoint_{tag}.pt")
        torch.save(
            {
                "global_step": self._global_step,
                "n_updates": self._n_updates,
                "policy_state_dict": self._policy.state_dict(),
                "optimizer_state_dict": self._optimizer.state_dict(),
            },
            weights_path,
        )
        logger.info("Checkpoint saved: %s  (weights: %s)", meta_path, weights_path)

    @staticmethod
    def _setup_logging() -> None:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )

    @staticmethod
    def _set_seeds(seed: int) -> None:
        """Seed Python's random module for reproducibility."""
        random.seed(seed)
