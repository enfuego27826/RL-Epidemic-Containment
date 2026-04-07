"""PPO baseline training loop for the epidemic containment environment.

This module implements a minimal, stdlib-only PPO scaffold.  All math is
performed with plain Python (no NumPy/PyTorch required) so that the smoke
test runs anywhere.  A real training run will swap the ``ActorCritic``
implementation for a PyTorch version in Phase 3.

Training loop sketch
--------------------
1. ``collect_rollout()`` — run the policy in the env for N steps, storing
   transitions in a ``RolloutBuffer``.
2. ``compute_advantages()`` — compute GAE-lambda returns (placeholder: uses
   simple discounted returns until the value head is properly wired).
3. ``optimize()`` — run ``n_epochs`` passes over mini-batches, computing PPO
   policy-gradient + value + entropy losses and updating parameters.
4. Repeat until ``total_timesteps`` is reached.

TODO (Phase 2): replace per-node discrete logits with hybrid action logits.
TODO (Phase 3): wire ST-GNN encoder into ActorCritic.forward().
TODO: replace pure-Python parameter updates with PyTorch autograd.
"""

from __future__ import annotations

import logging
import math
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rollout buffer
# ---------------------------------------------------------------------------

@dataclass
class RolloutBuffer:
    """Stores one rollout of (obs, action, log_prob, reward, done, value).

    All fields are plain Python lists to avoid external dependencies in this
    baseline scaffold.
    """

    obs: list[list[float]] = field(default_factory=list)
    actions: list[list[int]] = field(default_factory=list)
    log_probs: list[list[float]] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    dones: list[bool] = field(default_factory=list)
    values: list[float] = field(default_factory=list)

    # Filled by compute_advantages()
    returns: list[float] = field(default_factory=list)
    advantages: list[float] = field(default_factory=list)

    def clear(self) -> None:
        """Reset the buffer for the next rollout."""
        for lst in (
            self.obs, self.actions, self.log_probs,
            self.rewards, self.dones, self.values,
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

        self.log_interval: int = int(config.get("logging", {}).get("log_interval", 10))
        self.checkpoint_dir: str = config.get("logging", {}).get(
            "checkpoint_dir", "checkpoints/baseline"
        )
        self.checkpoint_interval: int = int(
            config.get("logging", {}).get("checkpoint_interval", 50)
        )

        hidden_dims: list[int] = list(model_cfg.get("hidden_dims", [128, 128]))

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
                    "policy_loss=%.4f  value_loss=%.4f  entropy=%.4f",
                    self._n_updates,
                    self._global_step,
                    fps,
                    metrics["policy_loss"],
                    metrics["value_loss"],
                    metrics["entropy"],
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
    # Optimisation step (pure-Python placeholder)
    # ------------------------------------------------------------------

    def update_policy(self) -> dict[str, float]:
        """Run ``n_epochs`` passes over the rollout buffer.

        This is a *placeholder* that computes PPO losses conceptually but
        does not perform real gradient updates (no PyTorch).  Replace with
        a proper autograd backward pass in the PyTorch refactor.

        Returns
        -------
        metrics:
            Dict with ``policy_loss``, ``value_loss``, ``entropy``.
        """
        T = len(self._buffer)
        indices = list(range(T))

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_batches = 0

        for _ in range(self.n_epochs):
            random.shuffle(indices)
            for start in range(0, T, self.batch_size):
                batch_idx = indices[start : start + self.batch_size]

                pl, vl, ent = self._compute_losses(batch_idx)
                total_policy_loss += pl
                total_value_loss += vl
                total_entropy += ent
                n_batches += 1

                # TODO: call optimizer.step() here after PyTorch port

        n_batches = max(n_batches, 1)
        return {
            "policy_loss": total_policy_loss / n_batches,
            "value_loss": total_value_loss / n_batches,
            "entropy": total_entropy / n_batches,
        }

    def _compute_losses(
        self, batch_idx: list[int]
    ) -> tuple[float, float, float]:
        """Compute PPO losses for a mini-batch.

        This pure-Python version computes the *values* of the losses as
        scalars for logging/smoke-test purposes.  A real implementation
        would keep these as torch.Tensor objects for backprop.

        Parameters
        ----------
        batch_idx:
            Indices into the rollout buffer for this mini-batch.

        Returns
        -------
        policy_loss, value_loss, entropy:
            Scalar loss values.
        """
        policy_loss = 0.0
        value_loss = 0.0
        entropy = 0.0

        for idx in batch_idx:
            obs = self._buffer.obs[idx]
            old_log_probs = self._buffer.log_probs[idx]  # list[float] per node
            adv = self._buffer.advantages[idx]
            ret = self._buffer.returns[idx]

            # Forward pass
            action_logits, value = self._policy.forward(obs)

            # Per-node contributions
            for node_i, (logits, old_lp, act) in enumerate(
                zip(action_logits, old_log_probs, self._buffer.actions[idx])
            ):
                from src.models.actor_critic import _softmax

                probs = _softmax(logits)
                new_lp = math.log(max(probs[act], 1e-8))

                # Importance ratio
                ratio = math.exp(new_lp - old_lp)
                clipped_ratio = max(
                    min(ratio, 1.0 + self.clip_eps), 1.0 - self.clip_eps
                )
                policy_loss -= min(ratio * adv, clipped_ratio * adv)

                # Entropy for this node
                ent_node = -sum(p * math.log(max(p, 1e-8)) for p in probs)
                entropy += ent_node

            # Value loss (clipped)
            value_loss += (value - ret) ** 2

        n = max(len(batch_idx), 1)
        policy_loss /= n
        value_loss = 0.5 * value_loss / n
        entropy /= n

        total_loss = (
            policy_loss
            + self.value_loss_coef * value_loss
            - self.entropy_coef * entropy
        )
        return policy_loss, value_loss, entropy

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _save_checkpoint(self, final: bool = False) -> None:
        """Persist a lightweight checkpoint (config + metadata).

        A full checkpoint would also serialise model weights.  For this
        scaffold we just write metadata to avoid external deps.
        """
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        tag = "final" if final else f"step_{self._global_step}"
        path = os.path.join(self.checkpoint_dir, f"checkpoint_{tag}.txt")
        with open(path, "w") as f:
            f.write(f"global_step={self._global_step}\n")
            f.write(f"n_updates={self._n_updates}\n")
            f.write(f"task={self.task_name}\n")
            f.write(f"seed={self.seed}\n")
        logger.info("Checkpoint saved: %s", path)

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
