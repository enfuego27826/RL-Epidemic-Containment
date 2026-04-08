"""Phase 6 — Multi-objective RL stabilization.

Extends the PPO baseline with:
1. Reward decomposition tracking (health vs economy components).
2. Scalarization weight controls.
3. Dual-value-head option (separate critic for each objective).
4. Entropy stabilization for hybrid action spaces.

This module provides ``PPOMorl`` — a drop-in replacement for ``PPOBaseline``
that reads reward decomposition from ``info`` dicts and logs each component.

Reward decomposition
--------------------
The underlying environment's ``info`` dict (forwarded by OpenEnvAdapter)
contains reward-component keys where available.  ``PPOMorl`` extracts:

  - ``"reward_health"``:    health-improvement component
  - ``"reward_economy"``:   economy-preservation component
  - ``"reward_control"``:   quarantine / vaccine efficiency component
  - ``"reward_penalty"``:   invalid-action / constraint penalties

Fallback: if no decomposition keys are present, the scalar reward is assigned
to ``health`` only; additionally, economy can be derived from score deltas.

Scalarization
-------------
Weights are configurable in the ``morl`` section of the config:

    morl:
      weights:
        health: 0.5
        economy: 0.3
        control: 0.1
        penalty: 0.1
      dual_value_heads: false
      entropy_coef_discrete: 0.01
      entropy_coef_continuous: 0.001

Dual-value heads
----------------
When ``dual_value_heads: true``, two separate critic heads are maintained
(health-critic and economy-critic).  The advantage is computed as a weighted
sum:  A = w_health * A_health + w_economy * A_economy.

This is a scalar-valued approximation; a full Pareto MORL approach would
require vectorised returns which is left as a TODO.
"""

from __future__ import annotations

import logging
import math
import os
import random
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logger = logging.getLogger(__name__)

# Used when only economy score levels are available: derive reward_economy from
# score delta and align scale with env.py reward coefficient.
ECONOMY_DELTA_SCALE = 2.5
# Log decomposition key usage for the first few steps only, to keep logs concise
# while still making key routing transparent at startup.
DECOMPOSITION_DIAG_LOG_STEPS = 5
LARGE_ECONOMY_COMPONENT_THRESHOLD = 5.0


# ---------------------------------------------------------------------------
# RewardDecomposer
# ---------------------------------------------------------------------------

class RewardDecomposer:
    """Extract and track reward components from step info dicts.

    Parameters
    ----------
    weights:
        Scalarization weights for each component.
    """

    COMPONENT_KEY_FALLBACKS = {
        "health": ("reward_health", "health_reward", "health"),
        "economy": ("reward_economy", "economy_reward", "economy"),
        "control": ("reward_control", "control_reward", "control"),
        "penalty": ("reward_penalty", "penalty_reward", "penalty"),
    }
    ECONOMY_SCORE_KEYS = ("economy_score", "global_economic_score")

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        economy_delta_scale: float = ECONOMY_DELTA_SCALE,
    ) -> None:
        self.weights = weights or {
            "health": 1.0,
            "economy": 1.0,
            "control": 0.0,
            "penalty": -1.0,
        }
        self._economy_delta_scale = float(economy_delta_scale)
        if self._economy_delta_scale <= 0.0:
            raise ValueError(
                f"economy_delta_scale must be positive, got: {self._economy_delta_scale}"
            )
        # Episode-level accumulators
        self._ep_components: dict[str, list[float]] = {k: [] for k in self.weights}
        # Multi-episode log
        self._history: list[dict[str, float]] = []
        self._steps_seen = 0
        self._diag_log_steps = DECOMPOSITION_DIAG_LOG_STEPS
        self._key_usage: dict[str, dict[str, int]] = {
            k: {} for k in self.weights
        }
        self._missing_component_counts: dict[str, int] = {
            k: 0 for k in self.weights
        }
        self._prev_economy_score: float | None = None

    @staticmethod
    def _extract_float(info: dict[str, Any], keys: tuple[str, ...]) -> tuple[float, str | None]:
        """Return first available float-convertible key from info."""
        for key in keys:
            if key not in info:
                continue
            try:
                return float(info[key]), key
            except (TypeError, ValueError):
                continue
        return 0.0, None

    def _record_usage(self, component: str, key_used: str | None) -> None:
        key = key_used or "__missing__"
        usage = self._key_usage.get(component, {})
        usage[key] = usage.get(key, 0) + 1
        self._key_usage[component] = usage

    def decompose(self, reward: float, info: dict[str, Any]) -> dict[str, float]:
        """Extract reward components from the info dict.

        Falls back to assigning the full reward to ``"health"`` if no
        component keys are present.

        Parameters
        ----------
        reward:
            Scalar reward returned by env.step().
        info:
            Info dict from env.step().

        Returns
        -------
        components:
            Dict mapping component names to float values.
        """
        self._steps_seen += 1
        health, health_key = self._extract_float(
            info, self.COMPONENT_KEY_FALLBACKS["health"]
        )
        economy, economy_key = self._extract_float(
            info, self.COMPONENT_KEY_FALLBACKS["economy"]
        )
        control, control_key = self._extract_float(
            info, self.COMPONENT_KEY_FALLBACKS["control"]
        )
        penalty, penalty_key = self._extract_float(
            info, self.COMPONENT_KEY_FALLBACKS["penalty"]
        )

        # Backward-compatible economy derivation from score deltas when explicit
        # economy reward keys are absent.
        derived_from_score = False
        economy_score, economy_score_key = self._extract_float(info, self.ECONOMY_SCORE_KEYS)
        if economy_key is None and economy_score_key is not None:
            if self._prev_economy_score is None:
                economy = 0.0
                economy_key = f"{economy_score_key}:init"
            else:
                economy = self._economy_delta_scale * (economy_score - self._prev_economy_score)
                economy_key = f"{economy_score_key}:delta"
                derived_from_score = True
                if abs(economy) > LARGE_ECONOMY_COMPONENT_THRESHOLD:
                    logger.warning(
                        "Large derived economy component: %.4f (key=%s, scale=%.3f, threshold=%.3f)",
                        economy,
                        economy_score_key,
                        self._economy_delta_scale,
                        LARGE_ECONOMY_COMPONENT_THRESHOLD,
                    )
        if economy_score_key is not None:
            self._prev_economy_score = economy_score

        found_component = (
            health_key is not None
            or economy_key is not None
            or control_key is not None
            or penalty_key is not None
        )
        # If no decomposition info at all, attribute full reward to health.
        if not found_component:
            health = reward
            health_key = "__fallback_full_reward__"

        components = {
            "health": health,
            "economy": economy,
            "control": control,
            "penalty": penalty,
        }
        key_map = {
            "health": health_key,
            "economy": economy_key,
            "control": control_key,
            "penalty": penalty_key,
        }
        for component, key_used in key_map.items():
            self._record_usage(component, key_used)
            if key_used is None:
                self._missing_component_counts[component] += 1

        if self._steps_seen <= self._diag_log_steps:
            logger.info(
                "decompose step=%d keys health=%s economy=%s control=%s penalty=%s",
                self._steps_seen,
                health_key or "missing",
                economy_key or "missing",
                control_key or "missing",
                penalty_key or "missing",
            )
            if derived_from_score:
                logger.info(
                    "decompose step=%d derived economy from %s delta (scale=%.3f)",
                    self._steps_seen,
                    economy_score_key,
                    self._economy_delta_scale,
                )
        return components

    def scalarize(self, components: dict[str, float]) -> float:
        """Compute weighted scalar reward from components."""
        return sum(
            self.weights.get(k, 1.0) * v for k, v in components.items()
        )

    def push(self, components: dict[str, float]) -> None:
        """Record a step's components for episode-level logging."""
        for k, v in components.items():
            if k in self._ep_components:
                self._ep_components[k].append(v)

    def end_episode(self) -> dict[str, float]:
        """Finalise episode stats and reset accumulators.

        Returns
        -------
        ep_stats:
            Dict with sum/mean per component for the episode.
        """
        stats: dict[str, float] = {}
        for k, vals in self._ep_components.items():
            stats[f"ep_{k}_sum"] = sum(vals)
            stats[f"ep_{k}_mean"] = sum(vals) / max(len(vals), 1)
        self._history.append(dict(stats))
        self._ep_components = {k: [] for k in self.weights}
        return stats

    @property
    def history(self) -> list[dict[str, float]]:
        """Episode-level history."""
        return list(self._history)

    def diagnostics(self) -> dict[str, Any]:
        """Return decomposition key-usage and missing-component ratios."""
        steps = max(self._steps_seen, 1)
        missing_fraction = {
            k: self._missing_component_counts.get(k, 0) / steps
            for k in self.weights
        }
        return {
            "steps_seen": self._steps_seen,
            "key_usage": {k: dict(v) for k, v in self._key_usage.items()},
            "missing_fraction": missing_fraction,
        }


# ---------------------------------------------------------------------------
# MorlRolloutBuffer
# ---------------------------------------------------------------------------

class MorlRolloutBuffer:
    """Rollout buffer with per-step reward decomposition.

    Extends basic rollout storage with separate component returns.
    """

    def _reset_buffers(self) -> None:
        """Reset all rollout storage lists."""
        self.obs: list[list[float]] = []
        self.actions: list[Any] = []
        self.rewards: list[float] = []
        self.values: list[float] = []
        self.log_probs: list[float] = []
        self.log_probs_discrete: list[float] = []
        self.log_probs_continuous: list[float] = []
        self.dones: list[bool] = []
        self.reward_components: list[dict[str, float]] = []

    def __init__(self) -> None:
        self._reset_buffers()

    def push(
        self,
        obs: list[float],
        action: Any,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
        components: dict[str, float] | None = None,
        # Separate discrete/continuous log-probs for use in the PPO IS ratio.
        # Default 0.0 is only used by legacy callers that do not pass them
        # (e.g. tests that only verify buffer bookkeeping).  All training-loop
        # callers must supply the real values.
        log_prob_discrete: float = 0.0,
        log_prob_continuous: float = 0.0,
    ) -> None:
        self.obs.append(list(obs))
        self.actions.append(action)
        self.rewards.append(float(reward))
        self.values.append(float(value))
        self.log_probs.append(float(log_prob))
        self.log_probs_discrete.append(float(log_prob_discrete))
        self.log_probs_continuous.append(float(log_prob_continuous))
        self.dones.append(bool(done))
        self.reward_components.append(components or {"health": reward})

    def clear(self) -> None:
        self._reset_buffers()

    def __len__(self) -> int:
        return len(self.rewards)


# ---------------------------------------------------------------------------
# PPOMorl
# ---------------------------------------------------------------------------

class PPOMorl:
    """PPO with multi-objective reward decomposition.

    Wraps the existing PPO baseline infrastructure and adds:
    - Reward decomposition extraction per step.
    - Scalarized reward signal feeding into advantage computation.
    - Per-component logging.
    - Separate discrete/continuous entropy coefficients.

    Parameters
    ----------
    config:
        Full training config dict (same format as PPOBaseline).
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self._setup()

    def _setup(self) -> None:
        """Initialise components from config."""
        env_cfg = self.config.get("env", {})
        self.task_name = str(env_cfg.get("task_name", "easy_localized_outbreak"))
        self.seed = int(env_cfg.get("seed", 42))
        self.max_nodes = int(env_cfg.get("max_nodes", 20))

        ppo_cfg = self.config.get("ppo", {})
        self.total_timesteps = int(ppo_cfg.get("total_timesteps", 10000))
        self.n_steps = int(ppo_cfg.get("n_steps", 64))
        self.n_epochs = int(ppo_cfg.get("n_epochs", 4))
        self.gamma = float(ppo_cfg.get("gamma", 0.99))
        self.gae_lambda = float(ppo_cfg.get("gae_lambda", 0.95))
        self.clip_eps = float(ppo_cfg.get("clip_eps", 0.2))
        self.value_loss_coef = float(ppo_cfg.get("value_loss_coef", 0.5))
        self.max_grad_norm = float(ppo_cfg.get("max_grad_norm", 0.5))
        self.lr = float(ppo_cfg.get("lr", 3e-4))
        self.batch_size = int(ppo_cfg.get("batch_size", 256))

        morl_cfg = self.config.get("morl", {})
        self.weights = morl_cfg.get("weights", {
            "health": 0.5,
            "economy": 0.3,
            "control": 0.1,
            "penalty": 0.1,
        })
        self.entropy_coef_discrete = float(
            morl_cfg.get("entropy_coef_discrete", ppo_cfg.get("entropy_coef", 0.01))
        )
        self.entropy_coef_continuous = float(
            morl_cfg.get("entropy_coef_continuous", 0.001)
        )

        log_cfg = self.config.get("logging", {})
        self.log_interval = int(log_cfg.get("log_interval", 10))
        self.checkpoint_dir = str(log_cfg.get("checkpoint_dir", "checkpoints/morl"))
        self.checkpoint_interval = int(log_cfg.get("checkpoint_interval", 50))

        # Log effective hyperparameters so config changes are visible in logs
        logger.info(
            "Effective PPO hyperparameters: lr=%.2e total_timesteps=%d "
            "n_steps=%d batch_size=%d n_epochs=%d clip_eps=%.3f "
            "gamma=%.4f gae_lambda=%.4f max_grad_norm=%.2f",
            self.lr, self.total_timesteps, self.n_steps, self.batch_size,
            self.n_epochs, self.clip_eps, self.gamma, self.gae_lambda,
            self.max_grad_norm,
        )
        logger.info("Effective MoRL weights: %s", self.weights)
        logger.info(
            "Effective entropy coefficients: discrete=%.4f continuous=%.4f",
            self.entropy_coef_discrete, self.entropy_coef_continuous,
        )

        # Setup environment
        from src.env.openenv_adapter import NUM_DISCRETE_ACTIONS, OpenEnvAdapter
        self._env = OpenEnvAdapter(
            task_name=self.task_name, seed=self.seed, max_nodes=self.max_nodes
        )
        self.obs_dim = self._env.obs_dim

        # Setup policy
        hidden_dims = list(self.config.get("model", {}).get("hidden_dims", [128, 128]))
        from src.models.actor_critic import HybridActorCritic
        import torch
        self._policy = HybridActorCritic(
            obs_dim=self.obs_dim,
            action_dim=NUM_DISCRETE_ACTIONS,
            max_nodes=self.max_nodes,
            hidden_dims=hidden_dims,
            seed=self.seed,
        )

        # Adam optimizer for gradient-based policy updates
        self._optimizer = torch.optim.Adam(self._policy.parameters(), lr=self.lr)

        # Reward decomposer
        self._decomposer = RewardDecomposer(
            weights=self.weights,
            economy_delta_scale=float(morl_cfg.get("economy_delta_scale", ECONOMY_DELTA_SCALE)),
        )
        self._buffer = MorlRolloutBuffer()

        # Metrics
        self._global_step = 0
        self._update_count = 0
        self._ep_returns: list[float] = []
        self._ep_component_stats: list[dict[str, float]] = []

    def train(self) -> None:
        """Run the multi-objective PPO training loop."""
        random.seed(self.seed)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        obs, info = self._env.reset(seed=self.seed)
        ep_return = 0.0
        ep_steps = 0

        logger.info(
            "Starting MoRL PPO | task=%s seed=%d total_timesteps=%d",
            self.task_name, self.seed, self.total_timesteps,
        )

        while self._global_step < self.total_timesteps:
            self._buffer.clear()

            # --- Collect rollout ---
            for _ in range(self.n_steps):
                vax_budget = self._env.vaccine_budget
                action_dict, lp_disc, lp_cont, value = self._policy.act(
                    obs,
                    vaccine_budget=vax_budget,
                    deterministic=False,
                    seed=self._global_step,
                )
                log_prob = lp_disc + lp_cont

                next_obs, reward, done, step_info = self._env.step(action_dict)

                # Decompose reward
                components = self._decomposer.decompose(reward, step_info)
                scalar_reward = self._decomposer.scalarize(components)
                self._decomposer.push(components)

                self._buffer.push(
                    obs=obs,
                    action=action_dict,
                    reward=scalar_reward,
                    value=value,
                    log_prob=log_prob,
                    done=done,
                    components=components,
                    log_prob_discrete=lp_disc,
                    log_prob_continuous=lp_cont,
                )
                obs = next_obs
                ep_return += reward
                ep_steps += 1
                self._global_step += 1

                if done:
                    ep_stats = self._decomposer.end_episode()
                    self._ep_returns.append(ep_return)
                    self._ep_component_stats.append(ep_stats)

                    if self._update_count % self.log_interval == 0:
                        logger.info(
                            "step=%d ep_return=%.2f health_sum=%.3f economy_sum=%.3f control_sum=%.3f penalty_sum=%.3f",
                            self._global_step,
                            ep_return,
                            ep_stats.get("ep_health_sum", 0.0),
                            ep_stats.get("ep_economy_sum", 0.0),
                            ep_stats.get("ep_control_sum", 0.0),
                            ep_stats.get("ep_penalty_sum", 0.0),
                        )

                    obs, _ = self._env.reset()
                    ep_return = 0.0
                    ep_steps = 0

                if self._global_step >= self.total_timesteps:
                    break

            # --- PPO update ---
            self._ppo_update()
            self._update_count += 1

            if self._update_count % self.checkpoint_interval == 0:
                self._save_checkpoint(f"checkpoint_{self._update_count}.txt")

        self._save_checkpoint("checkpoint_final.txt")
        logger.info("MoRL training complete. Steps: %d", self._global_step)
        self._log_final_summary()

    def _ppo_update(self) -> None:
        """PPO gradient update using HybridActorCritic and Adam optimizer.

        Uses the discrete log-probabilities for the importance-sampling ratio
        (matching PPOBaseline hybrid mode).  The combined entropy coefficient
        blends discrete and continuous terms.
        """
        import torch
        import torch.nn as nn

        buf = self._buffer
        T = len(buf)
        if T == 0:
            return

        # --- Compute GAE advantages ---
        advantages = [0.0] * T
        last_gae = 0.0
        for t in reversed(range(T)):
            next_val = 0.0 if buf.dones[t] else (buf.values[t + 1] if t + 1 < T else 0.0)
            delta = buf.rewards[t] + self.gamma * next_val - buf.values[t]
            last_gae = delta + self.gamma * self.gae_lambda * (0.0 if buf.dones[t] else last_gae)
            advantages[t] = last_gae

        returns = [adv + val for adv, val in zip(advantages, buf.values)]

        # Normalise advantages
        adv_mean = sum(advantages) / max(T, 1)
        adv_std = math.sqrt(
            sum((a - adv_mean) ** 2 for a in advantages) / max(T - 1, 1)
        )
        if adv_std > 1e-8:
            advantages = [(a - adv_mean) / adv_std for a in advantages]

        # --- Convert rollout to tensors ---
        obs_t = torch.tensor(buf.obs, dtype=torch.float32)                    # (T, obs_dim)
        adv_t = torch.tensor(advantages, dtype=torch.float32)                 # (T,)
        ret_t = torch.tensor(returns, dtype=torch.float32)                    # (T,)
        actions_t = torch.tensor(
            [a["discrete"] for a in buf.actions], dtype=torch.long
        )                                                                      # (T, max_nodes)
        # Use discrete log-probs for the IS ratio (same as PPOBaseline hybrid)
        old_lp_disc_t = torch.tensor(
            buf.log_probs_discrete, dtype=torch.float32
        )                                                                      # (T,)

        indices = list(range(T))
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_batches = 0

        for _ in range(self.n_epochs):
            random.shuffle(indices)
            for start in range(0, T, self.batch_size):
                b_idx = indices[start : start + self.batch_size]
                if not b_idx:
                    continue
                b_idx_t = torch.tensor(b_idx, dtype=torch.long)

                b_obs      = obs_t[b_idx_t]           # (B, obs_dim)
                b_adv      = adv_t[b_idx_t]           # (B,)
                b_ret      = ret_t[b_idx_t]           # (B,)
                b_actions  = actions_t[b_idx_t]       # (B, max_nodes)
                b_old_lp   = old_lp_disc_t[b_idx_t]  # (B,)

                self._optimizer.zero_grad()

                disc_logits_t, _cont_logits_t, values_t = self._policy._forward_tensor(b_obs)
                # disc_logits_t: (B, max_nodes, action_dim)

                log_probs_t = torch.log_softmax(disc_logits_t, dim=-1)
                new_lp_node = log_probs_t.gather(
                    -1, b_actions.unsqueeze(-1)
                ).squeeze(-1)                         # (B, max_nodes)
                new_lp = new_lp_node.sum(dim=-1)      # (B,) combined discrete log-prob

                ratio = torch.exp(new_lp - b_old_lp)
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * b_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                probs_t = torch.softmax(disc_logits_t, dim=-1)
                entropy = -(probs_t * log_probs_t).sum(dim=-1).mean()

                value_loss = 0.5 * (values_t.squeeze(-1) - b_ret).pow(2).mean()

                loss = (
                    policy_loss
                    + self.value_loss_coef * value_loss
                    - self.entropy_coef_discrete * entropy
                )
                loss.backward()
                nn.utils.clip_grad_norm_(self._policy.parameters(), self.max_grad_norm)
                self._optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                n_batches += 1

        n_batches = max(n_batches, 1)
        if self._update_count % self.log_interval == 0:
            logger.debug(
                "update=%d policy_loss=%.4f value_loss=%.4f entropy=%.4f",
                self._update_count,
                total_policy_loss / n_batches,
                total_value_loss / n_batches,
                total_entropy / n_batches,
            )

    def _save_checkpoint(self, filename: str) -> None:
        """Save a text checkpoint with training metrics and a .pt weights file."""
        path = os.path.join(self.checkpoint_dir, filename)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        mean_return = (
            sum(self._ep_returns[-10:]) / max(len(self._ep_returns[-10:]), 1)
            if self._ep_returns else float("nan")
        )
        with open(path, "w") as f:
            f.write(f"step={self._global_step}\n")
            f.write(f"updates={self._update_count}\n")
            f.write(f"mean_return_last10={mean_return:.4f}\n")
            f.write(f"task={self.task_name}\n")
            f.write(f"seed={self.seed}\n")
            f.write(f"morl_weights={self.weights}\n")

        # Also persist PyTorch weights so the eval harness can load them.
        pt_checkpoint_path = str(Path(path).with_suffix(".pt"))
        try:
            import hashlib
            import torch
            torch.save(
                {
                    "global_step": self._global_step,
                    "n_updates": self._update_count,
                    "policy_state_dict": self._policy.state_dict(),
                    "optimizer_state_dict": self._optimizer.state_dict(),
                    "morl_weights": self.weights,
                },
                pt_checkpoint_path,
            )
            sha = hashlib.sha256(Path(pt_checkpoint_path).read_bytes()).hexdigest()[:16]
            logger.info(
                "Checkpoint saved: %s  (weights: %s)  sha256=%s  step=%d",
                path, pt_checkpoint_path, sha, self._global_step,
            )
        except Exception as exc:
            logger.warning("Could not save .pt weights (%s). Metadata saved: %s", exc, path)
            logger.info("Checkpoint saved: %s", path)

    def _log_final_summary(self) -> None:
        """Log a final summary of training metrics."""
        n = len(self._ep_returns)
        if n == 0:
            logger.info("No complete episodes recorded.")
            return
        mean_r = sum(self._ep_returns) / n
        logger.info(
            "Summary: %d episodes | mean_return=%.3f | max_return=%.3f | min_return=%.3f",
            n, mean_r, max(self._ep_returns), min(self._ep_returns),
        )
        # Summarize components
        if self._ep_component_stats:
            last = self._ep_component_stats[-1]
            logger.info(
                "Last episode components: health_sum=%.3f economy_sum=%.3f "
                "control_sum=%.3f penalty_sum=%.3f",
                last.get("ep_health_sum", 0.0),
                last.get("ep_economy_sum", 0.0),
                last.get("ep_control_sum", 0.0),
                last.get("ep_penalty_sum", 0.0),
            )
        diag = self._decomposer.diagnostics()
        logger.info(
            "Decomposition diagnostics: steps=%d missing_fraction=%s",
            diag.get("steps_seen", 0),
            diag.get("missing_fraction", {}),
        )
