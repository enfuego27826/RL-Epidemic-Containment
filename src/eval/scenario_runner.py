"""Phase 7 — Robustness / Generalization Evaluation.

Provides a comprehensive evaluation harness that:
1. Runs a policy across multiple tasks, seeds, and optional lag settings.
2. Collects per-episode metrics.
3. Reports results as a formatted table (stdout) and optionally saves to
   JSON/CSV.

Usage
-----
Via ``scripts/run_eval_harness.py``:

    python scripts/run_eval_harness.py --config configs/full_pipeline.yaml

Or programmatically:

    from src.eval.scenario_runner import EvalHarness
    harness = EvalHarness(config)
    results = harness.run()
    harness.print_summary(results)
    harness.save_results(results, "results/eval_results.json")

Scenario randomization
----------------------
The harness supports basic randomization by varying:
- Task name (all three difficulty levels).
- Random seed.
- Optional lag (if config enables it).

This gives a simple but effective coverage of generalization.
"""

from __future__ import annotations

import csv
import json
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


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALL_TASKS = [
    "easy_localized_outbreak",
    "medium_multi_center_spread",
    "hard_asymptomatic_high_density",
]

DEFAULT_N_SEEDS = 3
DEFAULT_N_EPISODES = 3


# ---------------------------------------------------------------------------
# EpisodeResult
# ---------------------------------------------------------------------------

class EpisodeResult:
    """Metrics collected from a single evaluation episode."""

    def __init__(
        self,
        task_name: str,
        seed: int,
        episode_idx: int,
        ep_return: float,
        peak_infection: float,
        mean_economy: float,
        total_steps: int,
        invalid_action_rate: float,
        lag_steps: int = 0,
        extra: dict[str, float] | None = None,
    ) -> None:
        self.task_name = task_name
        self.seed = seed
        self.episode_idx = episode_idx
        self.ep_return = ep_return
        self.peak_infection = peak_infection
        self.mean_economy = mean_economy
        self.total_steps = total_steps
        self.invalid_action_rate = invalid_action_rate
        self.lag_steps = lag_steps
        self.extra = extra or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_name": self.task_name,
            "seed": self.seed,
            "episode_idx": self.episode_idx,
            "ep_return": self.ep_return,
            "peak_infection": self.peak_infection,
            "mean_economy": self.mean_economy,
            "total_steps": self.total_steps,
            "invalid_action_rate": self.invalid_action_rate,
            "lag_steps": self.lag_steps,
            **self.extra,
        }


# ---------------------------------------------------------------------------
# Policy wrapper (factory)
# ---------------------------------------------------------------------------

def _build_policy(config: dict[str, Any], obs_dim: int) -> Any:
    """Build a policy from config.

    Supports:
    - ``policy_type: "baseline"`` → ActorCritic (MLP)
    - ``policy_type: "hybrid"``  → HybridActorCritic
    - ``policy_type: "st"``      → STActorCritic (Phase 3)

    Falls back to ``"baseline"`` if not specified.
    """
    model_cfg = config.get("model", {})
    policy_type = str(model_cfg.get("policy_type", "baseline"))
    max_nodes = int(config.get("env", {}).get("max_nodes", 20))
    hidden_dims = list(model_cfg.get("hidden_dims", [128, 128]))
    seed = int(config.get("env", {}).get("seed", 42))

    from src.env.openenv_adapter import NUM_DISCRETE_ACTIONS

    if policy_type == "st":
        from src.models.st_encoder import STActorCritic
        policy = STActorCritic(
            node_feature_dim=4,
            max_nodes=max_nodes,
            action_dim=NUM_DISCRETE_ACTIONS,
            gcn_hidden_dim=int(model_cfg.get("gcn_hidden_dim", 32)),
            gru_hidden_dim=int(model_cfg.get("gru_hidden_dim", 32)),
            global_context_dim=int(model_cfg.get("global_context_dim", 32)),
            seed=seed,
        )
        policy.reset_episode()
        return policy

    if policy_type == "hybrid":
        from src.models.actor_critic import HybridActorCritic
        return HybridActorCritic(
            obs_dim=obs_dim,
            action_dim=NUM_DISCRETE_ACTIONS,
            max_nodes=max_nodes,
            hidden_dims=hidden_dims,
            seed=seed,
        )

    # Default: baseline MLP
    from src.models.actor_critic import ActorCritic
    return ActorCritic(
        obs_dim=obs_dim,
        action_dim=NUM_DISCRETE_ACTIONS,
        max_nodes=max_nodes,
        hidden_dims=hidden_dims,
        seed=seed,
    )


def _policy_act(policy: Any, obs: list[float], deterministic: bool, max_nodes: int, vax_budget: float) -> Any:
    """Call the right act() method for different policy types."""
    from src.models.st_encoder import STActorCritic
    from src.models.actor_critic import HybridActorCritic, ActorCritic

    if isinstance(policy, STActorCritic):
        action, _, _, _ = policy.act(obs, deterministic=deterministic, vaccine_budget=vax_budget)
        return action
    if isinstance(policy, HybridActorCritic):
        action, _, _, _ = policy.act(obs, deterministic=deterministic, vaccine_budget=vax_budget)
        return action
    # ActorCritic baseline
    actions, _, _ = policy.act(obs, deterministic=deterministic)
    return actions


def _load_checkpoint(policy: Any, checkpoint_path: str) -> bool:
    """Load weights from a .pt checkpoint into *policy* in-place.

    Parameters
    ----------
    policy:
        A ``torch.nn.Module`` policy object whose weights should be replaced.
    checkpoint_path:
        Path to the ``.pt`` file produced by the training loop.

    Returns
    -------
    bool
        ``True`` on success, ``False`` if loading was skipped or failed.
    """
    if not Path(checkpoint_path).exists():
        logger.warning(
            "Checkpoint not found — skipping weight load. "
            "Path: %s", checkpoint_path,
        )
        return False

    resolved = str(Path(checkpoint_path).resolve())

    try:
        import torch  # local import: not every user has torch
    except ImportError:
        logger.warning(
            "torch is not installed; cannot load checkpoint weights. "
            "Path: %s", resolved,
        )
        return False

    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
    except Exception as exc:
        logger.error(
            "Failed to read checkpoint file (%s). "
            "Path: %s", exc, resolved,
        )
        return False

    state_dict = ckpt.get("policy_state_dict")
    if state_dict is None:
        logger.error(
            "Checkpoint has no 'policy_state_dict' key — incompatible format. "
            "Available keys: %s. Path: %s",
            list(ckpt.keys()), resolved,
        )
        return False

    try:
        policy.load_state_dict(state_dict)
        logger.info("Checkpoint loaded: %s", resolved)
        return True
    except Exception as exc:
        logger.error(
            "load_state_dict failed (%s) — checkpoint may be incompatible with "
            "current policy architecture. Path: %s", exc, resolved,
        )
        return False


# ---------------------------------------------------------------------------
# EvalHarness
# ---------------------------------------------------------------------------

class EvalHarness:
    """Multi-task, multi-seed evaluation harness.

    Parameters
    ----------
    config:
        Full experiment config dict.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        eval_cfg = config.get("eval", {})
        self.tasks = list(eval_cfg.get("tasks", ALL_TASKS))
        self.seeds = list(eval_cfg.get("seeds", list(range(DEFAULT_N_SEEDS))))
        self.n_episodes = int(eval_cfg.get("n_episodes", DEFAULT_N_EPISODES))
        self.deterministic = bool(eval_cfg.get("deterministic", True))
        self.lag_steps = int(config.get("lag", {}).get("steps", 0))
        self.max_nodes = int(config.get("env", {}).get("max_nodes", 20))
        self._checkpoint_path = eval_cfg.get("checkpoint_path")

    def run(self) -> list[EpisodeResult]:
        """Run evaluation across all tasks and seeds.

        Returns
        -------
        results:
            List of ``EpisodeResult`` objects, one per episode.
        """
        all_results: list[EpisodeResult] = []

        for task_name in self.tasks:
            for seed in self.seeds:
                logger.info("Evaluating task=%s seed=%d", task_name, seed)
                try:
                    results = self._run_task_seed(task_name, seed)
                    all_results.extend(results)
                except Exception as exc:
                    logger.warning(
                        "Evaluation failed for task=%s seed=%d: %s",
                        task_name, seed, exc,
                    )

        return all_results

    def _run_task_seed(
        self,
        task_name: str,
        seed: int,
    ) -> list[EpisodeResult]:
        """Run n_episodes for a single (task, seed) combination."""
        from src.env.openenv_adapter import OpenEnvAdapter

        env = OpenEnvAdapter(task_name=task_name, seed=seed, max_nodes=self.max_nodes)
        obs_dim = env.obs_dim
        policy = _build_policy({**self.config, "env": {**self.config.get("env", {}), "seed": seed, "task_name": task_name}}, obs_dim)

        # Load checkpoint weights if a path was provided
        if self._checkpoint_path:
            _load_checkpoint(policy, self._checkpoint_path)

        # Reset episode state for ST models
        if hasattr(policy, "reset_episode"):
            policy.reset_episode()

        results = []
        for ep_idx in range(self.n_episodes):
            ep_seed = seed + ep_idx * 100
            obs, info = env.reset(seed=ep_seed)
            if hasattr(policy, "reset_episode"):
                policy.reset_episode()

            ep_return = 0.0
            max_inf = 0.0
            econ_sum = 0.0
            steps = 0
            done = False

            while not done:
                vax_budget = env.vaccine_budget
                action = _policy_act(
                    policy, obs, self.deterministic, self.max_nodes, vax_budget
                )
                obs, reward, done, step_info = env.step(action)
                ep_return += reward
                steps += 1

                # Track peak infection from obs
                # obs layout: [infection_rate, economy, quarantine, pop_frac] * max_nodes + globals
                node_feature_dim = 4
                for i in range(env.num_nodes):
                    inf_rate = obs[i * node_feature_dim]
                    econ = obs[i * node_feature_dim + 1]
                    max_inf = max(max_inf, inf_rate)
                    econ_sum += econ

            mean_economy = econ_sum / max(steps * max(env.num_nodes, 1), 1)
            invalid_rate = float(step_info.get("invalid_action_rate", 0.0))
            # Defensive clamp: rate must be in [0, 1] (fraction of decisions).
            if invalid_rate > 1.0:
                logger.warning(
                    "invalid_action_rate=%.4f exceeds 1.0 for task=%s seed=%d ep=%d; "
                    "clamping to 1.0.  Check denominator in OpenEnvAdapter.",
                    invalid_rate, task_name, seed, ep_idx,
                )
                invalid_rate = min(invalid_rate, 1.0)

            results.append(EpisodeResult(
                task_name=task_name,
                seed=seed,
                episode_idx=ep_idx,
                ep_return=ep_return,
                peak_infection=max_inf,
                mean_economy=mean_economy,
                total_steps=steps,
                invalid_action_rate=invalid_rate,
                lag_steps=self.lag_steps,
            ))

        return results

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def print_summary(self, results: list[EpisodeResult]) -> None:
        """Print a formatted summary table to stdout."""
        if not results:
            print("No results to report.")
            return

        # Aggregate per task
        by_task: dict[str, list[EpisodeResult]] = {}
        for r in results:
            by_task.setdefault(r.task_name, []).append(r)

        print()
        print("=" * 80)
        print(f"{'EVALUATION SUMMARY':^80}")
        print("=" * 80)
        print(
            f"{'Task':<35} {'N':>4} {'Return':>9} {'Peak Inf':>9} "
            f"{'Economy':>9} {'Inv%':>6}"
        )
        print("-" * 80)

        for task_name in ALL_TASKS:
            if task_name not in by_task:
                continue
            task_results = by_task[task_name]
            returns = [r.ep_return for r in task_results]
            peak_infs = [r.peak_infection for r in task_results]
            economies = [r.mean_economy for r in task_results]
            inv_rates = [min(r.invalid_action_rate, 1.0) * 100 for r in task_results]

            def _fmt_mean_std(vals: list[float]) -> str:
                m = sum(vals) / max(len(vals), 1)
                s = math.sqrt(sum((v - m) ** 2 for v in vals) / max(len(vals) - 1, 1))
                return f"{m:>6.2f}±{s:.2f}"

            print(
                f"{task_name:<35} {len(task_results):>4} "
                f"{_fmt_mean_std(returns):>14} "
                f"{_fmt_mean_std(peak_infs):>14} "
                f"{_fmt_mean_std(economies):>14} "
                f"{sum(inv_rates)/max(len(inv_rates),1):>5.1f}"
            )

        print("=" * 80)
        print()

    def save_results(
        self,
        results: list[EpisodeResult],
        output_path: str,
    ) -> None:
        """Save results to JSON or CSV depending on file extension.

        Parameters
        ----------
        results:
            List of ``EpisodeResult`` objects.
        output_path:
            Path ending in ``.json`` or ``.csv``.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        if output_path.endswith(".csv"):
            self._save_csv(results, output_path)
        else:
            self._save_json(results, output_path)

        logger.info("Results saved to: %s", output_path)

    def _save_json(self, results: list[EpisodeResult], path: str) -> None:
        data = [r.to_dict() for r in results]
        with open(path, "w") as f:
            json.dump({"results": data, "n_results": len(data)}, f, indent=2)

    def _save_csv(self, results: list[EpisodeResult], path: str) -> None:
        if not results:
            return
        fieldnames = list(results[0].to_dict().keys())
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow(r.to_dict())


# ---------------------------------------------------------------------------
# Aggregate helpers
# ---------------------------------------------------------------------------

def aggregate_results(results: list[EpisodeResult]) -> dict[str, Any]:
    """Compute aggregate statistics from a list of results.

    Returns
    -------
    Dict with mean/std/min/max for each key metric.
    """
    if not results:
        return {}

    def _stats(vals: list[float]) -> dict[str, float]:
        n = max(len(vals), 1)
        m = sum(vals) / n
        s = math.sqrt(sum((v - m) ** 2 for v in vals) / max(n - 1, 1))
        return {"mean": m, "std": s, "min": min(vals), "max": max(vals)}

    returns = [r.ep_return for r in results]
    peak_infs = [r.peak_infection for r in results]
    economies = [r.mean_economy for r in results]
    inv_rates = [r.invalid_action_rate for r in results]

    return {
        "n_episodes": len(results),
        "return": _stats(returns),
        "peak_infection": _stats(peak_infs),
        "mean_economy": _stats(economies),
        "invalid_action_rate": _stats(inv_rates),
    }
