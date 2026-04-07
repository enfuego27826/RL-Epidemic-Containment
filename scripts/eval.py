#!/usr/bin/env python3
"""Evaluation entry-point for a trained PPO checkpoint.

Usage
-----
    python scripts/eval.py --config configs/baseline.yaml
    python scripts/eval.py --config configs/baseline.yaml --checkpoint checkpoints/baseline/checkpoint_final.txt
    python scripts/eval.py --config configs/baseline.yaml --task hard_asymptomatic_high_density --n-episodes 5

The script runs ``n_episodes`` deterministic episodes and prints a per-task
score table to stdout in the same format as the README's baseline scores.
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Ensure repo root is on the Python path so ``src.*`` imports resolve.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _load_config(config_path: str) -> dict[str, Any]:
    """Load a YAML config file.  Falls back to minimal defaults if PyYAML is missing."""
    try:
        import yaml  # type: ignore[import]

        with open(config_path) as f:
            cfg: dict[str, Any] = yaml.safe_load(f)
        return cfg
    except ImportError:
        logging.warning("PyYAML not installed; using built-in defaults.")
        return {
            "env": {"task_name": "easy_localized_outbreak", "seed": 42, "max_nodes": 20},
            "ppo": {
                "total_timesteps": 0,
                "n_envs": 1,
                "n_steps": 5,
                "batch_size": 5,
                "n_epochs": 1,
                "lr": 3e-4,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_eps": 0.2,
                "entropy_coef": 0.01,
                "value_loss_coef": 0.5,
                "max_grad_norm": 0.5,
            },
            "model": {"hidden_dims": [128, 128]},
            "logging": {
                "log_interval": 1,
                "checkpoint_dir": "checkpoints/baseline",
                "checkpoint_interval": 10,
            },
            "eval": {"n_episodes": 3, "deterministic": True, "checkpoint_path": None},
        }


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def _run_eval(
    task_name: str,
    seed: int,
    max_nodes: int,
    n_episodes: int,
    deterministic: bool,
    policy: Any,
) -> dict[str, float]:
    """Run evaluation episodes and return aggregated metrics.

    Parameters
    ----------
    task_name, seed, max_nodes:
        Forwarded to ``OpenEnvAdapter``.
    n_episodes:
        Number of evaluation episodes.
    deterministic:
        Whether to use greedy action selection.
    policy:
        An ``ActorCritic`` instance with an ``act()`` method.

    Returns
    -------
    metrics:
        Dict with ``mean_return``, ``mean_task_score``, ``peak_infection``,
        ``mean_economy``.
    """
    from src.env.openenv_adapter import OpenEnvAdapter

    adapter = OpenEnvAdapter(task_name=task_name, seed=seed, max_nodes=max_nodes)

    ep_returns: list[float] = []
    ep_scores: list[float] = []
    peak_infections: list[float] = []
    mean_economies: list[float] = []

    for ep in range(n_episodes):
        obs, _ = adapter.reset(seed=seed + ep)
        ep_return = 0.0
        ep_scores_this_episode: list[float] = []

        done = False
        while not done:
            actions, _, _ = policy.act(obs, deterministic=deterministic)
            obs, reward, done, info = adapter.step(actions)
            ep_return += reward
            if "task_score" in info:
                ep_scores_this_episode.append(float(info["task_score"]))

        ep_returns.append(ep_return)
        if ep_scores_this_episode:
            ep_scores.append(ep_scores_this_episode[-1])

        # Retrieve detailed metrics from the underlying env state
        state = adapter._env.state()
        peak_infections.append(float(state.peak_infection_rate))
        mean_economies.append(float(state.global_economic_score))

    def _mean(lst: list[float]) -> float:
        return sum(lst) / max(len(lst), 1)

    return {
        "mean_return": _mean(ep_returns),
        "mean_task_score": _mean(ep_scores) if ep_scores else float("nan"),
        "peak_infection": _mean(peak_infections),
        "mean_economy": _mean(mean_economies),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained PPO baseline for epidemic containment."
    )
    parser.add_argument(
        "--config",
        default="configs/baseline.yaml",
        help="Path to the YAML config file.",
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint file.")
    parser.add_argument("--task", type=str, default=None, help="Override task name.")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed.")
    parser.add_argument("--n-episodes", type=int, default=None, help="Number of eval episodes.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

    cfg = _load_config(args.config)

    # CLI overrides
    if args.task:
        cfg["env"]["task_name"] = args.task
    if args.seed is not None:
        cfg["env"]["seed"] = args.seed
    eval_cfg = cfg.get("eval", {})
    if args.n_episodes is not None:
        eval_cfg["n_episodes"] = args.n_episodes
    if args.checkpoint is not None:
        eval_cfg["checkpoint_path"] = args.checkpoint

    task_name: str = cfg["env"]["task_name"]
    seed: int = int(cfg["env"].get("seed", 42))
    max_nodes: int = int(cfg["env"].get("max_nodes", 20))
    n_episodes: int = int(eval_cfg.get("n_episodes", 5))
    deterministic: bool = bool(eval_cfg.get("deterministic", True))

    _set_seeds(seed)

    # Build a fresh (untrained) policy for the scaffold
    from src.env.openenv_adapter import NUM_DISCRETE_ACTIONS, OpenEnvAdapter
    from src.models.actor_critic import ActorCritic

    tmp_adapter = OpenEnvAdapter(task_name=task_name, seed=seed, max_nodes=max_nodes)
    obs_dim = tmp_adapter.obs_dim
    hidden_dims: list[int] = list(cfg.get("model", {}).get("hidden_dims", [128, 128]))

    policy = ActorCritic(
        obs_dim=obs_dim,
        action_dim=NUM_DISCRETE_ACTIONS,
        max_nodes=max_nodes,
        hidden_dims=hidden_dims,
        seed=seed,
    )

    checkpoint_path = eval_cfg.get("checkpoint_path")
    if checkpoint_path and os.path.isfile(checkpoint_path):
        logger.info("Checkpoint file found: %s (weight loading not yet implemented)", checkpoint_path)
    elif checkpoint_path:
        logger.warning("Checkpoint file not found: %s — using random weights.", checkpoint_path)
    else:
        logger.info("No checkpoint specified — evaluating with randomly initialised weights.")

    logger.info(
        "Evaluating on task=%s  seed=%d  n_episodes=%d  deterministic=%s",
        task_name, seed, n_episodes, deterministic,
    )

    metrics = _run_eval(
        task_name=task_name,
        seed=seed,
        max_nodes=max_nodes,
        n_episodes=n_episodes,
        deterministic=deterministic,
        policy=policy,
    )

    # Print score table
    print("\n" + "=" * 55)
    print(f"{'Evaluation Results':^55}")
    print("=" * 55)
    print(f"  Task:              {task_name}")
    print(f"  Episodes:          {n_episodes}")
    print(f"  Deterministic:     {deterministic}")
    print("-" * 55)
    print(f"  Mean episode return:   {metrics['mean_return']:>8.3f}")
    print(f"  Mean task score:       {metrics['mean_task_score']:>8.3f}")
    print(f"  Peak infection rate:   {metrics['peak_infection']:>8.3f}")
    print(f"  Mean economy score:    {metrics['mean_economy']:>8.3f}")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
