#!/usr/bin/env python3
"""Inference script — run a trained policy on a single episode.

Loads a checkpoint (if available) and runs one episode deterministically,
printing step-by-step actions and metrics.

Usage
-----
    python scripts/inference.py --config configs/baseline.yaml
    python scripts/inference.py --config configs/full_pipeline.yaml --task easy
    python scripts/inference.py --config configs/baseline.yaml --checkpoint checkpoints/baseline/checkpoint_final.txt
    python scripts/inference.py --config configs/baseline.yaml --task hard --seed 7 --render
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

TASK_SHORTHAND = {
    "easy": "easy_localized_outbreak",
    "medium": "medium_multi_center_spread",
    "hard": "hard_asymptomatic_high_density",
}

ACTION_NAMES = {0: "noop", 1: "quarantine", 2: "lift", 3: "vaccinate"}


def _load_config(path: str) -> dict[str, Any]:
    try:
        import yaml  # type: ignore[import]
        with open(path) as f:
            return yaml.safe_load(f)
    except ImportError:
        return {
            "env": {"task_name": "easy_localized_outbreak", "seed": 42, "max_nodes": 20},
            "model": {"hidden_dims": [128, 128], "policy_type": "baseline"},
            "eval": {"deterministic": True},
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a trained policy inference.")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--task", default=None, help="Task shorthand or full name.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--render", action="store_true",
        help="Print step-by-step actions (always on by default).",
    )
    parser.add_argument("--policy-type", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger(__name__)

    cfg = _load_config(args.config)
    if args.task:
        cfg["env"]["task_name"] = TASK_SHORTHAND.get(args.task, args.task)
    if args.seed is not None:
        cfg["env"]["seed"] = args.seed
    if args.policy_type:
        cfg.setdefault("model", {})["policy_type"] = args.policy_type

    task_name = cfg["env"]["task_name"]
    seed = int(cfg["env"].get("seed", 42))
    max_nodes = int(cfg["env"].get("max_nodes", 20))
    deterministic = bool(cfg.get("eval", {}).get("deterministic", True))

    from src.env.openenv_adapter import OpenEnvAdapter
    from src.eval.scenario_runner import _build_policy, _policy_act, _load_checkpoint

    env = OpenEnvAdapter(task_name=task_name, seed=seed, max_nodes=max_nodes)
    obs, info = env.reset(seed=seed)
    policy = _build_policy(cfg, env.obs_dim)
    if hasattr(policy, "reset_episode"):
        policy.reset_episode()

    checkpoint = args.checkpoint or cfg.get("eval", {}).get("checkpoint_path")
    if checkpoint and os.path.isfile(checkpoint):
        # Supported input formats:
        #   1) direct torch weights checkpoint (.pt/.pth/etc.)
        #   2) legacy metadata text file (.txt) with adjacent same-name .pt
        weights_path = checkpoint
        if checkpoint.endswith(".txt"):
            base_path = Path(checkpoint)
            candidates = [
                str(base_path.with_suffix(".pt")),
                str(base_path.with_suffix(".pth")),
            ]
            resolved = next((c for c in candidates if os.path.isfile(c)), None)
            if resolved is not None:
                weights_path = resolved
            else:
                weights_path = ""
                logger.warning(
                    "Checkpoint metadata file found but no adjacent .pt/.pth weights file: %s",
                    checkpoint,
                )
        if not weights_path:
            logger.warning("Skipping checkpoint load and using random weights.")
        elif not _load_checkpoint(policy, weights_path):
            logger.warning(
                "Failed to load checkpoint weights from %s. "
                "Verify path, checkpoint integrity, and architecture compatibility. "
                "See earlier checkpoint-loader logs for detailed cause. "
                "Continuing inference with random policy weights.",
                weights_path,
            )
    elif checkpoint:
        logger.warning("Checkpoint not found: %s — using random weights.", checkpoint)

    logger.info("Task: %s | Seed: %d | Deterministic: %s", task_name, seed, deterministic)
    print()
    print("=" * 60)
    print(f" Task: {task_name}")
    print(f" Seed: {seed} | Max nodes: {max_nodes}")
    print("=" * 60)
    print(f"{'Step':>5} | {'Reward':>8} | {'CumRet':>8} | {'Actions'}")
    print("-" * 60)

    ep_return = 0.0
    step = 0
    done = False

    while not done:
        vax_budget = env.vaccine_budget
        action = _policy_act(policy, obs, deterministic, max_nodes, vax_budget)
        obs, reward, done, step_info = env.step(action)
        ep_return += reward
        step += 1

        # Format action summary
        if isinstance(action, dict):
            disc = action.get("discrete", [])
            cont = action.get("continuous", [])
            action_summary = ", ".join(
                f"{ACTION_NAMES.get(d, d)}"
                + (f"+{c:.2f}" if c > 1e-4 else "")
                for d, c in zip(disc[:env.num_nodes], cont[:env.num_nodes])
                if d != 0  # skip no-ops for brevity
            ) or "all-noop"
        else:
            action_summary = ", ".join(
                f"n{i}:{ACTION_NAMES.get(a, a)}"
                for i, a in enumerate(action[:env.num_nodes])
                if a != 0
            ) or "all-noop"

        print(f"{step:>5} | {reward:>8.3f} | {ep_return:>8.3f} | {action_summary}")

    print("-" * 60)
    print(f"Episode complete — return: {ep_return:.3f}  steps: {step}")

    # Final metrics from env state
    try:
        state = env._env.state()
        print(f"Peak infection: {state.peak_infection_rate:.4f}")
        print(f"Final economy:  {state.global_economic_score:.4f}")
    except Exception:
        pass

    print("=" * 60)


if __name__ == "__main__":
    main()
