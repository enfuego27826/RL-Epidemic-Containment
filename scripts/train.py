#!/usr/bin/env python3
"""Training entry-point for all PPO variants (baseline, hybrid, ST-GNN, MoRL).

Usage
-----
    # Phase 1 baseline
    python scripts/train.py --config configs/baseline.yaml

    # Phase 2 hybrid actions
    python scripts/train.py --config configs/phase2_hybrid.yaml

    # Phase 3 ST-GNN encoder
    python scripts/train.py --config configs/phase3_stgnn.yaml

    # Phase 6 multi-objective RL
    python scripts/train.py --config configs/phase6_morl.yaml

    # Full pipeline (Phases 3-6 combined)
    python scripts/train.py --config configs/full_pipeline.yaml

    # Common overrides
    python scripts/train.py --config configs/baseline.yaml --seed 1
    python scripts/train.py --config configs/baseline.yaml --task medium_multi_center_spread
    python scripts/train.py --config configs/baseline.yaml --smoke-test

The ``--smoke-test`` flag limits training to 10 steps for quick CI validation.

Trainer selection
-----------------
The trainer is chosen based on config keys:
  - ``morl`` key present → ``PPOMorl`` (Phase 6)
  - ``model.policy_type == "st"`` → ``PPOBaseline`` with ``STActorCritic``
  - otherwise → ``PPOBaseline`` (Phase 1/2)
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
    """Load a YAML config file, returning a nested dict.

    Falls back to a minimal inline default when PyYAML is not installed, so
    the smoke-test runs in environments with only stdlib.
    """
    try:
        import yaml  # type: ignore[import]

        with open(config_path) as f:
            cfg: dict[str, Any] = yaml.safe_load(f)
        return cfg
    except ImportError:
        # Provide minimal defaults so smoke-test doesn't crash.
        logging.warning(
            "PyYAML not installed; using built-in defaults.  "
            "Install PyYAML for full config support."
        )
        return {
            "env": {"task_name": "easy_localized_outbreak", "seed": 42, "max_nodes": 20},
            "ppo": {
                "total_timesteps": 10,
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
        }


def _apply_cli_overrides(
    cfg: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Apply command-line overrides to the loaded config dict (in-place)."""
    if args.seed is not None:
        cfg["env"]["seed"] = args.seed
    if args.task is not None:
        cfg["env"]["task_name"] = args.task
    if args.smoke_test:
        cfg["ppo"]["total_timesteps"] = 10
        cfg["ppo"]["n_steps"] = 5
        cfg["ppo"]["batch_size"] = 5
        cfg["ppo"]["n_epochs"] = 1
        cfg["logging"]["checkpoint_interval"] = 999999
    return cfg


def _set_seeds(seed: int) -> None:
    """Seed Python's random module for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a PPO baseline for epidemic containment."
    )
    parser.add_argument(
        "--config",
        default="configs/baseline.yaml",
        help="Path to the YAML config file (default: configs/baseline.yaml).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override random seed.")
    parser.add_argument("--task", type=str, default=None, help="Override task name.")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run only 10 steps for CI smoke testing.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

    logger.info("Loading config from: %s", args.config)
    cfg = _load_config(args.config)
    cfg = _apply_cli_overrides(cfg, args)

    seed: int = int(cfg["env"].get("seed", 42))
    _set_seeds(seed)
    logger.info("Seed set to %d", seed)

    # Select trainer based on config
    if "morl" in cfg:
        logger.info("MoRL config detected — using PPOMorl trainer (Phase 6)")
        from src.train.ppo_morl import PPOMorl
        trainer: Any = PPOMorl(config=cfg)
    else:
        from src.train.ppo_baseline import PPOBaseline
        trainer = PPOBaseline(config=cfg)

    trainer.train()
    logger.info("Training finished.")


if __name__ == "__main__":
    main()
