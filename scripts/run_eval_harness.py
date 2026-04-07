#!/usr/bin/env python3
"""Evaluation harness for robustness and generalization testing (Phase 7).

Runs the policy across multiple tasks, seeds, and optionally with lag,
then prints a summary table and saves results to JSON/CSV.

Usage
-----
    python scripts/run_eval_harness.py --config configs/full_pipeline.yaml
    python scripts/run_eval_harness.py --config configs/baseline.yaml --tasks easy medium hard
    python scripts/run_eval_harness.py --config configs/baseline.yaml --seeds 0 1 2 --n-episodes 5
    python scripts/run_eval_harness.py --config configs/baseline.yaml --output results/eval.json
    python scripts/run_eval_harness.py --config configs/baseline.yaml --output results/eval.csv

Task shorthand mapping
----------------------
  easy   → easy_localized_outbreak
  medium → medium_multi_center_spread
  hard   → hard_asymptomatic_high_density
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


def _load_config(path: str) -> dict[str, Any]:
    try:
        import yaml  # type: ignore[import]
        with open(path) as f:
            return yaml.safe_load(f)
    except ImportError:
        logging.warning("PyYAML not installed; using built-in defaults.")
        return {
            "env": {"task_name": "easy_localized_outbreak", "seed": 42, "max_nodes": 20},
            "model": {"hidden_dims": [128, 128], "policy_type": "baseline"},
            "eval": {"n_episodes": 3, "deterministic": True, "tasks": None, "seeds": [0, 1, 2]},
        }


def _resolve_tasks(task_args: list[str] | None) -> list[str]:
    if not task_args:
        return list(TASK_SHORTHAND.values())
    resolved = []
    for t in task_args:
        resolved.append(TASK_SHORTHAND.get(t, t))
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-scenario evaluation harness.")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument(
        "--tasks", nargs="+",
        help="Tasks to evaluate: easy|medium|hard or full task name. Default: all three.",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=None,
        help="Seeds to evaluate (default: 0 1 2).",
    )
    parser.add_argument("--n-episodes", type=int, default=None)
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save results (.json or .csv).")
    parser.add_argument("--policy-type", type=str, default=None,
                        help="Override policy type: baseline|hybrid|st.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cfg = _load_config(args.config)

    # Apply overrides
    tasks = _resolve_tasks(args.tasks)
    cfg.setdefault("eval", {})
    cfg["eval"]["tasks"] = tasks
    if args.seeds is not None:
        cfg["eval"]["seeds"] = args.seeds
    if args.n_episodes is not None:
        cfg["eval"]["n_episodes"] = args.n_episodes
    if args.policy_type is not None:
        cfg.setdefault("model", {})["policy_type"] = args.policy_type

    from src.eval.scenario_runner import EvalHarness

    harness = EvalHarness(cfg)
    results = harness.run()
    harness.print_summary(results)

    if args.output:
        harness.save_results(results, args.output)
        print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
