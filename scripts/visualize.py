#!/usr/bin/env python3
"""Presentation-quality visualizer for RL Epidemic Containment results.

Reads evaluation outputs (JSON/CSV from ``run_eval_harness.py``) and/or
runs a fresh evaluation episode, then generates:

* Summary charts (return, peak infection, economy, invalid-action rate)
* Epidemic graph visualization — cities as nodes, travel routes as edges,
  coloured by infection rate, with quarantine borders and vaccination stars
* Optional animated GIF of the graph over one episode
* A self-contained HTML report with all charts embedded
* A CSV summary table

Quick start
-----------
1. Run evaluation first (produces a JSON file):

    python scripts/run_eval_harness.py \\
        --config configs/baseline.yaml \\
        --output results/eval.json

2. Generate visuals from that JSON:

    python scripts/visualize.py --input results/eval.json

3. Or do both in one command (runs eval + generates graph for one episode):

    python scripts/visualize.py --config configs/baseline.yaml --graph

Full CLI
--------
    python scripts/visualize.py [OPTIONS]

Options
-------
  --input PATH          Path to existing eval JSON/CSV file.
                        If omitted, a fresh 3-task evaluation is run.
  --config PATH         Config YAML used for fresh eval (default: configs/baseline.yaml).
  --output-dir DIR      Where to write output artifacts (default: out/report).
  --title TEXT          Report title (default: auto).
  --graph               Also render the epidemic city-graph for one episode.
  --graph-task TASK     Task for graph episode (default: easy_localized_outbreak).
  --graph-seed INT      Seed for graph episode (default: 42).
  --animate             Save animated GIF of the graph episode (slow; needs Pillow).
  --format {png,html,all}
                        Output format selector (default: all).
  --n-episodes N        Episodes per task for fresh eval (default: 5).
  --seeds S [S ...]     Seeds for fresh eval (default: 42 43 44).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fresh evaluation helpers
# ---------------------------------------------------------------------------

TASK_SHORTHAND = {
    "easy": "easy_localized_outbreak",
    "medium": "medium_multi_center_spread",
    "hard": "hard_asymptomatic_high_density",
}


def _run_fresh_eval(
    config_path: str,
    n_episodes: int,
    seeds: list[int],
) -> list[dict]:
    """Run the eval harness and return raw result dicts."""
    try:
        import yaml
    except ImportError:
        logger.error("PyYAML is required. Install it with: pip install pyyaml")
        sys.exit(1)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    cfg.setdefault("eval", {})
    cfg["eval"]["n_episodes"] = n_episodes
    cfg["eval"]["seeds"] = seeds
    cfg["eval"]["tasks"] = [
        "easy_localized_outbreak",
        "medium_multi_center_spread",
        "hard_asymptomatic_high_density",
    ]

    from src.eval.scenario_runner import EvalHarness
    harness = EvalHarness(cfg)
    results = harness.run()
    if not results:
        logger.error(
            "Fresh evaluation produced no results. "
            "This usually means PyTorch is not installed. "
            "Either install torch (pip install torch) or provide "
            "an existing results file with --input results/eval.json."
        )
        sys.exit(1)
    harness.print_summary(results)
    return [r.to_dict() for r in results]


# ---------------------------------------------------------------------------
# Graph visualization helper
# ---------------------------------------------------------------------------

def _render_graph(
    config_path: str,
    task: str,
    seed: int,
    out_dir: str,
    animate: bool,
    title: str,
    gif_fps: int = 4,
) -> dict[str, str]:
    """Collect one episode and render graph snapshot + optional animation."""
    import yaml

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    max_nodes = int(cfg.get("env", {}).get("max_nodes", 20))

    from src.env.openenv_adapter import OpenEnvAdapter
    from src.viz.epidemic_graph import GraphVisualizer

    env = OpenEnvAdapter(task_name=task, seed=seed, max_nodes=max_nodes)
    viz = GraphVisualizer(env)

    logger.info("Collecting episode for graph visualization (task=%s seed=%d)…", task, seed)
    frames = viz.collect_episode(policy=None, seed=seed)
    logger.info("Collected %d frames.", len(frames))

    artifacts: dict[str, str] = {}
    os.makedirs(out_dir, exist_ok=True)

    # Snapshot of the last frame
    snap_path = os.path.join(out_dir, "graph_snapshot.png")
    viz.save_snapshot(
        frames[-1],
        snap_path,
        title=f"{title} — {task} (step {frames[-1].step})",
    )
    artifacts["graph_snapshot"] = snap_path
    logger.info("Graph snapshot saved to %s", snap_path)

    # Snapshot of peak-infection frame
    peak_frame = max(frames, key=lambda f: f.global_infection)
    peak_path = os.path.join(out_dir, "graph_peak_infection.png")
    viz.save_snapshot(
        peak_frame,
        peak_path,
        title=f"{title} — {task} (peak infection, step {peak_frame.step})",
    )
    artifacts["graph_peak"] = peak_path
    logger.info("Peak-infection snapshot saved to %s", peak_path)

    # Animation
    if animate:
        gif_path = os.path.join(out_dir, "graph_episode.gif")
        logger.info("Rendering animation (%d frames) → %s …", len(frames), gif_path)
        viz.save_animation(frames, gif_path, fps=gif_fps)
        artifacts["graph_animation"] = gif_path
        logger.info("Animation saved to %s", gif_path)

    return artifacts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate presentation-quality visuals for epidemic-control results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", type=str, default=None,
                        help="Path to existing eval JSON or CSV file.")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml",
                        help="Config YAML for fresh evaluation (used when --input is omitted).")
    parser.add_argument("--output-dir", type=str, default="out/report",
                        help="Directory for all output artifacts (default: out/report).")
    parser.add_argument("--title", type=str, default=None,
                        help="Report title.")
    parser.add_argument("--graph", action="store_true",
                        help="Render the epidemic city-graph for one episode.")
    parser.add_argument("--graph-task", type=str, default="easy_localized_outbreak",
                        help="Task to use for graph episode.")
    parser.add_argument("--graph-seed", type=int, default=42,
                        help="Seed for graph episode.")
    parser.add_argument("--animate", action="store_true",
                        help="Save animated GIF (requires Pillow).")
    parser.add_argument("--gif-fps", type=int, default=4,
                        help="Frames per second for the animated GIF (default: 4).")
    parser.add_argument("--format", choices=["png", "html", "all"], default="all",
                        help="Output format (default: all).")
    parser.add_argument("--n-episodes", type=int, default=5,
                        help="Episodes per task for fresh eval (default: 5).")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44],
                        help="Seeds for fresh eval (default: 42 43 44).")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    title = args.title or "RL Epidemic Containment — Evaluation Report"
    graph_task = TASK_SHORTHAND.get(args.graph_task, args.graph_task)

    # ── 1. Load or generate results ──────────────────────────────────────
    if args.input:
        logger.info("Loading results from %s", args.input)
        from src.viz.report import load_results
        results = load_results(args.input)
    else:
        logger.info("No --input provided; running fresh evaluation with %s", args.config)
        results = _run_fresh_eval(
            config_path=args.config,
            n_episodes=args.n_episodes,
            seeds=args.seeds,
        )

    # ── 2. Generate summary charts ────────────────────────────────────────
    from src.viz.report import ReportGenerator
    gen = ReportGenerator.from_results(
        results=results,
        out_dir=args.output_dir,
        title=title,
    )
    if args.input:
        gen.results_path = args.input

    artifacts = gen.run()

    # ── 3. Graph visualization ─────────────────────────────────────────────
    if args.graph:
        try:
            graph_artifacts = _render_graph(
                config_path=args.config,
                task=graph_task,
                seed=args.graph_seed,
                out_dir=args.output_dir,
                animate=args.animate,
                gif_fps=args.gif_fps,
                title=title,
            )
            artifacts.update(graph_artifacts)

            # Rebuild HTML with graph images embedded
            gen_with_graph = ReportGenerator.from_results(
                results=results,
                out_dir=args.output_dir,
                title=title,
            )
            if args.input:
                gen_with_graph.results_path = args.input
            gen_with_graph._results = results
            artifacts["html_report"] = gen_with_graph._save_html_report(artifacts)

        except Exception as exc:
            logger.warning("Graph rendering failed: %s", exc, exc_info=True)

    # ── 4. Print summary ───────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  Visualization complete — artifacts written to:", args.output_dir)
    print("=" * 60)
    for name, path in sorted(artifacts.items()):
        print(f"  {name:<25} {path}")
    print()
    html = artifacts.get("html_report")
    if html:
        print(f"  Open the report: file://{os.path.abspath(html)}")
    print()


if __name__ == "__main__":
    main()
