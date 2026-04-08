#!/usr/bin/env python3
"""Smoke tests for the visualization layer (src/viz).

Tests data parsing, aggregation, chart generation, and graph rendering
without requiring a real OpenEnv environment.  All checks use synthetic
data so they are fast (<5 s) and fully offline.

Usage
-----
    python src/tests/smoke_visualizer.py

Exit code 0 = all checks passed, 1 = at least one failure.
"""

from __future__ import annotations

import math
import os
import sys
import tempfile
import json
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_PASS = "  [PASS]"
_FAIL = "  [FAIL]"
_errors: list[str] = []


def _check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"{_PASS} {name}")
    else:
        msg = name + (f": {detail}" if detail else "")
        print(f"{_FAIL} {msg}")
        _errors.append(msg)


# ---------------------------------------------------------------------------
# Synthetic test data
# ---------------------------------------------------------------------------

ALL_TASKS = [
    "easy_localized_outbreak",
    "medium_multi_center_spread",
    "hard_asymptomatic_high_density",
]


def _make_results(n_per_task: int = 5) -> list[dict]:
    """Generate synthetic episode-result dicts covering all three tasks."""
    import random
    rng = random.Random(0)
    rows = []
    for task in ALL_TASKS:
        base_return = {"easy": -4.0, "medium": -15.0, "hard": -30.0}[
            task.split("_")[0]
        ]
        for i in range(n_per_task):
            rows.append({
                "task_name": task,
                "seed": i,
                "episode_idx": i,
                "ep_return": base_return + rng.gauss(0, 2),
                "peak_infection": rng.uniform(0.05, 0.8),
                "mean_economy": rng.uniform(0.5, 0.95),
                "total_steps": rng.randint(30, 60),
                "invalid_action_rate": rng.uniform(0.0, 0.3),
                "lag_steps": 0,
            })
    return rows


# ---------------------------------------------------------------------------
# 1. Data loading (JSON + CSV round-trip)
# ---------------------------------------------------------------------------

def check_data_loading() -> None:
    print("\n--- Data loading (JSON / CSV) ---")
    from src.viz.report import load_results

    results = _make_results(3)

    with tempfile.TemporaryDirectory() as tmp:
        # JSON
        json_path = os.path.join(tmp, "eval.json")
        with open(json_path, "w") as f:
            json.dump({"results": results}, f)
        loaded_json = load_results(json_path)
        _check("JSON load: correct count", len(loaded_json) == len(results),
               f"got {len(loaded_json)}")
        _check("JSON load: task_name preserved",
               loaded_json[0]["task_name"] == results[0]["task_name"])

        # CSV
        import csv
        csv_path = os.path.join(tmp, "eval.csv")
        fieldnames = list(results[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow(r)
        loaded_csv = load_results(csv_path)
        _check("CSV load: correct count", len(loaded_csv) == len(results),
               f"got {len(loaded_csv)}")
        _check("CSV load: ep_return is numeric",
               isinstance(loaded_csv[0]["ep_return"], float))


# ---------------------------------------------------------------------------
# 2. Aggregation
# ---------------------------------------------------------------------------

def check_aggregation() -> None:
    print("\n--- Aggregation ---")
    from src.viz.report import aggregate_by_task

    results = _make_results(10)
    agg = aggregate_by_task(results)

    _check("Aggregation: all tasks present",
           all(t in agg for t in ALL_TASKS))
    _check("Aggregation: ep_return tuple length",
           all(len(agg[t]["ep_return"]) == 2 for t in ALL_TASKS))
    _check("Aggregation: peak_infection mean in [0,1]",
           all(0.0 <= agg[t]["peak_infection"][0] <= 1.0 for t in ALL_TASKS))
    _check("Aggregation: invalid_action_pct in [0,100]",
           all(0.0 <= agg[t]["invalid_action_pct"][0] <= 100.0 for t in ALL_TASKS))


# ---------------------------------------------------------------------------
# 3. Report generation (charts + HTML)
# ---------------------------------------------------------------------------

def check_report_generation() -> None:
    print("\n--- Report generation ---")
    from src.viz.report import ReportGenerator

    results = _make_results(5)

    with tempfile.TemporaryDirectory() as tmp:
        gen = ReportGenerator.from_results(results, out_dir=tmp)
        artifacts = gen.run()

        _check("ReportGenerator returns dict", isinstance(artifacts, dict))
        _check("summary_overview PNG created",
               "summary_overview" in artifacts and os.path.exists(artifacts["summary_overview"]))
        _check("return_distribution PNG created",
               "return_distribution" in artifacts and os.path.exists(artifacts["return_distribution"]))
        _check("csv_summary created",
               "csv_summary" in artifacts and os.path.exists(artifacts["csv_summary"]))
        _check("html_report created",
               "html_report" in artifacts and os.path.exists(artifacts["html_report"]))

        # Validate CSV has right columns
        import csv
        with open(artifacts["csv_summary"]) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        _check("CSV summary has 3 rows", len(rows) == 3, f"got {len(rows)}")
        _check("CSV summary has return_mean column", "return_mean" in rows[0])

        # Validate HTML is non-empty
        html_size = os.path.getsize(artifacts["html_report"])
        _check("HTML report is non-empty", html_size > 1000, f"size={html_size}")


# ---------------------------------------------------------------------------
# 4. Graph layout helpers (pure Python, no env)
# ---------------------------------------------------------------------------

def check_graph_layout() -> None:
    print("\n--- Graph layout helpers ---")
    from src.viz.epidemic_graph import _circle_layout, _spring_layout, _infection_colour

    # Circle layout
    pos = _circle_layout(5)
    _check("circle_layout: 5 positions", len(pos) == 5)
    _check("circle_layout: all within radius 1.1",
           all(math.sqrt(x ** 2 + y ** 2) <= 1.1 for x, y in pos))

    # Spring layout
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
    spos = _spring_layout(edges, 5, iterations=20)
    _check("spring_layout: 5 positions", len(spos) == 5)
    _check("spring_layout: all within [-1, 1]",
           all(-1.0 <= x <= 1.0 and -1.0 <= y <= 1.0 for x, y in spos))

    # Colour helper
    c0 = _infection_colour(0.0)
    c1 = _infection_colour(1.0)
    c05 = _infection_colour(0.5)
    _check("infection_colour: low is green-ish", c0[1] >= 0.9)
    _check("infection_colour: high is red-ish", c1[0] >= 0.9 and c1[1] < 0.1)
    _check("infection_colour: mid is yellow-ish", c05[0] >= 0.9 and c05[1] >= 0.9)


# ---------------------------------------------------------------------------
# 5. EpisodeFrame construction (without running the env)
# ---------------------------------------------------------------------------

def check_episode_frame() -> None:
    print("\n--- EpisodeFrame & NodeFrame construction ---")
    from src.viz.epidemic_graph import NodeFrame, EpisodeFrame

    nf = NodeFrame(
        node_id="city_0",
        infection_rate=0.3,
        economic_health=0.8,
        is_quarantined=True,
        population_fraction=0.4,
        vaccinated_this_step=False,
        action_code=1,
    )
    _check("NodeFrame: node_id", nf.node_id == "city_0")
    _check("NodeFrame: is_quarantined", nf.is_quarantined is True)
    _check("NodeFrame: action_code quarantine", nf.action_code == 1)

    ef = EpisodeFrame(
        step=5,
        nodes=[nf],
        global_infection=0.3,
        global_economy=0.7,
        vaccine_budget_frac=0.8,
        reward=-2.5,
    )
    _check("EpisodeFrame: step", ef.step == 5)
    _check("EpisodeFrame: reward finite", math.isfinite(ef.reward))
    _check("EpisodeFrame: one node", len(ef.nodes) == 1)


# ---------------------------------------------------------------------------
# 6. Graph snapshot rendering (synthetic data, no env)
# ---------------------------------------------------------------------------

def check_graph_snapshot_rendering() -> None:
    print("\n--- Graph snapshot rendering ---")
    from src.viz.epidemic_graph import NodeFrame, EpisodeFrame, GraphVisualizer

    # Build a minimal mock env + visualizer that bypasses OpenEnv
    class _MockEnv:
        task_name = "easy_localized_outbreak"
        max_nodes = 5
        _initial_vaccine_budget = 1000.0
        num_nodes = 5
        _node_ids = ["city_0", "city_1", "city_2", "city_3", "city_4"]

        def reset(self, seed=42):
            return [0.1, 0.9, 0.0, 0.2] * 5 + [1.0, 0.8, 0.1, 0.0], {"num_nodes": 5}

        @property
        def vaccine_budget(self):
            return 1000.0

    viz = GraphVisualizer(_MockEnv())

    # Build two synthetic frames
    import random
    rng = random.Random(1)
    frames = []
    for step in range(3):
        nodes = [
            NodeFrame(
                node_id=f"city_{i}",
                infection_rate=rng.uniform(0.0, 0.6),
                economic_health=rng.uniform(0.5, 1.0),
                is_quarantined=(i == 2 and step > 0),
                population_fraction=0.2,
                vaccinated_this_step=(i == 1 and step == 1),
                action_code=3 if (i == 1 and step == 1) else 0,
            )
            for i in range(5)
        ]
        frames.append(EpisodeFrame(
            step=step,
            nodes=nodes,
            global_infection=rng.uniform(0.1, 0.5),
            global_economy=rng.uniform(0.5, 0.9),
            vaccine_budget_frac=1.0 - step * 0.1,
            reward=-rng.uniform(1, 5),
        ))

    with tempfile.TemporaryDirectory() as tmp:
        snap_path = os.path.join(tmp, "snap.png")
        viz.save_snapshot(frames[-1], snap_path, title="Smoke test snapshot")
        _check("save_snapshot: file created", os.path.exists(snap_path))
        _check("save_snapshot: file non-empty", os.path.getsize(snap_path) > 5000)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("  Visualization smoke tests")
    print("=" * 60)

    check_data_loading()
    check_aggregation()
    check_report_generation()
    check_graph_layout()
    check_episode_frame()
    check_graph_snapshot_rendering()

    print()
    if _errors:
        print(f"FAILED — {len(_errors)} check(s) failed:")
        for e in _errors:
            print(f"  ✗ {e}")
        sys.exit(1)
    else:
        total = 0
        for line in open(__file__):
            if "_check(" in line:
                total += 1
        print(f"All checks passed ({total} total).")
        sys.exit(0)


if __name__ == "__main__":
    main()
