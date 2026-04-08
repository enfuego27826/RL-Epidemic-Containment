"""Evaluation gate checks and baseline comparisons.

Provides:

1. ``GATE_THRESHOLDS`` — the canonical pass/fail thresholds per task.
2. :func:`check_gates` — evaluate a result dict against the thresholds.
3. :class:`BaselineComparator` — compare policy results vs random / heuristic
   baselines and a previous-best checkpoint.
4. :func:`generate_gate_report` — produce a compact summary string suitable
   for stdout or storing as an artifact.

Gate thresholds
---------------
The thresholds are set based on the roadmap targets:

  - Easy:   Peak Inf < 0.30, Economy > 0.85, Inv% < 5
  - Medium: Peak Inf < 0.45, Economy > 0.80, Inv% < 8
  - Hard:   Peak Inf < 0.60, Economy > 0.75, Inv% < 12

Usage
-----
::

    from src.eval.eval_gates import check_gates, generate_gate_report, GATE_THRESHOLDS

    results_by_task = {
        "easy_localized_outbreak": {"peak_infection": 0.25, "economy": 0.88, "inv_pct": 3.2},
        ...
    }
    gate_results = check_gates(results_by_task)
    report = generate_gate_report(gate_results)
    print(report)

The output includes a PASS/FAIL indicator per metric per task, plus an
overall PASS/FAIL summary.

Integration with eval harness
------------------------------
::

    from src.eval.scenario_runner import EvalHarness
    from src.eval.eval_gates import check_gates_from_episode_results

    harness = EvalHarness(config)
    results = harness.run()
    gate_results = check_gates_from_episode_results(results)
"""

from __future__ import annotations

import math
from typing import Any


# ---------------------------------------------------------------------------
# Gate thresholds
# ---------------------------------------------------------------------------

GATE_THRESHOLDS: dict[str, dict[str, float]] = {
    "easy_localized_outbreak": {
        "peak_infection_max": 0.30,
        "economy_min": 0.85,
        "inv_pct_max": 5.0,
    },
    "medium_multi_center_spread": {
        "peak_infection_max": 0.45,
        "economy_min": 0.80,
        "inv_pct_max": 8.0,
    },
    "hard_asymptomatic_high_density": {
        "peak_infection_max": 0.60,
        "economy_min": 0.75,
        "inv_pct_max": 12.0,
    },
}

# Shorthand aliases
TASK_SHORTHANDS = {
    "easy": "easy_localized_outbreak",
    "medium": "medium_multi_center_spread",
    "hard": "hard_asymptomatic_high_density",
}


# ---------------------------------------------------------------------------
# Gate result dataclass (plain dict-based)
# ---------------------------------------------------------------------------

def _gate_result(
    task: str,
    metric: str,
    value: float,
    threshold: float,
    passed: bool,
) -> dict[str, Any]:
    return {
        "task": task,
        "metric": metric,
        "value": value,
        "threshold": threshold,
        "passed": passed,
    }


# ---------------------------------------------------------------------------
# check_gates
# ---------------------------------------------------------------------------

def check_gates(
    results_by_task: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """Check metric results against the gate thresholds.

    Parameters
    ----------
    results_by_task:
        Dict mapping task name (or shorthand) to a dict of metric values.
        Expected metric keys:
          - ``"peak_infection"``:  float in [0, 1]
          - ``"economy"``:         float in [0, 1]
          - ``"inv_pct"``:         float, percentage (0–100)

    Returns
    -------
    Dict with:
      - ``"checks"``: list of per-metric gate dicts (see :func:`_gate_result`)
      - ``"tasks"``:  dict mapping task name to {passed: bool, details: list}
      - ``"all_passed"``: bool — True iff every gate passed
    """
    all_checks: list[dict[str, Any]] = []
    tasks_summary: dict[str, Any] = {}

    for task_key, metrics in results_by_task.items():
        # Resolve shorthands
        task_name = TASK_SHORTHANDS.get(task_key, task_key)
        thresholds = GATE_THRESHOLDS.get(task_name)
        if thresholds is None:
            # Unknown task — skip gracefully
            tasks_summary[task_key] = {"passed": None, "details": [], "note": "unknown task"}
            continue

        task_checks: list[dict[str, Any]] = []

        # Peak infection
        pi = float(metrics.get("peak_infection", float("nan")))
        pi_thr = thresholds["peak_infection_max"]
        pi_passed = math.isfinite(pi) and pi < pi_thr
        c = _gate_result(task_name, "peak_infection", pi, pi_thr, pi_passed)
        c["direction"] = "<"
        task_checks.append(c)
        all_checks.append(c)

        # Economy
        ec = float(metrics.get("economy", float("nan")))
        ec_thr = thresholds["economy_min"]
        ec_passed = math.isfinite(ec) and ec > ec_thr
        c = _gate_result(task_name, "economy", ec, ec_thr, ec_passed)
        c["direction"] = ">"
        task_checks.append(c)
        all_checks.append(c)

        # Invalid action percentage
        inv = float(metrics.get("inv_pct", float("nan")))
        inv_thr = thresholds["inv_pct_max"]
        inv_passed = math.isfinite(inv) and inv < inv_thr
        c = _gate_result(task_name, "inv_pct", inv, inv_thr, inv_passed)
        c["direction"] = "<"
        task_checks.append(c)
        all_checks.append(c)

        task_passed = all(ch["passed"] for ch in task_checks)
        tasks_summary[task_name] = {"passed": task_passed, "details": task_checks}

    all_passed = all(ch["passed"] for ch in all_checks if ch["passed"] is not None)
    return {
        "checks": all_checks,
        "tasks": tasks_summary,
        "all_passed": all_passed,
    }


# ---------------------------------------------------------------------------
# check_gates_from_episode_results (EvalHarness integration)
# ---------------------------------------------------------------------------

def check_gates_from_episode_results(results: list[Any]) -> dict[str, Any]:
    """Compute aggregate metrics from EpisodeResult objects and run gate checks.

    Parameters
    ----------
    results:
        List of ``src.eval.scenario_runner.EpisodeResult`` objects.

    Returns
    -------
    Same structure as :func:`check_gates`.
    """
    by_task: dict[str, list[Any]] = {}
    for r in results:
        by_task.setdefault(r.task_name, []).append(r)

    results_by_task: dict[str, dict[str, float]] = {}
    for task_name, task_results in by_task.items():
        n = max(len(task_results), 1)
        mean_peak = sum(r.peak_infection for r in task_results) / n
        mean_econ = sum(r.mean_economy for r in task_results) / n
        mean_inv_pct = sum(
            # Defensive clamp to [0,1] before converting to percentage.
            # The scenario_runner already applies this clamp, but we guard
            # here too in case episode results are constructed directly.
            min(r.invalid_action_rate, 1.0) * 100.0 for r in task_results
        ) / n
        results_by_task[task_name] = {
            "peak_infection": mean_peak,
            "economy": mean_econ,
            "inv_pct": mean_inv_pct,
        }

    return check_gates(results_by_task)


# ---------------------------------------------------------------------------
# Baseline comparator
# ---------------------------------------------------------------------------

class BaselineComparator:
    """Compare a policy's results against reference baselines.

    The class maintains a registry of reference baseline values (random
    policy, heuristic policy, previous-best checkpoint) and provides
    comparison utilities.

    Parameters
    ----------
    baselines:
        Optional dict of pre-registered baseline results.  Keys are
        baseline names (e.g. ``"random"``, ``"heuristic"``).
        Values are dicts mapping task_name → metric dict.
    """

    # Hard-coded approximate random-policy baselines (pessimistic estimates)
    # These represent expected performance of a random policy based on the
    # default environment dynamics.
    RANDOM_BASELINES: dict[str, dict[str, float]] = {
        "easy_localized_outbreak": {
            "peak_infection": 0.65,
            "economy": 0.65,
            "inv_pct": 35.0,
        },
        "medium_multi_center_spread": {
            "peak_infection": 0.75,
            "economy": 0.60,
            "inv_pct": 40.0,
        },
        "hard_asymptomatic_high_density": {
            "peak_infection": 0.85,
            "economy": 0.55,
            "inv_pct": 48.0,
        },
    }

    def __init__(
        self,
        baselines: dict[str, dict[str, dict[str, float]]] | None = None,
    ) -> None:
        self._baselines: dict[str, dict[str, dict[str, float]]] = {
            "random": self.RANDOM_BASELINES,
        }
        if baselines:
            self._baselines.update(baselines)

    def register_baseline(
        self,
        name: str,
        task_metrics: dict[str, dict[str, float]],
    ) -> None:
        """Register a new or updated baseline.

        Parameters
        ----------
        name:
            Baseline identifier (e.g. ``"previous_best"``).
        task_metrics:
            Dict mapping task_name → metric dict.
        """
        self._baselines[name] = task_metrics

    def compare(
        self,
        policy_results: dict[str, dict[str, float]],
        task_name: str,
    ) -> dict[str, Any]:
        """Compare policy results against all registered baselines for a task.

        Parameters
        ----------
        policy_results:
            Dict mapping task_name → metric dict for the evaluated policy.
        task_name:
            Task to compare on.

        Returns
        -------
        Dict with one entry per baseline, each containing per-metric deltas
        (positive = policy is better than baseline).
        """
        policy = policy_results.get(task_name, {})
        comparison: dict[str, Any] = {}

        for baseline_name, baseline_data in self._baselines.items():
            baseline = baseline_data.get(task_name, {})
            deltas: dict[str, float] = {}

            for metric in ("peak_infection", "economy", "inv_pct"):
                p_val = float(policy.get(metric, float("nan")))
                b_val = float(baseline.get(metric, float("nan")))
                if math.isfinite(p_val) and math.isfinite(b_val):
                    # For peak_infection and inv_pct: lower is better (delta > 0 means improvement)
                    # For economy: higher is better (delta > 0 means improvement)
                    if metric in ("peak_infection", "inv_pct"):
                        deltas[metric] = b_val - p_val
                    else:
                        deltas[metric] = p_val - b_val
                else:
                    deltas[metric] = float("nan")
            comparison[baseline_name] = {
                "policy": {m: policy.get(m) for m in ("peak_infection", "economy", "inv_pct")},
                "baseline": {m: baseline.get(m) for m in ("peak_infection", "economy", "inv_pct")},
                "deltas": deltas,
                "beats_baseline": all(
                    math.isfinite(d) and d > 0 for d in deltas.values()
                ),
            }

        return comparison


# ---------------------------------------------------------------------------
# Report generator
# ---------------------------------------------------------------------------

def generate_gate_report(gate_results: dict[str, Any]) -> str:
    """Format gate check results as a human-readable string.

    Parameters
    ----------
    gate_results:
        Output of :func:`check_gates` or :func:`check_gates_from_episode_results`.

    Returns
    -------
    Multi-line string suitable for printing to stdout or saving to a file.
    """
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append(f"{'EVALUATION GATE REPORT':^70}")
    lines.append("=" * 70)

    for task_name, task_info in gate_results.get("tasks", {}).items():
        status = "PASS" if task_info.get("passed") else "FAIL"
        lines.append(f"\n  Task: {task_name}  [{status}]")
        for detail in task_info.get("details", []):
            direction = detail.get("direction", "?")
            metric = detail["metric"]
            value = detail["value"]
            threshold = detail["threshold"]
            passed = detail["passed"]
            icon = "✓" if passed else "✗"
            lines.append(
                f"    {icon} {metric:<20} {value:>7.3f}  "
                f"{direction} {threshold:.3f}"
            )

    overall = "ALL GATES PASSED ✓" if gate_results.get("all_passed") else "SOME GATES FAILED ✗"
    lines.append(f"\n{'─' * 70}")
    lines.append(f"  {overall}")
    lines.append("=" * 70)
    return "\n".join(lines)
