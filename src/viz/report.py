"""Summary report generator.

Reads evaluation results (JSON / CSV produced by ``EvalHarness``) and
generates polished presentation charts:

* Bar chart of mean ± std episode return per task
* Peak infection rate and economy score comparison
* Invalid-action rate per task
* Time-series of global infection and economy (when step-level data is
  available via the ``timeseries`` key in the JSON)
* HTML summary page combining all charts

Usage
-----
    from src.viz.report import ReportGenerator
    gen = ReportGenerator("results/eval.json", out_dir="out/report")
    artifacts = gen.run()
    print(artifacts)
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# Colour palette (consistent with epidemic_graph.py)
# ---------------------------------------------------------------------------

TASK_COLOURS = {
    "easy_localized_outbreak": "#2ecc71",
    "medium_multi_center_spread": "#f39c12",
    "hard_asymptomatic_high_density": "#e74c3c",
}
DEFAULT_COLOUR = "#3498db"

ALL_TASKS = [
    "easy_localized_outbreak",
    "medium_multi_center_spread",
    "hard_asymptomatic_high_density",
]

TASK_LABELS = {
    "easy_localized_outbreak": "Easy\n(localized)",
    "medium_multi_center_spread": "Medium\n(multi-center)",
    "hard_asymptomatic_high_density": "Hard\n(asymptomatic)",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_json(path: str) -> list[dict[str, Any]]:
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "results" in data:
        return data["results"]
    raise ValueError(f"Unrecognised JSON structure in {path}")


def _load_csv(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed: dict[str, Any] = {}
            for k, v in row.items():
                try:
                    parsed[k] = float(v)
                except (ValueError, TypeError):
                    parsed[k] = v
            rows.append(parsed)
    return rows


def load_results(path: str) -> list[dict[str, Any]]:
    """Load evaluation results from a JSON or CSV file.

    Parameters
    ----------
    path:
        Path to the results file.  ``.json`` or ``.csv`` extensions are
        detected automatically.

    Returns
    -------
    list of episode-result dicts (keys: task_name, ep_return, peak_infection,
    mean_economy, invalid_action_rate, …).
    """
    p = path.lower()
    if p.endswith(".json"):
        return _load_json(path)
    if p.endswith(".csv"):
        return _load_csv(path)
    raise ValueError(f"Unsupported file extension: {path}")


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _mean_std(vals: list[float]) -> tuple[float, float]:
    if not vals:
        return 0.0, 0.0
    n = len(vals)
    m = sum(vals) / n
    s = math.sqrt(sum((v - m) ** 2 for v in vals) / max(n - 1, 1))
    return m, s


def aggregate_by_task(
    results: list[dict[str, Any]],
) -> dict[str, dict[str, tuple[float, float]]]:
    """Group results by task and compute mean ± std for each metric.

    Returns
    -------
    dict: task_name → {metric_key → (mean, std)}
    """
    by_task: dict[str, list[dict[str, Any]]] = {}
    for r in results:
        task = str(r.get("task_name", "unknown"))
        by_task.setdefault(task, []).append(r)

    agg: dict[str, dict[str, tuple[float, float]]] = {}
    for task, rows in by_task.items():
        agg[task] = {
            "ep_return": _mean_std([float(r.get("ep_return", 0)) for r in rows]),
            "peak_infection": _mean_std([float(r.get("peak_infection", 0)) for r in rows]),
            "mean_economy": _mean_std([float(r.get("mean_economy", 0)) for r in rows]),
            # Clamp to [0,1] before converting to percent
            "invalid_action_pct": _mean_std([
                min(float(r.get("invalid_action_rate", 0)), 1.0) * 100
                for r in rows
            ]),
        }
    return agg


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------

def _bar_with_error(
    ax: Any,
    tasks: list[str],
    means: list[float],
    stds: list[float],
    ylabel: str,
    title: str,
    colours: list[str],
    ylim: tuple[float, float] | None = None,
    fmt: str = ".2f",
) -> None:
    """Draw a bar chart with error bars on an existing axes."""
    import matplotlib.pyplot as plt

    x = list(range(len(tasks)))
    bars = ax.bar(x, means, yerr=stds, color=colours, alpha=0.85, capsize=5,
                  error_kw={"ecolor": "#555", "linewidth": 1.2})

    # Value labels above bars
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.01 * abs(m or 1), f"{m:{fmt}}", ha="center",
                va="bottom", fontsize=8, color="#333")

    labels = [TASK_LABELS.get(t, t) for t in tasks]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    if ylim:
        ax.set_ylim(*ylim)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)


# ---------------------------------------------------------------------------
# ReportGenerator
# ---------------------------------------------------------------------------

class ReportGenerator:
    """Generate a full visual report from evaluation results.

    Parameters
    ----------
    results_path:
        Path to ``.json`` or ``.csv`` produced by ``EvalHarness.save_results()``.
        May also be ``None`` when results are passed in-memory via
        :meth:`from_results`.
    out_dir:
        Directory where all output artifacts are written.
    title:
        Human-readable title embedded in charts and the HTML report.
    """

    def __init__(
        self,
        results_path: str | None,
        out_dir: str = "out/report",
        title: str = "RL Epidemic Containment — Evaluation Report",
    ) -> None:
        self.results_path = results_path
        self.out_dir = out_dir
        self.title = title
        self._results: list[dict[str, Any]] = []
        self._timeseries: list[dict[str, Any]] = []  # optional step-level data

        if results_path is not None:
            self._results = load_results(results_path)
            # Check for optional time-series embedded in JSON
            if results_path.lower().endswith(".json"):
                try:
                    with open(results_path) as f:
                        raw = json.load(f)
                    if isinstance(raw, dict) and "timeseries" in raw:
                        self._timeseries = raw["timeseries"]
                except Exception:
                    pass

    @classmethod
    def from_results(
        cls,
        results: list[dict[str, Any]],
        out_dir: str = "out/report",
        title: str = "RL Epidemic Containment — Evaluation Report",
        timeseries: list[dict[str, Any]] | None = None,
    ) -> "ReportGenerator":
        """Create a generator from in-memory result dicts."""
        gen = cls(results_path=None, out_dir=out_dir, title=title)
        gen._results = results
        gen._timeseries = timeseries or []
        return gen

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> dict[str, str]:
        """Generate all charts and save them.

        Returns
        -------
        dict mapping artifact name → file path.
        """
        import matplotlib
        matplotlib.use("Agg")

        os.makedirs(self.out_dir, exist_ok=True)

        if not self._results:
            print("[visualize] No results to plot.")
            return {}

        agg = aggregate_by_task(self._results)
        tasks = [t for t in ALL_TASKS if t in agg]
        if not tasks:
            tasks = list(agg.keys())

        artifacts: dict[str, str] = {}

        # 1. Summary overview (4 panels)
        path = self._plot_summary_overview(agg, tasks)
        artifacts["summary_overview"] = path

        # 2. Individual metric charts
        artifacts["returns"] = self._plot_single_metric(
            agg, tasks, "ep_return", "Mean Episode Return", "Episode Return",
            fmt=".2f",
        )
        artifacts["peak_infection"] = self._plot_single_metric(
            agg, tasks, "peak_infection", "Peak Infection Rate", "Infection Rate",
            ylim=(0.0, 1.0), fmt=".3f",
        )
        artifacts["economy"] = self._plot_single_metric(
            agg, tasks, "mean_economy", "Mean Economy Score", "Economy Score",
            ylim=(0.0, 1.0), fmt=".3f",
        )
        artifacts["invalid_actions"] = self._plot_single_metric(
            agg, tasks, "invalid_action_pct",
            "Invalid Action Rate (%)", "Invalid Actions (%)",
            ylim=(0.0, 100.0), fmt=".1f",
        )

        # 3. Return distribution (strip chart)
        artifacts["return_distribution"] = self._plot_return_distribution(tasks)

        # 4. Time-series (if available)
        if self._timeseries:
            path = self._plot_timeseries()
            if path:
                artifacts["timeseries"] = path

        # 5. CSV summary
        csv_path = self._save_csv_summary(agg, tasks)
        artifacts["csv_summary"] = csv_path

        # 6. HTML report
        html_path = self._save_html_report(artifacts)
        artifacts["html_report"] = html_path

        return artifacts

    # ------------------------------------------------------------------
    # Chart: 4-panel overview
    # ------------------------------------------------------------------

    def _plot_summary_overview(
        self,
        agg: dict[str, dict[str, tuple[float, float]]],
        tasks: list[str],
    ) -> str:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(self.title, fontsize=13, fontweight="bold", y=1.01)

        colours = [TASK_COLOURS.get(t, DEFAULT_COLOUR) for t in tasks]

        panels = [
            ("ep_return", "Episode Return", "Mean Episode Return", None, ".2f"),
            ("peak_infection", "Peak Infection Rate", "Infection Rate", (0, 1), ".3f"),
            ("mean_economy", "Economy Score", "Economy Score", (0, 1), ".3f"),
            ("invalid_action_pct", "Invalid Actions (%)", "Invalid Actions (%)", (0, 100), ".1f"),
        ]

        for ax, (key, title, ylabel, ylim, fmt) in zip(axes.flat, panels):
            means = [agg[t][key][0] for t in tasks]
            stds = [agg[t][key][1] for t in tasks]
            _bar_with_error(ax, tasks, means, stds, ylabel, title, colours, ylim, fmt)

        fig.tight_layout()
        path = os.path.join(self.out_dir, "summary_overview.png")
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        return path

    # ------------------------------------------------------------------
    # Chart: single metric
    # ------------------------------------------------------------------

    def _plot_single_metric(
        self,
        agg: dict[str, dict[str, tuple[float, float]]],
        tasks: list[str],
        key: str,
        title: str,
        ylabel: str,
        ylim: tuple[float, float] | None = None,
        fmt: str = ".2f",
    ) -> str:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4))
        colours = [TASK_COLOURS.get(t, DEFAULT_COLOUR) for t in tasks]
        means = [agg[t][key][0] for t in tasks]
        stds = [agg[t][key][1] for t in tasks]
        _bar_with_error(ax, tasks, means, stds, ylabel, title, colours, ylim, fmt)
        fig.tight_layout()
        fname = key.replace("_pct", "").replace("_", "-") + ".png"
        path = os.path.join(self.out_dir, fname)
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        return path

    # ------------------------------------------------------------------
    # Chart: return distribution strip/box plot
    # ------------------------------------------------------------------

    def _plot_return_distribution(self, tasks: list[str]) -> str:
        import matplotlib.pyplot as plt

        by_task: dict[str, list[float]] = {}
        for r in self._results:
            t = str(r.get("task_name", "unknown"))
            by_task.setdefault(t, []).append(float(r.get("ep_return", 0)))

        fig, ax = plt.subplots(figsize=(8, 5))

        present_tasks = [t for t in tasks if t in by_task]
        data = [by_task[t] for t in present_tasks]
        colours = [TASK_COLOURS.get(t, DEFAULT_COLOUR) for t in present_tasks]
        labels = [TASK_LABELS.get(t, t) for t in present_tasks]

        bp = ax.boxplot(
            data,
            patch_artist=True,
            notch=False,
            vert=True,
            widths=0.5,
        )
        for patch, colour in zip(bp["boxes"], colours):
            patch.set_facecolor(colour)
            patch.set_alpha(0.75)
        for median in bp["medians"]:
            median.set_color("#2c3e50")
            median.set_linewidth(2)

        # Overlay individual data points
        for i, (vals, colour) in enumerate(zip(data, colours), start=1):
            jitter = [i + 0.15 * (hash(str(v)) % 100 - 50) / 50.0 for v in vals]
            ax.scatter(jitter, vals, color=colour, alpha=0.6, s=20, zorder=3)

        ax.set_xticks(list(range(1, len(present_tasks) + 1)))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("Episode Return", fontsize=10)
        ax.set_title("Episode Return Distribution by Task", fontsize=11, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

        fig.tight_layout()
        path = os.path.join(self.out_dir, "return-distribution.png")
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        return path

    # ------------------------------------------------------------------
    # Chart: time-series (global infection + economy over steps)
    # ------------------------------------------------------------------

    def _plot_timeseries(self) -> str | None:
        """Plot infection + economy time-series if step-level data is present."""
        import matplotlib.pyplot as plt

        # Group by (task, seed) and average per step
        groups: dict[str, dict[int, list[float]]] = {}
        econ_groups: dict[str, dict[int, list[float]]] = {}

        for entry in self._timeseries:
            task = str(entry.get("task_name", "unknown"))
            step = int(entry.get("step", 0))
            inf = float(entry.get("global_infection", 0.0))
            econ = float(entry.get("global_economy", 1.0))
            groups.setdefault(task, {}).setdefault(step, []).append(inf)
            econ_groups.setdefault(task, {}).setdefault(step, []).append(econ)

        if not groups:
            return None

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

        for task, step_data in sorted(groups.items()):
            colour = TASK_COLOURS.get(task, DEFAULT_COLOUR)
            label = TASK_LABELS.get(task, task).replace("\n", " ")
            steps = sorted(step_data)
            means = [sum(step_data[s]) / len(step_data[s]) for s in steps]
            ax1.plot(steps, means, color=colour, label=label, linewidth=1.8)

        for task, step_data in sorted(econ_groups.items()):
            colour = TASK_COLOURS.get(task, DEFAULT_COLOUR)
            label = TASK_LABELS.get(task, task).replace("\n", " ")
            steps = sorted(step_data)
            means = [sum(step_data[s]) / len(step_data[s]) for s in steps]
            ax2.plot(steps, means, color=colour, label=label, linewidth=1.8)

        ax1.set_ylabel("Global Infection Rate", fontsize=9)
        ax1.set_ylim(0, 1)
        ax1.legend(fontsize=8)
        ax1.set_title("Infection & Economy Time-Series", fontsize=11, fontweight="bold")
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        ax1.grid(linestyle="--", alpha=0.4)

        ax2.set_ylabel("Economy Score", fontsize=9)
        ax2.set_xlabel("Step", fontsize=9)
        ax2.set_ylim(0, 1)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax2.grid(linestyle="--", alpha=0.4)

        fig.tight_layout()
        path = os.path.join(self.out_dir, "timeseries.png")
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        return path

    # ------------------------------------------------------------------
    # CSV summary
    # ------------------------------------------------------------------

    def _save_csv_summary(
        self,
        agg: dict[str, dict[str, tuple[float, float]]],
        tasks: list[str],
    ) -> str:
        import csv as _csv

        path = os.path.join(self.out_dir, "summary.csv")
        fieldnames = [
            "task",
            "n_episodes",
            "return_mean", "return_std",
            "peak_infection_mean", "peak_infection_std",
            "economy_mean", "economy_std",
            "invalid_action_pct_mean", "invalid_action_pct_std",
        ]
        by_task: dict[str, list[dict[str, Any]]] = {}
        for r in self._results:
            t = str(r.get("task_name", "unknown"))
            by_task.setdefault(t, []).append(r)

        with open(path, "w", newline="") as f:
            writer = _csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for task in tasks:
                m = agg[task]
                writer.writerow({
                    "task": task,
                    "n_episodes": len(by_task.get(task, [])),
                    "return_mean": f"{m['ep_return'][0]:.4f}",
                    "return_std": f"{m['ep_return'][1]:.4f}",
                    "peak_infection_mean": f"{m['peak_infection'][0]:.4f}",
                    "peak_infection_std": f"{m['peak_infection'][1]:.4f}",
                    "economy_mean": f"{m['mean_economy'][0]:.4f}",
                    "economy_std": f"{m['mean_economy'][1]:.4f}",
                    "invalid_action_pct_mean": f"{m['invalid_action_pct'][0]:.2f}",
                    "invalid_action_pct_std": f"{m['invalid_action_pct'][1]:.2f}",
                })
        return path

    # ------------------------------------------------------------------
    # HTML report
    # ------------------------------------------------------------------

    def _save_html_report(self, artifacts: dict[str, str]) -> str:
        """Generate a self-contained HTML summary page."""
        import base64

        def _img_tag(path: str, alt: str, width: str = "100%") -> str:
            if not path or not os.path.exists(path):
                return f"<p><em>({alt} not available)</em></p>"
            with open(path, "rb") as f:
                data = base64.b64encode(f.read()).decode()
            ext = Path(path).suffix.lstrip(".")
            mime = "image/gif" if ext == "gif" else "image/png"
            return (
                f'<figure style="margin:0 0 24px">'
                f'<img src="data:{mime};base64,{data}" alt="{alt}" '
                f'style="width:{width};border-radius:6px;box-shadow:0 2px 8px rgba(0,0,0,.15)">'
                f'<figcaption style="font-size:12px;color:#666;margin-top:4px">{alt}</figcaption>'
                f"</figure>"
            )

        rows_html = ""
        by_task: dict[str, list[dict[str, Any]]] = {}
        for r in self._results:
            t = str(r.get("task_name", "unknown"))
            by_task.setdefault(t, []).append(r)

        agg = aggregate_by_task(self._results)
        tasks = [t for t in ALL_TASKS if t in agg] or list(agg.keys())

        for task in tasks:
            m = agg[task]
            n = len(by_task.get(task, []))
            label = TASK_LABELS.get(task, task).replace("\n", " — ")
            colour = TASK_COLOURS.get(task, DEFAULT_COLOUR)
            rows_html += f"""
            <tr>
              <td style="border-left:4px solid {colour};padding-left:8px">{label}</td>
              <td>{n}</td>
              <td>{m['ep_return'][0]:.2f} ± {m['ep_return'][1]:.2f}</td>
              <td>{m['peak_infection'][0]:.3f} ± {m['peak_infection'][1]:.3f}</td>
              <td>{m['mean_economy'][0]:.3f} ± {m['mean_economy'][1]:.3f}</td>
              <td>{m['invalid_action_pct'][0]:.1f}% ± {m['invalid_action_pct'][1]:.1f}%</td>
            </tr>"""

        graph_gif = artifacts.get("graph_animation", "")
        graph_img_tag = _img_tag(graph_gif, "Epidemic graph animation") if graph_gif else ""
        graph_snap = artifacts.get("graph_snapshot", "")
        graph_snap_tag = _img_tag(graph_snap, "Graph snapshot (final step)") if graph_snap else ""

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>{self.title}</title>
  <style>
    body {{font-family:system-ui,sans-serif;background:#f8f9fa;color:#212529;margin:0;padding:24px}}
    h1 {{font-size:1.6rem;margin-bottom:4px}}
    h2 {{font-size:1.1rem;border-bottom:2px solid #dee2e6;padding-bottom:6px;margin-top:32px}}
    table {{width:100%;border-collapse:collapse;background:#fff;border-radius:8px;overflow:hidden;
            box-shadow:0 1px 4px rgba(0,0,0,.1)}}
    th {{background:#343a40;color:#fff;padding:10px 12px;text-align:left;font-size:13px}}
    td {{padding:8px 12px;border-bottom:1px solid #dee2e6;font-size:13px}}
    tr:last-child td {{border-bottom:none}}
    .grid {{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}}
    .card {{background:#fff;border-radius:8px;padding:16px;box-shadow:0 1px 4px rgba(0,0,0,.1)}}
    footer {{margin-top:40px;font-size:11px;color:#adb5bd;text-align:center}}
  </style>
</head>
<body>
  <h1>{self.title}</h1>
  <p style="color:#6c757d;font-size:13px">
    Generated from <code>{self.results_path or 'in-memory results'}</code>
    — {sum(len(v) for v in by_task.values())} total episodes across {len(tasks)} task(s).
  </p>

  <h2>Summary Table</h2>
  <table>
    <thead>
      <tr>
        <th>Task</th><th>N</th><th>Return (mean±std)</th>
        <th>Peak Infection</th><th>Economy Score</th><th>Invalid Actions</th>
      </tr>
    </thead>
    <tbody>{rows_html}</tbody>
  </table>

  <h2>Overview Charts</h2>
  <div class="card">
    {_img_tag(artifacts.get("summary_overview",""), "4-panel summary overview")}
  </div>

  <h2>Epidemic Graph</h2>
  <div class="grid">
    <div class="card">{graph_snap_tag or "<p><em>Run with --graph to generate graph visuals.</em></p>"}</div>
    <div class="card">{graph_img_tag}</div>
  </div>

  <h2>Detailed Charts</h2>
  <div class="grid">
    <div class="card">{_img_tag(artifacts.get("return_distribution",""), "Return distribution")}</div>
    <div class="card">{_img_tag(artifacts.get("returns",""), "Mean episode return")}</div>
    <div class="card">{_img_tag(artifacts.get("peak_infection",""), "Peak infection rate")}</div>
    <div class="card">{_img_tag(artifacts.get("economy",""), "Economy score")}</div>
    <div class="card">{_img_tag(artifacts.get("invalid_actions",""), "Invalid actions (%)")}</div>
    {"<div class='card'>" + _img_tag(artifacts.get("timeseries",""), "Time-series") + "</div>" if "timeseries" in artifacts else ""}
  </div>

  <footer>RL Epidemic Containment — Visualization Layer</footer>
</body>
</html>"""

        path = os.path.join(self.out_dir, "report.html")
        with open(path, "w") as f:
            f.write(html)
        return path
