"""Epidemic-graph visualizer.

Renders the city-travel-route network as a graph where each city node is
coloured by its current infection rate, with visual markers for quarantine
and vaccination events, and edges drawn between connected cities.

Can produce:
* A static PNG of the graph at a single timestep.
* An animated GIF/HTML walking through all steps of one episode.

Usage
-----
Run directly (quick demo against the easy task):

    python src/viz/epidemic_graph.py

Or import and use programmatically:

    from src.viz.epidemic_graph import GraphVisualizer
    from src.env.openenv_adapter import OpenEnvAdapter

    env = OpenEnvAdapter("easy_localized_outbreak", seed=42)
    viz = GraphVisualizer(env)

    # Collect a full episode
    frames = viz.collect_episode()

    # Save snapshot of last frame
    viz.save_snapshot(frames[-1], "out/graph_final.png")

    # Save animated walkthrough
    viz.save_animation(frames, "out/graph_episode.gif")
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path fix
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class NodeFrame:
    """State of one city at one timestep."""

    __slots__ = (
        "node_id", "infection_rate", "economic_health",
        "is_quarantined", "population_fraction",
        "vaccinated_this_step", "action_code",
    )

    def __init__(
        self,
        node_id: str,
        infection_rate: float,
        economic_health: float,
        is_quarantined: bool,
        population_fraction: float,
        vaccinated_this_step: bool = False,
        action_code: int = 0,
    ) -> None:
        self.node_id = node_id
        self.infection_rate = infection_rate
        self.economic_health = economic_health
        self.is_quarantined = is_quarantined
        self.population_fraction = population_fraction
        self.vaccinated_this_step = vaccinated_this_step
        self.action_code = action_code  # 0=noop,1=quar,2=lift,3=vax


class EpisodeFrame:
    """All city states + global metrics at one timestep."""

    def __init__(
        self,
        step: int,
        nodes: list[NodeFrame],
        global_infection: float,
        global_economy: float,
        vaccine_budget_frac: float,
        reward: float,
    ) -> None:
        self.step = step
        self.nodes = nodes
        self.global_infection = global_infection
        self.global_economy = global_economy
        self.vaccine_budget_frac = vaccine_budget_frac
        self.reward = reward


# ---------------------------------------------------------------------------
# Layout helpers (no networkx — pure Python)
# ---------------------------------------------------------------------------

def _circle_layout(n: int, radius: float = 1.0) -> list[tuple[float, float]]:
    """Arrange n nodes in a circle."""
    positions = []
    for i in range(n):
        angle = 2 * math.pi * i / max(n, 1)
        positions.append((radius * math.cos(angle), radius * math.sin(angle)))
    return positions


def _spring_layout(
    edges: list[tuple[int, int]],
    n: int,
    iterations: int = 80,
    seed: int = 0,
) -> list[tuple[float, float]]:
    """Fruchterman-Reingold spring layout (pure Python, no deps)."""
    import random
    rng = random.Random(seed)

    # Start from circle
    pos = list(_circle_layout(n))

    k = math.sqrt(1.0 / max(n, 1))
    temperature = 1.0
    cooling = temperature / (iterations + 1)

    for _ in range(iterations):
        disp = [(0.0, 0.0)] * n

        # Repulsive forces between all pairs
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[i][0] - pos[j][0]
                dy = pos[i][1] - pos[j][1]
                dist = math.sqrt(dx * dx + dy * dy) or 1e-6
                force = (k * k) / dist
                disp[i] = (disp[i][0] + dx / dist * force, disp[i][1] + dy / dist * force)
                disp[j] = (disp[j][0] - dx / dist * force, disp[j][1] - dy / dist * force)

        # Attractive forces along edges
        for (i, j) in edges:
            dx = pos[i][0] - pos[j][0]
            dy = pos[i][1] - pos[j][1]
            dist = math.sqrt(dx * dx + dy * dy) or 1e-6
            force = dist * dist / k
            disp[i] = (disp[i][0] - dx / dist * force, disp[i][1] - dy / dist * force)
            disp[j] = (disp[j][0] + dx / dist * force, disp[j][1] + dy / dist * force)

        # Update positions with clamped displacement
        for i in range(n):
            d = math.sqrt(disp[i][0] ** 2 + disp[i][1] ** 2) or 1e-6
            scale = min(d, temperature) / d
            x = pos[i][0] + disp[i][0] * scale
            y = pos[i][1] + disp[i][1] * scale
            # Clamp to unit square
            x = max(-1.0, min(1.0, x))
            y = max(-1.0, min(1.0, y))
            pos[i] = (x, y)

        temperature = max(temperature - cooling, 0.01)

    return pos


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

def _infection_colour(rate: float) -> tuple[float, float, float]:
    """Map infection rate [0,1] to an RGB tuple (green → yellow → red)."""
    rate = max(0.0, min(1.0, rate))
    if rate < 0.5:
        # green → yellow
        t = rate / 0.5
        return (t, 1.0, 0.0)
    else:
        # yellow → red
        t = (rate - 0.5) / 0.5
        return (1.0, 1.0 - t, 0.0)


# ---------------------------------------------------------------------------
# GraphVisualizer
# ---------------------------------------------------------------------------

class GraphVisualizer:
    """Collect and render epidemic graph animations.

    Parameters
    ----------
    env:
        An :class:`~src.env.openenv_adapter.OpenEnvAdapter` instance (already
        constructed but not necessarily reset).
    task_name:
        Task name — used to load the graph topology from ``TaskConfig``.
    """

    def __init__(self, env: Any) -> None:
        self._env = env
        task_name = getattr(env, "task_name", "easy_localized_outbreak")

        # Load graph topology
        from tasks import get_task_definition
        task_def = get_task_definition(task_name)
        cfg = task_def.config

        self._node_ids: list[str] = [n.node_id for n in cfg.nodes]
        self._edge_pairs: list[tuple[str, str]] = [(e.source, e.target) for e in cfg.edges]
        self._populations: dict[str, int] = {n.node_id: n.population for n in cfg.nodes}

        n = len(self._node_ids)
        node_idx = {nid: i for i, nid in enumerate(self._node_ids)}
        edge_idx = [(node_idx[s], node_idx[t]) for s, t in self._edge_pairs]
        self._positions: list[tuple[float, float]] = _spring_layout(edge_idx, n, seed=42)
        self._node_idx = node_idx

    # ------------------------------------------------------------------
    # Episode collection
    # ------------------------------------------------------------------

    def collect_episode(
        self,
        policy: Any = None,
        seed: int = 42,
        deterministic: bool = True,
        max_steps: int = 200,
    ) -> list[EpisodeFrame]:
        """Run one episode and collect per-step frames.

        Parameters
        ----------
        policy:
            Optional policy with an ``act()`` method.  If ``None``, random
            noop actions are used (zero vector).
        seed:
            Episode seed.
        deterministic:
            Passed to ``policy.act()`` if a policy is provided.
        max_steps:
            Hard cap on steps (safety guard).

        Returns
        -------
        frames:
            List of :class:`EpisodeFrame`, one per step (including step 0 =
            after reset).
        """
        from src.env.openenv_adapter import NODE_FEATURE_DIM, GLOBAL_FEATURE_DIM

        obs, info = self._env.reset(seed=seed)
        frames: list[EpisodeFrame] = []
        num_nodes = info["num_nodes"]
        initial_budget = float(self._env._initial_vaccine_budget)

        # Capture step-0 state
        frames.append(self._make_frame(obs, 0, 0.0, [0] * num_nodes, initial_budget))

        done = False
        step = 0
        while not done and step < max_steps:
            vax_budget = self._env.vaccine_budget

            # Choose action
            if policy is None:
                action = [0] * self._env.max_nodes
            else:
                from src.models.actor_critic import HybridActorCritic, ActorCritic
                from src.models.st_encoder import STActorCritic
                if isinstance(policy, (HybridActorCritic,)):
                    action, _, _, _ = policy.act(obs, deterministic=deterministic, vaccine_budget=vax_budget)
                elif isinstance(policy, STActorCritic):
                    action, _, _, _ = policy.act(obs, deterministic=deterministic, vaccine_budget=vax_budget)
                else:
                    action, _, _ = policy.act(obs, deterministic=deterministic)

            # Extract per-node discrete action codes for visualization
            if isinstance(action, dict):
                discrete = list(action.get("discrete", [0] * num_nodes))
            else:
                discrete = list(action)[:num_nodes]

            obs, reward, done, step_info = self._env.step(action)
            step += 1

            frames.append(self._make_frame(obs, step, reward, discrete, initial_budget))

        return frames

    def _make_frame(
        self,
        obs: list[float],
        step: int,
        reward: float,
        action_codes: list[int],
        initial_budget: float,
    ) -> EpisodeFrame:
        """Convert a raw observation tensor into an :class:`EpisodeFrame`."""
        from src.env.openenv_adapter import NODE_FEATURE_DIM, GLOBAL_FEATURE_DIM

        num_active = len(self._node_ids)
        node_frames: list[NodeFrame] = []

        for i, nid in enumerate(self._node_ids):
            base = i * NODE_FEATURE_DIM
            inf_rate = float(obs[base])
            econ = float(obs[base + 1])
            is_q = bool(obs[base + 2] > 0.5)
            pop_frac = float(obs[base + 3])
            act = int(action_codes[i]) if i < len(action_codes) else 0
            node_frames.append(NodeFrame(
                node_id=nid,
                infection_rate=inf_rate,
                economic_health=econ,
                is_quarantined=is_q,
                population_fraction=pop_frac,
                vaccinated_this_step=(act == 3),
                action_code=act,
            ))

        # Global scalars (last 4 elements of obs)
        global_budget_frac = float(obs[-4]) if len(obs) >= 4 else 1.0
        global_econ = float(obs[-3]) if len(obs) >= 3 else 1.0
        global_infection = float(obs[-2]) if len(obs) >= 2 else 0.0

        return EpisodeFrame(
            step=step,
            nodes=node_frames,
            global_infection=global_infection,
            global_economy=global_econ,
            vaccine_budget_frac=global_budget_frac,
            reward=reward,
        )

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def save_snapshot(
        self,
        frame: EpisodeFrame,
        path: str,
        title: str | None = None,
        dpi: int = 120,
    ) -> None:
        """Render a single timestep to a PNG file.

        Parameters
        ----------
        frame:
            The :class:`EpisodeFrame` to render.
        path:
            Output file path (should end in ``.png``).
        title:
            Optional plot title.
        dpi:
            Resolution of the saved PNG.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        fig = self._draw_frame(frame, title=title)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    def save_animation(
        self,
        frames: list[EpisodeFrame],
        path: str,
        fps: int = 4,
        dpi: int = 100,
    ) -> None:
        """Render a full episode as an animated GIF.

        Parameters
        ----------
        frames:
            List of :class:`EpisodeFrame` objects (from :meth:`collect_episode`).
        path:
            Output file path (should end in ``.gif``).
        fps:
            Frames per second for the animation.
        dpi:
            Resolution.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        # Render first frame to set figure
        fig = self._draw_frame(frames[0], title=None)

        # We'll re-draw into the same figure each tick
        def _animate(frame_idx: int) -> None:
            fig.clf()
            self._draw_frame(frames[frame_idx], title=None, fig=fig)

        ani = animation.FuncAnimation(
            fig,
            _animate,
            frames=len(frames),
            interval=int(1000 / fps),
            repeat=True,
        )
        ani.save(path, writer="pillow", dpi=dpi)
        plt.close(fig)

    # ------------------------------------------------------------------
    # Internal drawing
    # ------------------------------------------------------------------

    def _draw_frame(
        self,
        frame: EpisodeFrame,
        title: str | None = None,
        fig: Any = None,
    ) -> Any:
        """Draw one episode frame onto a matplotlib figure.

        Returns the figure object.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable

        if fig is None:
            fig = plt.figure(figsize=(10, 7))

        # ---- Layout: graph on left, mini time-bar on right ----
        ax_graph = fig.add_axes([0.0, 0.15, 0.72, 0.80])
        ax_cb = fig.add_axes([0.74, 0.15, 0.02, 0.80])   # colorbar
        ax_info = fig.add_axes([0.78, 0.15, 0.20, 0.80])  # info panel
        ax_footer = fig.add_axes([0.0, 0.00, 1.0, 0.12])  # footer bar

        ax_graph.set_aspect("equal")
        ax_graph.axis("off")
        ax_info.axis("off")
        ax_footer.axis("off")

        # ---- Draw edges ----
        node_map = {nf.node_id: nf for nf in frame.nodes}
        for (src, tgt) in self._edge_pairs:
            si = self._node_idx.get(src)
            ti = self._node_idx.get(tgt)
            if si is None or ti is None:
                continue
            px, py = self._positions[si]
            qx, qy = self._positions[ti]
            # Colour edge by average infection of the two endpoints
            avg_inf = 0.5 * (
                node_map.get(src, NodeFrame(src, 0, 1, False, 0)).infection_rate
                + node_map.get(tgt, NodeFrame(tgt, 0, 1, False, 0)).infection_rate
            )
            ec = _infection_colour(avg_inf)
            ax_graph.plot([px, qx], [py, qy], color=ec, linewidth=1.5, alpha=0.5, zorder=1)

        # ---- Draw nodes ----
        xs = [self._positions[self._node_idx[nf.node_id]][0] for nf in frame.nodes]
        ys = [self._positions[self._node_idx[nf.node_id]][1] for nf in frame.nodes]
        colors = [_infection_colour(nf.infection_rate) for nf in frame.nodes]

        # Node size proportional to population
        max_pop = max((nf.population_fraction for nf in frame.nodes), default=1e-6)
        node_sizes = [
            300 + 700 * (nf.population_fraction / max(max_pop, 1e-6))
            for nf in frame.nodes
        ]

        scatter = ax_graph.scatter(
            xs, ys,
            s=node_sizes,
            c=colors,
            edgecolors="none",
            zorder=2,
            clip_on=False,
        )

        # ---- Overlay markers for quarantine / vaccination ----
        for nf in frame.nodes:
            idx = self._node_idx[nf.node_id]
            px, py = self._positions[idx]
            if nf.is_quarantined:
                ax_graph.plot(
                    px, py, marker="s", markersize=18,
                    markerfacecolor="none", markeredgecolor="#c0392b",
                    markeredgewidth=2.0, zorder=3,
                )
            if nf.vaccinated_this_step:
                ax_graph.plot(
                    px, py, marker="*", markersize=14,
                    color="#27ae60", zorder=4,
                )

        # ---- Node labels ----
        for nf in frame.nodes:
            idx = self._node_idx[nf.node_id]
            px, py = self._positions[idx]
            label = nf.node_id.replace("city_", "C")
            ax_graph.text(
                px, py - 0.12, label,
                ha="center", va="top", fontsize=7,
                color="#2c3e50", fontweight="bold", zorder=5,
            )

        # ---- Infection colorbar ----
        norm = Normalize(vmin=0.0, vmax=1.0)
        sm = ScalarMappable(cmap=None, norm=norm)
        sm.set_array([])
        # Manual colour bar ticks
        ax_cb.imshow(
            [[[*_infection_colour(v / 100)] for v in range(101)]],
            aspect="auto",
            extent=[0, 1, 0, 1],
            origin="lower",
        )
        ax_cb.set_xticks([])
        ax_cb.set_yticks([0, 0.5, 1.0])
        ax_cb.set_yticklabels(["0%", "50%", "100%"], fontsize=7)
        ax_cb.yaxis.set_label_position("right")
        ax_cb.yaxis.tick_right()
        ax_cb.set_title("Inf.", fontsize=7, pad=2)

        # ---- Info panel ----
        lines = [
            ("Step", str(frame.step)),
            ("Global inf.", f"{frame.global_infection:.1%}"),
            ("Economy", f"{frame.global_economy:.1%}"),
            ("Vax budget", f"{frame.vaccine_budget_frac:.1%}"),
            ("Reward", f"{frame.reward:+.3f}"),
        ]
        ax_info.set_xlim(0, 1)
        ax_info.set_ylim(0, 1)
        y = 0.95
        for key, val in lines:
            ax_info.text(0.05, y, key, fontsize=8, color="#555", va="top")
            ax_info.text(0.98, y, val, fontsize=8, color="#222", va="top", ha="right", fontweight="bold")
            y -= 0.14

        # ---- Legend ----
        legend_patches = [
            mpatches.Patch(facecolor=_infection_colour(0.0), label="Low infection"),
            mpatches.Patch(facecolor=_infection_colour(0.5), label="Medium infection"),
            mpatches.Patch(facecolor=_infection_colour(1.0), label="High infection"),
            mpatches.Patch(facecolor="none", edgecolor="#c0392b", linewidth=1.5, label="Quarantined"),
            mpatches.Patch(facecolor="#27ae60", label="Vaccinated (this step)"),
        ]
        ax_graph.legend(
            handles=legend_patches,
            loc="lower left",
            fontsize=7,
            framealpha=0.85,
            ncol=2,
        )

        # ---- Footer ----
        plot_title = title or "Epidemic Containment — City Network"
        ax_footer.text(
            0.5, 0.7, plot_title,
            ha="center", va="center", fontsize=10, fontweight="bold",
            color="#2c3e50",
        )
        ax_footer.text(
            0.5, 0.2,
            "Node colour = infection rate  |  Red border = quarantine  |  ★ = vaccinated this step",
            ha="center", va="center", fontsize=7, color="#7f8c8d",
        )

        return fig
