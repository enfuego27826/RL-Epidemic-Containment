"""Pluggable encoder interface for RL policy networks.

This module defines the ``EncoderBase`` abstract base class that all observation
encoders must implement, along with a factory function ``build_encoder`` that
instantiates the correct encoder from a config dict.

Available encoders
------------------
``"mlp"``
    :class:`MLPEncoder` — simple multi-layer perceptron over the flat
    observation vector.  Baseline / ablation reference.

``"st"`` / ``"gnn_gru"``
    :class:`STEncoderWrapper` — wraps the Phase 3 ``STEncoder``
    (GCN spatial encoder + GRU temporal encoder + global readout).
    Requires only stdlib (no PyTorch Geometric).

Usage
-----
::

    from src.models.encoder_interface import build_encoder

    encoder = build_encoder(config, obs_dim=84)
    encoding = encoder.encode(obs_list)       # → list[float], length encoder.output_dim

Config switches
---------------
Set ``model.encoder_type`` in your YAML config to select the encoder:

    model:
      encoder_type: "mlp"    # default; fast, good baseline
      # encoder_type: "st"   # GNN+GRU; enable for Phase 3

Additional encoder-specific keys are read from the ``model`` section:
  - ``hidden_dims``: for MLP encoder layer widths
  - ``gcn_hidden_dim``, ``gru_hidden_dim``, ``global_context_dim``: for ST encoder

Ablation guide
--------------
To run ablation experiments comparing encoders, use the pre-built configs:

    # MLP baseline
    python scripts/train.py --config configs/ablation_mlp.yaml

    # ST-GNN + GRU
    python scripts/train.py --config configs/ablation_gnn_gru.yaml

Both configs are identical except for ``model.encoder_type``.
"""

from __future__ import annotations

import abc
import math
import random
from typing import Any


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class EncoderBase(abc.ABC):
    """Abstract base class for observation encoders.

    All encoders must:

    1. Accept a flat observation vector (``list[float]``) in ``encode()``.
    2. Expose ``output_dim: int`` so downstream actor-critic heads can
       size their input layers.
    3. Implement ``reset()`` to clear any episode-level recurrent state
       (no-op for stateless encoders).
    """

    @property
    @abc.abstractmethod
    def output_dim(self) -> int:
        """Dimension of the encoded representation vector."""

    @abc.abstractmethod
    def encode(self, obs: list[float], **kwargs: Any) -> list[float]:
        """Encode a flat observation vector into a fixed-size representation.

        Parameters
        ----------
        obs:
            Flat observation vector (length varies by environment config).
        **kwargs:
            Encoder-specific keyword arguments, e.g. ``num_active_nodes``
            for graph-based encoders.

        Returns
        -------
        list[float]
            Encoded representation of length ``self.output_dim``.
        """

    def reset(self) -> None:
        """Reset any recurrent/episodic state (default: no-op)."""


# ---------------------------------------------------------------------------
# MLP encoder (baseline)
# ---------------------------------------------------------------------------

class MLPEncoder(EncoderBase):
    """Pure-Python MLP encoder operating over the flat observation vector.

    This is the simplest possible encoder — no graph structure, no temporal
    memory.  It is fast, easy to understand, and serves as the ablation
    baseline for comparing against the ST-GNN encoder.

    Architecture: obs → Linear(hidden) → ReLU → ... → Linear(output_dim) → ReLU

    All weights are stored as plain Python lists so this class is importable
    without PyTorch.  For actual training, the actor-critic models use
    PyTorch-native layers (see :class:`src.models.actor_critic.ActorCritic`).
    This encoder is primarily used as the conceptual/config-level reference and
    for unit-testing the encoder interface.

    .. warning::
        The pure-Python matrix multiply in :meth:`encode` is O(n²) per layer
        and is **not intended for use in hot training paths**.  For training,
        the actor-critic uses PyTorch-native layers directly; this class serves
        as a stdlib-only reference implementation and configuration anchor.
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_dims: list[int] | None = None,
        output_dim: int = 64,
        seed: int = 0,
    ) -> None:
        self._obs_dim = obs_dim
        self._hidden_dims = list(hidden_dims or [128, 128])
        self._output_dim = output_dim

        # Build weight matrices (Xavier uniform init, plain Python)
        rng = random.Random(seed)
        dims = [obs_dim] + self._hidden_dims + [output_dim]
        self._weights: list[list[list[float]]] = []
        self._biases: list[list[float]] = []
        for in_d, out_d in zip(dims[:-1], dims[1:]):
            limit = math.sqrt(6.0 / (in_d + out_d))
            W = [[rng.uniform(-limit, limit) for _ in range(in_d)] for _ in range(out_d)]
            b = [0.0] * out_d
            self._weights.append(W)
            self._biases.append(b)

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def encode(self, obs: list[float], **kwargs: Any) -> list[float]:
        h = list(obs)
        # Pad/truncate to obs_dim
        if len(h) < self._obs_dim:
            h = h + [0.0] * (self._obs_dim - len(h))
        else:
            h = h[: self._obs_dim]

        for W, b in zip(self._weights, self._biases):
            new_h = [
                max(0.0, sum(W[j][i] * h[i] for i in range(len(h))) + b[j])
                for j in range(len(W))
            ]
            h = new_h
        return h


# ---------------------------------------------------------------------------
# ST-GNN + GRU encoder (Phase 3)
# ---------------------------------------------------------------------------

class STEncoderWrapper(EncoderBase):
    """Wraps :class:`src.models.st_encoder.STEncoder` as an ``EncoderBase``.

    This encoder applies:
    1. GCN-style spatial message passing over the city graph.
    2. Per-node GRU for temporal memory.
    3. Global mean-pool readout.

    It is stateful (GRU state persists across steps) — call ``reset()``
    at the start of each episode.
    """

    def __init__(
        self,
        node_feature_dim: int = 4,
        max_nodes: int = 20,
        gcn_hidden_dim: int = 32,
        gru_hidden_dim: int = 32,
        global_context_dim: int = 32,
        num_global_scalars: int = 4,
        seed: int = 0,
    ) -> None:
        from src.models.st_encoder import STEncoder
        self._enc = STEncoder(
            node_feature_dim=node_feature_dim,
            gcn_hidden_dim=gcn_hidden_dim,
            gru_hidden_dim=gru_hidden_dim,
            global_context_dim=global_context_dim,
            num_global_scalars=num_global_scalars,
            seed=seed,
        )
        self._max_nodes = max_nodes
        self._enc.reset(max_nodes)

    @property
    def output_dim(self) -> int:
        return self._enc.output_dim

    def encode(self, obs: list[float], **kwargs: Any) -> list[float]:
        num_active_nodes: int = kwargs.get("num_active_nodes", self._max_nodes)
        return self._enc.forward(obs, num_active_nodes=num_active_nodes)

    def reset(self) -> None:
        self._enc.reset(self._max_nodes)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_encoder(config: dict[str, Any], obs_dim: int) -> EncoderBase:
    """Instantiate an encoder from the ``model`` section of an experiment config.

    Parameters
    ----------
    config:
        Full experiment config dict.  Reads ``config["model"]["encoder_type"]``
        (default: ``"mlp"``).
    obs_dim:
        Dimensionality of the raw observation vector (used to size the MLP
        input layer).

    Returns
    -------
    EncoderBase
        An instantiated, ready-to-use encoder.

    Raises
    ------
    ValueError
        For unrecognised ``encoder_type`` values.
    """
    model_cfg = config.get("model", {})
    encoder_type = str(model_cfg.get("encoder_type", "mlp")).lower()
    seed = int(config.get("env", {}).get("seed", 0))
    max_nodes = int(config.get("env", {}).get("max_nodes", 20))

    if encoder_type == "mlp":
        hidden_dims = list(model_cfg.get("hidden_dims", [128, 128]))
        output_dim = int(model_cfg.get("encoder_output_dim", 64))
        return MLPEncoder(
            obs_dim=obs_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            seed=seed,
        )

    if encoder_type in ("st", "gnn_gru"):
        return STEncoderWrapper(
            node_feature_dim=4,
            max_nodes=max_nodes,
            gcn_hidden_dim=int(model_cfg.get("gcn_hidden_dim", 32)),
            gru_hidden_dim=int(model_cfg.get("gru_hidden_dim", 32)),
            global_context_dim=int(model_cfg.get("global_context_dim", 32)),
            num_global_scalars=4,
            seed=seed,
        )

    raise ValueError(
        f"Unknown encoder_type={encoder_type!r}. "
        f"Supported: 'mlp', 'st' (alias 'gnn_gru')."
    )
