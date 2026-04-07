"""Phase 3 — Spatiotemporal Graph Encoder (stdlib-only implementation).

Architecture overview
---------------------
The encoder processes node features over a graph through two stages:

1. **Graph Message Passing (spatial)**
   A simplified 1-layer graph convolution (GCN-style) that aggregates
   neighbour information using the adjacency structure.  Without PyTorch
   Geometric, we implement message passing as plain matrix operations.

   For node i: h_i = ReLU(W_self * x_i + W_neigh * mean(x_j for j in N(i)) + b)

2. **Temporal encoding (GRU-style)**
   A minimal GRU cell operating over the last K node-embedding snapshots.
   This gives temporal context without requiring PyTorch.

   GRU update equations (per node, per hidden unit):
     z = sigmoid(Wz * [h_prev; x_t] + bz)   (update gate)
     r = sigmoid(Wr * [h_prev; x_t] + br)   (reset gate)
     n = tanh(Wn * [r*h_prev; x_t] + bn)    (new gate)
     h_t = (1-z)*h_prev + z*n

3. **Global readout**
   Mean-pool over node embeddings to produce a global context vector.

4. **State assembly**
   [global_embedding || key_global_stats]
   → passed to actor-critic heads.

Variable graph size is handled via masking: padded node slots are zeroed
before message passing and excluded from the global mean-pool.

All weights are initialised with Xavier uniform and stored as plain Python
lists of lists (no external deps).
"""

from __future__ import annotations

import math
import random
from typing import Sequence


# ---------------------------------------------------------------------------
# Low-level math helpers
# ---------------------------------------------------------------------------

def _relu(x: float) -> float:
    return max(0.0, x)


def _sigmoid(x: float) -> float:
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


def _tanh(x: float) -> float:
    return math.tanh(x)


def _xavier_matrix(in_d: int, out_d: int, rng: random.Random) -> list[list[float]]:
    """Xavier uniform initialisation."""
    limit = math.sqrt(6.0 / (in_d + out_d))
    return [[rng.uniform(-limit, limit) for _ in range(in_d)] for _ in range(out_d)]


def _zero_vec(n: int) -> list[float]:
    return [0.0] * n


def _matvec(W: list[list[float]], x: list[float]) -> list[float]:
    """y = W @ x  (out_d × in_d matrix times in_d vector)."""
    return [sum(W[j][i] * x[i] for i in range(len(x))) for j in range(len(W))]


def _addvec(a: list[float], b: list[float]) -> list[float]:
    return [ai + bi for ai, bi in zip(a, b)]


def _scalevec(s: float, v: list[float]) -> list[float]:
    return [s * vi for vi in v]


def _mulvec(a: list[float], b: list[float]) -> list[float]:
    return [ai * bi for ai, bi in zip(a, b)]


# ---------------------------------------------------------------------------
# GRU cell
# ---------------------------------------------------------------------------

class GRUCell:
    """Single-step GRU cell (stdlib-only).

    Parameters
    ----------
    input_dim:
        Dimension of the input vector x_t.
    hidden_dim:
        Dimension of the hidden state h_t.
    seed:
        RNG seed for weight initialisation.
    """

    def __init__(self, input_dim: int, hidden_dim: int, seed: int = 0) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        rng = random.Random(seed)
        in_d = input_dim + hidden_dim
        # Update gate
        self.Wz = _xavier_matrix(in_d, hidden_dim, rng)
        self.bz = _zero_vec(hidden_dim)
        # Reset gate
        self.Wr = _xavier_matrix(in_d, hidden_dim, rng)
        self.br = _zero_vec(hidden_dim)
        # New gate (uses concatenated [r*h; x])
        self.Wn = _xavier_matrix(in_d, hidden_dim, rng)
        self.bn = _zero_vec(hidden_dim)

    def forward(self, x: list[float], h_prev: list[float]) -> list[float]:
        """Compute h_t given x_t and h_{t-1}.

        Parameters
        ----------
        x:
            Input vector of length input_dim.
        h_prev:
            Previous hidden state of length hidden_dim.

        Returns
        -------
        h_t:
            New hidden state of length hidden_dim.
        """
        xh = x + h_prev  # concatenate [x; h_prev]
        z = [_sigmoid(v) for v in _addvec(_matvec(self.Wz, xh), self.bz)]
        r = [_sigmoid(v) for v in _addvec(_matvec(self.Wr, xh), self.br)]
        rh = _mulvec(r, h_prev)
        n_input = x + rh  # concatenate [x; r*h_prev]
        n = [_tanh(v) for v in _addvec(_matvec(self.Wn, n_input), self.bn)]
        # h_t = (1 - z) * h_prev + z * n
        h_t = [((1.0 - zi) * hi + zi * ni) for zi, hi, ni in zip(z, h_prev, n)]
        return h_t


# ---------------------------------------------------------------------------
# Graph convolution layer
# ---------------------------------------------------------------------------

class GraphConvLayer:
    """One-layer graph convolution (mean-neighbour aggregation).

    For each node i with feature x_i and neighbour set N(i):

        h_i = ReLU(W_self @ x_i + W_neigh @ mean_j(x_j) + b)

    Padded nodes (is_active[i] == False) contribute x_i = 0 and are
    excluded from neighbour means.

    Parameters
    ----------
    in_dim:
        Input feature dimension per node.
    out_dim:
        Output embedding dimension per node.
    seed:
        RNG seed.
    """

    def __init__(self, in_dim: int, out_dim: int, seed: int = 0) -> None:
        rng = random.Random(seed)
        self.W_self = _xavier_matrix(in_dim, out_dim, rng)
        self.W_neigh = _xavier_matrix(in_dim, out_dim, rng)
        self.b = _zero_vec(out_dim)
        self.out_dim = out_dim
        self.in_dim = in_dim

    def forward(
        self,
        node_features: list[list[float]],
        adj: list[list[float]],
        is_active: list[bool],
    ) -> list[list[float]]:
        """Forward pass over the graph.

        Parameters
        ----------
        node_features:
            Shape (num_nodes, in_dim).  Padded nodes should have all-zero features.
        adj:
            Adjacency matrix (num_nodes × num_nodes) with float weights.
            Self-loops are NOT expected.
        is_active:
            Boolean list of length num_nodes; True = real node.

        Returns
        -------
        embeddings:
            Shape (num_nodes, out_dim).  Padded nodes return zero vectors.
        """
        num_nodes = len(node_features)
        embeddings: list[list[float]] = []

        for i in range(num_nodes):
            if not is_active[i]:
                embeddings.append(_zero_vec(self.out_dim))
                continue

            # Self contribution
            self_part = _matvec(self.W_self, node_features[i])

            # Neighbour mean
            neigh_sum = _zero_vec(self.in_dim)
            neigh_count = 0
            for j in range(num_nodes):
                if is_active[j] and adj[i][j] > 0.0:
                    neigh_sum = _addvec(neigh_sum, _scalevec(adj[i][j], node_features[j]))
                    neigh_count += 1

            if neigh_count > 0:
                neigh_mean = _scalevec(1.0 / neigh_count, neigh_sum)
            else:
                neigh_mean = _zero_vec(self.in_dim)

            neigh_part = _matvec(self.W_neigh, neigh_mean)
            h = [_relu(sv + nv + bv) for sv, nv, bv in zip(self_part, neigh_part, self.b)]
            embeddings.append(h)

        return embeddings


# ---------------------------------------------------------------------------
# Spatiotemporal Encoder
# ---------------------------------------------------------------------------

class STEncoder:
    """Spatiotemporal graph encoder combining graph conv + GRU over time.

    The encoder maintains a hidden state per node (GRU) across timesteps,
    enabling temporal context under partial observability.

    Parameters
    ----------
    node_feature_dim:
        Raw feature dimension per node (from OpenEnvAdapter).
    gcn_hidden_dim:
        Output dimension of the graph convolution layer.
    gru_hidden_dim:
        Hidden state dimension of the GRU cell (per node).
    global_context_dim:
        Dimension of the global readout (mean pool of GRU states).
        If 0, global readout is disabled and the encoder output is just
        the concatenated node hidden states.
    num_global_scalars:
        Number of global scalars appended to observations (default 4).
    seed:
        Weight init seed.
    """

    def __init__(
        self,
        node_feature_dim: int = 4,
        gcn_hidden_dim: int = 32,
        gru_hidden_dim: int = 32,
        global_context_dim: int = 32,
        num_global_scalars: int = 4,
        seed: int = 0,
    ) -> None:
        self.node_feature_dim = node_feature_dim
        self.gcn_hidden_dim = gcn_hidden_dim
        self.gru_hidden_dim = gru_hidden_dim
        self.global_context_dim = global_context_dim
        self.num_global_scalars = num_global_scalars

        # Graph convolution
        self._gcn = GraphConvLayer(node_feature_dim, gcn_hidden_dim, seed=seed)

        # GRU cell operating on per-node GCN output
        self._gru = GRUCell(gcn_hidden_dim, gru_hidden_dim, seed=seed + 1)

        # Global projection (mean pool → dense)
        rng = random.Random(seed + 2)
        self._W_global = _xavier_matrix(gru_hidden_dim, global_context_dim, rng)
        self._b_global = _zero_vec(global_context_dim)

        # Per-node GRU hidden states (initialised at reset)
        self._h_nodes: list[list[float]] = []
        self._max_nodes: int = 0

        # Encoder output dimension
        self.output_dim = global_context_dim + num_global_scalars

    def reset(self, max_nodes: int) -> None:
        """Reset GRU hidden states for a new episode.

        Parameters
        ----------
        max_nodes:
            Maximum number of node slots (including padding).
        """
        self._max_nodes = max_nodes
        self._h_nodes = [_zero_vec(self.gru_hidden_dim) for _ in range(max_nodes)]

    def forward(
        self,
        obs_tensor: list[float],
        adj: list[list[float]] | None = None,
        num_active_nodes: int | None = None,
    ) -> list[float]:
        """Encode a single observation step.

        Parameters
        ----------
        obs_tensor:
            Flat observation of length
            ``max_nodes * node_feature_dim + num_global_scalars``.
        adj:
            Adjacency matrix (max_nodes × max_nodes).  If None, a
            fully-connected uniform graph is assumed (all weights = 1/(N-1)).
        num_active_nodes:
            Number of real (non-padding) nodes.  Defaults to max_nodes.

        Returns
        -------
        encoding:
            Vector of length ``output_dim``.
        """
        max_nodes = self._max_nodes
        if max_nodes == 0:
            # Not yet reset
            return _zero_vec(self.output_dim)

        num_active = num_active_nodes if num_active_nodes is not None else max_nodes
        nfd = self.node_feature_dim
        gs = self.num_global_scalars

        # Extract node features from flat obs
        node_features: list[list[float]] = []
        for i in range(max_nodes):
            start = i * nfd
            node_features.append(list(obs_tensor[start : start + nfd]))

        # Extract global scalars (last num_global_scalars entries)
        global_scalars = list(obs_tensor[max_nodes * nfd : max_nodes * nfd + gs])

        # Active flags
        is_active = [i < num_active for i in range(max_nodes)]

        # Build default adjacency if not provided
        if adj is None:
            adj = _build_default_adj(max_nodes, num_active)

        # Graph convolution
        gcn_out = self._gcn.forward(node_features, adj, is_active)

        # GRU step per node
        new_h_nodes = []
        for i in range(max_nodes):
            if is_active[i]:
                h_new = self._gru.forward(gcn_out[i], self._h_nodes[i])
            else:
                h_new = _zero_vec(self.gru_hidden_dim)
            new_h_nodes.append(h_new)
        self._h_nodes = new_h_nodes

        # Global readout: mean pool over active node GRU states
        active_states = [self._h_nodes[i] for i in range(num_active)]
        if active_states:
            mean_state = _scalevec(1.0 / len(active_states), _sum_vecs(active_states))
        else:
            mean_state = _zero_vec(self.gru_hidden_dim)

        # Project to global_context_dim
        global_embed = [
            _relu(v)
            for v in _addvec(_matvec(self._W_global, mean_state), self._b_global)
        ]

        # Concatenate with global scalars
        encoding = global_embed + global_scalars
        assert len(encoding) == self.output_dim
        return encoding


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_default_adj(max_nodes: int, num_active: int) -> list[list[float]]:
    """Build a fully-connected (uniform weight) adjacency matrix.

    Active nodes are all connected to each other (weight 1.0).
    Padding nodes have no edges.
    """
    adj = [[0.0] * max_nodes for _ in range(max_nodes)]
    for i in range(num_active):
        for j in range(num_active):
            if i != j:
                adj[i][j] = 1.0
    return adj


def _sum_vecs(vecs: list[list[float]]) -> list[float]:
    """Element-wise sum of a list of vectors (all same length)."""
    if not vecs:
        return []
    result = list(vecs[0])
    for v in vecs[1:]:
        result = _addvec(result, v)
    return result


# ---------------------------------------------------------------------------
# STActorCritic — full model combining ST encoder + hybrid heads
# ---------------------------------------------------------------------------

class STActorCritic:
    """Phase 3 actor-critic with spatiotemporal graph encoder.

    Replaces the MLP trunk of ``HybridActorCritic`` with an ``STEncoder``
    that performs graph convolution followed by a GRU temporal update,
    producing a compact global embedding used as input to the actor/critic heads.

    Parameters
    ----------
    node_feature_dim:
        Features per node in the observation vector.
    max_nodes:
        Maximum number of node slots.
    action_dim:
        Number of discrete actions per node.
    gcn_hidden_dim:
        GCN output dimension.
    gru_hidden_dim:
        GRU hidden state dimension.
    global_context_dim:
        Global readout embedding dimension.
    num_global_scalars:
        Number of global scalars in observation (default 4).
    actor_hidden_dim:
        Hidden layer width in the actor/critic MLP heads.
    seed:
        Weight init seed.
    """

    def __init__(
        self,
        node_feature_dim: int = 4,
        max_nodes: int = 20,
        action_dim: int = 4,
        gcn_hidden_dim: int = 32,
        gru_hidden_dim: int = 32,
        global_context_dim: int = 32,
        num_global_scalars: int = 4,
        actor_hidden_dim: int = 64,
        seed: int = 0,
    ) -> None:
        self.node_feature_dim = node_feature_dim
        self.max_nodes = max_nodes
        self.action_dim = action_dim

        # ST encoder
        self._encoder = STEncoder(
            node_feature_dim=node_feature_dim,
            gcn_hidden_dim=gcn_hidden_dim,
            gru_hidden_dim=gru_hidden_dim,
            global_context_dim=global_context_dim,
            num_global_scalars=num_global_scalars,
            seed=seed,
        )
        enc_out_dim = self._encoder.output_dim

        rng = random.Random(seed + 10)

        # Actor: discrete head (max_nodes * action_dim logits)
        disc_out = max_nodes * action_dim
        self._W_actor_h = _xavier_matrix(enc_out_dim, actor_hidden_dim, rng)
        self._b_actor_h = _zero_vec(actor_hidden_dim)
        self._W_disc = _xavier_matrix(actor_hidden_dim, disc_out, rng)
        self._b_disc = _zero_vec(disc_out)

        # Actor: continuous head (max_nodes logits)
        self._W_cont = _xavier_matrix(actor_hidden_dim, max_nodes, rng)
        self._b_cont = _zero_vec(max_nodes)

        # Critic: scalar value
        self._W_critic_h = _xavier_matrix(enc_out_dim, actor_hidden_dim, rng)
        self._b_critic_h = _zero_vec(actor_hidden_dim)
        self._W_value = _xavier_matrix(actor_hidden_dim, 1, rng)
        self._b_value = _zero_vec(1)

    def reset_episode(self) -> None:
        """Reset GRU hidden states for a new episode."""
        self._encoder.reset(self.max_nodes)

    def forward(
        self,
        obs: list[float],
        adj: list[list[float]] | None = None,
        num_active_nodes: int | None = None,
    ) -> tuple[list[list[float]], list[float], float]:
        """Encode obs and produce per-node logits and value.

        Returns
        -------
        discrete_logits:
            Shape (max_nodes, action_dim).
        continuous_logits:
            Shape (max_nodes,).
        value:
            Scalar critic estimate.
        """
        encoding = self._encoder.forward(obs, adj=adj, num_active_nodes=num_active_nodes)

        # Actor hidden layer (shared for both heads)
        actor_h = [_relu(v) for v in _addvec(_matvec(self._W_actor_h, encoding), self._b_actor_h)]

        # Discrete logits
        flat_disc = _addvec(_matvec(self._W_disc, actor_h), self._b_disc)
        discrete_logits = [
            flat_disc[i * self.action_dim : (i + 1) * self.action_dim]
            for i in range(self.max_nodes)
        ]

        # Continuous logits
        cont_logits = _addvec(_matvec(self._W_cont, actor_h), self._b_cont)

        # Critic value
        critic_h = [_relu(v) for v in _addvec(_matvec(self._W_critic_h, encoding), self._b_critic_h)]
        value = _addvec(_matvec(self._W_value, critic_h), self._b_value)[0]

        return discrete_logits, cont_logits, value

    def act(
        self,
        obs: list[float],
        adj: list[list[float]] | None = None,
        num_active_nodes: int | None = None,
        mask: list[list[bool]] | None = None,
        vaccine_budget: float = 1.0,
        deterministic: bool = False,
        seed: int | None = None,
    ) -> tuple[dict[str, list], float, float, float]:
        """Sample a hybrid action.

        Returns
        -------
        action_dict:
            ``{"discrete": list[int], "continuous": list[float]}``.
        log_prob_discrete:
            Scalar.
        log_prob_continuous:
            Scalar.
        value:
            Critic estimate.
        """
        from src.models.hybrid_action import HybridActionDist

        disc_logits, cont_logits, value = self.forward(obs, adj=adj, num_active_nodes=num_active_nodes)
        dist = HybridActionDist(
            discrete_logits=disc_logits,
            continuous_logits=cont_logits,
            mask=mask,
            vaccine_budget=vaccine_budget,
            seed=seed,
        )
        sample = dist.mode() if deterministic else dist.sample()
        return (
            {"discrete": sample.discrete, "continuous": sample.continuous},
            sample.log_prob_discrete,
            sample.log_prob_continuous,
            value,
        )
