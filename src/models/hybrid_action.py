"""Hybrid action distribution for the PAMDP epidemic-containment agent.

Architecture
------------
The policy produces two action components per step:

Discrete head (per node):
    Categorical distribution over {no-op, quarantine, lift, vaccinate-candidate}.
    Invalid actions are zeroed out **before** sampling via an externally provided
    boolean mask of shape (max_nodes, NUM_DISCRETE_ACTIONS).

Continuous head (per node):
    Raw logits are passed through softplus to obtain non-negative allocations,
    then projected so that sum(allocations) <= vaccine_budget.

Combined log-probability:
    log π(a | s) = Σ_i log_prob_discrete(d_i) + Σ_i log_prob_continuous(v_i)

This module is pure Python (stdlib only) to match the Phase 1 scaffold.
A PyTorch port is planned for Phase 3.

TODO (Phase 3): replace MLP sub-calls with ST-GNN encoder outputs.
TODO (Phase 3): port to torch.distributions.Categorical + LogNormal.
"""

from __future__ import annotations

import math
import random
from typing import NamedTuple

# ---------------------------------------------------------------------------
# Discrete action codes (mirror of openenv_adapter constants)
# ---------------------------------------------------------------------------
ACTION_NOOP: int = 0
ACTION_QUARANTINE: int = 1
ACTION_LIFT: int = 2
ACTION_VACCINATE: int = 3
NUM_DISCRETE_ACTIONS: int = 4

# Minimum log-probability clamp to avoid extreme values in the continuous head
_MIN_LOG_PROB: float = -1e6


# ---------------------------------------------------------------------------
# Low-level math helpers (stdlib only)
# ---------------------------------------------------------------------------

def _softmax(logits: list[float]) -> list[float]:
    """Numerically stable softmax over a 1-D list."""
    max_l = max(logits)
    exps = [math.exp(l - max_l) for l in logits]
    total = sum(exps)
    return [e / total for e in exps]


def _masked_softmax(logits: list[float], mask: list[bool]) -> list[float]:
    """Softmax with boolean mask; masked positions get probability 0.

    Parameters
    ----------
    logits:
        Raw logits of length NUM_DISCRETE_ACTIONS.
    mask:
        Boolean list; True = action is valid.

    Returns
    -------
    Probability list of same length; masked entries are exactly 0.0.
    """
    NEG_INF = -1e30
    masked = [l if m else NEG_INF for l, m in zip(logits, mask)]
    probs = _softmax(masked)
    # Zero out any residual probability on masked slots due to numerical noise
    return [p if m else 0.0 for p, m in zip(probs, mask)]


def _softplus(x: float) -> float:
    """Numerically stable softplus: log(1 + exp(x))."""
    if x >= 20.0:
        return x
    return math.log1p(math.exp(x))


def _categorical_log_prob(probs: list[float], action: int) -> float:
    """Log-probability of ``action`` under the categorical distribution."""
    return math.log(max(probs[action], 1e-8))


def _categorical_entropy(probs: list[float]) -> float:
    """Shannon entropy H = -Σ p * log(p)."""
    return -sum(p * math.log(max(p, 1e-8)) for p in probs if p > 0.0)


def _sample_categorical(probs: list[float], rng: random.Random) -> int:
    """Draw a sample from a categorical distribution using inverse CDF."""
    r = rng.random()
    cumsum = 0.0
    for idx, p in enumerate(probs):
        cumsum += p
        if r < cumsum:
            return idx
    return len(probs) - 1


# ---------------------------------------------------------------------------
# HybridActionSample — lightweight result container
# ---------------------------------------------------------------------------

class HybridActionSample(NamedTuple):
    """Output of ``HybridActionDist.sample()``.

    Attributes
    ----------
    discrete:
        Per-node discrete action, list[int] of length ``num_nodes``.
    continuous:
        Per-node vaccine allocation, list[float] of length ``num_nodes``,
        non-negative and summing to at most ``vaccine_budget``.
    log_prob_discrete:
        Scalar log-probability for the discrete component (sum over nodes).
    log_prob_continuous:
        Scalar log-probability for the continuous component (sum over nodes).
    log_prob:
        Combined scalar = log_prob_discrete + log_prob_continuous.
    """

    discrete: list[int]
    continuous: list[float]
    log_prob_discrete: float
    log_prob_continuous: float
    log_prob: float


# ---------------------------------------------------------------------------
# HybridActionDist — main class
# ---------------------------------------------------------------------------

class HybridActionDist:
    """Mixed discrete/continuous action distribution for the PAMDP agent.

    Parameters
    ----------
    discrete_logits:
        Shape (num_nodes, NUM_DISCRETE_ACTIONS).  Raw (unmasked) logits from
        the actor head.
    continuous_logits:
        Shape (num_nodes,).  Raw logits for the continuous allocation head.
        Passed through softplus to produce non-negative raw allocations.
    mask:
        Shape (num_nodes, NUM_DISCRETE_ACTIONS).  Boolean validity mask;
        True = action is available.  If None, all actions are valid.
    vaccine_budget:
        Current available vaccine budget.  Continuous allocations are
        projected to sum to at most this value.
    seed:
        Optional RNG seed for reproducible sampling.
    """

    def __init__(
        self,
        discrete_logits: list[list[float]],
        continuous_logits: list[float],
        mask: list[list[bool]] | None = None,
        vaccine_budget: float = 1.0,
        seed: int | None = None,
    ) -> None:
        self.num_nodes = len(discrete_logits)
        self.vaccine_budget = max(vaccine_budget, 0.0)

        # Validate input shapes
        assert len(continuous_logits) == self.num_nodes, (
            f"continuous_logits length {len(continuous_logits)} != num_nodes {self.num_nodes}"
        )

        # Build default mask (all valid) if not provided
        if mask is None:
            mask = [[True] * NUM_DISCRETE_ACTIONS for _ in range(self.num_nodes)]
        assert len(mask) == self.num_nodes

        # Ensure no-op (action 0) is always valid to avoid degenerate distributions
        for node_mask in mask:
            node_mask[ACTION_NOOP] = True

        self._discrete_logits = discrete_logits
        self._continuous_logits = continuous_logits
        self._mask = mask
        self._rng = random.Random(seed)

        # Pre-compute masked probabilities for each node
        self._probs: list[list[float]] = [
            _masked_softmax(logits, node_mask)
            for logits, node_mask in zip(discrete_logits, mask)
        ]

        # Pre-compute continuous raw allocations and projection
        self._raw_alloc: list[float] = [
            _softplus(z) for z in continuous_logits
        ]
        self._alloc: list[float] = self._project_budget(self._raw_alloc)

    # ------------------------------------------------------------------
    # Budget projection
    # ------------------------------------------------------------------

    def _project_budget(self, raw: list[float]) -> list[float]:
        """Project raw non-negative allocations to sum <= vaccine_budget.

        Uses simple L1 renormalisation.  If all raw allocations are zero
        (budget is exhausted) returns a zero vector.

        Parameters
        ----------
        raw:
            Non-negative allocation amounts (after softplus).

        Returns
        -------
        Projected allocations summing to at most ``self.vaccine_budget``.
        """
        total = sum(raw)
        if total <= 0.0:
            return [0.0] * len(raw)
        if total <= self.vaccine_budget:
            return list(raw)
        scale = self.vaccine_budget / total
        return [r * scale for r in raw]

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(self) -> HybridActionSample:
        """Draw a stochastic sample from both heads.

        Returns
        -------
        HybridActionSample with both action components and log-probs.
        """
        discrete: list[int] = []
        lp_disc = 0.0

        for probs in self._probs:
            a = _sample_categorical(probs, self._rng)
            discrete.append(a)
            lp_disc += _categorical_log_prob(probs, a)

        continuous = list(self._alloc)
        lp_cont = self._continuous_log_prob(continuous)

        return HybridActionSample(
            discrete=discrete,
            continuous=continuous,
            log_prob_discrete=lp_disc,
            log_prob_continuous=lp_cont,
            log_prob=lp_disc + lp_cont,
        )

    def mode(self) -> HybridActionSample:
        """Return the deterministic (greedy) action from both heads.

        Discrete: argmax of masked probabilities.
        Continuous: use projected allocations directly (no randomness).

        Returns
        -------
        HybridActionSample with both action components and log-probs.
        """
        discrete: list[int] = []
        lp_disc = 0.0

        for probs in self._probs:
            a = max(range(len(probs)), key=lambda i: probs[i])
            discrete.append(a)
            lp_disc += _categorical_log_prob(probs, a)

        continuous = list(self._alloc)
        lp_cont = self._continuous_log_prob(continuous)

        return HybridActionSample(
            discrete=discrete,
            continuous=continuous,
            log_prob_discrete=lp_disc,
            log_prob_continuous=lp_cont,
            log_prob=lp_disc + lp_cont,
        )

    # ------------------------------------------------------------------
    # Log-probability re-evaluation (for PPO ratio computation)
    # ------------------------------------------------------------------

    def log_prob(
        self,
        discrete: list[int],
        continuous: list[float],
    ) -> tuple[float, float, float]:
        """Evaluate the log-probability of a given (discrete, continuous) pair.

        Used when computing the PPO importance ratio for stored rollout actions.

        Parameters
        ----------
        discrete:
            Per-node discrete actions.
        continuous:
            Per-node vaccine allocations.

        Returns
        -------
        lp_discrete, lp_continuous, lp_combined:
            Scalar log-probabilities.
        """
        lp_disc = 0.0
        for probs, a in zip(self._probs, discrete):
            lp_disc += _categorical_log_prob(probs, a)

        lp_cont = self._continuous_log_prob(continuous)
        return lp_disc, lp_cont, lp_disc + lp_cont

    def _continuous_log_prob(self, alloc: list[float]) -> float:
        """Approximate log-probability for the continuous allocation vector.

        We model the allocation as proportional to the projected softplus
        activations.  The log-probability is approximated as:

            Σ_i log(alloc_i / budget + ε)

        clamped to a finite range.  This is a scaffold approximation; a full
        implementation would use a proper Dirichlet or LogNormal distribution.
        """
        budget = max(self.vaccine_budget, 1e-8)
        lp = 0.0
        for v in alloc:
            lp += math.log(max(v / budget, 1e-8))
        # Clamp to avoid extreme values
        return max(lp, _MIN_LOG_PROB)

    # ------------------------------------------------------------------
    # Entropy
    # ------------------------------------------------------------------

    def entropy_discrete(self) -> float:
        """Mean per-node categorical entropy (averaged over nodes)."""
        if self.num_nodes == 0:
            return 0.0
        total = sum(_categorical_entropy(probs) for probs in self._probs)
        return total / self.num_nodes

    def entropy_continuous(self) -> float:
        """Proxy entropy for the continuous head.

        Uses the negative log of the allocation fractions as a diversity
        measure.  A more principled implementation in Phase 3 will use the
        differential entropy of the underlying parametric distribution.
        """
        budget = max(self.vaccine_budget, 1e-8)
        fracs = [v / budget for v in self._alloc]
        # Treat fractions as a discrete distribution approximation
        total = sum(fracs)
        if total <= 0.0:
            return 0.0
        normed = [f / total for f in fracs]
        return _categorical_entropy(normed)

    def entropy(self) -> tuple[float, float]:
        """Return (entropy_discrete, entropy_continuous)."""
        return self.entropy_discrete(), self.entropy_continuous()
