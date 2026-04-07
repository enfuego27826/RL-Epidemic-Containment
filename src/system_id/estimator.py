"""Phase 5 — Online System Identification.

Estimates hidden epidemic parameters (beta, gamma) from observed data
using a rolling statistical approach.  Inferred estimates and uncertainty
proxies are exposed as additional features for the policy.

Background
----------
The SIR model updates compartments as:

    new_infections_i ~ S_i * (1 - exp(-lambda_i))
    lambda_i = beta_i * (I_i / N_i) + ...  (cross-node terms omitted)
    recoveries_i = gamma_i * I_i

We estimate:
  - ``beta_hat_i``: effective transmission rate per node from the
    ratio of new infections to S_i * I_i / N_i
  - ``gamma_hat_i``: recovery rate from recoveries / I_i

Uncertainty is proxied by the coefficient of variation (CV) of recent
estimates and an explicit confidence weight based on sample count.

Safe fallbacks
--------------
When data are insufficient (e.g., I_i ≈ 0), estimates fall back to
population-level priors (configurable).  The ``is_uncertain`` flag signals
the policy when estimates are unreliable.

Public API
----------
- ``NodeEstimator``: single-node rolling beta/gamma estimator.
- ``SystemIdentifier``: multi-node wrapper; produces feature vectors.
- ``SystemIDFeatures``: named-tuple output with estimates + uncertainty.

All implementations are stdlib-only.
"""

from __future__ import annotations

import math
from typing import NamedTuple


# ---------------------------------------------------------------------------
# Constants / fallback priors
# ---------------------------------------------------------------------------

BETA_PRIOR: float = 0.3          # prior for transmission rate
GAMMA_PRIOR: float = 0.1         # prior for recovery rate
BETA_MIN: float = 0.0
BETA_MAX: float = 2.0
GAMMA_MIN: float = 0.01
GAMMA_MAX: float = 0.5

# Minimum infected fraction to attempt estimation
_MIN_INFECTED_FRAC: float = 1e-4

# Window size for rolling estimates
_DEFAULT_WINDOW: int = 8


# ---------------------------------------------------------------------------
# NodeEstimator
# ---------------------------------------------------------------------------

class NodeEstimator:
    """Rolling window estimator for beta and gamma at a single node.

    Parameters
    ----------
    window_size:
        Number of recent observations used for estimation.
    beta_prior:
        Fallback beta when estimation is unreliable.
    gamma_prior:
        Fallback gamma when estimation is unreliable.
    """

    def __init__(
        self,
        window_size: int = _DEFAULT_WINDOW,
        beta_prior: float = BETA_PRIOR,
        gamma_prior: float = GAMMA_PRIOR,
    ) -> None:
        self.window_size = window_size
        self.beta_prior = beta_prior
        self.gamma_prior = gamma_prior

        # Rolling buffers: (S, I, N, new_infections, recoveries)
        self._S_buf: list[float] = []
        self._I_buf: list[float] = []
        self._N_buf: list[float] = []
        self._new_inf_buf: list[float] = []
        self._rec_buf: list[float] = []

        # Cached estimates
        self._beta_hat: float = beta_prior
        self._gamma_hat: float = gamma_prior
        self._beta_cv: float = 1.0   # coefficient of variation (uncertainty)
        self._gamma_cv: float = 1.0
        self._n_samples: int = 0

    def push(
        self,
        S: float,
        I: float,
        N: float,
        new_infections: float,
        recoveries: float,
    ) -> None:
        """Record one timestep of compartment data.

        Parameters
        ----------
        S, I, N:
            Susceptible, infected, total population fractions or counts.
        new_infections:
            Newly infected individuals this step.
        recoveries:
            Newly recovered individuals this step.
        """
        self._S_buf.append(max(S, 0.0))
        self._I_buf.append(max(I, 0.0))
        self._N_buf.append(max(N, 1.0))
        self._new_inf_buf.append(max(new_infections, 0.0))
        self._rec_buf.append(max(recoveries, 0.0))

        # Trim to window
        if len(self._S_buf) > self.window_size:
            self._S_buf.pop(0)
            self._I_buf.pop(0)
            self._N_buf.pop(0)
            self._new_inf_buf.pop(0)
            self._rec_buf.pop(0)

        self._n_samples += 1
        self._update_estimates()

    def _update_estimates(self) -> None:
        """Recompute beta_hat and gamma_hat from the rolling window."""
        beta_samples: list[float] = []
        gamma_samples: list[float] = []

        for S, I, N, ni, rec in zip(
            self._S_buf, self._I_buf, self._N_buf,
            self._new_inf_buf, self._rec_buf,
        ):
            inf_frac = I / N
            # Beta estimate: lambda ≈ new_infections / S → beta = lambda / (I/N)
            if S > 0 and inf_frac > _MIN_INFECTED_FRAC:
                # Clamp ni/S to (0, 1) before taking log to avoid domain errors
                survival_prob = max(min(1.0 - ni / S, 1.0 - 1e-9), 1e-9)
                lam = -math.log(survival_prob)
                beta_est = lam / inf_frac
                beta_est = max(BETA_MIN, min(BETA_MAX, beta_est))
                beta_samples.append(beta_est)

            # Gamma estimate: recoveries / I
            if I > 0:
                gamma_est = rec / I
                gamma_est = max(GAMMA_MIN, min(GAMMA_MAX, gamma_est))
                gamma_samples.append(gamma_est)

        if beta_samples:
            self._beta_hat = _mean(beta_samples)
            self._beta_cv = _cv(beta_samples)
        else:
            self._beta_hat = self.beta_prior
            self._beta_cv = 1.0

        if gamma_samples:
            self._gamma_hat = _mean(gamma_samples)
            self._gamma_cv = _cv(gamma_samples)
        else:
            self._gamma_hat = self.gamma_prior
            self._gamma_cv = 1.0

    @property
    def beta_hat(self) -> float:
        return self._beta_hat

    @property
    def gamma_hat(self) -> float:
        return self._gamma_hat

    @property
    def beta_uncertainty(self) -> float:
        """Coefficient of variation of recent beta samples (proxy for uncertainty)."""
        return self._beta_cv

    @property
    def gamma_uncertainty(self) -> float:
        return self._gamma_cv

    @property
    def is_uncertain(self) -> bool:
        """True when estimates are unreliable (high CV or few samples)."""
        return self._n_samples < 3 or self._beta_cv > 0.5 or self._gamma_cv > 0.5

    @property
    def confidence(self) -> float:
        """Scalar in [0, 1] indicating estimation confidence."""
        sample_conf = min(1.0, self._n_samples / self.window_size)
        cv_conf = max(0.0, 1.0 - 0.5 * (self._beta_cv + self._gamma_cv))
        return 0.5 * (sample_conf + cv_conf)

    def reset(self) -> None:
        """Clear buffers for a new episode."""
        self._S_buf.clear()
        self._I_buf.clear()
        self._N_buf.clear()
        self._new_inf_buf.clear()
        self._rec_buf.clear()
        self._beta_hat = self.beta_prior
        self._gamma_hat = self.gamma_prior
        self._beta_cv = 1.0
        self._gamma_cv = 1.0
        self._n_samples = 0


# ---------------------------------------------------------------------------
# SystemIDFeatures — named-tuple result
# ---------------------------------------------------------------------------

class SystemIDFeatures(NamedTuple):
    """Output of ``SystemIdentifier.get_features()``.

    Attributes
    ----------
    beta_hats:
        Per-node estimated transmission rates, list[float] of length num_nodes.
    gamma_hats:
        Per-node estimated recovery rates.
    beta_uncertainties:
        Per-node CV of beta estimates (proxy for uncertainty).
    gamma_uncertainties:
        Per-node CV of gamma estimates.
    confidences:
        Per-node scalar confidence in [0, 1].
    flat_features:
        Concatenated flat feature vector for direct policy input.
        Layout: [beta_hats || gamma_hats || beta_unc || gamma_unc || confidences]
        Length = 5 * num_nodes.
    """

    beta_hats: list[float]
    gamma_hats: list[float]
    beta_uncertainties: list[float]
    gamma_uncertainties: list[float]
    confidences: list[float]
    flat_features: list[float]


# ---------------------------------------------------------------------------
# SystemIdentifier — multi-node wrapper
# ---------------------------------------------------------------------------

class SystemIdentifier:
    """Multi-node system identifier.

    Maintains one ``NodeEstimator`` per node and provides a flat feature
    vector suitable for concatenation with the policy's observation.

    Parameters
    ----------
    max_nodes:
        Maximum number of nodes.
    window_size:
        Rolling window size for each node estimator.
    beta_prior, gamma_prior:
        Priors used when estimates are unavailable.
    """

    def __init__(
        self,
        max_nodes: int = 20,
        window_size: int = _DEFAULT_WINDOW,
        beta_prior: float = BETA_PRIOR,
        gamma_prior: float = GAMMA_PRIOR,
    ) -> None:
        self.max_nodes = max_nodes
        self.window_size = window_size
        self._estimators: list[NodeEstimator] = [
            NodeEstimator(window_size=window_size, beta_prior=beta_prior, gamma_prior=gamma_prior)
            for _ in range(max_nodes)
        ]

    def reset(self) -> None:
        """Reset all node estimators for a new episode."""
        for est in self._estimators:
            est.reset()

    def update(
        self,
        node_index: int,
        S: float,
        I: float,
        N: float,
        new_infections: float,
        recoveries: float,
    ) -> None:
        """Push new observations for a single node.

        Parameters
        ----------
        node_index:
            Index of the node (0-based).
        S, I, N:
            Compartment values (raw counts or fractions).
        new_infections:
            Newly infected this step.
        recoveries:
            Newly recovered this step.
        """
        if 0 <= node_index < self.max_nodes:
            self._estimators[node_index].push(S, I, N, new_infections, recoveries)

    def update_from_obs(
        self,
        obs_tensor: list[float],
        prev_obs_tensor: list[float] | None = None,
        num_active_nodes: int | None = None,
        node_feature_dim: int = 4,
    ) -> None:
        """Infer compartment changes from consecutive observations.

        This is a proxy update when raw SIR counts are not available.
        It uses changes in ``known_infection_rate`` as a proxy for
        new infections and recovery.

        Parameters
        ----------
        obs_tensor:
            Current flat observation vector.
        prev_obs_tensor:
            Previous flat observation vector (None on first step).
        num_active_nodes:
            Number of real nodes.
        node_feature_dim:
            Features per node in the observation vector.
        """
        max_nodes = self.max_nodes
        num_active = num_active_nodes if num_active_nodes is not None else max_nodes

        for i in range(min(num_active, max_nodes)):
            base = i * node_feature_dim
            inf_rate = obs_tensor[base]           # known_infection_rate
            econ = obs_tensor[base + 1]            # economic_health
            pop_frac = obs_tensor[base + 3]        # population_fraction

            # Proxy N from population fraction (not exact, but proportional)
            N = max(pop_frac, 1e-6)
            I = inf_rate * N
            S = max(N - I, 0.0)

            # Approximate new_infections and recoveries from infection rate change
            if prev_obs_tensor is not None:
                prev_inf_rate = prev_obs_tensor[base]
                delta_inf = (inf_rate - prev_inf_rate) * N
                new_infections = max(delta_inf, 0.0)
                recoveries = max(-delta_inf * 0.5, 0.0)  # rough proxy
            else:
                new_infections = 0.0
                recoveries = 0.0

            self._estimators[i].push(S, I, N, new_infections, recoveries)

    def get_features(self, num_active_nodes: int | None = None) -> SystemIDFeatures:
        """Return system ID features for all nodes (padded to max_nodes).

        Parameters
        ----------
        num_active_nodes:
            Number of real nodes.  Padding slots get prior values.

        Returns
        -------
        SystemIDFeatures with per-node estimates and flat feature vector.
        """
        num_active = num_active_nodes if num_active_nodes is not None else self.max_nodes

        beta_hats: list[float] = []
        gamma_hats: list[float] = []
        beta_uncertainties: list[float] = []
        gamma_uncertainties: list[float] = []
        confidences: list[float] = []

        for i in range(self.max_nodes):
            if i < num_active:
                est = self._estimators[i]
                beta_hats.append(est.beta_hat)
                gamma_hats.append(est.gamma_hat)
                beta_uncertainties.append(est.beta_uncertainty)
                gamma_uncertainties.append(est.gamma_uncertainty)
                confidences.append(est.confidence)
            else:
                # Padding slot: use prior values, zero confidence
                beta_hats.append(BETA_PRIOR)
                gamma_hats.append(GAMMA_PRIOR)
                beta_uncertainties.append(1.0)
                gamma_uncertainties.append(1.0)
                confidences.append(0.0)

        flat = beta_hats + gamma_hats + beta_uncertainties + gamma_uncertainties + confidences
        return SystemIDFeatures(
            beta_hats=beta_hats,
            gamma_hats=gamma_hats,
            beta_uncertainties=beta_uncertainties,
            gamma_uncertainties=gamma_uncertainties,
            confidences=confidences,
            flat_features=flat,
        )

    @property
    def feature_dim(self) -> int:
        """Dimension of the flat feature vector returned by get_features()."""
        return 5 * self.max_nodes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    variance = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(max(variance, 0.0))


def _cv(values: list[float]) -> float:
    """Coefficient of variation (std / mean), clamped to [0, 1]."""
    m = _mean(values)
    if abs(m) < 1e-8:
        return 1.0
    return min(1.0, _std(values) / abs(m))
