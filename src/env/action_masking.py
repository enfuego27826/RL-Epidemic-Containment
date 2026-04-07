"""Action masking and budget validation utilities for Phase 2 hybrid actions.

This module provides:

1. ``build_mask(obs_tensor, max_nodes, quarantine_slot)``
   Build a (max_nodes, NUM_DISCRETE_ACTIONS) boolean mask from the current
   observation tensor.

2. ``project_budget(raw_alloc, vaccine_budget)``
   Project a raw non-negative allocation vector to sum <= vaccine_budget.

3. ``validate_hybrid_action(discrete, continuous, mask, vaccine_budget)``
   Check a hybrid action for constraint violations and return a diagnostics dict.

4. ``HybridActionValidator``
   Stateful validator that accumulates invalid-action counters across a rollout.

All functions use only the Python standard library.

TODO (Phase 3): extend mask to incorporate graph-level reachability constraints.
TODO (Phase 4): condition mask on belief-state (lag-aware quarantine flags).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Discrete action codes — must match openenv_adapter and hybrid_action
ACTION_NOOP: int = 0
ACTION_QUARANTINE: int = 1
ACTION_LIFT: int = 2
ACTION_VACCINATE: int = 3
NUM_DISCRETE_ACTIONS: int = 4

# Human-readable names for action codes (used in validation messages)
ACTION_NAMES: dict[int, str] = {
    ACTION_NOOP: "no-op",
    ACTION_QUARANTINE: "quarantine",
    ACTION_LIFT: "lift",
    ACTION_VACCINATE: "vaccinate",
}

# Observation tensor layout per node (must match openenv_adapter.NODE_FEATURE_DIM)
_NODE_FEATURE_DIM: int = 4
_QUARANTINE_SLOT: int = 2  # index within the per-node feature block


# ---------------------------------------------------------------------------
# Mask construction
# ---------------------------------------------------------------------------

def build_mask(
    obs_tensor: list[float],
    max_nodes: int,
    num_active_nodes: int | None = None,
    node_feature_dim: int = _NODE_FEATURE_DIM,
    quarantine_slot: int = _QUARANTINE_SLOT,
) -> list[list[bool]]:
    """Build a per-node action validity mask from the observation tensor.

    Rules applied
    -------------
    - Action 0 (no-op)           : always valid.
    - Action 1 (quarantine)      : valid only when node is **not** quarantined.
    - Action 2 (lift_quarantine) : valid only when node **is** quarantined.
    - Action 3 (vaccinate)       : always valid (budget enforcement is separate).
    - Padding nodes (i >= num_active_nodes): only no-op is valid.

    Parameters
    ----------
    obs_tensor:
        Flat observation vector as returned by ``OpenEnvAdapter``.
    max_nodes:
        Total number of node slots (padded).
    num_active_nodes:
        Number of real (non-padded) nodes.  Defaults to ``max_nodes``.
    node_feature_dim:
        Number of features per node (default 4, matching Phase 1 adapter).
    quarantine_slot:
        Offset within each node's feature block that holds the quarantine flag
        (default 2).

    Returns
    -------
    mask:
        Shape (max_nodes, NUM_DISCRETE_ACTIONS).  Entry is True iff the
        corresponding action is valid for that node.
    """
    if num_active_nodes is None:
        num_active_nodes = max_nodes

    mask: list[list[bool]] = []
    for i in range(max_nodes):
        if i >= num_active_nodes:
            # Padding node — only no-op is meaningful
            mask.append([True, False, False, False])
            continue

        # Read quarantine flag from obs tensor
        base = i * node_feature_dim
        is_quarantined = obs_tensor[base + quarantine_slot] > 0.5

        node_mask = [
            True,                   # 0: no-op — always valid
            not is_quarantined,     # 1: quarantine — only if not already quarantined
            is_quarantined,         # 2: lift — only if currently quarantined
            True,                   # 3: vaccinate — always valid
        ]
        mask.append(node_mask)

    return mask


# ---------------------------------------------------------------------------
# Budget projection
# ---------------------------------------------------------------------------

def project_budget(
    raw_alloc: list[float],
    vaccine_budget: float,
) -> list[float]:
    """Project a raw allocation vector to sum <= vaccine_budget.

    Non-negative inputs are preserved up to the budget cap using L1
    renormalisation.  Negative values are clamped to zero before projection.

    Parameters
    ----------
    raw_alloc:
        Per-node raw (unconstrained) allocations.  May be negative.
    vaccine_budget:
        Maximum total allocation allowed.

    Returns
    -------
    Projected allocations: non-negative, summing to at most ``vaccine_budget``.
    """
    clamped = [max(v, 0.0) for v in raw_alloc]
    total = sum(clamped)
    if vaccine_budget <= 0.0:
        return [0.0] * len(raw_alloc)
    if total <= vaccine_budget:
        return clamped
    scale = vaccine_budget / total
    return [v * scale for v in clamped]


# ---------------------------------------------------------------------------
# Hybrid action validator
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    """Result of validating a single hybrid action step."""

    valid: bool
    violations: list[str] = field(default_factory=list)
    budget_used: float = 0.0
    budget_overrun: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "violations": list(self.violations),
            "budget_used": self.budget_used,
            "budget_overrun": self.budget_overrun,
        }


def validate_hybrid_action(
    discrete: list[int],
    continuous: list[float],
    mask: list[list[bool]],
    vaccine_budget: float,
    budget_tolerance: float = 1e-6,
) -> ValidationResult:
    """Validate a hybrid action against mask and budget constraints.

    Parameters
    ----------
    discrete:
        Per-node discrete action codes.
    continuous:
        Per-node vaccine allocations (should be non-negative and budget-bounded).
    mask:
        Per-node validity mask (from ``build_mask``).
    vaccine_budget:
        Current available vaccine budget.
    budget_tolerance:
        Floating-point tolerance for budget constraint check.

    Returns
    -------
    ValidationResult with ``valid`` flag, violation messages, and budget stats.
    """
    violations: list[str] = []

    num_nodes = min(len(discrete), len(mask))

    # Check discrete action validity
    for i in range(num_nodes):
        act = discrete[i]
        if act < 0 or act >= NUM_DISCRETE_ACTIONS:
            violations.append(f"node {i}: action code {act} out of range")
            continue
        if not mask[i][act]:
            action_name = ACTION_NAMES.get(act, str(act))
            violations.append(f"node {i}: action '{action_name}' masked as invalid")

    # Check continuous allocation non-negativity
    for i, v in enumerate(continuous[:num_nodes]):
        if v < -budget_tolerance:
            violations.append(f"node {i}: negative allocation {v:.4f}")

    # Check budget constraint
    budget_used = sum(max(v, 0.0) for v in continuous[:num_nodes])
    budget_overrun = max(budget_used - vaccine_budget, 0.0)
    if budget_overrun > budget_tolerance:
        violations.append(
            f"budget overrun: used={budget_used:.4f} budget={vaccine_budget:.4f} "
            f"overrun={budget_overrun:.4f}"
        )

    return ValidationResult(
        valid=len(violations) == 0,
        violations=violations,
        budget_used=budget_used,
        budget_overrun=budget_overrun,
    )


class HybridActionValidator:
    """Stateful validator that accumulates diagnostics across a rollout.

    Maintains running counters for:
    - Total steps validated
    - Steps with at least one masked-action violation
    - Steps with budget overrun
    - Cumulative budget utilisation

    Usage
    -----
    ::

        validator = HybridActionValidator(vaccine_budget=100.0)
        for each step:
            result = validator.validate(discrete, continuous, mask, vaccine_budget)
        print(validator.summary())
    """

    def __init__(self, vaccine_budget: float = 1.0) -> None:
        self.vaccine_budget = vaccine_budget
        self._steps: int = 0
        self._mask_violations: int = 0
        self._budget_overruns: int = 0
        self._total_budget_used: float = 0.0
        self._total_budget_overrun: float = 0.0

    def validate(
        self,
        discrete: list[int],
        continuous: list[float],
        mask: list[list[bool]],
        vaccine_budget: float | None = None,
    ) -> ValidationResult:
        """Validate one step and update counters.

        Parameters
        ----------
        discrete, continuous, mask:
            As for ``validate_hybrid_action``.
        vaccine_budget:
            If provided, overrides the instance-level budget.

        Returns
        -------
        ValidationResult for this step.
        """
        budget = vaccine_budget if vaccine_budget is not None else self.vaccine_budget
        result = validate_hybrid_action(discrete, continuous, mask, budget)

        self._steps += 1
        if any("masked" in v for v in result.violations):
            self._mask_violations += 1
        if result.budget_overrun > 1e-6:
            self._budget_overruns += 1
        self._total_budget_used += result.budget_used
        self._total_budget_overrun += result.budget_overrun

        return result

    def summary(self) -> dict[str, Any]:
        """Return aggregated diagnostics over all validated steps."""
        n = max(self._steps, 1)
        return {
            "total_steps": self._steps,
            "mask_violation_rate": self._mask_violations / n,
            "budget_overrun_rate": self._budget_overruns / n,
            "mean_budget_used": self._total_budget_used / n,
            "mean_budget_overrun": self._total_budget_overrun / n,
        }

    def reset(self) -> None:
        """Reset all counters."""
        self._steps = 0
        self._mask_violations = 0
        self._budget_overruns = 0
        self._total_budget_used = 0.0
        self._total_budget_overrun = 0.0
