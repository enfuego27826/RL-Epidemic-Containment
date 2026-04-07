#!/usr/bin/env python3
"""Phase 2 smoke checks — lightweight validation for hybrid action modules.

These checks exercise the core Phase 2 components without requiring a full
training run.  They are designed to catch structural bugs early and can be
run in any environment with only the Python standard library.

Usage
-----
    python src/tests/smoke_phase2.py

Exit code 0 = all checks passed.
Exit code 1 = one or more checks failed.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Make sure repo root is importable
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# Also insert repo root as 'src' may not be a package from subprocess
_SRC = _REPO_ROOT / "src"
if str(_SRC.parent) not in sys.path:
    sys.path.insert(0, str(_SRC.parent))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PASS = "  [PASS]"
_FAIL = "  [FAIL]"
_errors: list[str] = []


def _check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"{_PASS} {name}")
    else:
        msg = f"{name}" + (f": {detail}" if detail else "")
        print(f"{_FAIL} {msg}")
        _errors.append(msg)


def _assert_finite(value: float, name: str) -> bool:
    ok = math.isfinite(value)
    _check(f"{name} is finite ({value:.4f})", ok)
    return ok


# ---------------------------------------------------------------------------
# Check 1: build_mask correctness
# ---------------------------------------------------------------------------

def check_mask_correctness() -> None:
    print("\n--- Check: build_mask correctness ---")
    from src.env.action_masking import build_mask, ACTION_NOOP, ACTION_QUARANTINE, ACTION_LIFT, ACTION_VACCINATE

    max_nodes = 5
    node_feature_dim = 4
    quarantine_slot = 2

    # Build a fake obs tensor: 5 nodes, alternating quarantined/not
    obs = []
    for i in range(max_nodes):
        is_q = float(i % 2)  # nodes 1, 3 are quarantined
        obs += [0.1, 0.9, is_q, 0.2]  # [infection, economy, quarantine, pop_frac]
    # Append 4 global scalars (not used by build_mask)
    obs += [1.0, 0.8, 0.1, 0.3]

    mask = build_mask(obs, max_nodes=max_nodes, num_active_nodes=max_nodes)

    _check("mask has correct shape (max_nodes rows)", len(mask) == max_nodes)
    _check("mask rows have NUM_DISCRETE_ACTIONS cols", all(len(row) == 4 for row in mask))

    for i in range(max_nodes):
        is_q = (i % 2 == 1)
        _check(f"node {i}: no-op always valid", mask[i][ACTION_NOOP] is True)
        _check(
            f"node {i}: quarantine valid iff not quarantined",
            mask[i][ACTION_QUARANTINE] == (not is_q),
        )
        _check(
            f"node {i}: lift valid iff quarantined",
            mask[i][ACTION_LIFT] == is_q,
        )
        _check(f"node {i}: vaccinate always valid", mask[i][ACTION_VACCINATE] is True)

    # Padding nodes: only no-op
    mask_padded = build_mask(obs, max_nodes=max_nodes, num_active_nodes=2)
    for i in range(2, max_nodes):
        _check(
            f"padding node {i}: only no-op valid",
            mask_padded[i] == [True, False, False, False],
        )


# ---------------------------------------------------------------------------
# Check 2: budget projection sums <= budget
# ---------------------------------------------------------------------------

def check_budget_projection() -> None:
    print("\n--- Check: budget projection sums <= budget ---")
    from src.env.action_masking import project_budget

    cases = [
        ([10.0, 20.0, 30.0], 50.0),
        ([1.0, 2.0, 3.0], 100.0),   # already under budget
        ([0.0, 0.0, 0.0], 50.0),   # all zero
        ([-5.0, 10.0, 5.0], 8.0),  # negative input
        ([1e-10, 1e-10], 1.0),     # near-zero
    ]

    for raw, budget in cases:
        projected = project_budget(raw, budget)
        total = sum(projected)
        _check(
            f"project_budget({raw}, {budget}) sum={total:.6f} <= {budget}",
            total <= budget + 1e-9,
        )
        _check(
            f"project_budget({raw}, {budget}) all non-negative",
            all(v >= -1e-9 for v in projected),
        )


# ---------------------------------------------------------------------------
# Check 3: HybridActionDist — combined log-prob is finite
# ---------------------------------------------------------------------------

def check_hybrid_dist_log_prob() -> None:
    print("\n--- Check: HybridActionDist combined log-prob finite ---")
    from src.models.hybrid_action import HybridActionDist

    num_nodes = 5
    import random
    rng = random.Random(42)

    discrete_logits = [
        [rng.gauss(0, 1) for _ in range(4)] for _ in range(num_nodes)
    ]
    continuous_logits = [rng.gauss(0, 1) for _ in range(num_nodes)]
    vaccine_budget = 100.0

    dist = HybridActionDist(
        discrete_logits=discrete_logits,
        continuous_logits=continuous_logits,
        vaccine_budget=vaccine_budget,
        seed=0,
    )

    sample = dist.sample()
    _assert_finite(sample.log_prob_discrete, "sample log_prob_discrete")
    _assert_finite(sample.log_prob_continuous, "sample log_prob_continuous")
    _assert_finite(sample.log_prob, "sample combined log_prob")

    mode_sample = dist.mode()
    _assert_finite(mode_sample.log_prob, "mode combined log_prob")

    # Re-evaluate stored action
    lp_disc, lp_cont, lp_combined = dist.log_prob(sample.discrete, sample.continuous)
    _assert_finite(lp_disc, "re-evaluated lp_discrete")
    _assert_finite(lp_cont, "re-evaluated lp_continuous")
    _assert_finite(lp_combined, "re-evaluated lp_combined")


# ---------------------------------------------------------------------------
# Check 4: HybridActionDist — budget constraint
# ---------------------------------------------------------------------------

def check_hybrid_dist_budget() -> None:
    print("\n--- Check: HybridActionDist budget constraint ---")
    from src.models.hybrid_action import HybridActionDist

    import random
    rng = random.Random(7)
    num_nodes = 10
    budget = 50.0

    for trial in range(5):
        discrete_logits = [[rng.gauss(0, 1) for _ in range(4)] for _ in range(num_nodes)]
        continuous_logits = [rng.gauss(0, 2) for _ in range(num_nodes)]
        dist = HybridActionDist(
            discrete_logits=discrete_logits,
            continuous_logits=continuous_logits,
            vaccine_budget=budget,
            seed=trial,
        )
        sample = dist.sample()
        total_alloc = sum(sample.continuous)
        _check(
            f"trial {trial}: sum(continuous)={total_alloc:.4f} <= budget={budget}",
            total_alloc <= budget + 1e-6,
        )
        _check(
            f"trial {trial}: all continuous >= 0",
            all(v >= -1e-9 for v in sample.continuous),
        )


# ---------------------------------------------------------------------------
# Check 5: validate_hybrid_action diagnostics
# ---------------------------------------------------------------------------

def check_validate_hybrid_action() -> None:
    print("\n--- Check: validate_hybrid_action diagnostics ---")
    from src.env.action_masking import (
        validate_hybrid_action,
        build_mask,
        ACTION_NOOP,
        ACTION_QUARANTINE,
        ACTION_LIFT,
    )

    max_nodes = 4
    obs = []
    # Node 0: not quarantined; Node 1: quarantined; Node 2: not; Node 3: quarantined
    quarantine_flags = [0.0, 1.0, 0.0, 1.0]
    for qf in quarantine_flags:
        obs += [0.1, 0.9, qf, 0.25]
    obs += [1.0, 0.8, 0.1, 0.3]

    mask = build_mask(obs, max_nodes=max_nodes, num_active_nodes=max_nodes)
    budget = 80.0

    # Valid action: no-op everywhere, small allocations
    valid_result = validate_hybrid_action(
        discrete=[ACTION_NOOP] * max_nodes,
        continuous=[5.0] * max_nodes,
        mask=mask,
        vaccine_budget=budget,
    )
    _check("valid action is valid", valid_result.valid, str(valid_result.violations))

    # Invalid: quarantine node 1 which is already quarantined
    invalid_q = validate_hybrid_action(
        discrete=[ACTION_NOOP, ACTION_QUARANTINE, ACTION_NOOP, ACTION_NOOP],
        continuous=[5.0] * max_nodes,
        mask=mask,
        vaccine_budget=budget,
    )
    _check("quarantine already-quarantined node is invalid", not invalid_q.valid)

    # Invalid: lift non-quarantined node 0
    invalid_l = validate_hybrid_action(
        discrete=[ACTION_LIFT, ACTION_NOOP, ACTION_NOOP, ACTION_NOOP],
        continuous=[5.0] * max_nodes,
        mask=mask,
        vaccine_budget=budget,
    )
    _check("lift non-quarantined node is invalid", not invalid_l.valid)

    # Invalid: budget overrun
    invalid_b = validate_hybrid_action(
        discrete=[ACTION_NOOP] * max_nodes,
        continuous=[50.0] * max_nodes,   # sum = 200 > 80
        mask=mask,
        vaccine_budget=budget,
    )
    _check("budget overrun detected", not invalid_b.valid)
    _check("budget_overrun > 0", invalid_b.budget_overrun > 0.0)


# ---------------------------------------------------------------------------
# Check 6: HybridActorCritic smoke forward pass
# ---------------------------------------------------------------------------

def check_hybrid_actor_critic() -> None:
    print("\n--- Check: HybridActorCritic forward pass ---")
    from src.models.actor_critic import HybridActorCritic

    obs_dim = 84   # 20 nodes * 4 features + 4 global
    max_nodes = 20
    policy = HybridActorCritic(obs_dim=obs_dim, max_nodes=max_nodes, seed=0)

    import random
    rng = random.Random(99)
    obs = [rng.gauss(0, 1) for _ in range(obs_dim)]

    disc_logits, cont_logits, value = policy.forward(obs)
    _check("discrete logits shape[0] == max_nodes", len(disc_logits) == max_nodes)
    _check("discrete logits shape[1] == 4", all(len(row) == 4 for row in disc_logits))
    _check("continuous logits length == max_nodes", len(cont_logits) == max_nodes)
    _assert_finite(value, "critic value")
    _check("all discrete logits finite", all(math.isfinite(l) for row in disc_logits for l in row))
    _check("all continuous logits finite", all(math.isfinite(l) for l in cont_logits))

    # act()
    action_dict, lp_disc, lp_cont, v = policy.act(obs, vaccine_budget=100.0, seed=0)
    _check("action_dict has 'discrete' key", "discrete" in action_dict)
    _check("action_dict has 'continuous' key", "continuous" in action_dict)
    _check("discrete length == max_nodes", len(action_dict["discrete"]) == max_nodes)
    _check("continuous length == max_nodes", len(action_dict["continuous"]) == max_nodes)
    _assert_finite(lp_disc, "act log_prob_discrete")
    _assert_finite(lp_cont, "act log_prob_continuous")


# ---------------------------------------------------------------------------
# Check 7: OpenEnvAdapter accepts hybrid action dict
# ---------------------------------------------------------------------------

def check_adapter_hybrid_step() -> None:
    print("\n--- Check: OpenEnvAdapter.step() with hybrid action dict ---")
    try:
        from src.env.openenv_adapter import OpenEnvAdapter, ACTION_NOOP

        adapter = OpenEnvAdapter(task_name="easy_localized_outbreak", seed=42, max_nodes=20)
        obs, info = adapter.reset()

        num_nodes = info["num_nodes"]
        hybrid_action = {
            "discrete": [ACTION_NOOP] * 20,
            "continuous": [1.0] * 20,
        }
        next_obs, reward, done, step_info = adapter.step(hybrid_action)

        _check("next_obs length == obs_dim", len(next_obs) == adapter.obs_dim)
        _check("reward is finite", math.isfinite(reward))
        _check("done is bool", isinstance(done, bool))
        _check("step_info has invalid_action_count", "invalid_action_count" in step_info)
        _check("step_info has invalid_action_rate", "invalid_action_rate" in step_info)
        _check("invalid_action_count >= 0", step_info["invalid_action_count"] >= 0)

    except Exception as exc:
        _check(f"adapter hybrid step (no exception)", False, str(exc))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Phase 2 smoke checks")
    print("=" * 60)

    check_mask_correctness()
    check_budget_projection()
    check_hybrid_dist_log_prob()
    check_hybrid_dist_budget()
    check_validate_hybrid_action()
    check_hybrid_actor_critic()
    check_adapter_hybrid_step()

    print("\n" + "=" * 60)
    if _errors:
        print(f"RESULT: {len(_errors)} check(s) FAILED")
        for e in _errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("RESULT: all checks PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()
