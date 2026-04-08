#!/usr/bin/env python3
"""Smoke tests for Phases 3–8.

Tests the ST encoder, belief state, system identification, MoRL trainer,
evaluation harness, and inference script components without requiring a
full training run.

Usage
-----
    python src/tests/smoke_phases3to8.py

Exit code 0 = all checks passed, 1 = failures.
"""

from __future__ import annotations

import math
import sys
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


def _finite(v: float) -> bool:
    return math.isfinite(v)


# ===========================================================================
# Phase 3 — ST Encoder
# ===========================================================================

def check_st_encoder() -> None:
    print("\n--- Phase 3: STEncoder ---")
    from src.models.st_encoder import STEncoder, _build_default_adj

    enc = STEncoder(
        node_feature_dim=4,
        gcn_hidden_dim=16,
        gru_hidden_dim=16,
        global_context_dim=16,
        num_global_scalars=4,
        seed=0,
    )
    max_nodes = 5
    enc.reset(max_nodes)
    _check("encoder.output_dim == 20", enc.output_dim == 20)

    obs = [0.1, 0.9, 0.0, 0.2] * max_nodes + [1.0, 0.8, 0.1, 0.05]
    encoding = enc.forward(obs, num_active_nodes=4)
    _check("encoding length == output_dim", len(encoding) == enc.output_dim)
    _check("encoding finite", all(_finite(v) for v in encoding))

    # Second forward pass (GRU step)
    encoding2 = enc.forward(obs, num_active_nodes=4)
    _check("second encoding finite", all(_finite(v) for v in encoding2))
    # GRU should produce different outputs after update
    changed = any(abs(a - b) > 1e-10 for a, b in zip(encoding, encoding2))
    _check("GRU state changes between steps", changed)


def check_st_actor_critic() -> None:
    print("\n--- Phase 3: STActorCritic ---")
    from src.models.st_encoder import STActorCritic

    max_nodes = 5
    model = STActorCritic(
        node_feature_dim=4,
        max_nodes=max_nodes,
        action_dim=4,
        gcn_hidden_dim=16,
        gru_hidden_dim=16,
        global_context_dim=16,
        seed=1,
    )
    model.reset_episode()

    obs = [0.1, 0.9, 0.0, 0.2] * max_nodes + [1.0, 0.8, 0.1, 0.05]
    disc_logits, cont_logits, value = model.forward(obs, num_active_nodes=4)

    _check("disc_logits shape (max_nodes, 4)", len(disc_logits) == max_nodes and all(len(r) == 4 for r in disc_logits))
    _check("cont_logits length == max_nodes", len(cont_logits) == max_nodes)
    _check("value is finite", _finite(value))

    action, lp_d, lp_c, v = model.act(obs, num_active_nodes=4, vaccine_budget=10.0)
    _check("action has 'discrete' key", "discrete" in action)
    _check("action has 'continuous' key", "continuous" in action)
    _check("act log_prob_discrete finite", _finite(lp_d))
    _check("act log_prob_continuous finite", _finite(lp_c))


# ===========================================================================
# Phase 4 — Belief State
# ===========================================================================

def check_belief_state() -> None:
    print("\n--- Phase 4: ObservationBuffer and BeliefStateBuilder ---")
    from src.env.belief_state import ObservationBuffer, BeliefStateBuilder

    obs_dim = 10
    buf = ObservationBuffer(obs_dim=obs_dim, max_history=6)
    initial = [float(i) for i in range(obs_dim)]
    buf.reset(initial)

    _check("buffer initialised", len(buf._obs_buf) == 6)

    new_obs = [float(i + 1) for i in range(obs_dim)]
    buf.push(new_obs)
    _check("buffer grows on push", buf.latest_obs == new_obs)

    builder_concat = BeliefStateBuilder(obs_dim=obs_dim, history_len=4, mode="concat")
    belief = builder_concat.build(buf)
    _check(
        "concat belief length == obs_dim * history_len",
        len(belief) == obs_dim * 4,
    )

    builder_mean = BeliefStateBuilder(obs_dim=obs_dim, history_len=4, mode="mean")
    mean_belief = builder_mean.build(buf)
    _check("mean belief length == obs_dim", len(mean_belief) == obs_dim)

    # Lagged obs
    for i in range(3):
        buf.push([float(i + 10)] * obs_dim)
    lagged = buf.get_lagged_obs(2)
    _check("lagged obs has correct length", len(lagged) == obs_dim)


def check_lag_aware_adapter() -> None:
    print("\n--- Phase 4: LagAwareAdapter ---")
    from src.env.belief_state import LagAwareAdapter

    adapter = LagAwareAdapter(
        task_name="easy_localized_outbreak",
        seed=42,
        max_nodes=20,
        lag_steps=2,
        history_len=4,
        mode="concat",
    )
    belief, info = adapter.reset(seed=42)
    _check("belief length == belief_dim", len(belief) == adapter.belief_dim)
    _check("belief_dim == obs_dim * history_len", adapter.belief_dim == adapter.obs_dim * 4)
    _check("info has lag_steps", "lag_steps" in info)

    noop_action = [0] * 20
    belief2, reward, done, step_info = adapter.step(noop_action)
    _check("step belief length == belief_dim", len(belief2) == adapter.belief_dim)
    _check("reward is finite", _finite(reward))


# ===========================================================================
# Phase 5 — System Identification
# ===========================================================================

def check_system_id() -> None:
    print("\n--- Phase 5: NodeEstimator ---")
    from src.system_id.estimator import NodeEstimator

    est = NodeEstimator(window_size=5)
    est.push(S=900, I=100, N=1000, new_infections=10, recoveries=5)
    est.push(S=890, I=110, N=1000, new_infections=12, recoveries=6)
    est.push(S=878, I=116, N=1000, new_infections=8, recoveries=7)

    _check("beta_hat in [0, 2]", 0 <= est.beta_hat <= 2)
    _check("gamma_hat in [0.01, 0.5]", 0.01 <= est.gamma_hat <= 0.5)
    _check("beta_uncertainty >= 0", est.beta_uncertainty >= 0)
    _check("confidence in [0, 1]", 0 <= est.confidence <= 1)

    est.reset()
    _check("reset: n_samples == 0", est._n_samples == 0)


def check_system_identifier() -> None:
    print("\n--- Phase 5: SystemIdentifier ---")
    from src.system_id.estimator import SystemIdentifier

    sysid = SystemIdentifier(max_nodes=5, window_size=4)
    sysid.reset()

    obs = [0.1, 0.9, 0.0, 0.2] * 5 + [1.0, 0.8, 0.1, 0.05]
    prev_obs = [0.05, 0.92, 0.0, 0.2] * 5 + [1.0, 0.85, 0.05, 0.04]

    sysid.update_from_obs(obs, prev_obs_tensor=prev_obs, num_active_nodes=5)
    features = sysid.get_features(num_active_nodes=5)

    _check("feature_dim == 5 * max_nodes", sysid.feature_dim == 25)
    _check("flat_features length == feature_dim", len(features.flat_features) == 25)
    _check("beta_hats length == max_nodes", len(features.beta_hats) == 5)
    _check("all confidences in [0,1]", all(0 <= c <= 1 for c in features.confidences))


# ===========================================================================
# Phase 6 — MoRL
# ===========================================================================

def check_reward_decomposer() -> None:
    print("\n--- Phase 6: RewardDecomposer ---")
    from src.train.ppo_morl import RewardDecomposer

    decomp = RewardDecomposer(weights={"health": 0.5, "economy": 0.3, "control": 0.1, "penalty": 0.1})

    # With no info components
    comps = decomp.decompose(1.5, {})
    _check("fallback: health == reward", abs(comps["health"] - 1.5) < 1e-9)
    _check("fallback: economy == 0", comps["economy"] == 0.0)

    # With explicit components
    info = {"reward_health": 0.8, "reward_economy": 0.4, "reward_control": 0.1, "reward_penalty": -0.1}
    comps2 = decomp.decompose(1.2, info)
    _check("health component == 0.8", abs(comps2["health"] - 0.8) < 1e-9)
    _check("economy component == 0.4", abs(comps2["economy"] - 0.4) < 1e-9)

    scalar = decomp.scalarize(comps2)
    _check("scalarized reward is finite", _finite(scalar))

    decomp.push(comps2)
    stats = decomp.end_episode()
    _check("ep_health_sum in stats", "ep_health_sum" in stats)
    _check("ep_economy_mean in stats", "ep_economy_mean" in stats)

    # Key fallback aliases for economy reward
    alias_comps = decomp.decompose(0.0, {"economy_reward": 0.25})
    _check("economy alias economy_reward works", abs(alias_comps["economy"] - 0.25) < 1e-9)
    alias_comps2 = decomp.decompose(0.0, {"economy": -0.4})
    _check("economy alias economy works", abs(alias_comps2["economy"] + 0.4) < 1e-9)

    # Economy derivation from score delta when explicit reward key is absent
    decomp_score = RewardDecomposer(weights={"health": 0.5, "economy": 0.3, "control": 0.1, "penalty": 0.1})
    d0 = decomp_score.decompose(0.0, {"global_economic_score": 0.70})
    d1 = decomp_score.decompose(0.0, {"global_economic_score": 0.74})
    _check("first economy score delta is zero", abs(d0["economy"]) < 1e-9)
    _check("economy score delta becomes non-zero", abs(d1["economy"]) > 1e-9)

    # No component keys at all -> full reward to health fallback
    decomp_fallback = RewardDecomposer(weights={"health": 0.5, "economy": 0.3, "control": 0.1, "penalty": 0.1})
    fallback = decomp_fallback.decompose(1.25, {"foo": "bar"})
    _check("absent-key fallback health == reward", abs(fallback["health"] - 1.25) < 1e-9)
    _check("absent-key fallback economy == 0", abs(fallback["economy"]) < 1e-9)


def check_morl_buffer() -> None:
    print("\n--- Phase 6: MorlRolloutBuffer ---")
    from src.train.ppo_morl import MorlRolloutBuffer

    buf = MorlRolloutBuffer()
    _check("empty buffer len == 0", len(buf) == 0)
    buf.push([0.1] * 10, {"discrete": [0] * 5}, 1.0, 0.5, -0.3, False, {"health": 0.8})
    _check("buffer len == 1 after push", len(buf) == 1)
    buf.clear()
    _check("buffer len == 0 after clear", len(buf) == 0)


def check_morl_smoke_train() -> None:
    print("\n--- Phase 6: PPOMorl smoke train ---")
    from src.train.ppo_morl import PPOMorl

    cfg = {
        "env": {"task_name": "easy_localized_outbreak", "seed": 42, "max_nodes": 20},
        "ppo": {
            "total_timesteps": 10, "n_steps": 5, "batch_size": 5,
            "n_epochs": 1, "lr": 3e-4, "gamma": 0.99, "gae_lambda": 0.95,
            "clip_eps": 0.2, "entropy_coef": 0.01, "value_loss_coef": 0.5,
            "max_grad_norm": 0.5,
        },
        "morl": {"weights": {"health": 0.5, "economy": 0.3, "control": 0.1, "penalty": 0.1}},
        "model": {"hidden_dims": [64, 64]},
        "logging": {"log_interval": 1, "checkpoint_dir": "/tmp/smoke_morl", "checkpoint_interval": 9999},
    }
    trainer = PPOMorl(config=cfg)
    try:
        trainer.train()
        _check("PPOMorl.train() completed without error", True)
    except Exception as e:
        _check("PPOMorl.train() completed without error", False, str(e))


def check_adapter_reward_components() -> None:
    print("\n--- Phase 6: OpenEnvAdapter reward-component diagnostics ---")
    from src.env.openenv_adapter import OpenEnvAdapter

    adapter = OpenEnvAdapter(task_name="easy_localized_outbreak", seed=42, max_nodes=20)
    _obs, _info = adapter.reset(seed=42)

    seen_non_zero_economy = False
    seen_non_zero_health = False
    for _ in range(20):
        _obs, _reward, done, step_info = adapter.step([0] * adapter.num_nodes)
        _check("step_info has reward_health", "reward_health" in step_info)
        _check("step_info has reward_economy", "reward_economy" in step_info)
        _check("step_info has global_economic_score", "global_economic_score" in step_info)
        if abs(float(step_info.get("reward_economy", 0.0))) > 1e-9:
            seen_non_zero_economy = True
        if abs(float(step_info.get("reward_health", 0.0))) > 1e-9:
            seen_non_zero_health = True
        if done:
            break

    _check("adapter emits non-zero economy component within rollout", seen_non_zero_economy)
    _check("adapter emits non-zero health component within rollout", seen_non_zero_health)


# ===========================================================================
# Phase 7 — Evaluation Harness
# ===========================================================================

def check_invalid_action_rate_bounds() -> None:
    """Unit test: invalid_action_rate must stay in [0, 1] regardless of num_nodes.

    This guards against the bug where the rate was computed as
    ``invalid_action_count / total_steps`` instead of
    ``invalid_action_count / (total_steps * num_nodes)``, which caused
    Inv% to exceed 100 in multi-node tasks.
    """
    print("\n--- Phase 7 (regression): invalid_action_rate bounded in [0,1] ---")
    from src.env.openenv_adapter import OpenEnvAdapter

    adapter = OpenEnvAdapter(task_name="easy_localized_outbreak", seed=42, max_nodes=20)
    obs, _info = adapter.reset(seed=42)

    # Force worst-case invalid actions by using all-quarantine (most nodes not already
    # quarantined) or all-lift (most nodes not quarantined → invalid lift).
    # A list of ACTION_LIFT (2) for all nodes is almost always invalid at step 0.
    num_nodes = adapter.num_nodes
    all_lift_action = [2] * num_nodes  # lift on non-quarantined → invalid per node

    max_steps = 30
    for _ in range(max_steps):
        obs, _reward, done, step_info = adapter.step(all_lift_action)
        rate = step_info["invalid_action_rate"]
        _check(
            "invalid_action_rate <= 1.0 after step",
            rate <= 1.0,
            f"got {rate:.4f}",
        )
        _check(
            "invalid_action_rate >= 0.0 after step",
            rate >= 0.0,
            f"got {rate:.4f}",
        )
        if done:
            break

    # Final rate check
    final_rate = step_info["invalid_action_rate"]
    _check(
        "final invalid_action_rate in [0,1]",
        0.0 <= final_rate <= 1.0,
        f"got {final_rate:.4f}",
    )


def check_eval_harness() -> None:
    print("\n--- Phase 7: EvalHarness ---")
    from src.eval.scenario_runner import EvalHarness, aggregate_results

    cfg = {
        "env": {"task_name": "easy_localized_outbreak", "seed": 42, "max_nodes": 20},
        "model": {"hidden_dims": [64, 64], "policy_type": "baseline"},
        "eval": {
            "tasks": ["easy_localized_outbreak"],
            "seeds": [42],
            "n_episodes": 1,
            "deterministic": True,
        },
    }
    harness = EvalHarness(cfg)
    results = harness.run()
    _check("harness returned results", len(results) > 0)
    _check("result has ep_return", hasattr(results[0], "ep_return"))
    _check("result ep_return finite", _finite(results[0].ep_return))
    _check("result peak_infection in [0,1]", 0 <= results[0].peak_infection <= 1)

    # --- Inv% regression guard ---
    for r in results:
        _check(
            "invalid_action_rate in [0, 1]",
            0.0 <= r.invalid_action_rate <= 1.0,
            f"got {r.invalid_action_rate:.4f} for task={r.task_name} seed={r.seed}",
        )
    inv_pct = sum(r.invalid_action_rate for r in results) / max(len(results), 1) * 100
    _check(
        "mean Inv% <= 100",
        inv_pct <= 100.0,
        f"got {inv_pct:.2f}%",
    )

    agg = aggregate_results(results)
    _check("aggregate has return stats", "return" in agg)
    _check("aggregate return.mean finite", _finite(agg["return"]["mean"]))
    _check(
        "aggregate invalid_action_rate.max <= 1",
        agg["invalid_action_rate"]["max"] <= 1.0,
        f"got {agg['invalid_action_rate']['max']:.4f}",
    )

    # Test save to JSON
    import tempfile
    import os
    import json
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "results.json")
        harness.save_results(results, path)
        _check("JSON file created", os.path.isfile(path))
        with open(path) as f:
            data = json.load(f)
        _check("JSON has results key", "results" in data)


# ===========================================================================
# Phase 8 — Scripts
# ===========================================================================

def check_scripts_importable() -> None:
    print("\n--- Phase 8: Script imports ---")
    import importlib
    import importlib.util
    import ast

    for script_name in ["scripts.train", "scripts.eval", "scripts.inference", "scripts.run_eval_harness"]:
        path = _REPO_ROOT / script_name.replace(".", "/")
        path = path.with_suffix(".py")
        try:
            spec = importlib.util.spec_from_file_location(script_name, path)
            # Don't exec the module — just check it parses
            ast.parse(path.read_text())
            _check(f"{script_name} parses without syntax errors", True)
        except Exception as e:
            _check(f"{script_name} parses without syntax errors", False, str(e))


def check_inference_script() -> None:
    print("\n--- Phase 8: inference.py smoke run ---")
    import subprocess
    import sys as _sys
    result = subprocess.run(
        [_sys.executable, str(_REPO_ROOT / "scripts" / "inference.py"),
         "--config", str(_REPO_ROOT / "configs" / "baseline.yaml"),
         "--task", "easy", "--seed", "42"],
        capture_output=True, text=True, cwd=str(_REPO_ROOT),
    )
    _check(
        "inference.py exits with code 0",
        result.returncode == 0,
        result.stderr[-500:] if result.returncode != 0 else "",
    )


# ===========================================================================
# Runner
# ===========================================================================

if __name__ == "__main__":
    print("=" * 60)
    print(" Smoke tests: Phases 3–8")
    print("=" * 60)

    check_st_encoder()
    check_st_actor_critic()
    check_belief_state()
    check_lag_aware_adapter()
    check_system_id()
    check_system_identifier()
    check_reward_decomposer()
    check_morl_buffer()
    check_morl_smoke_train()
    check_adapter_reward_components()
    check_invalid_action_rate_bounds()
    check_eval_harness()
    check_scripts_importable()
    check_inference_script()

    print()
    print("=" * 60)
    if _errors:
        print(f"RESULT: {len(_errors)} check(s) FAILED:")
        for e in _errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("RESULT: all checks PASSED")
