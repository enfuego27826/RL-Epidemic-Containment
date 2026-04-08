#!/usr/bin/env python3
"""Smoke tests for new features added in the RL quality improvement session.

Tests:
- TBLogger (with and without TensorBoard)
- PPOBaseline LR schedule and compute_advantages stats
- OpenEnvAdapter per-type invalid tracking and conformance assertion
- EncoderBase interface (MLPEncoder, STEncoderWrapper)
- build_encoder factory
- eval_gates (check_gates, generate_gate_report, BaselineComparator)
- Curriculum config loading

Usage
-----
    python src/tests/smoke_new_features.py

Exit code 0 = all checks passed, 1 = failures.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

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
# TBLogger
# ===========================================================================

def check_tb_logger() -> None:
    print("\n--- TBLogger ---")
    from src.train.tb_logger import TBLogger

    # Test with log_dir=None (disabled)
    tb = TBLogger(log_dir=None)
    _check("TBLogger(log_dir=None) creates without error", True)
    _check("is_active False when log_dir=None", not tb.is_active)

    # Test from_config (minimal config, TensorBoard may not be available)
    cfg = {
        "logging": {
            "checkpoint_dir": "checkpoints/test",
            "tensorboard_enabled": True,
        }
    }
    tb2 = TBLogger.from_config(cfg, run_tag="test")
    _check("from_config creates TBLogger", tb2 is not None)
    _check("from_config log_dir derived from checkpoint_dir",
           "test" in tb2.log_dir)

    # Test log_scalar is a no-op when inactive
    tb_noop = TBLogger(log_dir=None)
    try:
        tb_noop.log_scalar("test/val", 1.0, 0)
        tb_noop.log_scalars("test", {"a": 1.0, "b": 2.0}, 0)
        tb_noop.log_histogram("test/hist", [1.0, 2.0, 3.0], 0)
        tb_noop.log_text("test/text", "hello", 0)
        tb_noop.flush()
        tb_noop.close()
        _check("no-op TBLogger methods don't raise", True)
    except Exception as exc:
        _check("no-op TBLogger methods don't raise", False, str(exc))

    # Test repr
    r = repr(tb)
    _check("repr contains log_dir", "log_dir" in r or "TBLogger" in r)

    # Cleanup temp dir if it was created
    tb2.close()


# ===========================================================================
# PPOBaseline LR schedule + compute_advantages stats
# ===========================================================================

def check_ppo_lr_schedule() -> None:
    print("\n--- PPOBaseline LR schedule and compute_advantages ---")
    from src.train.ppo_baseline import PPOBaseline, RolloutBuffer

    # Minimal config
    cfg = {
        "env": {"task_name": "easy_localized_outbreak", "seed": 42, "max_nodes": 20},
        "ppo": {
            "total_timesteps": 10,
            "n_steps": 5,
            "batch_size": 5,
            "n_epochs": 1,
            "lr": 3e-4,
            "lr_schedule": "linear",
            "lr_final": 3e-5,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_eps": 0.2,
            "entropy_coef": 0.01,
            "value_loss_coef": 0.5,
            "max_grad_norm": 0.5,
        },
        "model": {"hidden_dims": [64, 64]},
        "logging": {
            "log_interval": 1,
            "checkpoint_dir": "/tmp/ckpt_test",
            "checkpoint_interval": 9999,
            "tensorboard_enabled": False,
        },
    }

    trainer = PPOBaseline(config=cfg)
    _check("PPOBaseline instantiation with lr_schedule='linear'", True)

    # Check LR computation
    trainer._global_step = 0
    lr_start = trainer._compute_lr()
    _check("LR at step 0 equals initial lr", abs(lr_start - 3e-4) < 1e-9)

    trainer._global_step = trainer.total_timesteps
    lr_end = trainer._compute_lr()
    _check("LR at total_timesteps equals lr_final",
           abs(lr_end - 3e-5) < 1e-8,
           f"got {lr_end}")

    trainer._global_step = 0  # reset

    # Test "none" schedule
    trainer.lr_schedule = "none"
    lr_none = trainer._compute_lr()
    _check("LR schedule 'none' returns constant lr", abs(lr_none - 3e-4) < 1e-9)

    # Test compute_advantages returns stats dict
    # Manually populate buffer with dummy data
    import random
    buf = trainer._buffer
    buf.clear()
    T = 10
    for _ in range(T):
        buf.obs.append([0.0] * trainer._adapter.obs_dim)
        buf.actions.append([0] * trainer.max_nodes)
        buf.log_probs.append([0.0] * trainer.max_nodes)
        buf.rewards.append(random.uniform(-1, 1))
        buf.dones.append(False)
        buf.values.append(random.uniform(-0.5, 0.5))

    stats = trainer.compute_advantages()
    _check("compute_advantages returns dict", isinstance(stats, dict))
    _check("stats has 'mean' key", "mean" in stats)
    _check("stats has 'std' key", "std" in stats)
    _check("stats has 'min' key", "min" in stats)
    _check("stats has 'max' key", "max" in stats)
    _check("stats has 'explained_variance' key", "explained_variance" in stats)
    _check("stats values are finite", all(_finite(v) for v in stats.values()))
    _check("buffer advantages normalised (mean≈0)",
           abs(sum(buf.advantages) / max(len(buf.advantages), 1)) < 0.1)

    # Test next_obs bootstrapping path (when last transition is non-terminal)
    for i in range(T):
        buf.rewards[i] = 0.0
        buf.values[i] = 0.0
        buf.dones[i] = False

    original_forward = trainer._policy.forward

    def _mock_forward(_obs_t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        device = next(trainer._policy.parameters()).device
        logits = torch.zeros(
            (trainer.max_nodes, trainer._policy.action_dim),
            dtype=torch.float32,
            device=device,
        )
        value = torch.tensor([1.0], dtype=torch.float32, device=device)
        return logits, value

    trainer._policy.forward = _mock_forward  # type: ignore[method-assign]
    trainer.compute_advantages(next_obs=[0.1] * trainer._adapter.obs_dim)
    trainer._policy.forward = original_forward

    _check(
        "next_obs bootstrap contributes positive return on final non-terminal step",
        len(buf.returns) > 0 and buf.returns[-1] > 0.5,
        f"got final return={buf.returns[-1] if buf.returns else 'empty'}",
    )


# ===========================================================================
# OpenEnvAdapter per-type invalid tracking + conformance
# ===========================================================================

def check_openenv_adapter() -> None:
    print("\n--- OpenEnvAdapter invalid-by-type + conformance ---")
    from src.env.openenv_adapter import OpenEnvAdapter

    # Conformance assertion
    try:
        OpenEnvAdapter.assert_openenv_conformance()
        _check("assert_openenv_conformance() passes", True)
    except AssertionError as e:
        _check("assert_openenv_conformance() passes", False, str(e))
    except ImportError as e:
        _check("assert_openenv_conformance() passes", False, str(e))

    # Per-type tracking
    adapter = OpenEnvAdapter(task_name="easy_localized_outbreak", seed=42, max_nodes=20)
    _check("adapter has _invalid_by_type", hasattr(adapter, "_invalid_by_type"))
    _check("_invalid_by_type has quarantine key", "quarantine" in adapter._invalid_by_type)
    _check("_invalid_by_type has lift key", "lift" in adapter._invalid_by_type)
    _check("_invalid_by_type has vaccinate key", "vaccinate" in adapter._invalid_by_type)

    obs, info = adapter.reset()
    _check("reset() clears _invalid_by_type",
           all(v == 0 for v in adapter._invalid_by_type.values()))

    # Step and check info dict has invalid_by_type
    action = [0] * 20  # all no-op
    obs2, reward, done, step_info = adapter.step(action)
    _check("step info has 'invalid_by_type'", "invalid_by_type" in step_info)
    _check("step info invalid_by_type is dict", isinstance(step_info["invalid_by_type"], dict))

    # Try a lift on a non-quarantined node → should increment lift invalid count
    adapter.reset()
    from src.env.openenv_adapter import ACTION_LIFT
    lift_action = [ACTION_LIFT] * 20  # lift_quarantine on all nodes (none quarantined)
    obs3, _, _, info3 = adapter.step(lift_action)
    _check("lift on non-quarantined increments lift invalid count",
           adapter._invalid_by_type["lift"] > 0)


# ===========================================================================
# EncoderBase interface
# ===========================================================================

def check_encoder_interface() -> None:
    print("\n--- EncoderBase / MLPEncoder / STEncoderWrapper ---")
    from src.models.encoder_interface import MLPEncoder, STEncoderWrapper, build_encoder, EncoderBase

    # MLPEncoder
    enc = MLPEncoder(obs_dim=84, hidden_dims=[64], output_dim=32, seed=0)
    _check("MLPEncoder is EncoderBase subclass", isinstance(enc, EncoderBase))
    _check("MLPEncoder.output_dim == 32", enc.output_dim == 32)

    obs = [0.1] * 84
    encoding = enc.encode(obs)
    _check("MLPEncoder.encode() returns list of length output_dim",
           len(encoding) == 32)
    _check("MLPEncoder.encode() values are finite",
           all(_finite(v) for v in encoding))

    enc.reset()  # no-op; should not raise
    _check("MLPEncoder.reset() is no-op (no exception)", True)

    # MLPEncoder with shorter obs (should pad)
    short_obs = [0.2] * 40
    enc2 = enc.encode(short_obs)
    _check("MLPEncoder pads short obs", len(enc2) == 32)

    # STEncoderWrapper
    st_enc = STEncoderWrapper(
        node_feature_dim=4,
        max_nodes=5,
        gcn_hidden_dim=8,
        gru_hidden_dim=8,
        global_context_dim=8,
        seed=1,
    )
    _check("STEncoderWrapper is EncoderBase subclass", isinstance(st_enc, EncoderBase))
    _check("STEncoderWrapper.output_dim > 0", st_enc.output_dim > 0)

    st_obs = [0.1, 0.9, 0.0, 0.2] * 5 + [1.0, 0.8, 0.1, 0.05]
    st_enc.reset()
    st_encoding = st_enc.encode(st_obs, num_active_nodes=4)
    _check("STEncoderWrapper.encode() returns list", isinstance(st_encoding, list))
    _check("STEncoderWrapper.encode() values are finite",
           all(_finite(v) for v in st_encoding))

    # build_encoder factory — MLP
    cfg_mlp = {
        "env": {"seed": 0, "max_nodes": 5},
        "model": {"encoder_type": "mlp", "hidden_dims": [32], "encoder_output_dim": 16},
    }
    built_mlp = build_encoder(cfg_mlp, obs_dim=20)
    _check("build_encoder('mlp') returns MLPEncoder", isinstance(built_mlp, MLPEncoder))

    # build_encoder factory — ST
    cfg_st = {
        "env": {"seed": 0, "max_nodes": 5},
        "model": {
            "encoder_type": "st",
            "gcn_hidden_dim": 8,
            "gru_hidden_dim": 8,
            "global_context_dim": 8,
        },
    }
    built_st = build_encoder(cfg_st, obs_dim=24)
    _check("build_encoder('st') returns STEncoderWrapper",
           isinstance(built_st, STEncoderWrapper))

    # build_encoder factory — unknown raises ValueError
    cfg_bad = {"env": {}, "model": {"encoder_type": "unknown_xyz"}}
    try:
        build_encoder(cfg_bad, obs_dim=10)
        _check("build_encoder unknown raises ValueError", False, "no exception raised")
    except ValueError:
        _check("build_encoder unknown raises ValueError", True)


# ===========================================================================
# Eval gates
# ===========================================================================

def check_eval_gates() -> None:
    print("\n--- eval_gates ---")
    from src.eval.eval_gates import (
        GATE_THRESHOLDS,
        check_gates,
        generate_gate_report,
        BaselineComparator,
    )

    # GATE_THRESHOLDS structure
    _check("GATE_THRESHOLDS has easy task", "easy_localized_outbreak" in GATE_THRESHOLDS)
    _check("GATE_THRESHOLDS has medium task", "medium_multi_center_spread" in GATE_THRESHOLDS)
    _check("GATE_THRESHOLDS has hard task", "hard_asymptomatic_high_density" in GATE_THRESHOLDS)

    # check_gates: all passing scenario
    passing = {
        "easy_localized_outbreak": {"peak_infection": 0.25, "economy": 0.90, "inv_pct": 3.0},
        "medium_multi_center_spread": {"peak_infection": 0.40, "economy": 0.82, "inv_pct": 6.0},
        "hard_asymptomatic_high_density": {"peak_infection": 0.55, "economy": 0.77, "inv_pct": 10.0},
    }
    gate_pass = check_gates(passing)
    _check("check_gates returns dict with 'all_passed'", "all_passed" in gate_pass)
    _check("check_gates: perfect results → all_passed=True", gate_pass["all_passed"])

    # check_gates: failing scenario (current baseline)
    failing = {
        "easy_localized_outbreak": {"peak_infection": 0.43, "economy": 0.87, "inv_pct": 13.3},
        "medium_multi_center_spread": {"peak_infection": 0.57, "economy": 0.80, "inv_pct": 20.9},
        "hard_asymptomatic_high_density": {"peak_infection": 0.71, "economy": 0.72, "inv_pct": 29.2},
    }
    gate_fail = check_gates(failing)
    _check("check_gates: baseline results → all_passed=False", not gate_fail["all_passed"])

    # Gate report is non-empty string
    report_pass = generate_gate_report(gate_pass)
    _check("generate_gate_report returns non-empty string",
           isinstance(report_pass, str) and len(report_pass) > 0)
    _check("report contains PASS", "PASS" in report_pass)

    report_fail = generate_gate_report(gate_fail)
    _check("failing report contains FAIL", "FAIL" in report_fail)

    # BaselineComparator
    comp = BaselineComparator()
    _check("BaselineComparator has random baseline",
           "random" in comp._baselines)

    comparison = comp.compare(
        {"easy_localized_outbreak": {"peak_infection": 0.25, "economy": 0.90, "inv_pct": 3.0}},
        task_name="easy_localized_outbreak",
    )
    _check("compare returns dict", isinstance(comparison, dict))
    _check("compare has 'random' key", "random" in comparison)
    _check("compare random deltas exist",
           "deltas" in comparison["random"])

    # check_gates_from_episode_results integration
    from src.eval.eval_gates import check_gates_from_episode_results

    class FakeResult:
        def __init__(self, task, peak, econ, inv):
            self.task_name = task
            self.peak_infection = peak
            self.mean_economy = econ
            self.invalid_action_rate = inv

    fake_results = [
        FakeResult("easy_localized_outbreak", 0.25, 0.90, 0.03),
        FakeResult("medium_multi_center_spread", 0.40, 0.82, 0.06),
    ]
    gate_ep = check_gates_from_episode_results(fake_results)
    _check("check_gates_from_episode_results returns dict", isinstance(gate_ep, dict))
    _check("check_gates_from_episode_results has all_passed key",
           "all_passed" in gate_ep)


# ===========================================================================
# Curriculum config
# ===========================================================================

def check_curriculum_config() -> None:
    print("\n--- Curriculum config ---")
    try:
        import yaml
        curriculum_path = _REPO_ROOT / "configs" / "curriculum.yaml"
        with open(curriculum_path) as f:
            cfg = yaml.safe_load(f)
        _check("configs/curriculum.yaml loads without error", True)
        _check("curriculum section present", "curriculum" in cfg)
        _check("curriculum.mode present", "mode" in cfg["curriculum"])
        _check("curriculum.phases is a list", isinstance(cfg["curriculum"]["phases"], list))
        _check("curriculum.phases has 3 entries", len(cfg["curriculum"]["phases"]) == 3)
        _check("domain_randomization section present", "domain_randomization" in cfg)
        _check("ppo.lr_schedule present", "lr_schedule" in cfg["ppo"])
        _check("logging.tensorboard_enabled present", "tensorboard_enabled" in cfg["logging"])
    except ImportError:
        print("  [SKIP] PyYAML not available — skipping YAML parse")
    except FileNotFoundError as exc:
        _check("configs/curriculum.yaml exists", False, str(exc))


# ===========================================================================
# Ablation configs
# ===========================================================================

def check_ablation_configs() -> None:
    print("\n--- Ablation configs ---")
    try:
        import yaml
        for name in ("ablation_mlp.yaml", "ablation_gnn_gru.yaml"):
            path = _REPO_ROOT / "configs" / name
            with open(path) as f:
                cfg = yaml.safe_load(f)
            _check(f"{name} loads without error", True)
            _check(f"{name} has model.encoder_type", "encoder_type" in cfg["model"])
    except ImportError:
        print("  [SKIP] PyYAML not available — skipping YAML parse")
    except FileNotFoundError as exc:
        _check("ablation config exists", False, str(exc))


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    print("=" * 60)
    print("  Smoke tests: new features (TBLogger, eval gates, encoder)")
    print("=" * 60)

    check_tb_logger()
    check_ppo_lr_schedule()
    check_openenv_adapter()
    check_encoder_interface()
    check_eval_gates()
    check_curriculum_config()
    check_ablation_configs()

    print()
    print("=" * 60)
    if _errors:
        print(f"RESULT: {len(_errors)} check(s) FAILED")
        for e in _errors:
            print(f"  ✗ {e}")
        sys.exit(1)
    else:
        print("RESULT: all checks PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()
