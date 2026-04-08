# RL Epidemic Containment — Roadmap Checklist

This document maps the RL quality improvement roadmap to **concrete engineering tasks**,
acceptance criteria, and run commands.  Work through phases in order; later phases build
on earlier ones.

---

## Phase 1 — PPO Correctness & Stability Hardening ✅

**Goal:** ensure the training signal is clean before adding complexity.

### Tasks

- [x] **Advantage normalisation** — already in `compute_advantages()`;
  now returns statistics (mean, std, min, max, explained_variance) for TB.
- [x] **Gradient clipping** — `nn.utils.clip_grad_norm_` in `update_policy()`;
  grad_norm tracked and logged.
- [x] **Ratio / KL / clip-frac** — computed per batch, averaged over all epochs,
  returned in metrics dict.
- [x] **Learning-rate schedule** — `lr_schedule: "linear"` in YAML config:

  ```yaml
  ppo:
    lr: 3.0e-4
    lr_schedule: "linear"   # "none" | "linear"
    lr_final: 3.0e-5
  ```

- [x] **TensorBoard logging** — `src/train/tb_logger.py` with graceful fallback
  to Python logging when `tensorboard` is not installed.

### Acceptance Criteria

- `policy_loss` varies across updates (not stuck at 0).
- `entropy` decreases gradually as the policy improves.
- `approx_kl` stays mostly < 0.02; `clip_frac` < 0.1.
- `explained_variance` rises above 0.5 within first 50K steps.

### Run Commands

```bash
# Stable PPO baseline (Run A)
python scripts/train.py --config configs/baseline.yaml

# With linear LR schedule
python scripts/train.py --config configs/baseline.yaml
# (set lr_schedule: linear in configs/baseline.yaml)

# View TensorBoard
tensorboard --logdir runs/baseline
```

---

## Phase 2 — Invalid-Action Reduction ✅

**Goal:** drive Inv% below target thresholds.

### Targets
| Task   | Target Inv% |
|--------|-------------|
| Easy   | < 5%        |
| Medium | < 8%        |
| Hard   | < 12%       |

### Tasks

- [x] **Strict pre-sampling masking** — `build_mask()` in `src/env/action_masking.py`
  prevents invalid discrete actions before sampling.
- [x] **Per-action-type diagnostics** — `OpenEnvAdapter._invalid_by_type` tracks
  quarantine / lift / vaccinate invalid counts separately.
- [x] **TensorBoard env metrics** — `env/invalid_action_rate`,
  `env/invalid_quarantine`, `env/invalid_lift`, `env/invalid_vaccinate` logged
  per update.
- [ ] **Invalid-action penalty in reward** — configure `reward_invalid_penalty`
  in YAML (optional tuning knob):

  ```yaml
  reward:
    invalid_action_penalty: -0.02  # applied per invalid node decision
  ```

### Acceptance Criteria

- `env/invalid_action_rate` trend is downward over training.
- Per-type counts reveal which action type dominates — tune masking accordingly.

### Run Commands

```bash
# Check per-type invalid counts after eval
python scripts/run_eval_harness.py --config configs/baseline.yaml

# Watch invalid rates live
tensorboard --logdir runs/baseline
# → env/invalid_action_rate, env/invalid_quarantine, env/invalid_lift
```

---

## Phase 3 — Stronger Encoder (GNN + GRU) 🔲

**Goal:** replace flat MLP trunk with spatiotemporal encoder for better
representation of networked spread.

### Tasks

- [x] **Pluggable encoder interface** — `src/models/encoder_interface.py`
  defines `EncoderBase` ABC with `encode()` / `output_dim` / `reset()`.
- [x] **MLP baseline encoder** — `MLPEncoder` (reference/ablation).
- [x] **GNN + GRU encoder** — `STEncoderWrapper` wraps `STEncoder` from
  Phase 3 (`src/models/st_encoder.py`).
- [x] **Config switch** — `model.encoder_type: "mlp" | "st"` in YAML.
- [x] **Ablation configs** — `configs/ablation_mlp.yaml` vs
  `configs/ablation_gnn_gru.yaml`.
- [ ] **Integrate encoder into PPOBaseline** — wire `build_encoder()` output
  into the actor-critic forward pass (currently policy directly processes raw obs).
- [ ] **PyTorch-native GNN** — replace stdlib GCN with torch_geometric for
  gradient flow through spatial encoder (requires Phase 3 PyTorch port TODO).

### Acceptance Criteria

- Ablation: `st` encoder beats `mlp` encoder on Peak Inf and Economy on
  medium/hard tasks across ≥ 3 seeds.
- `explained_variance` improves with richer encoder.

### Run Commands

```bash
# MLP encoder ablation
python scripts/train.py --config configs/ablation_mlp.yaml
python scripts/run_eval_harness.py --config configs/ablation_mlp.yaml

# GNN+GRU encoder ablation
python scripts/train.py --config configs/ablation_gnn_gru.yaml
python scripts/run_eval_harness.py --config configs/ablation_gnn_gru.yaml

# Compare TensorBoard runs side-by-side
tensorboard --logdir runs/
```

---

## Phase 4 — Reward & Objective Shaping 🔲

**Goal:** better Pareto tradeoff between health and economy.

### Tasks

- [ ] Sweep `morl.weights.health` / `economy` / `control` — use
  `configs/phase6_morl.yaml` as base.
- [ ] Add mild penalty for churning quarantine status (`flip_penalty`).
- [ ] Track per-component rewards in TensorBoard:
  `reward/health`, `reward/economy`, `reward/control`, `reward/penalty`.
- [ ] Generate Pareto frontier plot (`scripts/visualize.py`).

### Acceptance Criteria

- Economy score does not drop below 0.70 even on hard tasks.
- Pareto frontier shows visible tradeoff curve (not single collapsed point).

### Run Commands

```bash
python scripts/train.py --config configs/phase6_morl.yaml
python scripts/visualize.py --results results/eval.json
```

---

## Phase 5 — Curriculum + Robustness Training 🔲

**Goal:** policy generalises to unseen difficulty levels and environmental conditions.

### Tasks

- [x] **Config-driven curriculum** — `configs/curriculum.yaml` supports:
  - `mode: "sequential"` (easy → medium → hard)
  - `mode: "mixed"` (weighted task sampling)
  - `mode: "adaptive"` (promote on gate pass)
- [x] **Domain randomization hooks** — `domain_randomization` section in
  `curriculum.yaml` defines ranges for transmission/reporting lag/compliance.
- [ ] **Implement curriculum training loop** — extend `scripts/train.py` to
  switch tasks according to `curriculum.mode`:
  - Sequential: count timesteps per phase; switch task at boundary.
  - Mixed: sample task each episode from `task_weights`.
- [ ] **Domain randomization wiring** — pass randomized parameters to
  `EpidemicContainmentStrategyEnv` at each episode reset.

### Acceptance Criteria

- Sequential curriculum model outperforms direct hard-task training by ≥ 10%
  on Peak Inf on hard task.
- Domain-randomized model has lower variance across seeds on hard task (std
  reduced by ≥ 20%).

### Run Commands

```bash
python scripts/train.py --config configs/curriculum.yaml
python scripts/run_eval_harness.py --config configs/curriculum.yaml
```

---

## Phase 6 — Evaluation Gates & Reporting ✅

**Goal:** automated pass/fail gates for every checkpoint, with baseline comparison.

### Thresholds

| Task   | Peak Inf | Economy | Inv%  |
|--------|----------|---------|-------|
| Easy   | < 0.30   | > 0.85  | < 5   |
| Medium | < 0.45   | > 0.80  | < 8   |
| Hard   | < 0.60   | > 0.75  | < 12  |

### Tasks

- [x] **Gate thresholds** — `GATE_THRESHOLDS` in `src/eval/eval_gates.py`.
- [x] **`check_gates()`** — evaluates a result dict against thresholds.
- [x] **`check_gates_from_episode_results()`** — EvalHarness integration.
- [x] **`BaselineComparator`** — compare vs random/heuristic/previous-best.
- [x] **`generate_gate_report()`** — concise PASS/FAIL summary string.
- [ ] **Integrate gates into eval harness CLI** — add `--check-gates` flag
  to `scripts/run_eval_harness.py`.
- [ ] **Save gate report artifact** — write to `results/gate_report.txt`.

### Acceptance Criteria

- `generate_gate_report()` produces a non-empty, valid string.
- Gate check correctly classifies current baseline as FAIL (before training).
- Trained policy (300K steps) passes easy gate; medium gate targeted for 500K.

### Run Commands

```bash
# Run eval harness and check gates
python scripts/run_eval_harness.py --config configs/baseline.yaml

# Programmatic gate check
python - <<'EOF'
from src.eval.eval_gates import check_gates, generate_gate_report
results = {
    "easy_localized_outbreak":       {"peak_infection": 0.43, "economy": 0.87, "inv_pct": 13.3},
    "medium_multi_center_spread":    {"peak_infection": 0.57, "economy": 0.80, "inv_pct": 20.9},
    "hard_asymptomatic_high_density":{"peak_infection": 0.71, "economy": 0.72, "inv_pct": 29.2},
}
print(generate_gate_report(check_gates(results)))
EOF
```

---

## Phase 7 — OpenEnv Conformance ✅

**Goal:** verify and document that the entire train/eval stack runs through
OpenEnv integration points; prevent accidental non-OpenEnv execution paths.

### Tasks

- [x] **OpenEnv conformance assertion** — `OpenEnvAdapter.assert_openenv_conformance()`
  verifies `EpidemicContainmentStrategyEnv` is importable from the `openenv`
  stack and properly exported.
- [x] **Adapter wraps OpenEnv env** — `OpenEnvAdapter.__init__()` instantiates
  `EpidemicContainmentStrategyEnv` directly; all step/reset calls go through it.
- [x] **Eval harness uses OpenEnvAdapter** — `EvalHarness._run_task_seed()` imports
  `OpenEnvAdapter` explicitly.
- [x] **Training uses OpenEnvAdapter** — `PPOBaseline.__init__()` imports from
  `src.env.openenv_adapter`.
- [ ] **Add conformance check to train script** — call
  `OpenEnvAdapter.assert_openenv_conformance()` at start of `scripts/train.py`
  so non-OpenEnv environments fail loudly.

### Run Commands

```bash
# Verify OpenEnv conformance
python - <<'EOF'
from src.env.openenv_adapter import OpenEnvAdapter
OpenEnvAdapter.assert_openenv_conformance()
print("OpenEnv conformance: PASS")
EOF
```

---

## Phase 8 — TensorBoard Diagnostics ✅

**Goal:** rich, actionable dashboards to guide every training decision.

### Logged Metrics (per update, every `log_interval` updates)

| Group         | Metric                  | What to watch                                |
|---------------|-------------------------|----------------------------------------------|
| `train/`      | `policy_loss`           | Should vary; flat at 0 = no learning         |
| `train/`      | `value_loss`            | Should decrease as critic improves           |
| `train/`      | `entropy`               | Should decrease slowly; too fast = collapse  |
| `train/`      | `approx_kl`             | Keep < 0.02; spikes = too aggressive update  |
| `train/`      | `clip_frac`             | Keep < 0.1; high = overstepping              |
| `train/`      | `grad_norm`             | Should be stable; spikes = bad batches       |
| `train/`      | `explained_variance`    | Should rise toward 1.0                       |
| `train/`      | `learning_rate`         | Verify schedule is working                   |
| `advantages/` | `mean`, `std`           | Mean ≈ 0, std ≈ 1 after normalization        |
| `advantages/` | `min`, `max`            | Watch for extreme values                     |
| `advantages/` | `histogram`             | Distribution shape; should look roughly Gaussian |
| `env/`        | `invalid_action_rate`   | Should decrease over training                |
| `env/`        | `invalid_quarantine`    | Per-type diagnostic                          |
| `env/`        | `invalid_lift`          | Per-type diagnostic                          |
| `env/`        | `invalid_vaccinate`     | Per-type diagnostic                          |

### Setup

```bash
# Install TensorBoard
pip install tensorboard

# Launch TensorBoard dashboard
tensorboard --logdir runs/

# Or for a specific run
tensorboard --logdir runs/baseline_seed42

# Open in browser: http://localhost:6006
```

---

## Quick-Start Run Matrix

| Run  | Config                         | Goal                              | Command                                                      |
|------|--------------------------------|-----------------------------------|--------------------------------------------------------------|
| A    | `configs/baseline.yaml`        | Stable PPO (no LR schedule)       | `python scripts/train.py --config configs/baseline.yaml`     |
| B    | `configs/baseline.yaml`        | Linear LR schedule                | Set `lr_schedule: linear` then train                         |
| C    | `configs/ablation_mlp.yaml`    | MLP encoder ablation              | `python scripts/train.py --config configs/ablation_mlp.yaml` |
| D    | `configs/ablation_gnn_gru.yaml`| GNN+GRU encoder ablation          | `python scripts/train.py --config configs/ablation_gnn_gru.yaml` |
| E    | `configs/curriculum.yaml`      | Easy→medium→hard curriculum       | `python scripts/train.py --config configs/curriculum.yaml`   |
| F    | `configs/full_pipeline.yaml`   | All phases combined               | `python scripts/train.py --config configs/full_pipeline.yaml`|

---

## Before/After Baseline Numbers

### Before (Phase 0 baseline — random-weight policy)

| Task   | Return     | Peak Inf | Economy | Inv%  |
|--------|------------|----------|---------|-------|
| Easy   | -5.56±0.40 | 0.43     | 0.87    | 13.3  |
| Medium | -18.22±4.10| 0.57     | 0.80    | 20.9  |
| Hard   | -34.00±12.78| 0.71    | 0.72    | 29.2  |

### Target (after Phase 1-3 training)

| Task   | Peak Inf | Economy | Inv%  |
|--------|----------|---------|-------|
| Easy   | < 0.30   | > 0.85  | < 5   |
| Medium | < 0.45   | > 0.80  | < 8   |
| Hard   | < 0.60   | > 0.75  | < 12  |

---

## Smoke Tests

```bash
# Phase 1-2 smoke tests
python src/tests/smoke_phase2.py

# Phase 3-8 smoke tests (includes eval harness, encoder, etc.)
python src/tests/smoke_phases3to8.py

# New feature smoke tests (TBLogger, eval gates, encoder interface)
python src/tests/smoke_new_features.py

# Full train smoke test (10 steps)
python scripts/train.py --config configs/baseline.yaml --smoke-test
python scripts/train.py --config configs/curriculum.yaml --smoke-test
```
