# RL Epidemic Containment Strategy — Phased Implementation Plan

## Objective

Implement a production-grade reinforcement learning system for graph-coupled epidemic containment
in OpenEnv-style settings with:

- Hidden transmission/recovery parameters
- Optional delayed observations (DOMDP)
- Hybrid action space (discrete quarantine/lift + continuous vaccination)
- Multi-objective rewards (health + economy)

This plan focuses on **delivery sequence**, **interfaces**, **validation gates**, and **risk
controls**.

---

## Guiding Principles

1. **Build vertical slices early**: get an end-to-end baseline quickly, then improve modules.
2. **Separate concerns**:
   - Environment adapters
   - State estimation
   - Policy architecture
   - Training/evaluation
3. **Always keep a strong baseline** for regression checks.
4. **Measure before sophistication**: each advanced module must beat prior phase KPIs.

---

## Target Architecture (Final State)

- **State Encoder**
  - Graph feature builder
  - Spatiotemporal encoder (GNN + GRU/LSTM)
  - Belief-state updater for lagged observations
- **System ID module**
  - Online transmission/recovery estimation (Bayesian + optional PINN-inspired constraints)
- **Policy**
  - Hybrid actor head:
    - Discrete: quarantine/lift/no-op
    - Continuous: vaccine allocation vector under budget
  - Critic(s):
    - Single scalar critic initially
    - Optional dual critics for MORL
- **Algorithm**
  - PPO (hybrid action compatible)
  - GAE, entropy regularization, clipping
- **Evaluation**
  - Deterministic + stochastic scenarios
  - Pareto-style health/economy tradeoff reporting

---

## Phase 0 — Project Setup & Reproducibility

### Goals
- Establish repository structure, experiment tracking, and deterministic training runs.

### Deliverables
- `README.md` (run instructions)
- `requirements.txt` / `pyproject.toml`
- Config system (YAML/TOML + CLI overrides)
- Seeding utility (Python/NumPy/Torch/env)
- Logging (TensorBoard/W&B optional)
- Checkpointing + resume

### Suggested Structure
- `src/env/` — OpenEnv adapters/wrappers
- `src/models/` — encoders, actor-critic, heads
- `src/train/` — PPO loop, rollout storage
- `src/eval/` — metrics, scenario suite
- `src/system_id/` — parameter estimation
- `configs/` — experiment configs
- `scripts/` — train/eval entrypoints
- `docs/` — design docs and reports

### Exit Criteria
- One command to launch train and eval
- Deterministic smoke test passes (same seed → same first N metrics)

---

## Phase 1 — Environment Contract & Baseline Agent (Fast Vertical Slice)

### Goals
- Implement a robust interface to observations/actions/rewards.
- Train a baseline PPO agent without advanced modules.

### Tasks
1. **Environment wrapper**
   - Normalize observation schema into tensors:
     - Node-level epidemiology features
     - Node-level economy features
     - Quarantine flags
     - Graph adjacency / edge attributes
   - Handle variable graph sizes (masking/padding or fixed max V)
2. **Action adapter (v0)**
   - Discrete simplification first (e.g., limited action templates)
   - Vaccination as coarse discrete bins (temporary baseline)
3. **Baseline model**
   - MLP or simple GNN without temporal memory
4. **Baseline PPO**
   - Standard PPO + GAE + clipping
5. **Metrics pipeline**
   - Episode return
   - Max infection per node
   - Economy-collapse events
   - Invalid action rate

### Exit Criteria
- Baseline policy trains stably and beats random policy on at least 3 core metrics.

---

## Phase 2 — Proper Hybrid Action Space (PAMDP)

### Goals
Replace coarse action approximation with true hybrid control.

### Tasks
1. **Discrete action head**
   - Per-node logits for:
     - no-op / quarantine / lift / vaccinate-target-candidate
2. **Continuous vaccination head**
   - Produce nonnegative vaccine allocation vector `v`
   - Enforce budget: `sum(v) <= B` via normalization/projection
3. **Action validity layer**
   - Mask impossible actions (e.g., lifting non-quarantined nodes)
4. **Log-prob composition**
   - Correct PPO objective for mixed action distributions
5. **Entropy terms**
   - Separate entropy coefficients for discrete and continuous heads

### Exit Criteria
- Hybrid PPO converges
- Invalid action rate significantly reduced
- Better health-economy tradeoff than Phase 1

---

## Phase 3 — Spatiotemporal Graph Encoder

### Goals
Capture diffusion structure + temporal momentum under partial observability.

### Tasks
1. **Spatial encoder**
   - GATv2/GraphConv layers over city graph
2. **Temporal encoder**
   - GRU/LSTM over per-node embeddings across recent K steps
3. **Global readout**
   - Attention/mean pooling to produce global context vector
4. **State assembly**
   - Concatenate:
     - global embedding
     - key global stats (infection burden, economy, budget)
5. **Ablations**
   - Compare MLP vs GNN vs GNN+RNN

### Exit Criteria
- GNN+temporal model outperforms prior phase under networked spread scenarios.

---

## Phase 4 — Delayed Observation Handling (Belief State / DOMDP)

### Goals
Make policy robust to reporting lag.

### Tasks
1. **History buffer**
   - Store lagged observations and executed actions
2. **Belief updater**
   - Roll forward latent state using learned transition model OR recurrent latent filter
3. **Lag-aware training curriculum**
   - Train on mixed lag levels (0…k)
4. **Consistency losses (optional)**
   - Penalize impossible compartment transitions

### Exit Criteria
- Performance degradation under lag is materially lower than lag-unaware model.

---

## Phase 5 — Online System Identification (Hidden Parameter Inference)

### Goals
Infer hidden `beta_i`, `beta_ji`, `gamma_i` online and inject into policy features.

### Tasks
1. **Estimator v1 (statistical)**
   - Rolling-window MLE/Bayesian updates for transmission/recovery proxies
2. **Uncertainty features**
   - Add posterior variance/confidence as risk indicators
3. **Estimator-policy fusion**
   - Concatenate inferred params to node embeddings
4. **PINN-inspired constraints (v2)**
   - Physics regularizer for mass conservation

### Exit Criteria
- Better control under hidden-parameter scenarios, especially early episode stages.
- Calibration sanity checks on inferred parameters.

---

## Phase 6 — Multi-Objective RL Stabilization

### Goals
Improve Pareto behavior and reduce reward hacking.

### Tasks
1. **Reward decomposition tracking**
   - Log each component separately (health reward, ruin penalties, invalid penalties)
2. **Scalarization tuning**
   - Sweep objective weights
3. **Dual-critic option**
   - Separate value heads for health and economy
4. **Constraint-aware training**
   - Penalty schedules / Lagrangian-style adjustments (optional)

### Exit Criteria
- Demonstrated Pareto frontier improvement across validation scenarios.
- Reduced pathological behaviors (oscillatory quarantines, economy crashes).

---

## Phase 7 — Robustness, Generalization, and Stress Testing

### Goals
Ensure policy reliability across unseen graphs and shocks.

### Tasks
1. **Domain randomization**
   - Graph topology, transmission profiles, lag, budget
2. **Stress scenarios**
   - Sudden cluster outbreak
   - High-centrality node super-spread event
   - Low-budget emergencies
3. **OOD evaluation**
   - Unseen graph sizes/topologies
4. **Safety checks**
   - Hard guards for invalid/unethical action patterns

### Exit Criteria
- Stable performance with bounded variance across randomized test suite.

---

## Phase 8 — Packaging, Inference Policy, and Documentation

### Goals
Make the solution usable and reproducible by others.

### Deliverables
- Trained model artifacts
- Inference runner script
- Eval report (tables + plots)
- "How it works" architecture doc
- Reproducibility guide (seeds, configs, hardware)

### Exit Criteria
- One-command eval on pretrained checkpoint
- Full documentation sufficient for third-party replication

---

## Implementation Order (Practical)

1. Phase 0
2. Phase 1
3. Phase 2
4. Phase 3
5. Phase 6 (basic reward tuning early)
6. Phase 4
7. Phase 5
8. Phase 7
9. Phase 8

> Rationale: get hybrid actions and graph learning working before expensive system-ID complexity;
> tune objectives continuously.

---

## Core KPIs (Track Every Phase)

- **Health**
  - Peak infection ratio
  - Area under infection curve
  - Catastrophic infection events (`I_i/N_i >= 0.8`)
- **Economy**
  - Mean economy score
  - Economy floor violations (`E_i <= epsilon`)
  - Quarantine-days per node
- **Control quality**
  - Invalid action rate
  - Vaccine budget utilization efficiency
  - Policy entropy / collapse indicators
- **Training**
  - Sample efficiency
  - Stability across seeds (mean ± std)

---

## Risk Register & Mitigations

1. **Reward hacking**
   - Mitigation: component-level monitoring + scenario-based eval, not return alone.
2. **Hybrid action instability**
   - Mitigation: action masks, separate entropy coefficients, clipped continuous outputs.
3. **Overfitting to one graph**
   - Mitigation: domain randomization and held-out topology suite.
4. **Lag sensitivity**
   - Mitigation: lag curriculum and recurrent belief module.
5. **Estimator drift**
   - Mitigation: bounded parameter ranges, uncertainty-aware features, fallback priors.

---

## Definition of Done (Program-Level)

The implementation is complete when:

- Hybrid PPO + ST-GNN + lag-aware belief module trains end-to-end.
- Hidden-parameter scenarios are handled with online estimation features.
- Policy demonstrates superior Pareto tradeoff vs baseline across held-out scenarios.
- Reproducible training/evaluation artifacts and docs are published in-repo.

---

## Immediate Next Step (Execution)

Create and commit these first artifacts:

1. `docs/PHASE_1_BASELINE_PLAN.md` (task breakdown with acceptance criteria)
2. `configs/baseline.yaml`
3. `scripts/train.py` + `scripts/eval.py` stubs
4. `src/env/openenv_adapter.py`
5. `src/train/ppo_baseline.py`

Then run a 24-hour baseline experiment sweep (3 seeds) and record KPI table.
