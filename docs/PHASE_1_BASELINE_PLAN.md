# Phase 1 — Baseline PPO Agent: Practical Delivery Plan

## Goal

Deliver a runnable PPO baseline that trains stably against the
`EpidemicContainmentStrategyEnv` and beats the random policy on at least
three core KPIs.  All implementation should be self-contained in the `src/`
and `scripts/` directories described in `IMPLEMENTATION_PLAN.md`.

---

## Milestones

| # | Milestone | Acceptance Criteria |
|---|-----------|---------------------|
| M1 | Scaffold compiles and smoke-runs | `python scripts/train.py --config configs/baseline.yaml` exits without error |
| M2 | Adapter contract locked | `OpenEnvAdapter.reset()` and `.step()` return correctly-shaped tensors |
| M3 | PPO rollout loop runs | Collects N steps, fills rollout buffer, runs backward pass |
| M4 | Training converges | Mean episode return improves over 500 episodes (3 seeds) |
| M5 | Beats random baseline | PPO ≥ random on peak-infection, economy score, invalid-action rate |
| M6 | Evaluation report | `scripts/eval.py` produces per-task score table matching README format |

---

## Practical Checklist

### Environment Adapter (`src/env/openenv_adapter.py`)

- [x] `OpenEnvAdapter` class wrapping `EpidemicContainmentStrategyEnv`
- [x] `reset()` returns flat observation tensor + info dict
- [x] `step(action_tensor)` maps tensor → `EpidemicAction`, calls env, returns
      `(obs_tensor, reward, done, info)`
- [x] Observation normalisation: infection rates, economy scores, quarantine flags
- [x] Action space documented with shape/dtype annotations
- [ ] Graph adjacency tensor construction (Phase 3 hook, TODO)
- [ ] Action mask generation (Phase 2 hook, TODO)
- [ ] Hybrid continuous vaccination head wiring (Phase 2 hook, TODO)

### Baseline Config (`configs/baseline.yaml`)

- [x] Task name + seed fields
- [x] PPO hyperparameters: `lr`, `gamma`, `gae_lambda`, `clip_eps`, `entropy_coef`,
      `value_loss_coef`, `max_grad_norm`, `n_envs`, `n_steps`, `batch_size`,
      `n_epochs`, `total_timesteps`
- [x] Network architecture dims
- [x] Logging and checkpoint paths

### PPO Baseline (`src/train/ppo_baseline.py`)

- [x] `RolloutBuffer` dataclass
- [x] `PPOBaseline` class with `collect_rollout()`, `compute_advantages()`,
      `optimize()`, `train()` methods
- [x] `ActorCritic` MLP placeholder in `src/models/actor_critic.py`
- [x] Advantage / GAE computation (placeholder returning zeros until model is
      wired)
- [x] Policy gradient + value loss optimisation step
- [x] Entropy regularisation term
- [ ] Full advantage computation with bootstrapped value (wire after M2)
- [ ] Gradient clipping + learning-rate schedule (wire during M3 tuning)

### Script Entry-Points

- [x] `scripts/train.py` — loads YAML config, sets seeds, calls `PPOBaseline.train()`
- [x] `scripts/eval.py` — loads YAML config + checkpoint, runs eval episodes,
      prints score table

### Package Init Files

- [x] `src/__init__.py`
- [x] `src/env/__init__.py`
- [x] `src/models/__init__.py`
- [x] `src/train/__init__.py`
- [x] `src/eval/__init__.py`

---

## Acceptance Criteria (Phase 1 Exit Gate)

1. **Smoke test**: `python scripts/train.py --config configs/baseline.yaml` runs
   10 steps without crashing.
2. **Tensor contract**: adapter returns observation tensor of shape
   `(num_nodes * NODE_FEATURE_DIM,)` and reward is a Python float.
3. **Training stability**: loss curves show no NaN/Inf in first 100 updates
   across 3 independent seeds.
4. **KPI comparison** (deterministic eval, seed 42):

   | Metric | Random | PPO Baseline | Pass? |
   |--------|--------|--------------|-------|
   | Peak infection rate | TBD | < random | ✅/❌ |
   | Mean economy score | TBD | > random | ✅/❌ |
   | Invalid action rate | TBD | < random | ✅/❌ |

   (Fill in after first training run.)

---

## Time Estimates

| Task | Estimate |
|------|----------|
| Adapter skeleton + tests | 2 h |
| Config + seed wiring | 1 h |
| PPO loop + buffer | 3 h |
| Actor-critic MLP | 1 h |
| Script entry-points | 1 h |
| Smoke testing + debugging | 2 h |
| KPI eval run (3 seeds) | 4 h (compute) |
| **Total** | **~14 h** |

---

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Obs tensor shape mismatch | Unit-test adapter shapes before wiring into PPO |
| Unstable value loss early | Clip value targets, start with small `value_loss_coef` |
| Episode too short for learning | Ensure `n_steps` ≤ task `max_steps`; use multiple envs |
| Invalid-action spikes | Add action mask as early as Phase 1 (cheap to add) |

---

## Next Phase Hooks (already stubbed)

- Graph adjacency tensor construction → Phase 3 ST-GNN
- Action mask generation → Phase 2 hybrid action space
- Hybrid continuous head → Phase 2 PAMDP
- Belief-state buffer → Phase 4 DOMDP
- System-ID features → Phase 5
