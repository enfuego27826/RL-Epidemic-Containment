# Phase 2 — Proper Hybrid Action Space (PAMDP): Delivery Plan

## 1. Overview & Goals

Phase 2 replaces the coarse discrete-only action approximation from Phase 1 with a
true **Parameterised Action MDP (PAMDP)** where the policy simultaneously outputs:

- **Discrete head** — a per-node categorical decision:
  `{0: no-op, 1: quarantine, 2: lift_quarantine, 3: vaccinate-candidate}`
- **Continuous head** — a non-negative vaccine allocation vector `v ∈ ℝ≥0^N`
  projected so that `sum(v) ≤ B` (vaccine budget).

The goal is a PPO training loop that is **correct under mixed distributions**, with
separate entropy coefficients for each head and an invalid-action masking layer that
prevents structurally impossible choices before sampling.

---

## 2. Scope & Non-Goals

| In scope | Out of scope |
|----------|-------------|
| Per-node discrete action head | ST-GNN encoder (Phase 3) |
| Continuous vaccine allocation head | Lagged-observation belief state (Phase 4) |
| Budget projection (simplex / renorm) | Online system-ID / PINN (Phase 5) |
| Invalid action masking | Multi-objective dual-critic (Phase 6) |
| PPO log-prob composition | Full PyTorch autograd port |
| Separate discrete/continuous entropy | Distributed / multi-env training |

---

## 3. Assumptions

1. The environment `EpidemicAction` accepts a list of `interventions` where each
   vaccinate entry carries an explicit `amount`.  The Phase 2 adapter distributes
   the continuous budget allocation as these amounts.
2. Vaccine budget is tracked by the environment; the adapter reads
   `engine.vaccine_budget` before each step to know the current available budget.
3. A node-level quarantine flag is available in the observation tensor (slot `[2]`
   per node) and is used to build the action mask.
4. The pure-Python compute graph does **not** support autograd — parameter updates
   remain placeholder scalars for logging.  A PyTorch port is planned for Phase 3.

---

## 4. Action Schema

### 4.1 Policy Output

```
HybridAction = {
    "discrete":    list[int]   # length max_nodes, each in {0,1,2,3}
    "continuous":  list[float] # length max_nodes, each >= 0, sum <= vaccine_budget
}
```

### 4.2 Discrete Action Codes

| Code | Meaning | Valid when |
|------|---------|-----------|
| 0 | no-op | always |
| 1 | quarantine | node not already quarantined |
| 2 | lift_quarantine | node is currently quarantined |
| 3 | vaccinate-candidate | continuous amount for this node > threshold |

### 4.3 Continuous Head

The actor outputs a raw vector `z ∈ ℝ^N`.  Budget projection is applied as:

```
v_raw = softplus(z)                  # non-negativity
v     = v_raw / max(sum(v_raw), 1) * B   # renormalise to sum <= B
```

`softplus(x) = log(1 + exp(x))` is smooth and strictly positive.

---

## 5. Invalid Action Masking Strategy

Masking is applied **before sampling**, not as a penalty:

1. **Build mask** from current observation:
   - If `is_quarantined[i] == 1` → disable action `1` (quarantine) for node `i`
   - If `is_quarantined[i] == 0` → disable action `2` (lift) for node `i`
2. **Apply mask** by setting corresponding logits to `-inf` before softmax.
3. **Enforce budget** during continuous allocation via projection (not masking).
4. Diagnostics counter `invalid_action_count` is incremented when a structurally
   invalid action reaches the environment step despite masking (defensive check).

---

## 6. PPO Objective for Mixed Distributions

The combined log-probability is:

```
log π(a | s) = Σ_i log Cat(d_i | logits_i)          # discrete head
             + Σ_i log LogNormal(v_i | μ_i, σ_i)    # continuous head (alt: Normal + clip)
```

In this Phase 2 scaffold we use:
- **Discrete**: categorical log-prob from masked softmax distribution.
- **Continuous**: treat allocation fractions as a Dirichlet-like sample; approximate
  log-prob with the log of the allocation fraction normalised by the budget.

The PPO ratio uses the combined log-prob:

```
ratio = exp(log_π_new - log_π_old)
```

where `log_π_old` is stored per rollout step as a scalar (sum over nodes and both heads).

### 6.1 Entropy

```
H_total = α_disc * H_discrete + α_cont * H_continuous
```

Both `α_disc` and `α_cont` are configurable in `configs/phase2_hybrid.yaml`.

---

## 7. Metrics & Acceptance Criteria

### Phase 2 Exit Gate

| Criterion | Target |
|-----------|--------|
| Smoke test passes (10 steps) | Zero crashes |
| Invalid action rate | < 5 % of steps |
| Budget constraint respected | `sum(v) <= B + 1e-6` at every step |
| Combined log-prob finite | No NaN/Inf over first 100 updates |
| Mean episode return | ≥ Phase 1 baseline on `easy_localized_outbreak` |

### Tracked KPIs (all phases)

- Peak infection ratio
- Mean economy score
- Invalid action rate (%)
- Vaccine budget utilisation efficiency (`sum(v) / B` average)
- Policy entropy (discrete + continuous)
- Training stability across 3 seeds

---

## 8. File Map

| File | Role |
|------|------|
| `src/models/hybrid_action.py` | HybridActionDist class: sample, log_prob, entropy |
| `src/env/action_masking.py` | build_mask(), project_budget(), validate_hybrid_action() |
| `src/env/openenv_adapter.py` | Updated to accept hybrid action dict |
| `src/train/ppo_baseline.py` | Updated rollout collection + PPO objective |
| `configs/phase2_hybrid.yaml` | Phase 2 hyperparameters |
| `scripts/train.py` | Unchanged — works with any config |
| `src/tests/smoke_phase2.py` | Lightweight smoke checks |

---

## 9. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| NaN log-probs from zero allocation | Medium | High | Clamp `v_i >= 1e-8` before log |
| Budget creep (floating-point rounding) | Low | Medium | Add `1e-6` tolerance in assertion |
| Discrete mask makes all actions invalid | Low | High | Always allow no-op (code 0) |
| Hybrid rollout buffer misaligned shapes | Medium | Medium | Store dict; validate shapes in buffer |
| Entropy collapse in continuous head | Medium | Medium | Separate entropy coef; log entropy per head |

---

## 10. Rollout Plan

| Step | Description |
|------|-------------|
| 1 | Implement `src/models/hybrid_action.py` |
| 2 | Implement `src/env/action_masking.py` |
| 3 | Update `src/env/openenv_adapter.py` |
| 4 | Update `src/train/ppo_baseline.py` |
| 5 | Add `configs/phase2_hybrid.yaml` |
| 6 | Add `src/tests/smoke_phase2.py` |
| 7 | Run smoke test end-to-end |
| 8 | Update docs/README |

---

## 11. Next Phase Hooks

- **Phase 3**: Replace MLP trunk in `HybridActorCritic` with ST-GNN encoder.
- **Phase 4**: Extend rollout buffer to store observation history for belief-state module.
- **Phase 5**: Inject system-ID features into the observation tensor fed to both heads.
