# RL Epidemic Containment

Reinforcement-learning agent for the **Epidemic Containment Strategy** [OpenEnv](https://github.com/openenv) benchmark. The agent acts as a public-health command centre that must contain an outbreak on a travel graph while preserving economic activity — with hidden transmission rates and (optionally) lagged infection reports.

---

## Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.10+ |
| pip | latest |

No GPU is required. All core modules run on CPU with the Python standard library.

---

## Install

```bash
# Clone the repo
git clone https://github.com/enfuego27826/RL-Epidemic-Containment.git
cd RL-Epidemic-Containment

# Install Python dependencies
pip install -r requirements.txt

# Optional: install PyYAML for config support (strongly recommended)
pip install pyyaml
```

`requirements.txt` includes `openenv-core`, `openai`, `fastapi`, `uvicorn`, and `pydantic`. All RL components use the Python standard library (no PyTorch dependency).

---

## Quick Start

### Smoke Test (< 5 seconds)
```bash
python scripts/train.py --config configs/baseline.yaml --smoke-test
```

### Run Inference (one episode)
```bash
python scripts/inference.py --config configs/baseline.yaml --task easy
```

### Evaluate Policy
```bash
python scripts/eval.py --config configs/baseline.yaml
```

---

## Training

### Phase 1 — Baseline MLP PPO
```bash
python scripts/train.py --config configs/baseline.yaml
```

### Phase 2 — Hybrid Action Space (discrete + continuous vaccination)
```bash
python scripts/train.py --config configs/phase2_hybrid.yaml
```

### Phase 3 — Spatiotemporal GNN Encoder
```bash
python scripts/train.py --config configs/phase3_stgnn.yaml
```

### Phase 4 — Delayed Observations / Belief State
```bash
python scripts/train.py --config configs/phase4_delayed.yaml
```

### Phase 6 — Multi-objective RL
```bash
python scripts/train.py --config configs/phase6_morl.yaml
```

### Full Pipeline (Phases 3–6 combined)
```bash
python scripts/train.py --config configs/full_pipeline.yaml
```

#### Common CLI overrides
```bash
python scripts/train.py --config configs/baseline.yaml \
    --seed 1 \
    --task hard_asymptomatic_high_density \
    --smoke-test
```

---

## Evaluation

### Single task evaluation
```bash
python scripts/eval.py --config configs/baseline.yaml
python scripts/eval.py --config configs/baseline.yaml --task hard --n-episodes 5
```

### Robustness / generalization harness (Phase 7)
```bash
# All three tasks × 3 seeds × 3 episodes
python scripts/run_eval_harness.py --config configs/full_pipeline.yaml

# Specific tasks and seeds
python scripts/run_eval_harness.py --config configs/baseline.yaml \
    --tasks easy medium hard \
    --seeds 0 1 2 \
    --n-episodes 5

# Save results to file
python scripts/run_eval_harness.py --config configs/baseline.yaml \
    --output results/eval.json
python scripts/run_eval_harness.py --config configs/baseline.yaml \
    --output results/eval.csv
```

---

## Inference

Run a trained checkpoint on one episode with step-by-step output:

```bash
python scripts/inference.py --config configs/baseline.yaml --task easy
python scripts/inference.py --config configs/baseline.yaml --task hard --seed 7
python scripts/inference.py --config configs/baseline.yaml \
    --checkpoint checkpoints/baseline/checkpoint_final.txt
```

---

## Tasks

| Shorthand | Full name | Description |
|---|---|---|
| `easy` | `easy_localized_outbreak` | Single cluster, low density |
| `medium` | `medium_multi_center_spread` | Multi-city spread |
| `hard` | `hard_asymptomatic_high_density` | Lagged reporting, high density |

---

## Config Overview

All configs live in `configs/`. Key sections:

```yaml
env:
  task_name: "easy_localized_outbreak"  # task to train on
  seed: 42
  max_nodes: 20                          # node padding target

ppo:
  total_timesteps: 200000
  n_steps: 64
  lr: 3.0e-4
  gamma: 0.99
  clip_eps: 0.2

model:
  policy_type: "baseline"  # baseline | hybrid | st
  hidden_dims: [128, 128]
  # Phase 3 ST-GNN options:
  gcn_hidden_dim: 32
  gru_hidden_dim: 32
  global_context_dim: 32

hybrid:                    # Phase 2 — hybrid action space
  enabled: true
  entropy_coef_discrete: 0.01
  entropy_coef_continuous: 0.001

lag:                       # Phase 4 — delayed observations
  steps: 2
  history_len: 4
  mode: "concat"           # "concat" | "mean"

system_id:                 # Phase 5 — online system identification
  enabled: true
  window_size: 8

morl:                      # Phase 6 — multi-objective RL
  weights:
    health: 0.5
    economy: 0.3
    control: 0.1
    penalty: 0.1

eval:
  tasks: [easy_localized_outbreak, medium_multi_center_spread, hard_asymptomatic_high_density]
  seeds: [42, 43, 44]
  n_episodes: 5
  deterministic: true
```

### Config files

| File | Description |
|---|---|
| `configs/baseline.yaml` | Phase 1 MLP baseline |
| `configs/phase2_hybrid.yaml` | Phase 2 hybrid actions |
| `configs/phase3_stgnn.yaml` | Phase 3 ST-GNN encoder |
| `configs/phase4_delayed.yaml` | Phase 4 delayed observations |
| `configs/phase6_morl.yaml` | Phase 6 multi-objective RL |
| `configs/full_pipeline.yaml` | All phases combined |

---

## Expected Outputs

After training, checkpoints are saved to the directory specified in `logging.checkpoint_dir`:

```
checkpoints/
  baseline/
    checkpoint_final.txt    # training summary
  phase2/
    checkpoint_final.txt
  ...
```

The eval harness outputs a summary table:

```
================================================================================
                               EVALUATION SUMMARY
================================================================================
Task                                   N    Return  Peak Inf   Economy   Inv%
--------------------------------------------------------------------------------
easy_localized_outbreak                3    -3.72±0.1   0.40±0.05   0.86±0.01   0.0
medium_multi_center_spread             3    -8.14±0.3   0.61±0.08   0.74±0.02   0.2
hard_asymptomatic_high_density         3   -12.6±0.5   0.75±0.12   0.62±0.03   0.4
================================================================================
```

---

## Repository Structure

```
├── configs/                # experiment configs
│   ├── baseline.yaml
│   ├── phase2_hybrid.yaml
│   ├── phase3_stgnn.yaml
│   ├── phase4_delayed.yaml
│   ├── phase6_morl.yaml
│   └── full_pipeline.yaml
├── scripts/
│   ├── train.py            # training entry-point
│   ├── eval.py             # evaluation entry-point
│   ├── inference.py        # run one episode
│   └── run_eval_harness.py # multi-task/seed evaluation
├── src/
│   ├── env/
│   │   ├── openenv_adapter.py    # env wrapper + normalisation
│   │   ├── action_masking.py     # invalid-action masks
│   │   └── belief_state.py       # Phase 4: lag + history buffer
│   ├── models/
│   │   ├── actor_critic.py       # Phase 1/2 MLP actor-critic
│   │   ├── hybrid_action.py      # Phase 2 hybrid action dist
│   │   └── st_encoder.py         # Phase 3 ST-GNN + GRU encoder
│   ├── system_id/
│   │   └── estimator.py          # Phase 5 rolling beta/gamma estimator
│   ├── train/
│   │   ├── ppo_baseline.py       # Phase 1/2 PPO training loop
│   │   └── ppo_morl.py           # Phase 6 multi-objective PPO
│   ├── eval/
│   │   └── scenario_runner.py    # Phase 7 evaluation harness
│   └── tests/
│       ├── smoke_phase2.py       # Phase 2 smoke tests
│       └── smoke_phases3to8.py   # Phases 3–8 smoke tests
├── env.py                  # EpidemicContainmentStrategyEnv
├── engine.py               # GraphEpidemicEngine (SIR dynamics)
├── models.py               # Pydantic observation/action models
├── tasks.py                # task definitions and grading
└── requirements.txt
```

---

## Running Tests

```bash
# Phase 2 smoke tests
python src/tests/smoke_phase2.py

# Phases 3–8 smoke tests
python src/tests/smoke_phases3to8.py
```

Both scripts exit with code 0 on success and print `[PASS]` / `[FAIL]` per check.

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError: No module named 'openenv'` | Run `pip install openenv-core` |
| `ModuleNotFoundError: No module named 'yaml'` | Run `pip install pyyaml` |
| `AssertionError: Episode already completed` | The env was stepped after `done=True`; call `reset()` first |
| Low economy scores | Increase `morl.weights.economy` in the config |
| High invalid action rate | Ensure `hybrid.masking_enabled: true` in config |
| Training is slow | Reduce `ppo.total_timesteps` or use `--smoke-test` |
| Checkpoint not found warning | Pass `--checkpoint path/to/checkpoint.txt` or set `eval.checkpoint_path` in config |

---

## Mathematical Model

Each node tracks compartments S (susceptible), I (infected), R (recovered) and economic health:

```
lambda_i = beta_i * (I_i / N_i) + Σ_j(beta_ji * I_j / N_j)
new_infections_i = S_i * (1 − exp(−lambda_i))
recoveries_i     = gamma_i * I_i
```

Quarantine zeroes cross-node transmission from node *i*. Vaccination moves *S → R* subject to a finite budget. The reward balances infection control against economic preservation.

---

## Implementation Phases

| Phase | Component | Status |
|---|---|---|
| 1 | MLP baseline PPO | ✅ |
| 2 | Hybrid action space (PAMDP) | ✅ |
| 3 | Spatiotemporal GNN encoder | ✅ |
| 4 | Delayed observations / belief state | ✅ |
| 5 | Online system identification | ✅ |
| 6 | Multi-objective RL stabilization | ✅ |
| 7 | Robustness / generalization evaluation | ✅ |
| 8 | Packaging, scripts, docs | ✅ |


## Action Space

`EpidemicAction` is a strict Pydantic model with:

- `interventions[]`

`interventions[]` is a discriminated union of:

- `{"kind": "quarantine", "node_id": "..."}`
- `{"kind": "vaccinate", "node_id": "...", "amount": 80.0}`
- `{"kind": "lift_quarantine", "node_id": "..."}`

An empty `interventions` list is the supported no-op / monitor action.

## Internal State

`state()` returns exact internal state, including:

- hidden node transmission and recovery rates
- hidden edge transmission rates
- exact `S/I/R` counts
- quarantine streaks
- reward / event history
- grader-relevant metrics such as peak infection and unnecessary quarantine steps

## Reward Design

The environment uses dense per-step rewards instead of a sparse terminal signal.

- Positive shaping for reducing infection and preserving the economy
- `+1.0` when infections stabilize or fall while the economy is above `50%`
- Penalties when any node reaches `80%+` infection
- Penalties when quarantine drives a node economy to `0`
- Penalties for unnecessary quarantines and invalid actions

## Tasks

Exactly three tasks are implemented in [`tasks.py`](/d:/openenv/tasks.py).

### Easy: `easy_localized_outbreak`

- 5 nodes
- 1 seeded outbreak
- High vaccine budget
- Horizon: 10 steps
- Goal: keep total infection below `20%`
- Grader: `1.0` if successful, otherwise decays with peak infection rate

### Medium: `medium_multi_center_spread`

- 10 nodes
- 3 seeded outbreaks
- Moderate vaccine budget
- Horizon: 15 steps
- Goal: keep infections below `30%` and economy above `60%`
- Grader: weighted sum of containment and economic survival

### Hard: `hard_asymptomatic_high_density`

- 20 nodes
- Dense clustered routing
- Low vaccine budget
- Reported infections lag actual infections by 1 step
- Horizon: 20 steps
- Goal: maximize final score while avoiding unnecessary quarantines
- Grader: strict multi-objective geometric score across containment, economy, and quarantine efficiency

## Files

- [`engine.py`](/d:/openenv/engine.py): graph SIR + economy engine
- [`models.py`](/d:/openenv/models.py): Pydantic action, observation, and state schemas
- [`env.py`](/d:/openenv/env.py): local env API plus OpenEnv adapter
- [`tasks.py`](/d:/openenv/tasks.py): task definitions, graders, and rule-based baseline
- [`server_app.py`](/d:/openenv/server_app.py): FastAPI / OpenEnv server entrypoint
- [`inference.py`](/d:/openenv/inference.py): LLM evaluation script with strict stdout formatting

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the environment server locally:

```bash
uvicorn server_app:app --host 0.0.0.0 --port 8000
```

Open the visual dashboard:

```text
http://localhost:8000/dashboard
```

Smoke-test the OpenEnv reset endpoint:

```bash
curl -X POST http://localhost:8000/reset -H "content-type: application/json" -d "{}"
```

Run a local Python episode:

```python
from env import EpidemicContainmentStrategyEnv
from tasks import baseline_policy

env = EpidemicContainmentStrategyEnv("easy_localized_outbreak", seed=42)
obs = env.reset(seed=42)

done = False
while not done:
    action = baseline_policy("easy_localized_outbreak", obs)
    obs, reward, done, info = env.step(action)

print(env.latest_evaluation())
```

## Visual Dashboard

The server includes a lightweight browser dashboard at `/dashboard`.

It lets you:

- reset any of the three tasks
- step the environment manually
- queue multiple interventions before stepping
- run the built-in baseline policy one step at a time
- run a server-side LLM policy one step at a time
- autoplay the server-side LLM policy through the episode
- autoplay the baseline policy through the full episode
- compare reported infections with hidden actual infections
- inspect the travel graph, per-node economy, and recent trajectory history

This is meant for debugging and environment tuning, so it intentionally exposes both observable and hidden state to the human operator.

### Dashboard LLM Setup

The dashboard reads model configuration from the server process environment. Set these in PowerShell before starting `uvicorn`:

```powershell
$env:API_BASE_URL="https://router.huggingface.co/v1"
$env:MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
$env:HF_TOKEN="your_hf_token"
$env:REQUEST_TIMEOUT_S="45"
uvicorn server_app:app --host 0.0.0.0 --port 8000
```

Then open:

```text
http://localhost:8000/dashboard
```

The dashboard will show whether the server sees a valid LLM configuration, and the `LLM Step` / `Play LLM` buttons will use that server-side model configuration.

## Inference Script

`inference.py` uses the `openai` Python client and reads credentials from these exact environment variables:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`

Example:

```bash
export API_BASE_URL="https://your-openai-compatible-endpoint/v1"
export MODEL_NAME="your-model"
export HF_TOKEN="your-token"
python inference.py --task hard_asymptomatic_high_density
```

### Running With Ollama

You can run the benchmark locally with Ollama while still using the `openai` Python client.

1. Install Ollama and start the local server:

```bash
ollama serve
```

2. Pull a model, for example:

```bash
ollama pull llama3.2:3b
```

3. Run inference with the built-in Ollama defaults:

```bash
python inference.py --ollama --task medium_multi_center_spread
```

You can also set the required environment variables explicitly:

```bash
export API_BASE_URL="http://localhost:11434/v1"
export MODEL_NAME="llama3.2:3b"
export HF_TOKEN="ollama"
python inference.py --task medium_multi_center_spread
```

For Windows PowerShell:

```powershell
$env:API_BASE_URL="http://localhost:11434/v1"
$env:MODEL_NAME="llama3.2:3b"
$env:HF_TOKEN="ollama"
python inference.py --task medium_multi_center_spread
```

Notes:

- `HF_TOKEN` can be any non-empty string for Ollama; the local server ignores it.
- `--ollama-model` lets you swap models quickly, for example `--ollama-model qwen2.5:7b`.
- If Ollama is unavailable or the model response is invalid, the script falls back step-by-step to the built-in heuristic policy.

The script prints only:

- one `[START]` line
- one `[STEP]` line after each `env.step()`
- one `[END]` line even on failure

## Docker

Build:

```bash
docker build -t epidemic-containment-strategy:latest .
```

Run:

```bash
docker run --rm -p 8000:8000 epidemic-containment-strategy:latest
```

The container starts `uvicorn server_app:app` and is intended to be deployable as a Hugging Face Docker Space.

## Baseline Scores Achieved

Using the built-in rule-based policy in [`tasks.py`](/d:/openenv/tasks.py) with seed `42`:

- Easy: `1.00`
- Medium: `0.84`
- Hard: `0.35`

For comparison, a pure no-op policy scores approximately:

- Easy: `0.68`
- Medium: `0.78`
- Hard: `0.31`

These scores are deterministic for the current task definitions and are intended as a sanity-check baseline rather than a ceiling.

## RL Baseline Scaffold

An initial PPO baseline scaffold is included for training RL agents against this environment.
See [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) for the full phased roadmap,
[`docs/PHASE_1_BASELINE_PLAN.md`](docs/PHASE_1_BASELINE_PLAN.md) for Phase 1 milestones, and
[`docs/PHASE_2_HYBRID_ACTION_PLAN.md`](docs/PHASE_2_HYBRID_ACTION_PLAN.md) for Phase 2 design.

### Project Structure

```
src/
  env/
    openenv_adapter.py   # Environment adapter (supports Phase 1 discrete + Phase 2 hybrid)
    action_masking.py    # Phase 2: mask builders, budget projection, action validation
  models/
    actor_critic.py      # ActorCritic (Phase 1) + HybridActorCritic (Phase 2)
    hybrid_action.py     # Phase 2: HybridActionDist (discrete + continuous heads)
  train/
    ppo_baseline.py      # PPO training loop (Phase 1 & 2 compatible)
  tests/
    smoke_phase2.py      # Phase 2 lightweight smoke checks
  eval/                  # Evaluation utilities
configs/
  baseline.yaml          # Phase 1 PPO hyperparameters
  phase2_hybrid.yaml     # Phase 2 hybrid action hyperparameters
scripts/
  train.py               # Training entry-point
  eval.py                # Evaluation entry-point
docs/
  PHASE_1_BASELINE_PLAN.md
  PHASE_2_HYBRID_ACTION_PLAN.md
```

### Phase 1 — Discrete-Only Baseline

The Phase 1 scaffold uses a per-node categorical action space
`{0: no-op, 1: quarantine, 2: lift_quarantine, 3: vaccinate}` with a fixed
vaccine dose per step.

### Phase 2 — Hybrid Action Space (PAMDP)

Phase 2 replaces the coarse discrete approximation with a true hybrid action space:

- **Discrete head** — per-node `{no-op, quarantine, lift, vaccinate-candidate}` with
  invalid-action masking (e.g., prevents lifting non-quarantined nodes).
- **Continuous head** — per-node vaccine allocation vector, non-negative and projected
  to sum ≤ vaccine budget via L1 renormalisation.
- **HybridActionDist** — samples both heads, computes combined log-probability for PPO.
- **Separate entropy coefficients** — `entropy_coef_discrete` and `entropy_coef_continuous`
  control exploration independently.

#### Phase 2 Action Schema

```python
# Policy output (Phase 2)
hybrid_action = {
    "discrete":   [0, 1, 0, 3, ...],   # list[int], per-node
    "continuous": [0.0, 0.0, 5.2, ...], # list[float], vaccine allocation, sum <= budget
}
```

#### Enabling Phase 2

Set `hybrid.enabled: true` in the config (already set in `configs/phase2_hybrid.yaml`):

```yaml
hybrid:
  enabled: true
  entropy_coef_discrete: 0.01
  entropy_coef_continuous: 0.001
  masking_enabled: true
```

### Running the Baseline Scaffold

Train with Phase 1 discrete-only config:

```bash
python scripts/train.py --config configs/baseline.yaml
```

Train with Phase 2 hybrid action config:

```bash
python scripts/train.py --config configs/phase2_hybrid.yaml
```

Smoke-test Phase 2 (10 steps, no output files):

```bash
python scripts/train.py --config configs/phase2_hybrid.yaml --smoke-test
```

Run Phase 2 unit smoke checks:

```bash
python src/tests/smoke_phase2.py
```

Evaluate (uses randomly initialised weights until a trained checkpoint is available):

```bash
python scripts/eval.py --config configs/phase2_hybrid.yaml --n-episodes 5
```

Override task or seed at the CLI:

```bash
python scripts/train.py --config configs/phase2_hybrid.yaml --task medium_multi_center_spread --seed 1
```

---

## OpenEnv Alignment

The server and manifest are aligned to the current OpenEnv interfaces and package layout:

- GitHub: https://github.com/meta-pytorch/OpenEnv
- PyPI: https://pypi.org/project/openenv-core/
