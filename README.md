# Epidemic Containment Strategy

Epidemic Containment Strategy is an OpenEnv benchmark where an agent acts like a public-health command center. It must contain an outbreak on a travel graph while preserving economic activity, operating under hidden transmission rates and, in the hardest task, lagged infection reporting.

The environment is designed to feel operational rather than toy-like: quarantine shuts down inter-city transmission but drains local economies, vaccination consumes a finite budget, and the agent is judged on both short-horizon control and end-of-episode outcomes.

## Mathematical Model

Each city node tracks:

- `S`: susceptible population
- `I`: infected population
- `R`: recovered / immune population
- `economic_health`: local economy score in `[0, 1]`

At each step, infections evolve with a graph-coupled SIR update:

```text
lambda_i = beta_i * (I_i / N_i) + sum_j(edge_beta_ji * (I_j / N_j))
new_infections_i = S_i * (1 - exp(-lambda_i))
recoveries_i = gamma_i * I_i
```

Quarantine sets edge transmission involving that node to zero for the step, while vaccination moves people from `S` to `R` subject to budget. Non-quarantined cities recover economically over time, while infected or quarantined cities lose economic health.

## Observation Space

`EpidemicObservation` is a strict Pydantic model with:

- `benchmark`
- `task_name`
- `difficulty`
- `goal`
- `step_count`
- `max_steps`
- `reporting_lag_steps`
- `vaccine_budget`
- `global_economic_score`
- `reported_total_infection_rate`
- `nodes[]`

Each `nodes[]` item contains:

- `node_id`
- `population`
- `known_infection_rate`
- `economic_health`
- `is_quarantined`

The agent sees graph topology through task metadata and prompts, but edge transmission rates remain hidden and are only exposed through `state()`.

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

## OpenEnv Alignment

The server and manifest are aligned to the current OpenEnv interfaces and package layout:

- GitHub: https://github.com/meta-pytorch/OpenEnv
- PyPI: https://pypi.org/project/openenv-core/
