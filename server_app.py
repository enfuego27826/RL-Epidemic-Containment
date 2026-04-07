from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from openenv.core.env_server import create_app
from pydantic import BaseModel, Field, ValidationError

from dashboard_ui import DASHBOARD_HTML
from env import EpidemicContainmentStrategyEnv
from env import OpenEnvEpidemicContainmentEnv
from llm_policy import (
    create_openai_client_from_env,
    llm_status_from_env,
    request_llm_action,
    sanitize_text,
)
from models import EpidemicAction, EpidemicObservation
from tasks import baseline_policy


app: FastAPI = create_app(
    OpenEnvEpidemicContainmentEnv,
    EpidemicAction,
    EpidemicObservation,
    env_name="epidemic_containment_strategy",
    max_concurrent_envs=1,
)

_dashboard_env = EpidemicContainmentStrategyEnv(task_name="easy_localized_outbreak", seed=42)
_dashboard_previous_llm_error: str | None = None


class DashboardResetRequest(BaseModel):
    task_name: str = Field(default="easy_localized_outbreak")
    seed: int = Field(default=42)


class DashboardStepRequest(BaseModel):
    interventions: list[dict[str, Any]] = Field(default_factory=list)


def _dashboard_payload(
    observation: EpidemicObservation,
    reward: float | None = None,
    done: bool | None = None,
    info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    state = _dashboard_env.state()
    evaluation = _dashboard_env.latest_evaluation()
    return {
        "observation": observation.model_dump(),
        "state": state.model_dump(),
        "evaluation": evaluation.model_dump(),
        "reward": reward if reward is not None else observation.reward,
        "done": done if done is not None else observation.done,
        "info": info or {},
        "llm": llm_status_from_env(),
    }


def _dashboard_current_observation() -> EpidemicObservation:
    state = _dashboard_env.state()
    return _dashboard_env._build_observation(
        reward=0.0,
        done=state.step_count >= state.max_steps,
    )


def _dashboard_action_payload(action: EpidemicAction | dict[str, Any]) -> EpidemicAction:
    try:
        return action if isinstance(action, EpidemicAction) else EpidemicAction.model_validate(action)
    except ValidationError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid action payload: {sanitize_text(str(exc))}",
        ) from exc


def _dashboard_blocked_step(
    decision_source: str,
    errors: list[str],
    dropped_action_descriptions: list[str],
    extra_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    info = {
        "decision_source": decision_source,
        "action_descriptions": [],
        "dropped_action_descriptions": dropped_action_descriptions,
        "sanitized_action": {"interventions": []},
        "action_blocked": True,
        "errors": errors,
    }
    if extra_info:
        info.update(extra_info)
    observation = _dashboard_current_observation()
    return _dashboard_payload(
        observation=observation,
        reward=0.0,
        done=observation.done,
        info=info,
    )


def _dashboard_step_with_guardrail(
    action: EpidemicAction | dict[str, Any],
    decision_source: str,
    extra_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalized_action = _dashboard_action_payload(action)
    preview = _dashboard_env.sanitize_action(normalized_action)
    if preview.blocked and preview.errors:
        return _dashboard_blocked_step(
            decision_source=decision_source,
            errors=preview.errors,
            dropped_action_descriptions=preview.dropped_action_descriptions,
            extra_info=extra_info,
        )

    observation, reward, done, info = _dashboard_env.step(normalized_action)
    info = dict(info)
    info["decision_source"] = decision_source
    if extra_info:
        info.update(extra_info)
    return _dashboard_payload(
        observation=observation,
        reward=reward,
        done=done,
        info=info,
    )


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard() -> str:
    return DASHBOARD_HTML


@app.post("/dashboard/api/reset")
def dashboard_reset(request: DashboardResetRequest) -> dict[str, Any]:
    global _dashboard_previous_llm_error
    observation = _dashboard_env.reset(seed=request.seed, task_name=request.task_name)
    _dashboard_previous_llm_error = None
    return _dashboard_payload(observation=observation, reward=0.0, done=False, info={})


@app.post("/dashboard/api/step")
def dashboard_step(request: DashboardStepRequest) -> dict[str, Any]:
    return _dashboard_step_with_guardrail(
        action={"interventions": request.interventions},
        decision_source="manual/no-op",
    )


@app.post("/dashboard/api/baseline-step")
def dashboard_baseline_step() -> dict[str, Any]:
    state = _dashboard_env.state()
    if state.step_count >= state.max_steps:
        observation = _dashboard_current_observation()
        return _dashboard_payload(observation=observation, reward=0.0, done=True, info={})
    observation = _dashboard_current_observation()
    action = baseline_policy(state.task_name, observation)
    return _dashboard_step_with_guardrail(
        action=action,
        decision_source="baseline",
    )


@app.post("/dashboard/api/llm-step")
def dashboard_llm_step() -> dict[str, Any]:
    global _dashboard_previous_llm_error

    state = _dashboard_env.state()
    if state.step_count >= state.max_steps:
        observation = _dashboard_current_observation()
        return _dashboard_payload(observation=observation, reward=0.0, done=True, info={})

    try:
        client, config = create_openai_client_from_env()
        observation = _dashboard_current_observation()
        action = request_llm_action(
            client=client,
            model_name=config.model_name,
            task_name=state.task_name,
            observation=observation,
            previous_error=_dashboard_previous_llm_error,
        )
    except Exception as exc:
        _dashboard_previous_llm_error = sanitize_text(str(exc))
        raise HTTPException(
            status_code=400,
            detail=f"LLM step failed: {sanitize_text(str(exc))}",
        ) from exc

    preview = _dashboard_env.sanitize_action(action)
    if preview.blocked and preview.errors:
        _dashboard_previous_llm_error = sanitize_text("; ".join(preview.errors))
        return _dashboard_blocked_step(
            decision_source="llm",
            errors=preview.errors,
            dropped_action_descriptions=preview.dropped_action_descriptions,
            extra_info={"llm_action": action.model_dump()},
        )

    payload = _dashboard_step_with_guardrail(
        action=action,
        decision_source="llm",
        extra_info={"llm_action": action.model_dump()},
    )
    env_errors = payload["info"].get("errors", [])
    _dashboard_previous_llm_error = (
        sanitize_text("; ".join(env_errors)) if env_errors else None
    )
    return payload


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
