from __future__ import annotations

from typing import Any

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from models import EpidemicAction, EpidemicObservation, EpidemicState


class EpidemicContainmentClient(
    EnvClient[EpidemicAction, EpidemicObservation, EpidemicState]
):
    def _step_payload(self, action: EpidemicAction) -> dict[str, Any]:
        payload = action.model_dump()
        payload.pop("metadata", None)
        return payload

    def _parse_result(self, payload: dict[str, Any]) -> StepResult[EpidemicObservation]:
        obs_data = dict(payload.get("observation", {}))
        obs_data["reward"] = payload.get("reward")
        obs_data["done"] = payload.get("done", False)
        observation = EpidemicObservation.model_validate(obs_data)
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict[str, Any]) -> EpidemicState:
        return EpidemicState.model_validate(payload)
