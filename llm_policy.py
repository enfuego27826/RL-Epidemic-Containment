from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from models import EpidemicAction
from tasks import get_task_definition


@dataclass(frozen=True)
class LLMRuntimeConfig:
    api_base_url: str
    model_name: str
    api_key: str
    timeout_s: float


def sanitize_text(value: str | None) -> str:
    if not value:
        return "null"
    compact = " ".join(value.replace("\n", " ").replace("\r", " ").split())
    return compact if compact else "null"


def extract_json_object(text: str) -> str:
    start = text.find("{")
    if start < 0:
        raise ValueError("Model response did not contain a JSON object.")

    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    raise ValueError("Model response contained an unterminated JSON object.")


def build_prompt(task_name: str, observation: Any, previous_error: str | None) -> str:
    task_definition = get_task_definition(task_name)
    quarantined_node_ids = [
        node.node_id for node in observation.nodes if getattr(node, "is_quarantined", False)
    ]
    available_node_ids = [node.node_id for node in observation.nodes]
    prompt_payload = {
        "task_context": task_definition.prompt_context(),
        "observation": observation.model_dump(),
        "action_schema": EpidemicAction.model_json_schema(),
        "instructions": [
            "Return exactly one JSON object.",
            "The JSON must validate against the action_schema.",
            "Use only node_id values that appear in observation.nodes.",
            "Choose at most one intervention per node in a step. Never repeat a node.",
            "Do not quarantine a node that is already quarantined.",
            "Do not lift quarantine for a node that is not currently quarantined.",
            "Do not request more total vaccination than the remaining vaccine_budget.",
            "If every candidate action looks invalid, uncertain, or redundant, return {\"interventions\":[]}.",
            "Use an empty interventions list when monitoring is better than acting.",
            "Do not explain your reasoning.",
        ],
        "valid_action_hints": {
            "available_node_ids": available_node_ids,
            "currently_quarantined_node_ids": quarantined_node_ids,
            "max_interventions_per_step": 3,
        },
    }
    if previous_error:
        prompt_payload["previous_error"] = previous_error
    return json.dumps(prompt_payload, separators=(",", ":"))


def request_llm_action(
    client: OpenAI,
    model_name: str,
    task_name: str,
    observation: Any,
    previous_error: str | None,
) -> EpidemicAction:
    response = client.chat.completions.create(
        model=model_name,
        temperature=0.1,
        max_tokens=350,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a public health control policy. Output only JSON that matches "
                    "the provided action schema. Prefer an empty interventions list over any "
                    "duplicate, contradictory, or state-invalid action."
                ),
            },
            {
                "role": "user",
                "content": build_prompt(task_name, observation, previous_error),
            },
        ],
    )
    content = response.choices[0].message.content or ""
    action_json = extract_json_object(content)
    return EpidemicAction.model_validate_json(action_json)


def load_llm_runtime_config_from_env(timeout_s: float | None = None) -> LLMRuntimeConfig:
    api_base_url = os.getenv("API_BASE_URL")
    model_name = os.getenv("MODEL_NAME")
    api_key = os.getenv("HF_TOKEN")

    if timeout_s is None:
        raw_timeout = os.getenv("REQUEST_TIMEOUT_S")
        timeout_s = float(raw_timeout) if raw_timeout else 45.0

    if not api_base_url or not model_name or not api_key:
        raise EnvironmentError(
            "Missing API_BASE_URL, MODEL_NAME, or HF_TOKEN in the server environment."
        )

    return LLMRuntimeConfig(
        api_base_url=api_base_url,
        model_name=model_name,
        api_key=api_key,
        timeout_s=timeout_s,
    )


def create_openai_client_from_env(timeout_s: float | None = None) -> tuple[OpenAI, LLMRuntimeConfig]:
    config = load_llm_runtime_config_from_env(timeout_s=timeout_s)
    client = OpenAI(
        base_url=config.api_base_url,
        api_key=config.api_key,
        timeout=config.timeout_s,
        max_retries=1,
    )
    return client, config


def llm_status_from_env() -> dict[str, Any]:
    api_base_url = os.getenv("API_BASE_URL")
    model_name = os.getenv("MODEL_NAME")
    api_key = os.getenv("HF_TOKEN")
    timeout_value = os.getenv("REQUEST_TIMEOUT_S")
    return {
        "configured": bool(api_base_url and model_name and api_key),
        "api_base_url": api_base_url or "",
        "model_name": model_name or "",
        "timeout_s": float(timeout_value) if timeout_value else 45.0,
    }
