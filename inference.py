from __future__ import annotations

import argparse
import os

from openai import OpenAI

from env import EpidemicContainmentStrategyEnv
from llm_policy import request_llm_action, sanitize_text
from models import format_action
from tasks import BENCHMARK_NAME, baseline_policy


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _combine_error(existing: str | None, detail: str | None) -> str | None:
    if not detail or detail == "null":
        return existing
    return detail if existing is None else f"{existing};{detail}"


def _resolve_client_config(args: argparse.Namespace) -> tuple[str, str, str, float]:
    api_base_url = os.getenv("API_BASE_URL")
    model_name = os.getenv("MODEL_NAME")
    hf_token = os.getenv("HF_TOKEN")

    if args.ollama:
        api_base_url = api_base_url or args.ollama_base_url
        model_name = model_name or args.ollama_model
        hf_token = hf_token or "ollama"
        timeout_s = args.request_timeout_s or 120.0
    else:
        timeout_s = args.request_timeout_s or 20.0

    if not api_base_url or not hf_token or not model_name:
        raise EnvironmentError(
            "API_BASE_URL, MODEL_NAME, and HF_TOKEN environment variables are required. "
            "For local Ollama, pass --ollama or set API_BASE_URL=http://localhost:11434/v1, "
            "MODEL_NAME=<ollama_model>, HF_TOKEN=ollama."
        )

    return api_base_url, model_name, hf_token, timeout_s


def _should_disable_llm(args: argparse.Namespace, exc: Exception) -> bool:
    name = exc.__class__.__name__
    text = sanitize_text(str(exc)).lower()
    if name in {"APIConnectionError", "APITimeoutError"}:
        return True
    if args.ollama and name == "NotFoundError" and 'model "' in text and "not found" in text:
        return True
    if args.ollama and (
        "connection" in text
        or "refused" in text
        or "timed out" in text
        or "failed to establish a new connection" in text
    ):
        return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        default="easy_localized_outbreak",
        choices=sorted(
            [
                "easy_localized_outbreak",
                "medium_multi_center_spread",
                "hard_asymptomatic_high_density",
            ]
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--ollama",
        action="store_true",
        help="Use local Ollama defaults while still calling through the openai Python client.",
    )
    parser.add_argument(
        "--ollama-model",
        default="llama3.2:3b",
        help="Default Ollama model to use when MODEL_NAME is not set.",
    )
    parser.add_argument(
        "--ollama-base-url",
        default="http://localhost:11434/v1",
        help="Default OpenAI-compatible base URL for Ollama when API_BASE_URL is not set.",
    )
    parser.add_argument(
        "--request-timeout-s",
        type=float,
        default=None,
        help="Optional override for the model request timeout in seconds.",
    )
    args = parser.parse_args()

    rewards: list[float] = []
    total_steps = 0
    final_score = 0.0
    final_success = False
    previous_error: str | None = None
    env: EpidemicContainmentStrategyEnv | None = None
    llm_enabled = True
    start_model_name = os.getenv("MODEL_NAME") or (
        args.ollama_model if args.ollama else "missing"
    )

    print(
        f"[START] task={args.task} env={BENCHMARK_NAME} model={sanitize_text(start_model_name)}"
    )

    try:
        api_base_url, model_name, hf_token, timeout_s = _resolve_client_config(args)
        client = OpenAI(
            base_url=api_base_url,
            api_key=hf_token,
            timeout=timeout_s,
            max_retries=1,
        )

        env = EpidemicContainmentStrategyEnv(task_name=args.task, seed=args.seed)
        observation = env.reset(seed=args.seed)
        done = False

        while not done:
            action_error: str | None = None
            action_source = "baseline"
            if llm_enabled:
                try:
                    action = request_llm_action(
                        client=client,
                        model_name=model_name,
                        task_name=args.task,
                        observation=observation,
                        previous_error=previous_error,
                    )
                    action_source = "llm"
                except Exception as exc:
                    if _should_disable_llm(args, exc):
                        llm_enabled = False
                    action = baseline_policy(args.task, observation)
                    action_error = (
                        f"llm_fallback:{exc.__class__.__name__}:{sanitize_text(str(exc))}"
                    )
            else:
                action = baseline_policy(args.task, observation)
                action_error = "llm_disabled"

            preview = env.sanitize_action(action)
            if action.interventions and preview.blocked:
                blocked_detail = sanitize_text("; ".join(preview.errors))
                action_error = _combine_error(
                    action_error,
                    f"{action_source}_blocked:{blocked_detail}",
                )
                if action_source == "llm":
                    fallback_action = baseline_policy(args.task, observation)
                    fallback_preview = env.sanitize_action(fallback_action)
                    if not fallback_preview.blocked:
                        action = fallback_action
                    else:
                        action = preview.action
                else:
                    action = preview.action

            observation, reward, done, info = env.step(action)
            rewards.append(reward)
            total_steps += 1

            env_errors = info.get("errors", [])
            combined_error = _combine_error(
                action_error,
                sanitize_text("; ".join(env_errors)) if env_errors else None,
            )

            previous_error = combined_error if combined_error != "null" else None
            print(
                "[STEP] "
                f"step={total_steps} "
                f"action={format_action(action)} "
                f"reward={reward:.2f} "
                f"done={_bool_text(done)} "
                f"error={sanitize_text(combined_error)}"
            )

        evaluation = env.latest_evaluation()
        final_score = evaluation.score
        final_success = evaluation.success

    except Exception:
        final_success = False
        if env is not None:
            try:
                final_score = env.latest_evaluation().score
            except Exception:
                final_score = 0.0
        else:
            final_score = 0.0

    finally:
        rewards_text = ",".join(f"{reward:.2f}" for reward in rewards)
        print(
            "[END] "
            f"success={_bool_text(final_success)} "
            f"steps={total_steps} "
            f"score={final_score:.2f} "
            f"rewards={rewards_text}"
        )


if __name__ == "__main__":
    main()
