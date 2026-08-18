from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timezone
from typing import Any

import requests


def get_api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key
    try:
        from google.colab import userdata

        key = userdata.get("OPENROUTER_API_KEY")
    except Exception:
        key = None
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not configured.")
    return key


def parse_strict_ab(text: str | None) -> str | None:
    if not text:
        return None
    match = re.fullmatch(r"\s*[\"']?([ABab])[\"']?[.\s]*", text)
    return match.group(1).upper() if match else None


def _headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}


def _binding_payload(
    config: dict[str, Any], model_label: str, model: dict[str, Any]
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model["id"],
        "messages": [
            {
                "role": "user",
                "content": f"Return exactly A. Provider-binding probe {config['experiment_id']}:{model_label}.",
            }
        ],
        "max_tokens": int(config["api"]["max_tokens"]),
        "provider": {"require_parameters": True},
    }
    if model["send_temperature"]:
        payload["temperature"] = float(config["analysis"]["validation_temperature"])
    if model["send_reasoning"]:
        payload["reasoning"] = dict(config["api"]["reasoning"])
    return payload


def discover_providers(config: dict[str, Any], api_key: str) -> dict[str, dict[str, Any]]:
    timeout = int(config["api"]["timeout_seconds"])
    provider_response = requests.get(
        config["api"]["providers_url"],
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=timeout,
    )
    provider_response.raise_for_status()
    provider_slugs = {
        item["name"].casefold(): item["slug"]
        for item in provider_response.json().get("data", [])
        if item.get("name") and item.get("slug")
    }

    bindings = {}
    for label, model in config["models"].items():
        response = requests.post(
            config["api"]["openrouter_url"],
            headers=_headers(api_key),
            json=_binding_payload(config, label, model),
            timeout=timeout,
        )
        if not response.ok:
            raise RuntimeError(
                f"Provider probe failed for {label}: HTTP {response.status_code}: {response.text[:1000]}"
            )
        data = response.json()
        provider_name = data.get("provider")
        provider_slug = provider_slugs.get((provider_name or "").casefold())
        if not provider_slug:
            raise RuntimeError(f"Could not resolve provider {provider_name!r} for {label}.")
        bindings[label] = {
            "model_label": label,
            "requested_model": model["id"],
            "provider_name": provider_name,
            "provider_slug": provider_slug,
            "probe_http_status": response.status_code,
            "probe_response_id": data.get("id"),
            "probe_resolved_model": data.get("model"),
        }
    return bindings


def explicitly_rejects_reasoning(response: requests.Response) -> bool:
    body = response.text.lower()
    return response.status_code in {400, 422} and "reasoning" in body and any(
        token in body
        for token in ["unsupported", "not supported", "invalid", "unrecognized", "unknown"]
    )


def call_model(
    row: dict[str, Any], config: dict[str, Any], api_key: str
) -> dict[str, Any]:
    started = time.perf_counter()
    result = dict(row)
    send_reasoning = row["reasoning_parameter_mode"] == "effort_none"
    send_temperature = row.get("temperature_requested") is not None
    result.update(
        {
            "requested_at_utc": datetime.now(timezone.utc).isoformat(),
            "http_status": None,
            "api_ok": False,
            "parse_ok": False,
            "parsed_choice": None,
            "raw_response": None,
            "response_id": None,
            "resolved_model": None,
            "provider": None,
            "provider_match": None,
            "finish_reason": None,
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "reported_cost": None,
            "http_attempts": 0,
            "reasoning_requested": send_reasoning,
            "reasoning_request": json.dumps(config["api"]["reasoning"], sort_keys=True) if send_reasoning else None,
            "reasoning_request_http_accepted": None,
            "reasoning_fallback_used": False,
            "reasoning_mode_used": "unsupported_parameter_omitted" if not send_reasoning else None,
            "reasoning_attempt_http_status": None,
            "reasoning_rejection_text": None,
            "reasoning_content_present": None,
            "temperature_parameter_sent": send_temperature,
            "error": None,
        }
    )

    payload: dict[str, Any] = {
        "model": row["requested_model"],
        "messages": [{"role": "user", "content": row["prompt"]}],
        "max_tokens": int(config["api"]["max_tokens"]),
        "provider": {
            "only": [row["frozen_provider_slug"]],
            "allow_fallbacks": False,
            "require_parameters": True,
        },
    }
    if send_temperature:
        payload["temperature"] = float(row["temperature_requested"])
    if send_reasoning:
        payload["reasoning"] = dict(config["api"]["reasoning"])

    try:
        response = requests.post(
            config["api"]["openrouter_url"],
            headers=_headers(api_key),
            json=payload,
            timeout=int(config["api"]["timeout_seconds"]),
        )
        result["http_attempts"] = 1
        result["reasoning_attempt_http_status"] = response.status_code

        if response.ok and send_reasoning:
            result["reasoning_request_http_accepted"] = True
            result["reasoning_mode_used"] = "effort_none_request_accepted"
        elif send_reasoning and explicitly_rejects_reasoning(response):
            result["reasoning_fallback_used"] = True
            result["reasoning_rejection_text"] = response.text[:2000]
            fallback = dict(payload)
            fallback.pop("reasoning", None)
            response = requests.post(
                config["api"]["openrouter_url"],
                headers=_headers(api_key),
                json=fallback,
                timeout=int(config["api"]["timeout_seconds"]),
            )
            result["http_attempts"] = 2
            result["reasoning_mode_used"] = "endpoint_default_after_fallback"
        elif send_reasoning:
            result["reasoning_mode_used"] = "reasoning_request_failed_no_fallback"

        result["http_status"] = response.status_code
        if not response.ok:
            result["error"] = response.text[:2000]
            return result

        data = response.json()
        choice = data["choices"][0]
        message = choice["message"]
        raw_text = message.get("content")
        usage = data.get("usage") or {}
        provider = data.get("provider")
        parsed = parse_strict_ab(raw_text)
        result.update(
            {
                "api_ok": True,
                "parse_ok": parsed is not None,
                "parsed_choice": parsed,
                "raw_response": raw_text,
                "response_id": data.get("id"),
                "resolved_model": data.get("model"),
                "provider": provider,
                "provider_match": bool(provider) and provider.casefold() == row["frozen_provider_name"].casefold(),
                "finish_reason": choice.get("finish_reason"),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
                "reported_cost": usage.get("cost"),
                "reasoning_content_present": bool(message.get("reasoning") or message.get("reasoning_details")),
            }
        )
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    finally:
        result["latency_seconds"] = time.perf_counter() - started
    return result

