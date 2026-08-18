from __future__ import annotations

import itertools
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .prompts import donation_option, load_prompt_assets, render_choice, valued_target


def build_base_schedule(
    config: dict[str, Any], outcomes: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    assets = load_prompt_assets(config)
    settings = config["elicitation"]
    rows: list[dict[str, Any]] = []

    for left, right in itertools.combinations(outcomes, 2):
        for temperature, repetition, left_is_a in itertools.product(
            settings["direct_temperatures"],
            range(1, int(settings["direct_repetitions_per_order"]) + 1),
            [True, False],
        ):
            option_a, option_b = (
                (left["text"], right["text"])
                if left_is_a
                else (right["text"], left["text"])
            )
            rows.append(
                {
                    "phase": "direct_tier4",
                    "left_outcome_id": left["outcome_id"],
                    "right_outcome_id": right["outcome_id"],
                    "left_valence": left["valence"],
                    "right_valence": right["valence"],
                    "left_text": left["text"],
                    "right_text": right["text"],
                    "left_side": "A" if left_is_a else "B",
                    "right_side": "B" if left_is_a else "A",
                    "source_tier": int(config["selection"]["tier"]),
                    "temperature": float(temperature),
                    "temperature_role": "targeted_direct_validation",
                    "repetition": repetition,
                    "option_a": option_a,
                    "option_b": option_b,
                    "prompt": render_choice(assets["forced_choice"], option_a, option_b),
                }
            )

    for outcome in outcomes:
        target, target_kind, multiplier = valued_target(
            assets, outcome["text"], outcome["valence"]
        )
        for frame, frame_config in settings["charity_frames"].items():
            for amount, temperature, repetition, target_is_a in itertools.product(
                settings["donation_amounts_usd"],
                frame_config["temperatures"],
                range(1, int(settings["donation_repetitions_per_order"]) + 1),
                [True, False],
            ):
                donation = donation_option(assets, frame, float(amount))
                option_a, option_b = (target, donation) if target_is_a else (donation, target)
                rows.append(
                    {
                        "phase": "money_tier4",
                        "outcome_id": outcome["outcome_id"],
                        "valence": outcome["valence"],
                        "category": outcome["category"],
                        "identified_property": outcome["identified_property"],
                        "source_tier": int(config["selection"]["tier"]),
                        "outcome_text": outcome["text"],
                        "donation_usd": float(amount),
                        "charity_frame": frame,
                        "charity_frame_role": frame_config["role"],
                        "target_side": "A" if target_is_a else "B",
                        "donation_side": "B" if target_is_a else "A",
                        "target_kind": target_kind,
                        "signed_value_multiplier": multiplier,
                        "temperature": float(temperature),
                        "temperature_role": (
                            "direct_validation_match"
                            if float(temperature) == float(config["analysis"]["validation_temperature"])
                            else "temperature_sensitivity"
                        ),
                        "repetition": repetition,
                        "option_a": option_a,
                        "option_b": option_b,
                        "prompt": render_choice(assets["forced_choice"], option_a, option_b),
                    }
                )
    return rows


def build_schedule(
    config: dict[str, Any], outcomes: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    base = build_base_schedule(config, outcomes)
    schedule: list[dict[str, Any]] = []
    reference_temperature = float(config["analysis"]["validation_temperature"])

    for model_label, model in config["models"].items():
        for row in base:
            if not model["send_temperature"] and float(row["temperature"]) != reference_temperature:
                continue
            scheduled = {"model_label": model_label, "requested_model": model["id"], **row}
            scheduled["temperature_requested"] = (
                float(row["temperature"]) if model["send_temperature"] else None
            )
            scheduled["temperature_mode"] = (
                "explicit" if model["send_temperature"] else "endpoint_default"
            )
            scheduled["reasoning_parameter_mode"] = (
                "effort_none" if model["send_reasoning"] else "unsupported_omitted"
            )
            schedule.append(scheduled)

    random.Random(int(config["selection"]["schedule_seed"])).shuffle(schedule)
    for index, row in enumerate(schedule, start=1):
        row["request_index"] = index
        row["experiment_id"] = config["experiment_id"]
    return schedule


def schedule_counts(config: dict[str, Any], outcome_count: int) -> dict[str, int]:
    settings = config["elicitation"]
    direct = (
        math.comb(outcome_count, 2)
        * len(settings["direct_temperatures"])
        * int(settings["direct_repetitions_per_order"])
        * 2
    )
    per_model = {}
    for label, model in config["models"].items():
        frame_temperature_count = sum(
            len(frame["temperatures"]) if model["send_temperature"] else 1
            for frame in settings["charity_frames"].values()
        )
        donation = (
            outcome_count
            * len(settings["donation_amounts_usd"])
            * frame_temperature_count
            * int(settings["donation_repetitions_per_order"])
            * 2
        )
        per_model[label] = direct + donation
    return per_model


def save_schedule(config: dict[str, Any], schedule: list[dict[str, Any]]) -> Path:
    path: Path = config["resolved_paths"]["schedule"]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(schedule).to_csv(path, index=False)
    return path


def load_schedule(config: dict[str, Any]) -> list[dict[str, Any]]:
    path: Path = config["resolved_paths"]["schedule"]
    if not path.exists():
        raise FileNotFoundError(f"Run --stage prepare-data first: {path}")
    records = pd.read_csv(path, low_memory=False).to_dict("records")
    return [
        {key: (None if isinstance(value, float) and np.isnan(value) else value) for key, value in row.items()}
        for row in records
    ]

