from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_prompt_assets(config: dict[str, Any]) -> dict[str, Any]:
    paths = config["resolved_paths"]
    frames = yaml.safe_load(Path(paths["charity_frames"]).read_text(encoding="utf-8"))
    expected_frames = set(config["elicitation"]["charity_frames"])
    if set(frames) != expected_frames:
        raise ValueError("Prompt charity frames do not match the configuration.")
    return {
        "forced_choice": Path(paths["prompt_template"]).read_text(encoding="utf-8").strip(),
        "negative_prevention": Path(paths["negative_prevention_template"]).read_text(encoding="utf-8").strip(),
        "charity_frames": frames,
    }


def format_usd(amount: float) -> str:
    amount = float(amount)
    if amount.is_integer():
        return f"${int(amount):,}"
    return f"${amount:,.2f}".rstrip("0").rstrip(".")


def render_choice(template: str, option_a: str, option_b: str) -> str:
    return template.format(option_a=option_a, option_b=option_b)


def donation_option(assets: dict[str, Any], frame: str, amount: float) -> str:
    return assets["charity_frames"][frame].format(amount=format_usd(amount))


def valued_target(
    assets: dict[str, Any], statement: str, valence: str
) -> tuple[str, str, int]:
    if valence == "positive":
        return statement, "positive_outcome", 1
    if valence == "negative":
        return (
            assets["negative_prevention"].format(outcome=statement),
            "negative_prevention",
            -1,
        )
    raise ValueError(f"Unexpected valence: {valence}")

