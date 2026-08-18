from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG = Path("config/experiment.yaml")


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    root = config_path.parent.parent
    config["_config_path"] = config_path
    config["_project_root"] = root
    config["resolved_paths"] = {
        name: (root / value).resolve()
        for name, value in config["paths"].items()
    }
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    models = config.get("models", {})
    if not models:
        raise ValueError("Configuration contains no models.")

    model_ids = [item["id"] for item in models.values()]
    if len(model_ids) != len(set(model_ids)):
        raise ValueError("Model IDs must be unique.")

    amounts = config["elicitation"]["donation_amounts_usd"]
    if not amounts or any(float(amount) <= 0 for amount in amounts):
        raise ValueError("Donation amounts must be positive.")
    if list(map(float, amounts)) != sorted(map(float, amounts)):
        raise ValueError("Donation amounts must be strictly ordered.")

    frames = set(config["elicitation"]["charity_frames"])
    primary = config["analysis"]["primary_frame"]
    if primary not in frames:
        raise ValueError(f"Unknown primary charity frame: {primary}")


def serializable_config(config: dict[str, Any]) -> dict[str, Any]:
    output = deepcopy(config)
    output.pop("_config_path", None)
    output.pop("_project_root", None)
    output.pop("resolved_paths", None)
    return output

