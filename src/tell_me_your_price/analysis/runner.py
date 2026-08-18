from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .common import load_responses, output_paths
from .cross_instrument import analyze_cross_instrument
from .direct_preferences import analyze_direct_preferences
from .donation_values import (
    estimate_donation_values,
    plot_donation_choice_curves,
    plot_frame_sensitivity,
)
from .outcome_table import create_outcome_table
from .quality import analyze_quality
from .temperature import run_temperature_analysis


def run_primary_analysis(config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    responses = load_responses(config)
    quality, model_quality = analyze_quality(responses, config)
    pairs, cycles, ranks = analyze_direct_preferences(responses, model_quality, config)
    money, cells, d50 = estimate_donation_values(responses, config)
    plot_frame_sensitivity(d50, model_quality, config)
    plot_donation_choice_curves(cells, config)
    prediction_pairs, validation = analyze_cross_instrument(
        pairs, ranks, d50, model_quality, config
    )
    return {
        "responses": responses,
        "quality": quality,
        "model_quality": model_quality,
        "direct_pairs": pairs,
        "direct_cycles": cycles,
        "donation_cells": cells,
        "donation_d50": d50,
        "prediction_pairs": prediction_pairs,
        "validation": validation,
    }


def run_temperature_stage(config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    responses = load_responses(config)
    _, model_quality = analyze_quality(responses, config)
    tables, _ = output_paths(config)
    path = tables / "donation_d50.csv"
    d50 = pd.read_csv(path) if path.exists() else estimate_donation_values(responses, config)[2]
    return run_temperature_analysis(responses, d50, model_quality, config)


def run_outcome_table_stage(config: dict[str, Any]) -> pd.DataFrame:
    tables, _ = output_paths(config)
    d50_path = tables / "donation_d50.csv"
    validation_path = tables / "cross_instrument_validation.csv"
    if not d50_path.exists() or not validation_path.exists():
        raise FileNotFoundError("Run --stage analyze before --stage outcome-table.")
    return create_outcome_table(
        pd.read_csv(d50_path), pd.read_csv(validation_path), config
    )


def run_all_analysis(config: dict[str, Any]) -> dict[str, Any]:
    primary = run_primary_analysis(config)
    temperature = run_temperature_analysis(
        primary["responses"], primary["donation_d50"], primary["model_quality"], config
    )
    table = create_outcome_table(primary["donation_d50"], primary["validation"], config)
    return {"primary": primary, "temperature": temperature, "outcome_table": table}
