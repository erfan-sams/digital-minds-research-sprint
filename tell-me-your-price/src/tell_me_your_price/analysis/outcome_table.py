from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .common import output_paths


def format_usd(value: float) -> str:
    if pd.isna(value):
        return "—"
    value = float(value)
    if value >= 1_000_000_000:
        return f"${value / 1_000_000_000:g}B"
    if value >= 1_000_000:
        return f"${value / 1_000_000:g}M"
    if value >= 1_000:
        return f"${value / 1_000:g}K"
    if value >= 1:
        return f"${value:,.0f}"
    return f"${value:.2f}".rstrip("0").rstrip(".")


def display_d50(row: pd.Series) -> str:
    if row["d50_status"] == "identified_within_grid":
        return (
            f"≈{format_usd(row['d50_usd'])} "
            f"[{format_usd(row['d50_lower_usd'])}–{format_usd(row['d50_upper_usd'])}]"
        )
    if row["d50_status"] == "at_or_below_grid":
        return f"≤{format_usd(row['d50_upper_usd'])}"
    if row["d50_status"] == "above_grid":
        return f">{format_usd(row['d50_lower_usd'])}"
    return "Not estimable"


def create_outcome_table(
    d50: pd.DataFrame, validation: pd.DataFrame, config: dict[str, Any]
) -> pd.DataFrame:
    models = validation.loc[
        validation["significant_conservative"].fillna(False), "model_label"
    ].tolist()
    if not models:
        raise RuntimeError("No model passed the conservative cross-instrument test.")

    metadata_path: Path = config["resolved_paths"]["selected_outcomes"]
    metadata = pd.DataFrame(json.loads(metadata_path.read_text(encoding="utf-8")))
    metadata["outcome_order"] = np.arange(len(metadata))
    frame = config["analysis"]["primary_frame"]
    temperature = float(config["analysis"]["validation_temperature"])
    data = d50.loc[
        d50["model_label"].isin(models)
        & d50["charity_frame"].eq(frame)
        & d50["temperature"].eq(temperature)
    ].merge(
        metadata[["outcome_id", "text", "valence", "category", "outcome_order"]],
        on="outcome_id",
        how="left",
        suffixes=("", "_metadata"),
    )
    data["display_d50"] = data.apply(display_d50, axis=1)
    data["valued_target"] = np.where(
        data["valence"].eq("negative"), "Prevention of: " + data["text"], data["text"]
    )
    wide = (
        data.pivot_table(
            index=["outcome_order", "valued_target", "valence", "category"],
            columns="model_label",
            values="display_d50",
            aggfunc="first",
        )
        .reset_index()
        .sort_values("outcome_order")
        .drop(columns="outcome_order")
        .rename(
            columns={
                "valued_target": "outcome_being_valued",
                "valence": "original_valence",
            }
        )
    )
    ordered = ["outcome_being_valued", "original_valence", "category", *models]
    wide = wide[[column for column in ordered if column in wide]]
    tables, _ = output_paths(config)
    wide.to_csv(tables / "outcome_donation_equivalents.csv", index=False)
    return wide

