from __future__ import annotations

from typing import Any

import pandas as pd

from .common import output_paths


def analyze_quality(responses: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    thresholds = config["analysis"]
    quality = (
        responses.groupby(["model_label", "phase"], as_index=False)
        .agg(
            scheduled=("model_label", "size"),
            api_success_rate=("api_ok_flag", "mean"),
            parse_yield=("parse_ok_flag", "mean"),
            provider_match_rate=("provider_match", "mean"),
            length_finish_rate=("hit_max_tokens", "mean"),
        )
    )
    quality["conditional_parse_rate"] = quality["parse_yield"] / quality["api_success_rate"]
    quality["phase_quality_pass"] = (
        quality["api_success_rate"].ge(float(thresholds["minimum_api_success"]))
        & quality["conditional_parse_rate"].ge(float(thresholds["minimum_conditional_parse"]))
        & quality["length_finish_rate"].le(float(thresholds["maximum_length_finish"]))
    )
    model_quality = (
        quality.groupby("model_label", as_index=False)
        .agg(
            quality_eligible=("phase_quality_pass", "all"),
            minimum_api_success=("api_success_rate", "min"),
            minimum_conditional_parse=("conditional_parse_rate", "min"),
            maximum_length_finish=("length_finish_rate", "max"),
        )
    )
    quality = quality.merge(model_quality[["model_label", "quality_eligible"]], on="model_label")
    tables, _ = output_paths(config)
    quality.to_csv(tables / "operational_quality.csv", index=False)
    model_quality.to_csv(tables / "model_quality.csv", index=False)
    return quality, model_quality

