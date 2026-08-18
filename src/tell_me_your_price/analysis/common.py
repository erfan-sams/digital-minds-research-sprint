from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm


def output_paths(config: dict[str, Any]) -> tuple[Path, Path]:
    root: Path = config["resolved_paths"]["results_dir"]
    tables, figures = root / "tables", root / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    return tables, figures


def as_bool(series: pd.Series) -> pd.Series:
    if str(series.dtype) in {"bool", "boolean"}:
        return series.astype("boolean")
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
        .astype("boolean")
    )


def load_responses(config: dict[str, Any]) -> pd.DataFrame:
    directory: Path = config["resolved_paths"]["collected_dir"]
    path = directory / "responses.csv"
    legacy = directory / "stage1_responses.csv"
    if not path.exists() and legacy.exists():
        path = legacy
    if not path.exists():
        raise FileNotFoundError(f"Missing collected responses: {path}")

    data = pd.read_csv(path, low_memory=False)
    for column in ["api_ok", "parse_ok", "provider_match", "reasoning_fallback_used"]:
        if column in data:
            data[column] = as_bool(data[column])
    data["api_ok_flag"] = data["api_ok"].fillna(False).astype(bool)
    data["parse_ok_flag"] = data["parse_ok"].fillna(False).astype(bool)
    data["temperature"] = pd.to_numeric(data["temperature"], errors="coerce")
    data["donation_usd"] = pd.to_numeric(data["donation_usd"], errors="coerce")
    data["hit_max_tokens"] = np.where(
        data["api_ok_flag"], data["finish_reason"].eq("length"), np.nan
    )
    return data


def bh_adjust(values: np.ndarray | list[float]) -> np.ndarray:
    p = np.asarray(values, dtype=float)
    output = np.full(len(p), np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    if not len(valid):
        return output
    order = np.argsort(p[valid])
    ranked = p[valid][order]
    adjusted_ranked = np.minimum.accumulate(
        (ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1]
    )[::-1]
    adjusted = np.empty(len(ranked))
    adjusted[order] = np.minimum(adjusted_ranked, 1)
    output[valid] = adjusted
    return output


def holm_adjust(values: np.ndarray | list[float]) -> np.ndarray:
    p = np.asarray(values, dtype=float)
    output = np.full(len(p), np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    if not len(valid):
        return output
    order = np.argsort(p[valid])
    ranked = p[valid][order]
    adjusted_ranked = np.maximum.accumulate(
        (len(ranked) - np.arange(len(ranked))) * ranked
    )
    adjusted = np.empty(len(ranked))
    adjusted[order] = np.minimum(adjusted_ranked, 1)
    output[valid] = adjusted
    return output


def wilson_interval(successes: int, total: int, alpha: float = 0.05) -> tuple[float, float]:
    if total == 0:
        return np.nan, np.nan
    z = norm.ppf(1 - alpha / 2)
    p = successes / total
    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    radius = z * math.sqrt(
        p * (1 - p) / total + z**2 / (4 * total**2)
    ) / denominator
    return center - radius, center + radius


def cochran_q(matrix: np.ndarray) -> tuple[float, float]:
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[1] < 2:
        return np.nan, np.nan
    row_sums, column_sums = values.sum(1), values.sum(0)
    total, k = values.sum(), values.shape[1]
    denominator = k * total - np.sum(row_sums**2)
    if denominator <= 0:
        return 0.0, 1.0
    statistic = (k - 1) * (k * np.sum(column_sums**2) - total**2) / denominator
    return float(statistic), float(chi2.sf(statistic, k - 1))


def decreasing_pava(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    blocks = []
    for index, (value, weight) in enumerate(zip(values, weights)):
        blocks.append([index, index, float(weight), float(value * weight)])
        while len(blocks) >= 2:
            previous = blocks[-2][3] / blocks[-2][2]
            current = blocks[-1][3] / blocks[-1][2]
            if previous >= current:
                break
            right, left = blocks.pop(), blocks.pop()
            blocks.append([left[0], right[1], left[2] + right[2], left[3] + right[3]])
    fitted = np.empty(len(values), dtype=float)
    for start, end, weight, total in blocks:
        fitted[start : end + 1] = total / weight
    return fitted


def estimate_d50(
    amounts: np.ndarray, probabilities: np.ndarray, weights: np.ndarray
) -> dict[str, Any]:
    order = np.argsort(amounts)
    amounts, probabilities, weights = amounts[order], probabilities[order], weights[order]
    fitted = decreasing_pava(probabilities, weights)
    reversals = int(np.sum(np.diff(probabilities) > 0))
    result = {
        "raw_reversal_count": reversals,
        "strict_raw_monotonicity": reversals == 0,
        "d50_status": None,
        "d50_usd": np.nan,
        "d50_lower_usd": np.nan,
        "d50_upper_usd": np.nan,
    }
    weighted_mean = np.average(probabilities, weights=weights)
    total = np.sum(weights * (probabilities - weighted_mean) ** 2)
    residual = np.sum(weights * (probabilities - fitted) ** 2)
    result["isotonic_r2"] = 1 - residual / total if total > 0 else np.nan

    if fitted[0] <= 0.5:
        result.update(d50_status="at_or_below_grid", d50_upper_usd=float(amounts[0]))
        return result
    if fitted[-1] > 0.5:
        result.update(d50_status="above_grid", d50_lower_usd=float(amounts[-1]))
        return result

    upper_index = int(np.flatnonzero(fitted <= 0.5)[0])
    lower_index = upper_index - 1
    p_high, p_low = fitted[lower_index], fitted[upper_index]
    fraction = 0.5 if p_high == p_low else (0.5 - p_high) / (p_low - p_high)
    low_amount, high_amount = amounts[lower_index], amounts[upper_index]
    log_d50 = np.log10(low_amount) + fraction * (
        np.log10(high_amount) - np.log10(low_amount)
    )
    result.update(
        d50_status="identified_within_grid",
        d50_usd=float(10**log_d50),
        d50_lower_usd=float(low_amount),
        d50_upper_usd=float(high_amount),
    )
    return result

