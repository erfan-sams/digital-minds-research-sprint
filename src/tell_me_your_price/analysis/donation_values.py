from __future__ import annotations

import warnings
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

from .common import bh_adjust, estimate_d50, output_paths, wilson_interval


CURVE_KEYS = [
    "model_label",
    "temperature",
    "temperature_mode",
    "charity_frame",
    "outcome_id",
    "valence",
    "target_kind",
    "signed_value_multiplier",
]


def prepare_money(responses: pd.DataFrame) -> pd.DataFrame:
    money = responses.loc[
        responses["phase"].eq("money_tier4")
        & responses["api_ok_flag"]
        & responses["parse_ok_flag"]
    ].copy()
    money["prefer_target"] = money["parsed_choice"].eq(money["target_side"]).astype(int)
    money["target_is_A"] = money["target_side"].eq("A").astype(int)
    money["log10_donation"] = np.log10(money["donation_usd"].astype(float))
    return money


def estimate_donation_values(
    responses: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    money = prepare_money(responses)
    expected_amounts = len(config["elicitation"]["donation_amounts_usd"])
    cells = (
        money.groupby(CURVE_KEYS + ["donation_usd"], as_index=False, dropna=False)
        .agg(p_prefer_target=("prefer_target", "mean"), n=("prefer_target", "size"))
    )

    rows = []
    for keys, group in money.groupby(CURVE_KEYS, dropna=False, sort=True):
        X = sm.add_constant(
            group[["log10_donation", "target_is_A"]].astype(float), has_constant="add"
        )
        slope = standard_error = p_negative = np.nan
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = sm.GLM(
                    group["prefer_target"].astype(float), X, family=sm.families.Binomial()
                ).fit(cov_type="HC3", maxiter=200)
            slope = float(fit.params["log10_donation"])
            standard_error = float(fit.bse["log10_donation"])
            p_negative = float(norm.cdf(slope / standard_error)) if standard_error > 0 else np.nan
        except Exception:
            pass

        cell_group = cells
        for column, value in zip(CURVE_KEYS, keys):
            cell_group = cell_group.loc[
                cell_group[column].isna() if pd.isna(value) else cell_group[column].eq(value)
            ]
        cell_group = cell_group.sort_values("donation_usd")
        if len(cell_group) == expected_amounts:
            descriptives = estimate_d50(
                cell_group["donation_usd"].to_numpy(float),
                cell_group["p_prefer_target"].to_numpy(float),
                cell_group["n"].to_numpy(float),
            )
        else:
            descriptives = {
                "raw_reversal_count": np.nan,
                "strict_raw_monotonicity": False,
                "isotonic_r2": np.nan,
                "d50_status": "incomplete_grid",
                "d50_usd": np.nan,
                "d50_lower_usd": np.nan,
                "d50_upper_usd": np.nan,
            }
        rows.append(
            {
                **dict(zip(CURVE_KEYS, keys)),
                "n_valid_responses": len(group),
                "n_amounts": len(cell_group),
                "donation_slope": slope,
                "donation_slope_se_hc3": standard_error,
                "one_sided_p_negative": p_negative,
                **descriptives,
            }
        )

    d50 = pd.DataFrame(rows)
    d50["q_bh"] = np.nan
    correction_keys = ["model_label", "temperature", "temperature_mode", "charity_frame"]
    for _, indices in d50.groupby(correction_keys, dropna=False).groups.items():
        indices = list(indices)
        d50.loc[indices, "q_bh"] = bh_adjust(d50.loc[indices, "one_sided_p_negative"])
    d50["significant_decreasing_trend"] = d50["donation_slope"].lt(0) & d50["q_bh"].lt(
        float(config["analysis"]["alpha"])
    )
    d50["signed_d50_usd"] = np.where(
        d50["d50_status"].eq("identified_within_grid"),
        d50["signed_value_multiplier"] * d50["d50_usd"],
        np.nan,
    )

    summary_rows = []
    summary_keys = [
        "model_label",
        "temperature",
        "temperature_mode",
        "charity_frame",
        "target_kind",
    ]
    for keys, group in d50.groupby(summary_keys, dropna=False):
        significant = int(group["significant_decreasing_trend"].sum())
        total = len(group)
        low, high = wilson_interval(significant, total)
        summary_rows.append(
            {
                **dict(zip(summary_keys, keys)),
                "outcome_curves": total,
                "significant_negative_slopes": significant,
                "significant_trend_rate": significant / total,
                "significant_wilson_low": low,
                "significant_wilson_high": high,
                "strict_monotonicity_rate": group["strict_raw_monotonicity"].mean(),
                "finite_d50_outcomes": group["d50_status"].eq("identified_within_grid").sum(),
                "below_grid_outcomes": group["d50_status"].eq("at_or_below_grid").sum(),
                "above_grid_outcomes": group["d50_status"].eq("above_grid").sum(),
            }
        )
    summary = pd.DataFrame(summary_rows)
    tables, _ = output_paths(config)
    d50.to_csv(tables / "donation_d50.csv", index=False)
    summary.to_csv(tables / "donation_monotonicity.csv", index=False)
    return money, cells, d50


def plot_frame_sensitivity(
    d50: pd.DataFrame,
    model_quality: pd.DataFrame,
    config: dict[str, Any],
) -> None:
    _, figures = output_paths(config)
    validation_temperature = float(config["analysis"]["validation_temperature"])
    eligible = set(model_quality.loc[model_quality["quality_eligible"], "model_label"])
    data = d50.loc[
        d50["model_label"].isin(eligible) & d50["temperature"].eq(validation_temperature)
    ]
    frames = list(config["elicitation"]["charity_frames"])
    frame_labels = {
        "wfp_emergency_food": "WFP emergency food",
        "generic_human_welfare": "Generic human welfare",
        "effective_human_welfare": "Effective human welfare",
        "generic_ai_welfare": "AI welfare",
    }
    models = sorted(data["model_label"].unique())
    width = 0.8 / len(frames)
    figure, axis = plt.subplots(figsize=(14, 6))
    x = np.arange(len(models))
    for index, frame in enumerate(frames):
        rates = []
        for model in models:
            group = data.loc[data["model_label"].eq(model) & data["charity_frame"].eq(frame)]
            rates.append(100 * group["significant_decreasing_trend"].mean())
        axis.bar(
            x + (index - (len(frames) - 1) / 2) * width,
            rates,
            width=width,
            label=frame_labels.get(frame, frame),
            edgecolor="black",
            linewidth=0.4,
        )
    axis.set_xticks(x)
    axis.set_xticklabels(models, rotation=35, ha="right")
    axis.set_ylim(0, 105)
    axis.set_ylabel("Outcomes with a BH-significant negative donation slope (%)")
    axis.set_xlabel("Model")
    axis.set_title("Donation sensitivity across charity frames")
    axis.legend(title="Donation beneficiary")
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(figures / "donation_sensitivity_by_frame.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_donation_choice_curves(cells: pd.DataFrame, config: dict[str, Any]) -> None:
    _, figures = output_paths(config)
    temperature = float(config["analysis"]["validation_temperature"])
    data = cells.loc[cells["temperature"].eq(temperature)]
    models = sorted(data["model_label"].unique())
    columns, rows = 2, int(np.ceil(len(models) / 2))
    figure, axes = plt.subplots(rows, columns, figsize=(14, 4.3 * rows), squeeze=False)
    colors = {"positive_outcome": "#377eb8", "negative_prevention": "#ff7f00"}
    styles = {
        "wfp_emergency_food": "-",
        "generic_human_welfare": "--",
        "effective_human_welfare": ":",
        "generic_ai_welfare": "-.",
    }
    for axis, model in zip(axes.flat, models):
        group = data.loc[data["model_label"].eq(model)]
        for (target, frame), curve in group.groupby(["target_kind", "charity_frame"]):
            means = curve.groupby("donation_usd")["p_prefer_target"].mean()
            axis.plot(
                means.index,
                means.values,
                color=colors[target],
                linestyle=styles[frame],
                marker="o",
                markersize=3,
                label=f"{target.replace('_', ' ')} — {frame.replace('_', ' ')}",
            )
        axis.axhline(0.5, color="black", linestyle="--", linewidth=0.8)
        axis.set_xscale("log")
        axis.set_ylim(-0.02, 1.02)
        axis.set_title(model)
        axis.set_xlabel("Donation amount (USD, log scale)")
        axis.set_ylabel("P(choose outcome or prevention)")
        axis.grid(alpha=0.2)
    for axis in axes.flat[len(models) :]:
        axis.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2, fontsize=8)
    figure.suptitle("Donation choice curves by model and charity frame", y=1.01)
    figure.tight_layout()
    figure.savefig(figures / "donation_choice_curves.png", dpi=220, bbox_inches="tight")
    plt.close(figure)
