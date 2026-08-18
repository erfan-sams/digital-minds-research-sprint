from __future__ import annotations

import warnings
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import friedmanchisquare

from .common import cochran_q, holm_adjust, output_paths, wilson_interval
from .donation_values import prepare_money


def _friedman(pivot: pd.DataFrame, temperatures: list[float]) -> tuple[int, float, float]:
    complete = pivot.reindex(columns=temperatures).dropna()
    if len(complete) < 3:
        return len(complete), np.nan, np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        statistic, p_value = friedmanchisquare(
            *[complete[value].to_numpy() for value in temperatures]
        )
    if not np.isfinite(p_value):
        statistic, p_value = 0.0, 1.0
    return len(complete), float(statistic), float(p_value)


def _eligible_temperature_data(
    responses: pd.DataFrame, model_quality: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, list[str], list[float]]:
    frame = config["analysis"]["primary_frame"]
    temperatures = sorted(
        map(float, config["elicitation"]["charity_frames"][frame]["temperatures"])
    )
    eligible = set(model_quality.loc[model_quality["quality_eligible"], "model_label"])
    raw = responses.loc[
        responses["phase"].eq("money_tier4")
        & responses["charity_frame"].eq(frame)
        & responses["temperature_mode"].eq("explicit")
        & responses["temperature"].isin(temperatures)
    ]
    complete_models = set(
        raw.groupby("model_label")["temperature"].nunique().loc[lambda x: x == len(temperatures)].index
    )
    models = sorted(eligible & complete_models)
    money = prepare_money(responses)
    money = money.loc[
        money["model_label"].isin(models)
        & money["charity_frame"].eq(frame)
        & money["temperature_mode"].eq("explicit")
        & money["temperature"].isin(temperatures)
    ].copy()
    return money, models, temperatures


def _curve_tests(money: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    rows = []
    for model in models:
        group = money.loc[money["model_label"].eq(model)].copy()
        group["temperature_label"] = group["temperature"].astype(str)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = smf.glm(
                    "prefer_target ~ C(outcome_id) + target_is_A + log10_donation * C(temperature_label)",
                    data=group,
                    family=sm.families.Binomial(),
                ).fit(
                    cov_type="cluster",
                    cov_kwds={"groups": group["outcome_id"]},
                    maxiter=200,
                )
            names = list(fit.params.index)

            def test(indices: list[int]) -> float:
                restriction = np.zeros((len(indices), len(names)))
                for row, index in enumerate(indices):
                    restriction[row, index] = 1
                return float(fit.wald_test(restriction, scalar=True).pvalue)

            all_terms = [i for i, name in enumerate(names) if "C(temperature_label)" in name]
            slope_terms = [i for i in all_terms if ":" in names[i]]
            rows.append(
                {
                    "model_label": model,
                    "joint_curve_temperature_p": test(all_terms),
                    "slope_interaction_p": test(slope_terms),
                }
            )
        except Exception as error:
            rows.append(
                {
                    "model_label": model,
                    "joint_curve_temperature_p": np.nan,
                    "slope_interaction_p": np.nan,
                    "error": str(error),
                }
            )
    output = pd.DataFrame(rows)
    output["joint_curve_p_holm"] = holm_adjust(output["joint_curve_temperature_p"])
    output["slope_interaction_p_holm"] = holm_adjust(output["slope_interaction_p"])
    return output


def _slope_prevalence(
    d50: pd.DataFrame, models: list[str], temperatures: list[float], config: dict[str, Any]
) -> pd.DataFrame:
    frame = config["analysis"]["primary_frame"]
    data = d50.loc[
        d50["model_label"].isin(models)
        & d50["charity_frame"].eq(frame)
        & d50["temperature"].isin(temperatures)
        & d50["temperature_mode"].eq("explicit")
    ].copy()
    summaries = []
    for keys, group in data.groupby(["model_label", "valence", "temperature"]):
        model, valence, temperature = keys
        successes, total = int(group["significant_decreasing_trend"].sum()), len(group)
        low, high = wilson_interval(successes, total)
        summaries.append(
            {
                "model_label": model,
                "valence": valence,
                "temperature": temperature,
                "significant_outcomes": successes,
                "total_outcomes": total,
                "significant_rate": successes / total,
                "wilson_low": low,
                "wilson_high": high,
            }
        )
    summary = pd.DataFrame(summaries)
    tests = []
    for keys, group in data.groupby(["model_label", "valence"]):
        pivot = group.pivot(index="outcome_id", columns="temperature", values="significant_decreasing_trend")
        complete = pivot.reindex(columns=temperatures).dropna()
        statistic, p_value = cochran_q(complete.astype(int).to_numpy())
        tests.append(
            {
                "model_label": keys[0],
                "valence": keys[1],
                "complete_outcomes": len(complete),
                "cochran_q": statistic,
                "temperature_p": p_value,
            }
        )
    tests = pd.DataFrame(tests)
    tests["temperature_p_holm"] = holm_adjust(tests["temperature_p"])
    return summary.merge(tests, on=["model_label", "valence"], how="left")


def _d50_tests(
    d50: pd.DataFrame, models: list[str], temperatures: list[float], config: dict[str, Any]
) -> pd.DataFrame:
    frame = config["analysis"]["primary_frame"]
    data = d50.loc[
        d50["model_label"].isin(models)
        & d50["charity_frame"].eq(frame)
        & d50["temperature"].isin(temperatures)
        & d50["temperature_mode"].eq("explicit")
    ].copy()
    data["finite_d50"] = data["d50_status"].eq("identified_within_grid")
    data["log10_d50"] = np.log10(data["d50_usd"])
    rows = []
    for keys, group in data.groupby(["model_label", "valence"]):
        finite = group.pivot(index="outcome_id", columns="temperature", values="finite_d50")
        finite_complete = finite.reindex(columns=temperatures).dropna()
        q_statistic, q_p = cochran_q(finite_complete.astype(int).to_numpy())
        value = group.pivot(index="outcome_id", columns="temperature", values="log10_d50")
        n, f_statistic, f_p = _friedman(value, temperatures)
        rows.extend(
            [
                {
                    "model_label": keys[0],
                    "valence": keys[1],
                    "test": "finite_d50_coverage",
                    "complete_outcomes": len(finite_complete),
                    "statistic": q_statistic,
                    "p_value": q_p,
                },
                {
                    "model_label": keys[0],
                    "valence": keys[1],
                    "test": "finite_d50_value",
                    "complete_outcomes": n,
                    "statistic": f_statistic,
                    "p_value": f_p,
                },
            ]
        )
    output = pd.DataFrame(rows)
    output["p_holm_within_test"] = np.nan
    for _, indices in output.groupby("test").groups.items():
        indices = list(indices)
        output.loc[indices, "p_holm_within_test"] = holm_adjust(output.loc[indices, "p_value"])
    return output


def _variability_tests(
    money: pd.DataFrame, temperatures: list[float]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cells = (
        money.groupby(
            ["model_label", "temperature", "outcome_id", "valence", "donation_usd", "target_side"],
            as_index=False,
        )
        .agg(p_target=("prefer_target", "mean"), n_valid=("prefer_target", "size"))
    )
    cells["choice_dispersion"] = 1 - np.abs(2 * cells["p_target"] - 1)
    outcomes = (
        cells.groupby(["model_label", "temperature", "outcome_id", "valence"], as_index=False)
        .agg(mean_choice_dispersion=("choice_dispersion", "mean"))
    )
    rows = []
    for keys, group in outcomes.groupby(["model_label", "valence"]):
        pivot = group.pivot(index="outcome_id", columns="temperature", values="mean_choice_dispersion")
        n, statistic, p_value = _friedman(pivot, temperatures)
        rows.append(
            {
                "model_label": keys[0],
                "valence": keys[1],
                "complete_outcomes": n,
                "friedman_statistic": statistic,
                "temperature_p": p_value,
            }
        )
    tests = pd.DataFrame(rows)
    tests["temperature_p_holm"] = holm_adjust(tests["temperature_p"])
    means = (
        outcomes.groupby(["model_label", "valence", "temperature"], as_index=False)
        .agg(mean_choice_dispersion=("mean_choice_dispersion", "mean"))
        .merge(tests, on=["model_label", "valence"], how="left")
    )
    return means, outcomes


def _plot_temperature_curves(
    money: pd.DataFrame,
    models: list[str],
    temperatures: list[float],
    config: dict[str, Any],
) -> None:
    _, figures = output_paths(config)
    cells = (
        money.groupby(["model_label", "valence", "temperature", "outcome_id", "donation_usd"], as_index=False)
        .agg(p_prefer_target=("prefer_target", "mean"))
    )
    figure, axes = plt.subplots(len(models), 2, figsize=(13, 3.5 * len(models)), sharex=True, sharey=True, squeeze=False)
    colors = dict(zip(temperatures, ["#1f77b4", "#ff7f0e", "#2ca02c"]))
    for row, model in enumerate(models):
        for column, valence in enumerate(["positive", "negative"]):
            axis = axes[row, column]
            for temperature in temperatures:
                group = cells.loc[
                    cells["model_label"].eq(model)
                    & cells["valence"].eq(valence)
                    & cells["temperature"].eq(temperature)
                ]
                curve = group.groupby("donation_usd")["p_prefer_target"].mean()
                axis.plot(curve.index, curve.values, marker="o", color=colors[temperature], label=f"T={temperature:g}")
            axis.axhline(0.5, color="black", linestyle="--", linewidth=0.8)
            axis.set_xscale("log")
            axis.set_ylim(-0.02, 1.02)
            axis.set_title(f"{model} — {'positive outcomes' if valence == 'positive' else 'preventing negative outcomes'}")
            if row == len(models) - 1:
                axis.set_xlabel("Donation amount (USD, log scale)")
            if column == 0:
                axis.set_ylabel("P(choose outcome or prevention)")
            axis.grid(alpha=0.2)
    axes[0, 1].legend()
    figure.suptitle("Donation-choice curves across sampling temperatures")
    figure.tight_layout()
    figure.savefig(figures / "temperature_donation_curves.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def _plot_grouped_rates(
    data: pd.DataFrame,
    models: list[str],
    temperatures: list[float],
    value: str,
    low: str,
    high: str,
    ylabel: str,
    title: str,
    path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharey=True)
    width = 0.8 / len(temperatures)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for axis, valence in zip(axes, ["positive", "negative"]):
        subset = data.loc[data["valence"].eq(valence)]
        x = np.arange(len(models))
        for index, temperature in enumerate(temperatures):
            rows = subset.loc[subset["temperature"].eq(temperature)].set_index("model_label").reindex(models)
            values = 100 * rows[value].to_numpy(float)
            lower, upper = 100 * rows[low].to_numpy(float), 100 * rows[high].to_numpy(float)
            axis.bar(
                x + (index - (len(temperatures) - 1) / 2) * width,
                values,
                width,
                yerr=np.vstack([values - lower, upper - values]),
                capsize=3,
                color=colors[index],
                edgecolor="black",
                linewidth=0.4,
                label=f"T={temperature:g}",
            )
        axis.set_xticks(x)
        axis.set_xticklabels(models, rotation=35, ha="right")
        axis.set_ylim(0, 108)
        axis.set_title("Positive outcomes" if valence == "positive" else "Preventing negative outcomes")
        axis.set_xlabel("Model")
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel(ylabel)
    axes[1].legend(title="Temperature")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def _plot_temperature_summaries(
    prevalence: pd.DataFrame,
    variability: pd.DataFrame,
    d50: pd.DataFrame,
    models: list[str],
    temperatures: list[float],
    config: dict[str, Any],
) -> None:
    _, figures = output_paths(config)
    _plot_grouped_rates(
        prevalence,
        models,
        temperatures,
        "significant_rate",
        "wilson_low",
        "wilson_high",
        "Outcomes with a BH-significant\nnegative donation slope (%)",
        "Prevalence of detectable donation effects across temperature",
        figures / "temperature_slope_prevalence.png",
    )

    frame = config["analysis"]["primary_frame"]
    data = d50.loc[
        d50["model_label"].isin(models)
        & d50["charity_frame"].eq(frame)
        & d50["temperature"].isin(temperatures)
        & d50["temperature_mode"].eq("explicit")
    ].copy()
    coverage_rows = []
    for keys, group in data.groupby(["model_label", "valence", "temperature"]):
        successes = int(group["d50_status"].eq("identified_within_grid").sum())
        low, high = wilson_interval(successes, len(group))
        coverage_rows.append(
            {
                "model_label": keys[0],
                "valence": keys[1],
                "temperature": keys[2],
                "coverage_rate": successes / len(group),
                "wilson_low": low,
                "wilson_high": high,
            }
        )
    _plot_grouped_rates(
        pd.DataFrame(coverage_rows),
        models,
        temperatures,
        "coverage_rate",
        "wilson_low",
        "wilson_high",
        "Outcomes with finite $d_{50}$ (%)",
        "Finite donation-equivalent coverage across temperature",
        figures / "temperature_d50_coverage.png",
    )

    figure, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharey=True)
    colors = dict(zip(models, plt.cm.tab10(np.arange(len(models)) % 10)))
    for axis, valence in zip(axes, ["positive", "negative"]):
        subset = variability.loc[variability["valence"].eq(valence)]
        for model in models:
            rows = subset.loc[subset["model_label"].eq(model)].set_index("temperature").reindex(temperatures)
            axis.plot(temperatures, rows["mean_choice_dispersion"], marker="o", label=model, color=colors[model])
        axis.set_xticks(temperatures)
        axis.set_ylim(-0.02, 1.02)
        axis.set_xlabel("Sampling temperature")
        axis.set_title("Positive outcomes" if valence == "positive" else "Preventing negative outcomes")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Mean choice dispersion\n(0 = unanimous; 1 = 50/50)")
    axes[1].legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
    figure.suptitle("Repeated-response variability across temperature")
    figure.tight_layout()
    figure.savefig(figures / "temperature_response_variability.png", dpi=220, bbox_inches="tight")
    plt.close(figure)


def run_temperature_analysis(
    responses: pd.DataFrame,
    d50: pd.DataFrame,
    model_quality: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    money, models, temperatures = _eligible_temperature_data(responses, model_quality, config)
    if not models:
        raise RuntimeError("No quality-eligible model has all explicit temperature conditions.")
    curve = _curve_tests(money, models)
    prevalence = _slope_prevalence(d50, models, temperatures, config)
    d50_tests = _d50_tests(d50, models, temperatures, config)
    variability, _ = _variability_tests(money, temperatures)
    tables, _ = output_paths(config)
    curve.to_csv(tables / "temperature_curve_tests.csv", index=False)
    prevalence.to_csv(tables / "temperature_slope_prevalence.csv", index=False)
    d50_tests.to_csv(tables / "temperature_d50_tests.csv", index=False)
    variability.to_csv(tables / "temperature_response_variability.csv", index=False)
    _plot_temperature_curves(money, models, temperatures, config)
    _plot_temperature_summaries(
        prevalence, variability, d50, models, temperatures, config
    )
    return {
        "curve_tests": curve,
        "slope_prevalence": prevalence,
        "d50_tests": d50_tests,
        "response_variability": variability,
    }
