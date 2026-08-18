from __future__ import annotations

import math
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr

from .common import holm_adjust, output_paths
from .direct_preferences import fit_bradley_terry


def permutation_null(
    pairs: pd.DataFrame,
    scores: pd.DataFrame,
    permutations: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if pairs.empty:
        return np.array([])
    scores = scores.drop_duplicates("outcome_id")
    outcome_ids = scores["outcome_id"].tolist()
    lookup = {outcome: index for index, outcome in enumerate(outcome_ids)}
    values = scores["signed_d50_usd"].to_numpy(float)
    valences = scores["valence"].to_numpy()
    left = np.array([lookup[value] for value in pairs["left_outcome_id"]])
    right = np.array([lookup[value] for value in pairs["right_outcome_id"]])
    observed_left = pairs["observed_left"].to_numpy(bool)
    output = np.empty(permutations)
    batch_size = 2000
    position = 0
    while position < permutations:
        batch = min(batch_size, permutations - position)
        permuted = np.broadcast_to(values, (batch, len(values))).copy()
        for valence in ["positive", "negative"]:
            indices = np.flatnonzero(valences == valence)
            if len(indices) > 1:
                order = np.argsort(rng.random((batch, len(indices))), axis=1)
                permuted[:, indices] = values[indices][order]
        differences = permuted[:, left] - permuted[:, right]
        credit = np.where(
            differences > 0,
            observed_left[None, :],
            np.where(differences < 0, ~observed_left[None, :], 0.5),
        )
        output[position : position + batch] = credit.mean(axis=1)
        position += batch
    return output


def analyze_cross_instrument(
    pairs: pd.DataFrame,
    direct_ranks: pd.DataFrame,
    d50: pd.DataFrame,
    model_quality: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = config["analysis"]["primary_frame"]
    temperature = float(config["analysis"]["validation_temperature"])
    finite = d50.loc[
        d50["charity_frame"].eq(frame)
        & d50["temperature"].eq(temperature)
        & d50["d50_status"].eq("identified_within_grid"),
        [
            "model_label",
            "temperature",
            "temperature_mode",
            "outcome_id",
            "valence",
            "signed_d50_usd",
        ],
    ].copy()
    direct = pairs.loc[
        pairs["temperature"].eq(temperature) & pairs["majority_winner"].notna()
    ].copy()
    direct["observed_left"] = direct["majority_winner"].eq(direct["left_outcome_id"])

    left_scores = finite.rename(
        columns={
            "outcome_id": "left_outcome_id",
            "valence": "left_score_valence",
            "signed_d50_usd": "left_score",
        }
    )
    right_scores = finite.rename(
        columns={
            "outcome_id": "right_outcome_id",
            "valence": "right_score_valence",
            "signed_d50_usd": "right_score",
        }
    )
    setting_keys = ["model_label", "temperature", "temperature_mode"]
    prediction_pairs = direct.merge(
        left_scores,
        on=setting_keys + ["left_outcome_id"],
        how="inner",
    ).merge(
        right_scores,
        on=setting_keys + ["right_outcome_id"],
        how="inner",
    )
    prediction_pairs["pair_scope"] = np.select(
        [
            prediction_pairs["left_score_valence"].eq("positive")
            & prediction_pairs["right_score_valence"].eq("positive"),
            prediction_pairs["left_score_valence"].eq("negative")
            & prediction_pairs["right_score_valence"].eq("negative"),
        ],
        ["positive_positive", "negative_negative"],
        default="mixed_valence",
    )
    difference = prediction_pairs["left_score"] - prediction_pairs["right_score"]
    prediction_pairs["prediction_credit"] = np.where(
        difference > 0,
        prediction_pairs["observed_left"].astype(float),
        np.where(difference < 0, (~prediction_pairs["observed_left"]).astype(float), 0.5),
    )

    permutations = int(config["analysis"]["permutations"])
    rng = np.random.default_rng(int(config["analysis"]["random_seed"]))
    rows = []
    for setting, group in prediction_pairs.groupby(setting_keys, dropna=False):
        within = group.loc[group["pair_scope"].ne("mixed_valence")]
        if within.empty:
            continue
        score_group = finite
        for column, value in zip(setting_keys, setting):
            score_group = score_group.loc[score_group[column].eq(value)]
        null = permutation_null(within, score_group, permutations, rng)
        accuracy = within["prediction_credit"].mean()
        positive = within.loc[within["pair_scope"].eq("positive_positive")]
        negative = within.loc[within["pair_scope"].eq("negative_negative")]
        rows.append(
            {
                **dict(zip(setting_keys, setting)),
                "n_within_pairs": len(within),
                "within_accuracy": accuracy,
                "positive_pairs": len(positive),
                "positive_accuracy": positive["prediction_credit"].mean() if len(positive) else np.nan,
                "negative_pairs": len(negative),
                "negative_accuracy": negative["prediction_credit"].mean() if len(negative) else np.nan,
                "null_mean": null.mean(),
                "null_low_95": np.quantile(null, 0.025),
                "null_high_95": np.quantile(null, 0.975),
                "permutation_p": (1 + np.sum(null >= accuracy)) / (1 + len(null)),
            }
        )

    all_models = pd.DataFrame({"model_label": list(config["models"])})
    summary = all_models.merge(pd.DataFrame(rows), on="model_label", how="left").merge(
        model_quality, on="model_label", how="left"
    )
    summary["permutation_p_holm_all_models"] = holm_adjust(
        summary["permutation_p"].fillna(1).to_numpy(float)
    )
    summary["significant_conservative"] = summary["quality_eligible"] & summary[
        "permutation_p_holm_all_models"
    ].lt(float(config["analysis"]["alpha"]))

    tables, figures = output_paths(config)
    prediction_pairs.to_csv(tables / "cross_instrument_pairs.csv", index=False)
    summary.to_csv(tables / "cross_instrument_validation.csv", index=False)

    plot = summary.loc[summary["quality_eligible"] & summary["n_within_pairs"].notna()].sort_values(
        "within_accuracy"
    )
    figure, axis = plt.subplots(figsize=(10, max(5, 0.65 * len(plot))))
    y = np.arange(len(plot))
    axis.barh(y, plot["within_accuracy"], color="#4c78a8")
    axis.errorbar(
        plot["null_mean"],
        y,
        xerr=np.vstack(
            [plot["null_mean"] - plot["null_low_95"], plot["null_high_95"] - plot["null_mean"]]
        ),
        fmt="o",
        color="black",
        capsize=4,
        label="Task-label permutation null (95%)",
    )
    for index, row in enumerate(plot.itertuples(index=False)):
        if row.significant_conservative:
            axis.text(row.within_accuracy + 0.015, index, "*", va="center", fontsize=16)
    axis.set_yticks(y)
    axis.set_yticklabels(plot["model_label"])
    axis.set_xlim(0, 1.03)
    axis.set_xlabel("Within-valence prediction accuracy")
    axis.set_title("Donation scores predict independent direct choices\nPrimary WFP frame")
    axis.legend(fontsize=8)
    axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    figure.savefig(figures / "within_valence_prediction.png", dpi=220, bbox_inches="tight")
    plt.close(figure)

    rank_table = build_rank_table(pairs, finite, temperature)
    rank_table.to_csv(tables / "cross_instrument_ranks.csv", index=False)
    plot_rank_agreement(rank_table, summary, figures / "cross_instrument_rank_agreement.png")
    return prediction_pairs, summary


def build_rank_table(
    pairs: pd.DataFrame, finite: pd.DataFrame, temperature: float
) -> pd.DataFrame:
    rows = []
    direct = pairs.loc[pairs["temperature"].eq(temperature)]
    for setting, score_group in finite.groupby(
        ["model_label", "temperature_mode"], dropna=False
    ):
        model, mode = setting
        pair_group = direct.loc[
            direct["model_label"].eq(model) & direct["temperature_mode"].eq(mode)
        ]
        for valence in ["positive", "negative"]:
            valence_pairs = pair_group.loc[
                pair_group["left_valence"].eq(valence)
                & pair_group["right_valence"].eq(valence)
            ]
            direct_rank = fit_bradley_terry(valence_pairs)
            donation = score_group.loc[score_group["valence"].eq(valence)].copy()
            donation["donation_rank"] = donation["signed_d50_usd"].rank(method="average")
            merged = donation.merge(direct_rank, on="outcome_id", how="inner")
            for row in merged.to_dict("records"):
                rows.append({"model_label": model, "temperature_mode": mode, "valence": valence, **row})
    return pd.DataFrame(rows)


def plot_rank_agreement(rank_table: pd.DataFrame, summary: pd.DataFrame, path) -> None:
    eligible = summary.loc[summary["quality_eligible"], "model_label"].tolist()
    models = [model for model in eligible if model in set(rank_table["model_label"])]
    if not models:
        return
    columns = min(3, len(models))
    rows = math.ceil(len(models) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(5 * columns, 4.3 * rows), squeeze=False)
    for axis, model in zip(axes.flat, models):
        group = rank_table.loc[rank_table["model_label"].eq(model)]
        for valence, color in [("positive", "#377eb8"), ("negative", "#ff7f00")]:
            subset = group.loc[group["valence"].eq(valence)]
            axis.scatter(subset["donation_rank"], subset["direct_rank"], label=valence, color=color, alpha=0.8)
        if len(group) >= 3:
            sns.regplot(
                data=group,
                x="donation_rank",
                y="direct_rank",
                scatter=False,
                color="black",
                ci=95,
                ax=axis,
            )
            rho = spearmanr(group["donation_rank"], group["direct_rank"]).statistic
        else:
            rho = np.nan
        result = summary.loc[summary["model_label"].eq(model)].iloc[0]
        axis.set_title(
            f"{model}\nn={len(group)}, ρ={rho:.2f}, Holm p={result['permutation_p_holm_all_models']:.3g}"
        )
        axis.set_xlabel("Donation-equivalent rank")
        axis.set_ylabel("Direct-preference rank")
        axis.grid(alpha=0.2)
    for axis in axes.flat[len(models) :]:
        axis.axis("off")
    axes.flat[0].legend(title="Original valence")
    figure.suptitle("Agreement between donation-equivalent and direct preference rankings")
    figure.tight_layout()
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)

