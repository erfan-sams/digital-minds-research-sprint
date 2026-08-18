from __future__ import annotations

import itertools
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from .common import output_paths, wilson_interval


def _cycle(triad: tuple[str, str, str], winners: dict[frozenset[str], str]) -> bool:
    outdegree = {outcome: 0 for outcome in triad}
    for left, right in itertools.combinations(triad, 2):
        winner = winners[frozenset((left, right))]
        outdegree[winner] += 1
    return all(value == 1 for value in outdegree.values())


def fit_bradley_terry(pair_group: pd.DataFrame) -> pd.DataFrame:
    outcomes = sorted(
        set(pair_group["left_outcome_id"]) | set(pair_group["right_outcome_id"])
    )
    if len(outcomes) < 2:
        return pd.DataFrame(columns=["outcome_id", "bt_utility", "direct_rank"])
    reference, estimated = outcomes[-1], outcomes[:-1]
    columns = {outcome: index for index, outcome in enumerate(estimated)}
    design = np.zeros((len(pair_group), len(estimated)))
    for row_index, row in enumerate(pair_group.itertuples(index=False)):
        if row.left_outcome_id != reference:
            design[row_index, columns[row.left_outcome_id]] += 1
        if row.right_outcome_id != reference:
            design[row_index, columns[row.right_outcome_id]] -= 1
    try:
        fit = sm.GLM(
            pair_group["p_prefer_left"].to_numpy(float),
            design,
            family=sm.families.Binomial(),
            freq_weights=pair_group["n_valid"].to_numpy(float),
        ).fit(maxiter=300)
        utilities = dict(zip(estimated, fit.params)) | {reference: 0.0}
    except Exception:
        utilities = {
            outcome: float(
                np.mean(
                    np.r_[
                        pair_group.loc[pair_group["left_outcome_id"].eq(outcome), "p_prefer_left"],
                        1 - pair_group.loc[pair_group["right_outcome_id"].eq(outcome), "p_prefer_left"],
                    ]
                )
            )
            for outcome in outcomes
        }
    table = pd.DataFrame({"outcome_id": outcomes, "bt_utility": [utilities[x] for x in outcomes]})
    table["direct_rank"] = table["bt_utility"].rank(method="average")
    return table


def analyze_direct_preferences(
    responses: pd.DataFrame, model_quality: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    direct = responses.loc[
        responses["phase"].eq("direct_tier4")
        & responses["api_ok_flag"]
        & responses["parse_ok_flag"]
    ].copy()
    direct["prefer_left"] = direct["parsed_choice"].eq(direct["left_side"]).astype(int)
    keys = [
        "model_label",
        "temperature",
        "temperature_mode",
        "left_outcome_id",
        "right_outcome_id",
        "left_valence",
        "right_valence",
    ]
    pairs = (
        direct.groupby(keys, as_index=False, dropna=False)
        .agg(p_prefer_left=("prefer_left", "mean"), n_valid=("prefer_left", "size"))
    )
    pairs["decisiveness"] = 2 * (pairs["p_prefer_left"] - 0.5).abs()
    pairs["majority_winner"] = np.where(
        pairs["p_prefer_left"].gt(0.5),
        pairs["left_outcome_id"],
        np.where(pairs["p_prefer_left"].lt(0.5), pairs["right_outcome_id"], None),
    )
    pairs["pair_type"] = np.where(
        pairs["left_valence"].eq(pairs["right_valence"]),
        pairs["left_valence"].astype(str) + "_" + pairs["right_valence"].astype(str),
        "mixed",
    )

    cycle_rows, rank_rows = [], []
    setting_keys = ["model_label", "temperature", "temperature_mode"]
    for setting, group in pairs.groupby(setting_keys, dropna=False):
        strict = group.loc[group["majority_winner"].notna()]
        winners = {
            frozenset((row.left_outcome_id, row.right_outcome_id)): row.majority_winner
            for row in strict.itertuples(index=False)
        }
        outcomes = sorted(set(group["left_outcome_id"]) | set(group["right_outcome_id"]))
        scorable = cycles = 0
        for triad in itertools.combinations(outcomes, 3):
            edges = [frozenset(edge) for edge in itertools.combinations(triad, 2)]
            if not all(edge in winners for edge in edges):
                continue
            scorable += 1
            cycles += int(_cycle(triad, winners))
        low, high = wilson_interval(cycles, scorable)
        cycle_rows.append(
            {
                **dict(zip(setting_keys, setting)),
                "scorable_pairs": len(strict),
                "mean_decisiveness": group["decisiveness"].mean(),
                "mean_valid_responses_per_pair": group["n_valid"].mean(),
                "scorable_triads": scorable,
                "cyclic_triads": cycles,
                "cycle_rate": cycles / scorable if scorable else np.nan,
                "cycle_wilson_low": low,
                "cycle_wilson_high": high,
            }
        )
        ranks = fit_bradley_terry(group)
        for row in ranks.to_dict("records"):
            rank_rows.append({**dict(zip(setting_keys, setting)), **row})

    cycles = pd.DataFrame(cycle_rows).merge(model_quality, on="model_label", how="left")
    ranks = pd.DataFrame(rank_rows)
    tables, figures = output_paths(config)
    pairs.to_csv(tables / "direct_pair_results.csv", index=False)
    cycles.to_csv(tables / "direct_cycles.csv", index=False)
    ranks.to_csv(tables / "direct_bradley_terry_ranks.csv", index=False)

    plot_data = cycles.sort_values("cycle_rate", ascending=True)
    figure, axis = plt.subplots(figsize=(9, 5.5))
    axis.barh(plot_data["model_label"], plot_data["cycle_rate"], color="#377eb8")
    axis.axvline(0.25, color="black", linestyle="--", linewidth=1, label="Random tournament: 0.25")
    axis.set_xlabel("Observed majority-cycle rate")
    axis.set_ylabel("Model")
    axis.set_title("Direct Tier-4 transitivity")
    axis.legend()
    axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    figure.savefig(figures / "direct_cycle_rate.png", dpi=220, bbox_inches="tight")
    plt.close(figure)
    return pairs, cycles, ranks

