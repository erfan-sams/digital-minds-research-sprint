from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import requests


def tier_text(ladder: dict[str, Any], tier: int) -> str:
    return next(
        variation["text"]
        for variation in ladder["variations"]
        if int(variation["tier"]) == tier
    )


def category_balanced_sample(
    ladders: list[dict[str, Any]], valence: str, n: int, seed: int
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    groups: dict[str, list[dict[str, Any]]] = {}
    for ladder in ladders:
        if ladder["valence"] == valence:
            groups.setdefault(ladder["category"], []).append(ladder)

    for group in groups.values():
        rng.shuffle(group)
    categories = sorted(groups)
    rng.shuffle(categories)

    selected: list[dict[str, Any]] = []
    while len(selected) < n:
        added = False
        for category in categories:
            if groups[category] and len(selected) < n:
                selected.append(groups[category].pop())
                added = True
        if not added:
            raise ValueError(f"Not enough {valence} outcomes to select {n}.")
        rng.shuffle(categories)
    return selected


def prepare_data(config: dict[str, Any]) -> list[dict[str, Any]]:
    paths = config["resolved_paths"]
    source_path: Path = paths["mint_data"]
    selected_path: Path = paths["selected_outcomes"]
    source_path.parent.mkdir(parents=True, exist_ok=True)
    selected_path.parent.mkdir(parents=True, exist_ok=True)

    response = requests.get(config["source"]["url"], timeout=60)
    response.raise_for_status()
    ladders = response.json()
    source_path.write_text(
        json.dumps(ladders, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    expected = int(config["source"]["expected_ladders"])
    if len(ladders) != expected:
        raise ValueError(f"Expected {expected} ladders, found {len(ladders)}.")

    tier = int(config["selection"]["tier"])
    eligible = ladders
    if config["selection"].get("require_original_tier_match", True):
        eligible = [
            row for row in ladders if row["original_text"] == tier_text(row, tier)
        ]

    seed = int(config["selection"]["seed"])
    selected = category_balanced_sample(
        eligible,
        "positive",
        int(config["selection"]["positive_outcomes"]),
        seed,
    ) + category_balanced_sample(
        eligible,
        "negative",
        int(config["selection"]["negative_outcomes"]),
        seed + 1,
    )
    selected.sort(key=lambda row: (row["valence"], row["original_statement_id"]))

    output = [
        {
            "outcome_id": row["original_statement_id"],
            "text": row["original_text"],
            "valence": row["valence"],
            "category": row["category"],
            "identified_property": row["identified_property"],
            "source_tier": tier,
        }
        for row in selected
    ]
    selected_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return output


def load_selected_outcomes(config: dict[str, Any]) -> list[dict[str, Any]]:
    path: Path = config["resolved_paths"]["selected_outcomes"]
    if not path.exists():
        raise FileNotFoundError(f"Run --stage prepare-data first: {path}")
    return json.loads(path.read_text(encoding="utf-8"))

