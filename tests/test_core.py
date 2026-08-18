from copy import deepcopy

from tell_me_your_price.analysis.common import estimate_d50
from tell_me_your_price.collection.openrouter import parse_strict_ab
from tell_me_your_price.collection.prompts import (
    donation_option,
    format_usd,
    load_prompt_assets,
    render_choice,
    valued_target,
)
from tell_me_your_price.collection.runner import append_checkpoint, load_checkpoint
from tell_me_your_price.collection.schedule import build_schedule, schedule_counts
from tell_me_your_price.config import load_config

import numpy as np


def config():
    return load_config("config/experiment.yaml")


def outcomes():
    return [
        {
            "outcome_id": f"o{index}",
            "text": f"Outcome {index}",
            "valence": "positive" if index < 2 else "negative",
            "category": "test",
            "identified_property": "test property",
            "source_tier": 4,
        }
        for index in range(4)
    ]


def test_default_schedule_count():
    assert sum(schedule_counts(config(), 30).values()) == 475_800


def test_small_schedule_is_balanced():
    cfg = deepcopy(config())
    cfg["models"] = {
        "temperature_model": {
            "id": "example/temperature",
            "send_temperature": True,
            "send_reasoning": True,
        },
        "default_model": {
            "id": "example/default",
            "send_temperature": False,
            "send_reasoning": False,
        },
    }
    cfg["elicitation"]["donation_amounts_usd"] = [1, 10]
    cfg["elicitation"]["direct_repetitions_per_order"] = 1
    cfg["elicitation"]["donation_repetitions_per_order"] = 1
    cfg["elicitation"]["charity_frames"]["wfp_emergency_food"]["temperatures"] = [0, 1]
    schedule = build_schedule(cfg, outcomes())
    assert len(schedule) == sum(schedule_counts(cfg, 4).values()) == 168
    assert len({row["request_index"] for row in schedule}) == len(schedule)
    money = [row for row in schedule if row["phase"] == "money_tier4"]
    for key in {(row["model_label"], row["outcome_id"], row["donation_usd"], row["charity_frame"], row["temperature"]) for row in money}:
        rows = [
            row
            for row in money
            if (row["model_label"], row["outcome_id"], row["donation_usd"], row["charity_frame"], row["temperature"]) == key
        ]
        assert {row["target_side"] for row in rows} == {"A", "B"}


def test_prompt_rendering_and_parser():
    assets = load_prompt_assets(config())
    negative, kind, multiplier = valued_target(assets, "A harmful event occurs.", "negative")
    donation = donation_option(assets, "wfp_emergency_food", 0.01)
    prompt = render_choice(assets["forced_choice"], negative, donation)
    assert kind == "negative_prevention" and multiplier == -1
    assert "A harmful event occurs." in prompt
    assert format_usd(0.01) == "$0.01"
    assert prompt.count("Option A:") == prompt.count("Option B:") == 1
    assert parse_strict_ab(' "a". ') == "A"
    assert parse_strict_ab("I choose A") is None


def test_d50_crossing_and_censoring():
    amounts = np.array([1, 10, 100], dtype=float)
    weights = np.full(3, 20.0)
    finite = estimate_d50(amounts, np.array([0.9, 0.4, 0.1]), weights)
    below = estimate_d50(amounts, np.array([0.4, 0.2, 0.1]), weights)
    above = estimate_d50(amounts, np.array([0.9, 0.8, 0.7]), weights)
    assert finite["d50_status"] == "identified_within_grid"
    assert 1 < finite["d50_usd"] < 10
    assert below["d50_status"] == "at_or_below_grid"
    assert above["d50_status"] == "above_grid"


def test_checkpoint_repairs_truncated_tail(tmp_path):
    path = tmp_path / "checkpoint.jsonl"
    append_checkpoint(path, [{"request_index": 1}, {"request_index": 2}])
    with path.open("ab") as handle:
        handle.write(b'{"request_index": 3')
    loaded = load_checkpoint(path)
    assert sorted(loaded) == [1, 2]
    assert path.read_bytes().endswith(b"\n")

