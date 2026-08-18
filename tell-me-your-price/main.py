from __future__ import annotations

import argparse

from tell_me_your_price.analysis.runner import (
    run_all_analysis,
    run_outcome_table_stage,
    run_primary_analysis,
    run_temperature_stage,
)
from tell_me_your_price.collection.prepare_data import load_selected_outcomes, prepare_data
from tell_me_your_price.collection.runner import run_collection
from tell_me_your_price.collection.schedule import build_schedule, save_schedule, schedule_counts
from tell_me_your_price.config import load_config


STAGES = [
    "prepare-data",
    "collect",
    "analyze",
    "temperature",
    "outcome-table",
    "all-analysis",
]


def prepare(config):
    outcomes = prepare_data(config)
    schedule = build_schedule(config, outcomes)
    expected = sum(schedule_counts(config, len(outcomes)).values())
    if len(schedule) != expected:
        raise RuntimeError(f"Schedule size mismatch: {len(schedule):,} != {expected:,}")
    path = save_schedule(config, schedule)
    print(f"Selected outcomes: {len(outcomes)}")
    print(f"Scheduled requests: {len(schedule):,}")
    print(f"Saved schedule: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tell Me Your Price experiment pipeline")
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument("--config", default="config/experiment.yaml")
    args = parser.parse_args()
    config = load_config(args.config)

    if args.stage == "prepare-data":
        prepare(config)
    elif args.stage == "collect":
        load_selected_outcomes(config)
        run_collection(config)
    elif args.stage == "analyze":
        run_primary_analysis(config)
    elif args.stage == "temperature":
        run_temperature_stage(config)
    elif args.stage == "outcome-table":
        run_outcome_table_stage(config)
    elif args.stage == "all-analysis":
        run_all_analysis(config)


if __name__ == "__main__":
    main()

