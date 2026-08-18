from __future__ import annotations

import json
import os
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from tqdm.auto import tqdm

from tell_me_your_price.config import serializable_config

from .openrouter import call_model, discover_providers, get_api_key
from .schedule import load_schedule


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def repair_jsonl_tail(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        return
    with path.open("rb+") as handle:
        handle.seek(-1, os.SEEK_END)
        if handle.read(1) == b"\n":
            return
        end = handle.tell()
        while end > 0:
            start = max(0, end - 65_536)
            handle.seek(start)
            block = handle.read(end - start)
            newline = block.rfind(b"\n")
            if newline >= 0:
                handle.truncate(start + newline + 1)
                return
            end = start
        handle.truncate(0)


def load_checkpoint(path: Path) -> dict[int, dict[str, Any]]:
    repair_jsonl_tail(path)
    rows: dict[int, dict[str, Any]] = {}
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            index = int(row["request_index"])
            if index in rows:
                raise RuntimeError(f"Duplicate request_index {index} at line {line_number}.")
            rows[index] = row
    return rows


def append_checkpoint(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("a", encoding="utf-8") as handle:
        handle.write("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
        handle.flush()
        os.fsync(handle.fileno())


def _load_or_bind_providers(
    config: dict[str, Any], api_key: str, path: Path
) -> dict[str, dict[str, Any]]:
    if path.exists():
        bindings = json.loads(path.read_text(encoding="utf-8"))
    else:
        bindings = discover_providers(config, api_key)
        atomic_json(path, bindings)

    if set(bindings) != set(config["models"]):
        raise RuntimeError("Provider bindings do not match the configured models.")
    for label, model in config["models"].items():
        binding = bindings[label]
        if binding.get("requested_model") != model["id"] or not binding.get("provider_slug"):
            raise RuntimeError(f"Invalid provider binding for {label}.")
    return bindings


def run_collection(config: dict[str, Any]) -> Path:
    collected_dir: Path = config["resolved_paths"]["collected_dir"]
    collected_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = collected_dir / "responses_checkpoint.jsonl"
    binding_path = collected_dir / "provider_bindings.json"
    api_key = get_api_key()
    schedule = load_schedule(config)
    bindings = _load_or_bind_providers(config, api_key, binding_path)

    for row in schedule:
        binding = bindings[row["model_label"]]
        row["frozen_provider_name"] = binding["provider_name"]
        row["frozen_provider_slug"] = binding["provider_slug"]

    completed = load_checkpoint(checkpoint_path)
    for index, saved in completed.items():
        if index < 1 or index > len(schedule):
            raise RuntimeError(f"Out-of-range checkpoint request_index: {index}")
        expected = schedule[index - 1]
        for field in ["experiment_id", "requested_model", "prompt", "frozen_provider_slug"]:
            if saved.get(field) != expected.get(field):
                raise RuntimeError(f"Checkpoint mismatch at request {index}: {field}")

    pending = [row for row in schedule if int(row["request_index"]) not in completed]
    batch_size = int(config["api"]["checkpoint_every"])
    max_workers = int(config["api"]["max_workers"])
    print(f"Checkpointed: {len(completed):,}; remaining: {len(pending):,}")

    buffer: list[dict[str, Any]] = []
    if pending:
        iterator = iter(pending)
        executor = ThreadPoolExecutor(max_workers=max_workers)
        in_flight = {}
        try:
            for _ in range(min(max_workers, len(pending))):
                row = next(iterator)
                in_flight[executor.submit(call_model, row, config, api_key)] = row

            with tqdm(total=len(pending), desc="OpenRouter requests") as progress:
                while in_flight:
                    done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                    for future in done:
                        submitted = in_flight.pop(future)
                        result = future.result()
                        completed[int(submitted["request_index"])] = result
                        buffer.append(result)
                        progress.update(1)
                        if len(buffer) >= batch_size:
                            append_checkpoint(checkpoint_path, buffer)
                            buffer.clear()
                        try:
                            next_row = next(iterator)
                        except StopIteration:
                            continue
                        in_flight[executor.submit(call_model, next_row, config, api_key)] = next_row
        finally:
            append_checkpoint(checkpoint_path, buffer)
            buffer.clear()
            executor.shutdown(wait=False, cancel_futures=True)

    if len(completed) != len(schedule):
        raise RuntimeError(f"Collection incomplete: {len(completed):,}/{len(schedule):,}")

    ordered = [completed[index] for index in range(1, len(schedule) + 1)]
    responses = pd.DataFrame(ordered)
    responses["hit_max_tokens"] = np.where(
        responses["api_ok"], responses["finish_reason"].eq("length"), np.nan
    )
    csv_path = collected_dir / "responses.csv"
    jsonl_path = collected_dir / "responses.jsonl"
    responses.to_csv(csv_path, index=False)
    with jsonl_path.open("w", encoding="utf-8") as handle:
        handle.write("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in ordered))

    frozen = serializable_config(config)
    frozen["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    frozen["provider_bindings"] = bindings
    frozen["total_experimental_requests"] = len(schedule)
    (collected_dir / "run_config.yaml").write_text(
        yaml.safe_dump(frozen, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    print(f"Saved {len(responses):,} responses to {csv_path}")
    return csv_path

