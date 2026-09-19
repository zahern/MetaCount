from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class SearchOutputConfig:
    output_dir: str = "results"
    experiment_name: str = "experiment"
    search_description: str = ""
    save_json: bool = True
    # Best-so-far checkpointing during the search (release valve for walltime
    # kills): write the incumbent decision vector + score every N SA
    # generations. 0 disables checkpointing. The file is written atomically
    # (tmp + os.replace) so a kill mid-write never corrupts it, and a later
    # run can warm-start from it via init_solutions=[checkpoint["best_decision"]].
    checkpoint_every: int = 0
    checkpoint_name: str = ""   # defaults to "{experiment_name}_checkpoint.json"


def checkpoint_path(config: SearchOutputConfig) -> Path:
    name = config.checkpoint_name or f"{config.experiment_name}_checkpoint.json"
    return Path(config.output_dir) / name


def save_search_checkpoint(
    config: SearchOutputConfig,
    payload: dict[str, Any],
) -> Path:
    """Atomically write the best-so-far checkpoint for a running search.

    Unlike ``save_search_result`` (one timestamped file at the end of a run),
    this overwrites a single well-known file during the search so that a
    walltime kill / OOM leaves a resumable incumbent behind. The payload is
    normalised the same way as the final result JSON.
    """
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = checkpoint_path(config)

    body = {
        "config": asdict(config),
        "timestamp_iso": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "checkpoint": _normalize(payload),
    }
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(body, indent=2), encoding="utf-8")
    os.replace(tmp, target)
    return target


def _normalize(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _normalize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize(v) for v in value]
    if hasattr(value, "__dict__"):
        return _normalize(vars(value))
    return value


def save_search_result(
    result: dict[str, Any],
    config: SearchOutputConfig,
    family: str,
    algorithm: str,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write the search result JSON.

    `metadata` is an optional dict for run context that isn't part of
    `result` itself -- e.g. hyperparameters actually used, candidate
    variable names, objective/criterion name(s), elapsed wall time, and
    train/test split details. Callers (ExperimentBuilder.run()) populate
    this from what they already have on hand; the field is additive and
    backward compatible (old payloads without it still parse fine).
    """
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target = output_dir / f"{config.experiment_name}_{family}_{algorithm}_{timestamp}.json"

    payload = {
        "config": asdict(config),
        "family": family,
        "algorithm": algorithm,
        "metadata": _normalize(metadata or {}),
        "result": _normalize(result),
    }
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return target
