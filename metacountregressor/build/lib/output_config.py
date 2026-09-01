from __future__ import annotations

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
