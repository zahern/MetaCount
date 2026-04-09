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
) -> Path:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target = output_dir / f"{config.experiment_name}_{family}_{algorithm}_{timestamp}.json"

    payload = {
        "config": asdict(config),
        "family": family,
        "algorithm": algorithm,
        "result": _normalize(result),
    }
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return target
