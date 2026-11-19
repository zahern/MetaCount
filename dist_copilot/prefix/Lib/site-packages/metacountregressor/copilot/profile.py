"""Data loading and profiling for the experiment copilot."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .schema import ColumnProfile, DataProfile


def load_data(path: str, **kwargs) -> pd.DataFrame:
    """Read a tabular data source by extension."""
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"data source not found: {p}")
    suffix = p.suffix.lower()
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(p, **kwargs)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(p, **kwargs)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(p, **kwargs)
    if suffix in {".json", ".jsonl"}:
        return pd.read_json(p, lines=suffix == ".jsonl", **kwargs)
    if suffix in {".dta"}:
        return pd.read_stata(p, **kwargs)
    raise ValueError(
        f"unsupported data source {suffix!r}; use CSV, Parquet, Excel, JSON or Stata"
    )


def _numeric_summary(series: pd.Series) -> Dict[str, Any]:
    s = pd.to_numeric(series, errors="coerce")
    valid = s.dropna()
    if valid.empty:
        return {}
    return {
        "min": float(valid.min()),
        "max": float(valid.max()),
        "mean": float(valid.mean()),
        "is_integer": bool(
            np.allclose(valid.to_numpy(), np.round(valid.to_numpy()))
        ),
        "is_nonnegative": bool((valid >= 0).all()),
    }


def profile_dataframe(df: pd.DataFrame, path: str = "<memory>") -> DataProfile:
    """Summarise columns and propose response / ID / offset / group candidates."""
    columns: List[ColumnProfile] = []
    for name in df.columns:
        series = df[name]
        info = ColumnProfile(
            name=str(name),
            dtype=str(series.dtype),
            n_missing=int(series.isna().sum()),
            n_unique=int(series.nunique(dropna=True)),
            samples=[
                _jsonable(v) for v in series.dropna().head(5).tolist()
            ],
        )
        summary = _numeric_summary(series) if pd.api.types.is_numeric_dtype(series) else {}
        for key, value in summary.items():
            setattr(info, key, value)
        info.is_binary = bool(info.n_unique <= 2)
        columns.append(info)

    names = [c.name for c in columns]
    lower = {c.name: c.name.lower() for c in columns}

    def named(*tokens: str) -> List[str]:
        # Match whole name tokens, not substrings: "ID" must not match
        # "WIDTH" or "MEDWIDTH".
        out = []
        for c in columns:
            parts = re.split(r"[^a-z0-9]+", lower[c.name])
            if any(tok in parts for tok in tokens):
                out.append(c.name)
        return out

    response_candidates = [
        c.name
        for c in columns
        if c.is_integer
        and c.is_nonnegative
        and c.n_unique > 2
        and (c.max is None or c.max > 2)
    ]
    preferred = [
        c.name
        for c in columns
        if c.name in response_candidates
        and any(tok in lower[c.name] for tok in ("freq", "count", "crash", "response", "outcome", "y"))
    ]
    response_candidates = list(dict.fromkeys(preferred + response_candidates))

    id_candidates = list(
        dict.fromkeys(
            named("id", "site", "segment", "location", "zone", "person", "household")
        )
    )

    offset_candidates = named("offset", "exposure", "log_length", "loglen", "log_aadt")
    group_candidates = [
        c.name
        for c in columns
        if c.name in named("group", "site", "zone", "class", "cluster")
        and c.n_unique < max(2, len(df))
    ]

    return DataProfile(
        path=str(path),
        n_rows=int(len(df)),
        n_cols=int(df.shape[1]),
        columns=columns,
        response_candidates=response_candidates,
        id_candidates=id_candidates,
        offset_candidates=offset_candidates,
        group_candidates=list(dict.fromkeys(group_candidates)),
    )


def load_and_profile(path: str, **kwargs) -> DataProfile:
    return profile_dataframe(load_data(path, **kwargs), path=path)


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value
