from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _example16_3_path() -> Path:
    return Path(__file__).resolve().parent / "metacountregressor" / "data" / "Ex-16-3.csv"


def load_example16_3_raw_data() -> pd.DataFrame:
    return pd.read_csv(_example16_3_path())


def load_example16_3_model_data() -> pd.DataFrame:
    df = load_example16_3_raw_data().copy()
    df["OFFSET"] = np.log(np.clip(df["AADT"] * df["LENGTH"] * 365 / 1e8, 1e-12, None))
    fc_categories = sorted(df["FC"].dropna().unique().tolist())
    df["FC_ENCODED"] = pd.Categorical(df["FC"], categories=fc_categories, ordered=True).codes
    df["FC_LABEL"] = df["FC"].map(lambda value: f"FC_{int(value)}")
    return df


def load_example_crash_data() -> pd.DataFrame:
    return load_example16_3_model_data()


def load_example_duration_data() -> pd.DataFrame:
    df = load_example16_3_model_data().copy()
    # Provide a simple positive duration-style target derived from the original columns.
    df["DURATION"] = np.exp(
        0.5
        + 0.01 * df["WIDTH"]
        + 0.02 * df["CURVES"]
        + 0.00002 * df["AADT"]
        + 0.05 * df["FC_ENCODED"]
    )
    return df


def load_example_linear_data() -> pd.DataFrame:
    df = load_example16_3_model_data().copy()
    df["LINEAR_TARGET"] = (
        0.1 * df["WIDTH"]
        + 0.2 * df["CURVES"]
        + 0.05 * df["SLOPE"]
        + 0.01 * df["AADT"] / 100.0
    )
    return df


def load_example_panel_data() -> pd.DataFrame:
    return load_example16_3_model_data()
