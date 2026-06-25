from __future__ import annotations

from importlib import resources
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd


def _example16_3_path() -> Path:
    return Path(__file__).resolve().parent / "metacountregressor" / "data" / "Ex-16-3.csv"


def _load_example16_3_bytes() -> bytes:
    local_candidates = [
        _example16_3_path(),
        Path(__file__).resolve().parent / "data" / "Ex-16-3.csv",
    ]
    for path in local_candidates:
        if path.exists():
            return path.read_bytes()

    try:
        resource = resources.files("metacountregressor").joinpath("data", "Ex-16-3.csv")
        return resource.read_bytes()
    except (FileNotFoundError, ModuleNotFoundError, AttributeError):
        raise FileNotFoundError(
            "Could not locate packaged Example 16-3 data. "
            "Expected metacountregressor/data/Ex-16-3.csv to be installed."
        )


def load_example16_3_raw_data() -> pd.DataFrame:
    return pd.read_csv(BytesIO(_load_example16_3_bytes()))


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


def load_example_platform_speed_data() -> pd.DataFrame:
    rng = np.random.default_rng(123)
    n_platforms = 50
    obs_per_platform = 30
    platform_ids = np.repeat(np.arange(1, n_platforms + 1), obs_per_platform)
    distances = np.tile(np.linspace(-120, 120, obs_per_platform), n_platforms)

    platform_type = np.repeat(rng.integers(0, 3, size=n_platforms), obs_per_platform)
    platform_height = np.repeat(rng.uniform(0.06, 0.12, size=n_platforms), obs_per_platform)
    platform_width = np.repeat(rng.uniform(3.5, 7.5, size=n_platforms), obs_per_platform)
    posted_speed = np.repeat(rng.choice([30, 40, 50, 60], size=n_platforms), obs_per_platform)
    approach_accel = rng.normal(0.0, 0.35, size=platform_ids.size)
    vehicle_speed = (
        posted_speed
        - 7.0 * np.exp(-(distances / 28.0) ** 2)
        - 12.0 * platform_height
        - 1.0 * platform_type
        + 0.12 * distances / 10.0
        + 1.5 * approach_accel
        + rng.normal(0, 2.0, size=platform_ids.size)
    )
    relative_speed = vehicle_speed - posted_speed

    return pd.DataFrame(
        {
            "PLATFORM_ID": platform_ids,
            "DIST_TO_PLATFORM": distances,
            "SPEED": vehicle_speed,
            "VEHICLE_SPEED": vehicle_speed,
            "RELATIVE_SPEED": relative_speed,
            "POSTED_SPEED": posted_speed,
            "APPROACH_ACCEL": approach_accel,
            "PLATFORM_TYPE": platform_type,
            "PLATFORM_HEIGHT": platform_height,
            "PLATFORM_WIDTH": platform_width,
            "AT_PLATFORM": (np.abs(distances) < 5).astype(int),
        }
    )


def load_example_platform_gap_duration_data() -> pd.DataFrame:
    rng = np.random.default_rng(456)
    n_platforms = 40
    obs_per_platform = 24
    platform_ids = np.repeat(np.arange(1, n_platforms + 1), obs_per_platform)

    posted_speed = np.repeat(rng.choice([30, 40, 50, 60], size=n_platforms), obs_per_platform)
    preceding_vehicle_speed = posted_speed + rng.normal(2.0, 4.0, size=platform_ids.size)
    following_vehicle_speed = posted_speed + rng.normal(1.5, 4.5, size=platform_ids.size)
    platform_height = np.repeat(rng.uniform(0.06, 0.12, size=n_platforms), obs_per_platform)
    platform_width = np.repeat(rng.uniform(3.5, 7.5, size=n_platforms), obs_per_platform)
    approach_volume = np.repeat(rng.uniform(150, 650, size=n_platforms), obs_per_platform)

    raw_gap = (
        2.0
        + 0.025 * np.maximum(preceding_vehicle_speed - posted_speed, 0)
        + 0.018 * np.maximum(following_vehicle_speed - posted_speed, 0)
        + 5.0 * platform_height
        - 0.0012 * approach_volume
        + rng.normal(0, 0.15, size=platform_ids.size)
    )
    duration_until_next_speeding = np.exp(np.clip(raw_gap, 0.2, 4.5))

    return pd.DataFrame(
        {
            "PLATFORM_ID": platform_ids,
            "DURATION_UNTIL_NEXT_SPEEDING": duration_until_next_speeding,
            "PRECEDING_VEHICLE_SPEED": preceding_vehicle_speed,
            "FOLLOWING_VEHICLE_SPEED": following_vehicle_speed,
            "POSTED_SPEED": posted_speed,
            "PLATFORM_HEIGHT": platform_height,
            "PLATFORM_WIDTH": platform_width,
            "APPROACH_VOLUME": approach_volume,
        }
    )
