from __future__ import annotations

import numpy as np
import pandas as pd


def load_example_panel_data() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    ids = np.repeat(np.arange(1, 13), 3)
    periods = np.tile(np.arange(1, 4), 12)
    functional_class = np.repeat([0, 1] * 6, 3)

    aadt = np.repeat(np.linspace(8000, 22000, 12), 3) + periods * 150
    length = np.repeat(np.linspace(0.8, 2.2, 12), 3)
    urban = functional_class.copy()
    grade = (ids % 3 == 0).astype(int)
    lighting = (periods == 3).astype(int)
    curve = ((ids + periods) % 2).astype(int)
    lanewidth = np.repeat(np.linspace(2.8, 3.8, 12), 3)
    shoulder = np.repeat(np.linspace(0.5, 1.8, 12), 3)
    median = np.repeat(np.linspace(0.0, 1.0, 12), 3)
    rain = (periods == 2).astype(int)
    zero_flag = ((ids + periods) % 4 == 0).astype(int)
    memb_urban = urban.copy()
    intersection_density = np.repeat(np.linspace(0.1, 0.9, 12), 3)
    speed = np.repeat(np.linspace(50, 100, 12), 3)
    lanes = np.repeat(np.tile([2, 4], 6), 3)
    budget = np.repeat(np.linspace(50, 120, 12), 3)

    linear_signal = (
        0.4
        + 0.00005 * aadt
        + 0.2 * curve
        - 0.15 * rain
        + 0.1 * urban
    )
    mu = np.exp(-2.2 + linear_signal + 0.2 * functional_class)
    y = rng.poisson(np.clip(mu, 1e-3, None))
    duration = np.exp(1.5 + 0.12 * shoulder + 0.08 * lanes + 0.2 * functional_class + rng.normal(0, 0.15, len(ids)))

    df = pd.DataFrame(
        {
            "ID": ids,
            "PERIOD": periods,
            "TRUE_FUNCTIONAL_CLASS": functional_class,
            "FACILITY_CLASS": np.where(functional_class == 1, "arterial", "local"),
            "Y": y,
            "AADT": aadt,
            "LENGTH": length,
            "GRADE": grade,
            "LIGHTING": lighting,
            "CURVE": curve,
            "LANEWIDTH": lanewidth,
            "SHOULDER": shoulder,
            "MEDIAN": median,
            "RAIN": rain,
            "ZERO_FLAG": zero_flag,
            "MEMB_URBAN": memb_urban,
            "URBAN": urban,
            "INTERSECTION_DENSITY": intersection_density,
            "SPEED": speed,
            "LANES": lanes,
            "B": budget,
            "DURATION": duration,
            "LINEAR_X1": shoulder,
            "LINEAR_X2": median,
            "LINEAR_X3": speed / 100.0,
        }
    )
    df["OFFSET"] = np.log(np.clip(df["AADT"] * df["LENGTH"] * 365 / 1e8, 1e-12, None))
    return df


def load_example_crash_data() -> pd.DataFrame:
    return load_example_panel_data().copy()


def load_example_duration_data() -> pd.DataFrame:
    return load_example_panel_data().copy()
