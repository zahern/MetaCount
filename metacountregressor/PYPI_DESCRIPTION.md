# MetaCountRegressor

**Metaheuristic-guided automated structure search for mixed count-data regression models.**

[![PyPI version](https://badge.fury.io/py/metacountregressor.svg)](https://pypi.org/project/metacountregressor/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

`MetaCountRegressor` automates the hardest part of applied count-data regression:
**choosing which variables to include, what role they play, and how many latent sub-populations exist in your data.**

It combines a **JAX back-end** (fast autodiff + JIT compilation) with **metaheuristic search** (Simulated Annealing, Differential Evolution, Harmony Search) to explore an exponentially large model space and return the specification with the best BIC — or a full Pareto front of BIC / test-RMSE trade-offs.

---

## What problems does it solve?

Typical applications include:

- **Traffic safety** — crash-frequency models for road segments or intersections
- **Health outcomes** — count outcomes (hospitalisations, events) with site-level heterogeneity
- **Demand forecasting** — arrival counts, visit counts, purchase counts
- **Any panel count dataset** where the right model structure is unknown

---

## Key features

- **Poisson and Negative Binomial** (automatic dispersion selection)
- **Zero-inflation** — role 6 allows variables to enter the ZI probability equation
- **Random effects** — independent, correlated (Cholesky), and grouped
- **Heterogeneity in means** — covariates that shift random-effect distributions
- **Latent class models (1–N classes)** with EM warm-start for numerical stability
- **Membership variables** (roles 7 & 8) — covariates that predict which latent class an observation belongs to
- **Single-objective** (minimise BIC) and **multi-objective** (BIC + test RMSE Pareto front)
- **Automatic variable type inference** and role/distribution suggestions
- **Full summary output** to `.txt` files in a `results/` folder

---

## Installation

```bash
pip install metacountregressor
```

---

## 5-minute quickstart

```python
import pandas as pd
from metacountregressor.solution import ObjectiveFunction
from metacountregressor.metaheuristics import harmony_search, differential_evolution, simulated_annealing

df = pd.read_csv("crash_data.csv")
df["OFFSET"] = 0          # or log(exposure)
df.rename(columns={"FREQ": "Y"}, inplace=True)

builder = ExperimentBuilder(
    df           = df,
    id_col       = "SITE_ID",
    y_col        = "Y",
    offset_col   = "OFFSET",
    group_id_col = "FC",
)

builder.describe()         # data summary + role guide
builder.suggest_config()   # automatic role/distribution suggestions

evaluator = builder.build_evaluator(mode="single", R=200)
result    = builder.run(evaluator, algo="sa", max_iter=2000)
```

---

## Role codes

| Code | Name | Description |
|---|---|---|
| 0 | Excluded | Not in the model |
| 1 | Fixed | Single coefficient, constant across observations |
| 2 | Random Independent | Per-site coefficient, independent draws |
| 3 | Random Correlated | Per-site coefficient, jointly estimated Cholesky covariance |
| 4 | Grouped | Coefficient shared within a group (e.g. road class) |
| 5 | Heterogeneity | Shifts the mean of a random coefficient |
| 6 | Zero Inflation | Enters the zero-inflation probability equation |
| 7 | Membership only | Enters the class-membership equation; no direct outcome effect |
| 8 | Membership + Fixed | Enters class-membership AND outcome equations |

---

## Algorithm options

```python
# Simulated Annealing — recommended for single-objective
result = builder.run(evaluator, algo="sa", max_iter=3000)

# Adaptive DE + NSGA-II — recommended for multi-objective
result = builder.run(evaluator, algo="de", max_iter=2000, population_size=30)

# Dynamic Harmony Search + NSGA-II
result = builder.run(evaluator, algo="hs", max_iter=2000, population_size=30)
```

---

## Latent class example

```python
evaluator = builder.build_evaluator(
    mode                = "single",
    max_latent_classes  = 2,
    R                   = 150,
    membership_override = {
        "URB":   [0, 7],     # can influence class membership only
        "SPEED": [0, 7, 8],  # can influence membership and outcome
    },
)
result = builder.run(evaluator, algo="sa", max_iter=3000)
```

---

## Links

- **Source code**: [github.com/zahern/MetaCount](https://github.com/zahern/MetaCount)
- **Issue tracker**: [github.com/zahern/MetaCount/issues](https://github.com/zahern/MetaCount/issues)

---

## Citation

```bibtex
@misc{Ahern2024Meta,
  author = {Zeke Ahern, Paul Corry and Alexander Paz},
  title  = {MetaCountRegressor: Automated structure search for mixed count-data models},
  year   = {2024},
  url    = {https://pypi.org/project/metacountregressor/},
}
```

---

MIT License. See `LICENSE` for details.
