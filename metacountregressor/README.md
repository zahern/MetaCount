# metajax-regression

**JAX-accelerated automated structure search for mixed count-data regression models.**

`metajax-regression` lets you automatically discover the best model specification — variable roles, random-effect distributions, dispersion, and latent class structure — for panel count-data problems (crash frequency, health events, demand counts, etc.).  
It combines a JAX back-end for fast gradient computation with metaheuristic search (Simulated Annealing, Differential Evolution, Harmony Search + NSGA-II) to explore a combinatorially large model space and return the structure with the best BIC or Pareto-optimal BIC / test-RMSE trade-off.

---

## Feature overview

| Feature | Detail |
|---|---|
| **Model families** | Poisson · Negative Binomial · Zero-Inflated variants |
| **Random effects** | Independent · Correlated (Cholesky) · Grouped |
| **Distributions** | Normal · Log-Normal · Triangular · Uniform |
| **Heterogeneity in means** | Covariates that shift random-effect means |
| **Latent classes** | 1–N classes with EM warm-start |
| **Membership equations** | Variables that explain class membership (roles 7 & 8) |
| **Optimisers** | Simulated Annealing · Adaptive DE · Dynamic Harmony Search · NSGA-II |
| **Objective** | Single (BIC) or multi-objective (BIC + test RMSE) |

---

## Installation

```bash
pip install metajax-regression
```

### GPU support (optional)

```bash
pip install "metajax-regression[gpu]"
```

### Development / editable install

```bash
git clone https://github.com/your-org/metajax-regression.git
cd metajax-regression
pip install -e ".[dev]"
```

**Python ≥ 3.10 required.**  
Core dependencies: `jax`, `jaxlib`, `jaxopt`, `numpy`, `pandas`, `scipy`, `joblib`.

---

## Quickstart (5 minutes)

```python
import pandas as pd
from metajax_regression import ExperimentBuilder

# 1. Load your panel count dataset
df = pd.read_csv("crash_data.csv")

# 2. (Optional) engineer an exposure offset
df["EXPOSE"] = df["LENGTH"] * df["AADT"] * 365 / 1e8
df["OFFSET"] = 0          # zero → log(1), i.e. no offset by default
df.rename(columns={"FREQ": "Y"}, inplace=True)

# 3. Initialise the builder
builder = ExperimentBuilder(
    df           = df,
    id_col       = "ID",
    y_col        = "Y",
    offset_col   = "OFFSET",
    group_id_col = "FC",      # optional: functional class → grouped effects
)

# 4. Inspect data and get automatic suggestions
builder.describe()
builder.suggest_config()

# 5. Build the evaluator  (single-objective, BIC)
evaluator = builder.build_evaluator(
    mode   = "single",
    R      = 200,       # Halton simulation draws
)

# 6. Run Simulated Annealing
result = builder.run(evaluator, algo="sa", max_iter=1000, seed=42)
```

The best model specification and a full coefficient table are printed automatically.  
A `results/` folder is created with a timestamped `.txt` summary.

---

## Core concepts

### Role codes

Every candidate variable is assigned a **role** that determines how it enters the model.

| Code | Name | Description |
|---|---|---|
| `0` | Excluded | Not in the model |
| `1` | Fixed | Single coefficient, same across all observations |
| `2` | Random Independent | Per-site coefficient, independent draws |
| `3` | Random Correlated | Per-site coefficient, jointly estimated covariance |
| `4` | Grouped | Coefficient shared within a group (e.g. road class) |
| `5` | Heterogeneity | Shifts the *mean* of random coefficients |
| `6` | Zero Inflation | Enters the zero-inflation probability equation |
| `7` | Membership only | Enters the class-probability equation; no direct outcome effect |
| `8` | Membership + Fixed | Enters *both* the class-probability equation and the outcome equation |

Roles 7 and 8 are only meaningful when `max_latent_classes > 1`.

### Decision vector

Internally, each candidate model is encoded as an integer array:

```
[ roles(D)  |  dist_codes(D)  |  dispersion_bit  |  lc_code ]
```

- `roles`: one role code per variable  
- `dist_codes`: distribution for random/grouped variables  
- `dispersion_bit`: 0 = Poisson, 1 = Negative Binomial  
- `lc_code`: `(lc_code % max_latent_classes) + 1` → number of latent classes

### Algorithms

| `algo=` | When to use |
|---|---|
| `"sa"` | **Recommended default.** Single-objective (BIC). Fast, reliable. |
| `"de"` | Multi-objective (BIC + RMSE). Adaptive Differential Evolution + NSGA-II. |
| `"hs"` | Multi-objective (BIC + RMSE). Dynamic Harmony Search + NSGA-II. |

---

## API reference

### `ExperimentBuilder`

```python
builder = ExperimentBuilder(
    df,
    id_col       = "SITE_ID",
    y_col        = "CRASHES",
    offset_col   = "EXPOSURE",   # optional log-offset column
    group_id_col = "ROAD_CLASS", # optional grouping variable
)
```

#### `.describe()`
Prints a full data summary: outcome statistics, overdispersion index, variable types, and the role-code guide.

#### `.suggest_config(max_latent_classes=1)`
Prints per-variable recommended roles and distributions based on automatic type inference.

#### `.build_evaluator(...) → StructureEvaluatorLC`

```python
evaluator = builder.build_evaluator(
    variables           = None,      # default: all candidate columns
    fixed_override      = {"EXPO": [1]},        # force variable to fixed
    membership_override = {"URB": [0, 7, 8]},   # allow membership roles
    exclude             = ["YEAR"],             # always exclude
    mode                = "single",  # "single" or "multi"
    max_latent_classes  = 2,         # 1 disables LC search
    R                   = 200,       # Halton draws
    default_roles       = None,      # override default search space
)
```

#### `.run(evaluator, ...) → dict`

```python
result = builder.run(
    evaluator  = evaluator,
    algo       = "sa",        # "sa" | "de" | "hs"
    max_iter   = 3000,
    n_jobs     = 1,
    seed       = 0,
    # SA-specific kwargs:
    mutation_rate = 0.3,
    n_starts      = 1,
    alpha         = 0.995,
)
```

Returns a dict with keys: `algorithm`, `seed`, `solutions`, `scores`, `best_solution`, `best_score`.

---

## Examples

### 1 · Fixed-effects Poisson baseline

```python
evaluator = builder.build_evaluator(
    default_roles = [0, 1],   # only Excluded or Fixed
    mode          = "single",
    R             = 50,
)
result = builder.run(evaluator, algo="sa", max_iter=500)
```

### 2 · Mixed Negative Binomial search

```python
evaluator = builder.build_evaluator(
    mode          = "single",
    R             = 200,
    default_roles = [0, 1, 2, 3],  # allow random effects
)
result = builder.run(evaluator, algo="sa", max_iter=2000)
```

### 3 · Zero-inflated model search

```python
evaluator = builder.build_evaluator(
    default_roles = [0, 1, 2, 6],  # 6 = ZI role
    mode          = "single",
    R             = 150,
)
result = builder.run(evaluator, algo="sa", max_iter=1500)
```

### 4 · Latent class search with membership variables

```python
evaluator = builder.build_evaluator(
    mode                = "single",
    max_latent_classes  = 2,
    R                   = 150,
    membership_override = {
        "URB":   [0, 7],     # may only influence class membership
        "SPEED": [0, 7, 8],  # may influence membership AND outcome
    },
)
result = builder.run(evaluator, algo="sa", max_iter=3000)
```

### 5 · Multi-objective search (BIC + test RMSE Pareto front)

```python
evaluator = builder.build_evaluator(
    mode               = "multi",
    max_latent_classes = 2,
    R                  = 200,
)
result = builder.run(evaluator, algo="de", max_iter=2000, population_size=30)
```

### 6 · Fixing specific variables and excluding others

```python
evaluator = builder.build_evaluator(
    fixed_override = {
        "AADT":  [1],        # always fixed
        "SPEED": [0, 1, 2],  # only Excluded / Fixed / Random-Ind
    },
    exclude        = ["SEGMENT_ID", "YEAR"],
    mode           = "single",
    R              = 200,
)
result = builder.run(evaluator, algo="sa", max_iter=2000)
```

### 7 · Grouped random effects (shared within road class)

```python
builder = ExperimentBuilder(
    df           = df,
    id_col       = "SITE_ID",
    y_col        = "CRASHES",
    offset_col   = "EXPOSURE",
    group_id_col = "FC",       # <-- enables role 4
)
evaluator = builder.build_evaluator(
    default_roles = [0, 1, 2, 4],
    mode          = "single",
    R             = 150,
)
result = builder.run(evaluator, algo="sa", max_iter=2000)
```

---

## Output files

Every run writes to `results/` (auto-created):

| File | Content |
|---|---|
| `sa_summary_seed0_config0_<timestamp>.txt` | Full coefficient table, BIC, AIC, train/test/validation metrics |
| `de_pareto_seed0_config0_<timestamp>.txt` | Pareto-front summary (multi-objective only) |

---

## Citation

If you use `metajax-regression` in academic work, please cite:

```bibtex
@software{metajax_regression,
  title   = {metajax-regression: Automated structure search for mixed count-data models},
  year    = {2024},
  url     = {https://github.com/your-org/metajax-regression},
}
```

---

## Contributing

Pull requests are welcome. Please open an issue first to discuss major changes.  
See `CONTRIBUTING.md` for development setup, code style, and test instructions.

---

## License

MIT — see `LICENSE`.
