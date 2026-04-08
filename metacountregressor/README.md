# metacountregressor

`metacountregressor` searches count-model structure for you. It can explore fixed effects, random effects, grouped effects, heterogeneity in means, zero inflation, and latent classes, then score candidate models with BIC or a multi-objective criterion.

The package now exposes a stable package import path:

```python
from metacountregressor.experiment_package import ExperimentBuilder
from metacountregressor.cmf_package import CMFExperimentBuilder
```

## Installation

### Install from a local clone

```bash
git clone https://github.com/zahern/MetaCount.git
cd metacountregressor
pip install -e ".[dev]"
```

### Build a wheel

```bash
python -m build
pip install dist/metacountregressor-0.1.0-py3-none-any.whl
```

### Quick import check

```bash
python -c "from metacountregressor.experiment_package import ExperimentBuilder; print(ExperimentBuilder)"
```

## What The Search Can Explore

### Role codes

| Code | Meaning |
| --- | --- |
| `0` | Excluded |
| `1` | Fixed effect |
| `2` | Random independent effect |
| `3` | Random correlated effect |
| `4` | Grouped random effect |
| `5` | Heterogeneity in random-effect means |
| `6` | Zero-inflation equation |
| `7` | Latent-class membership only |
| `8` | Latent-class membership plus fixed outcome effect |

### Model dimensions covered by the search

- Poisson and Negative Binomial
- Fixed-only models
- Linear mixed count models with independent random effects
- Linear mixed count models with correlated random effects
- Grouped random effects
- Heterogeneity in means
- Zero-inflated specifications
- Latent-class specifications with membership equations

## Running A General Experiment

### 1. Prepare a dataframe

You need:

- an ID column
- a count outcome column
- optional offset column
- optional group column for grouped random effects

Example:

```python
import numpy as np
import pandas as pd

df = pd.read_csv("crash_data.csv")
df["OFFSET"] = np.log(np.clip(df["AADT"] * df["LENGTH"] * 365 / 1e8, 1e-12, None))
df = df.rename(columns={"FREQ": "Y"})
```

### 2. Create the experiment builder

```python
from metacountregressor.experiment_package import ExperimentBuilder

builder = ExperimentBuilder(
    df=df,
    id_col="ID",
    y_col="Y",
    offset_col="OFFSET",
    group_id_col="FC",
)
```

### 3. Inspect the data and suggestions

```python
builder.describe()
builder.suggest_config(max_latent_classes=2)
```

This prints:

- outcome diagnostics
- candidate variable summaries
- suggested roles
- suggested random-effect distributions

### 4. Build the evaluator

#### Standard mixed-model search

```python
evaluator = builder.build_evaluator(
    mode="single",
    R=200,
    default_roles=[0, 1, 2, 3, 4, 5],
)
```

#### Latent-class mixed-model search

```python
evaluator = builder.build_evaluator(
    mode="single",
    max_latent_classes=3,
    R=200,
    default_roles=[0, 1, 2, 3, 4, 5, 7, 8],
    membership_override={
        "URB": [0, 7, 8],
        "SPEED": [0, 1, 7, 8],
    },
)
```

#### Search only a selected subset of variables

```python
evaluator = builder.build_evaluator(
    variables=["AADT", "LENGTH", "URB", "SPEED", "GRADE"],
    mode="single",
    R=150,
)
```

### 5. Run the search

#### Simulated annealing

```python
result = builder.run(
    evaluator=evaluator,
    algo="sa",
    max_iter=1500,
    seed=42,
)
```

#### Differential evolution / Pareto search

```python
result = builder.run(
    evaluator=evaluator,
    algo="de",
    max_iter=1000,
    n_jobs=1,
    seed=42,
    population_size=20,
)
```

### 6. Read the result

The returned dictionary includes:

- `best_solution`
- `best_score`
- `solutions`
- `scores`
- `algorithm`
- `seed`

The run also prints the decoded best structure and writes a summary file to `results/`.

## Common Experiment Recipes

### Fixed-only baseline

```python
evaluator = builder.build_evaluator(
    mode="single",
    R=100,
    default_roles=[0, 1],
)
result = builder.run(evaluator, algo="sa", max_iter=500)
```

### Independent and correlated random effects

```python
evaluator = builder.build_evaluator(
    mode="single",
    R=200,
    default_roles=[0, 1, 2, 3],
)
result = builder.run(evaluator, algo="sa", max_iter=1500)
```

### Grouped random effects plus heterogeneity in means

```python
evaluator = builder.build_evaluator(
    mode="single",
    R=200,
    default_roles=[0, 1, 2, 3, 4, 5],
)
result = builder.run(evaluator, algo="sa", max_iter=2000)
```

### Zero inflation

```python
evaluator = builder.build_evaluator(
    mode="single",
    R=200,
    default_roles=[0, 1, 2, 6],
)
result = builder.run(evaluator, algo="sa", max_iter=1500)
```

### Latent classes with membership variables

```python
evaluator = builder.build_evaluator(
    mode="single",
    max_latent_classes=2,
    R=200,
    default_roles=[0, 1, 2, 3, 5, 7, 8],
    membership_override={
        "URB": [0, 7],
        "AADT": [0, 1, 7, 8],
    },
)
result = builder.run(evaluator, algo="sa", max_iter=2500)
```

## Running CMF Experiments

The GA CMF code is now wrapped by `CMFExperimentBuilder`, so you can run the classic AADT-based CMF search through the package instead of calling the standalone script directly.

### 1. Create the CMF builder

```python
import pandas as pd
from metacountregressor.cmf_package import CMFExperimentBuilder

df = pd.read_csv("cmf_data.csv")

cmf_builder = CMFExperimentBuilder(
    df=df,
    y_col="Y",
    aadt_col="AADT",
    baseline_vars=["URB", "HISNOW", "SLOPE"],
    local_vars=["GBRPM", "EXPOSE", "INTPM", "SPEED"],
)
```

### 2. Search the CMF structure

```python
search_result = cmf_builder.run_search(R=200)
print(search_result)
```

This search chooses:

- which CMF variables are active
- whether each active CMF is fixed or random
- Poisson vs Negative Binomial
- Halton draws vs pseudo-random draws

### 3. Fit the final CMF model and print the report

```python
fit_result = cmf_builder.fit_best_model(search_result, final_R=500)
cmf_builder.print_report(search_result, fit_result)
```

### 4. Bridge CMF variables into latent-class search

If you want latent-class search over the same CMF covariates, move from the CMF builder into the general package search:

```python
general_builder, evaluator = cmf_builder.build_latent_class_evaluator(
    id_col="ID",
    offset_col="OFFSET",
    max_latent_classes=2,
    R=200,
    default_roles=[0, 1, 2, 3, 5, 7, 8],
    membership_override={
        "AADT": [0, 1, 7, 8],
        "URB": [0, 7, 8],
    },
)

result = general_builder.run(
    evaluator=evaluator,
    algo="sa",
    max_iter=2000,
    seed=7,
)
```

That path is the recommended way to set up latent-class CMF-style experiments in the package.

## Test Suite

Run the package tests with:

```bash
pytest
```

The current test suite checks:

- package imports
- the `metacountregressor.experiment_package` import path
- role decoding for fixed, random, grouped, heterogeneity, zero inflation, and membership roles
- CMF builder integration with the general latent-class experiment API

## Output Files

Search runs create result files under `results/`, including summaries and Pareto-front exports for multi-objective runs.

## Notes

- Python 3.10+ is required.
- JAX should be installed in an environment that matches your CPU/GPU setup.
- The latent-class search space can get large quickly. Start with fewer variables and fewer draws, then scale up once the pipeline is working.
