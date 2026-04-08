# metacountregressor

`metacountregressor` is a JAX-first package for model-structure search.

The package has one main architecture:

- the default engine is JAX
- the default search family is the standard count-model search
- CMF search now defaults to the same main JAX count-search architecture, using a CMF design transformation
- the legacy GA-CMF routine is still available, but it is no longer the primary path

The main entry point is:

```python
from metacountregressor.experiment_package import ExperimentBuilder
```

There is also a lower-level CMF helper:

```python
from metacountregressor.cmf_package import CMFExperimentBuilder
```

## What The Package Searches

There are three core model families:

1. Standard count search
   This is the default workflow. It searches Poisson and Negative Binomial count models with fixed effects, random effects, grouped effects, heterogeneity in means, zero inflation, and latent classes.
2. CMF search
   This uses the CMF-style functional form
   `log(mu) = baseline block + local block * log(AADT)`
   but now runs through the same main JAX count-search architecture by default, so it can use the broader structural search machinery.
3. Duration search
   This uses the duration-model workflow.

## Install

### Local editable install

```bash
python -m pip install -e .
```

### Required runtime stack

```bash
python -m pip install jax jaxlib jaxopt
```

The package metadata requires these JAX packages at install time.

### Build and inspect a wheel

```bash
python -m build
python - <<'PY'
import glob, zipfile
wheel = glob.glob("dist/*.whl")[-1]
with zipfile.ZipFile(wheel) as zf:
    metadata = [n for n in zf.namelist() if n.endswith("METADATA")][0]
    text = zf.read(metadata).decode("utf-8")
    for line in text.splitlines():
        if line.startswith("Requires-Dist:"):
            print(line)
PY
```

Make sure the wheel lists:

- `Requires-Dist: jax`
- `Requires-Dist: jaxlib`
- `Requires-Dist: jaxopt`

### Quick import check

```bash
python -c "from metacountregressor.experiment_package import ExperimentBuilder; print(ExperimentBuilder)"
```

## Workflow Order

The easiest way to use the package is:

1. create an `ExperimentBuilder`
2. choose the model family
3. build the evaluator or search problem
4. run the search
5. inspect the best structure and refit output

## 1. Build The Main ExperimentBuilder

```python
import numpy as np
import pandas as pd
from metacountregressor.experiment_package import ExperimentBuilder

df = pd.read_csv("crash_data.csv")
df["OFFSET"] = np.log(np.clip(df["AADT"] * df["LENGTH"] * 365 / 1e8, 1e-12, None))
df = df.rename(columns={"FREQ": "Y"})

builder = ExperimentBuilder(
    df=df,
    id_col="ID",
    y_col="Y",
    offset_col="OFFSET",
    group_id_col="FC",
)
```

Helpful inspection steps:

```python
builder.describe()
builder.suggest_config(max_latent_classes=2)
```

## 2. Standard Count Search

This is the default model family.

It searches over count-model structures such as:

- Poisson
- Negative Binomial
- fixed effects
- random independent effects
- random correlated effects
- grouped random effects
- heterogeneity in means
- zero inflation
- latent classes
- membership equations

### Count role codes

| Code | Meaning |
| --- | --- |
| `0` | Excluded |
| `1` | Fixed |
| `2` | Random independent |
| `3` | Random correlated |
| `4` | Grouped random |
| `5` | Heterogeneity in means |
| `6` | Zero inflation |
| `7` | Latent-class membership only |
| `8` | Latent-class membership plus fixed outcome |

### Standard count example

```python
evaluator = builder.build_count_evaluator(
    variables=[
        "SPEED",
        "LANES",
        "SHOULDER",
        "MEDIAN",
        "URBAN",
        "RAIN",
        "ZERO_FLAG",
        "MEMB_URBAN",
    ],
    mode="single",
    max_latent_classes=2,
    R=200,
    default_roles=[0, 1, 2, 3, 4, 5, 6, 7, 8],
)

result = builder.run(
    evaluator=evaluator,
    algo="sa",
    max_iter=2000,
    seed=42,
)
```

### Focused count examples

Poisson/NB mixed search:

```python
evaluator = builder.build_count_evaluator(
    variables=["SPEED", "LANES", "AADT_LOG"],
    mode="single",
    R=200,
    default_roles=[0, 1, 2, 3],
)
result = builder.run(evaluator=evaluator, algo="sa", max_iter=1500, seed=1)
```

Zero-inflated count search:

```python
evaluator = builder.build_count_evaluator(
    variables=["SPEED", "LANES", "ZERO_FLAG"],
    mode="single",
    R=200,
    default_roles=[0, 1, 2, 6],
)
result = builder.run(evaluator=evaluator, algo="sa", max_iter=1500, seed=2)
```

Latent-class count search:

```python
evaluator = builder.build_count_evaluator(
    variables=["SPEED", "LANES", "URBAN", "MEMB_URBAN"],
    mode="single",
    max_latent_classes=2,
    R=200,
    default_roles=[0, 1, 2, 3, 7, 8],
)
result = builder.run(evaluator=evaluator, algo="sa", max_iter=2000, seed=3)
```

## 3. CMF Search

CMF search is different from the standard count search in functional form.

The CMF family uses:

```text
log(mu) = baseline block + local block * log(AADT)
```

In the package, the default CMF path now transforms this into the main JAX count-search architecture by creating:

- baseline terms
- `log(AADT)` as the main elasticity term
- `local_var * log(AADT)` interaction terms

This means the default CMF path can use the same main search machinery for:

- fixed effects
- random independent effects
- random correlated effects
- grouped random effects
- heterogeneity in means
- zero inflation
- latent classes
- membership equations

### Default CMF search on the main JAX architecture

```python
cmf_search = builder.build_evaluator(
    model_family="cmf",
    aadt_col="AADT",
    baseline_vars=["GRADE", "LIGHTING", "CURVE"],
    local_vars=["LANEWIDTH", "SHOULDER", "MEDIAN"],
    variables=["RAIN", "ZERO_FLAG", "MEMB_URBAN"],
    mode="single",
    max_latent_classes=2,
    R=200,
)

result = builder.run_search(
    cmf_search,
    algo="sa",
    max_iter=2000,
    seed=7,
)
```

What this does:

- keeps the CMF-style functional form
- uses the main JAX count-search architecture as the driver
- lets auxiliary variables enter as heterogeneity, zero-inflation, or membership terms through the normal role system

### CMF with explicit advanced roles

```python
cmf_search = builder.build_evaluator(
    model_family="cmf",
    aadt_col="AADT",
    baseline_vars=["GRADE", "LIGHTING"],
    local_vars=["LANEWIDTH", "MEDIAN"],
    variables=["HET_RAIN", "ZERO_FLAG", "MEMB_URBAN"],
    mode="single",
    max_latent_classes=2,
    R=200,
    default_roles=[0, 1, 2, 3, 4, 5, 6, 7, 8],
)
```

In that setup:

- baseline and CMF local terms are searched in the CMF-transformed outcome equation
- extra variables can be assigned roles like heterogeneity in means, zero inflation, and latent-class membership

### Legacy GA-CMF path

If you want the older GA-specific CMF driver explicitly:

```python
cmf_search = builder.build_evaluator(
    model_family="cmf",
    cmf_driver="ga",
    aadt_col="AADT",
    baseline_vars=["GRADE", "LIGHTING"],
    local_vars=["LANEWIDTH", "MEDIAN"],
)

result = builder.run_search(cmf_search, algo="ga", R=200, fit_final=True, final_R=500)
```

Use this only if you specifically want the legacy GA routine. The default recommendation is the JAX-count-backed CMF path.

## 4. Duration Search

Use the duration family for the duration-model workflow.

```python
duration_search = builder.build_evaluator(
    model_family="duration",
    variables=["x1", "x2", "x3"],
    budget_col="B",
)

result = builder.run_search(
    duration_search,
    objective="budget_penalty",
    lambda_penalty=10.0,
)
```

## 5. Lower-Level CMF Helper

If you want to start from CMF-specific inputs directly, you can use `CMFExperimentBuilder`.

```python
from metacountregressor.cmf_package import CMFExperimentBuilder

cmf_builder = CMFExperimentBuilder(
    df=df,
    y_col="Y",
    aadt_col="AADT",
    baseline_vars=["GRADE", "LIGHTING"],
    local_vars=["LANEWIDTH", "MEDIAN"],
)
```

### Build a CMF evaluator on the main JAX count architecture

```python
general_builder, evaluator, metadata = cmf_builder.build_jax_count_evaluator(
    id_col="ID",
    offset_col="OFFSET",
    group_id_col="FC",
    variables=["HET_RAIN", "ZERO_FLAG", "MEMB_URBAN"],
    max_latent_classes=2,
    R=200,
)

result = general_builder.run(evaluator=evaluator, algo="sa", max_iter=2000, seed=9)
```

### Run the legacy GA-CMF search directly

```python
search_result = cmf_builder.run_search(R=200)
fit_result = cmf_builder.fit_best_model(search_result, final_R=500)
cmf_builder.print_report(search_result, fit_result)
```

## 6. How To Think About Testing Models

A practical way to work through experiments is:

1. start with the standard count search
2. check whether Poisson or NB is preferred
3. turn on random effects
4. add heterogeneity in means if random parameters are consistently selected
5. test zero inflation if there is excess zero mass
6. test latent classes if you suspect segmented sub-populations
7. then compare against a CMF search if your theory is specifically CMF-based

For CMF experiments, a practical progression is:

1. run the default JAX-count-backed CMF search
2. allow random effects on baseline and CMF local terms
3. add heterogeneity variables
4. add zero-inflation candidates
5. add latent-class membership variables
6. only fall back to the legacy GA-CMF routine if you specifically want that older driver

## Notes

- `ExperimentBuilder` is JAX-first and only supports the JAX engine through the current package API.
- The default family is `count`.
- The default CMF path is also JAX-first now, but it remains a different functional form from the normal count model.
- The duration workflow is separate from both count and CMF.
