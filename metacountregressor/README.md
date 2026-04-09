# metacountregressor Cookbook

`metacountregressor` is a JAX-first package for fitting and searching hierarchical models.

This cookbook is built around the actual public package API and the bundled example dataset, so every variable name shown below exists in the data returned by the package.

## 1. What The Package Covers

There are four main model families:

1. Count models
   Poisson and Negative Binomial models on the main JAX hierarchical architecture.
2. CMF models
   Crash Modification Factor models with functional form
   `log(mu) = baseline block + local block * log(AADT)`.
   The default CMF route also uses the main JAX hierarchical architecture.
3. Duration models
   The default duration route now uses the main JAX hierarchical architecture with a lognormal family.
4. Linear models
   The default linear route now uses the main JAX hierarchical architecture with a Gaussian family.

Across the main JAX architecture, the package is designed to support:

- random parameters
- correlated random parameters
- grouped random parameters
- heterogeneity in the means
- zero inflation
- latent classes
- membership equations
- random-parameter distribution assumptions
- metaheuristic structure search

## 2. Install

```bash
python -m pip install -e .
python -m pip install jax jaxlib jaxopt
```

Quick import check:

```bash
python -c "from metacountregressor import __version__, ExperimentBuilder, load_example_crash_data; print(__version__, ExperimentBuilder, load_example_crash_data().shape)"
```

The repo version and the uploaded PyPI version should move together. This repo is currently prepared for release `1.0.24`.

## 3. Use The Bundled Example Data

The package ships with accessible example data:

```python
from metacountregressor import load_example_crash_data

df = load_example_crash_data()
print(df.columns.tolist())
```

Important columns in the bundled dataset:

- `ID`
- `Y`
- `OFFSET`
- `FACILITY_CLASS`
- `TRUE_FUNCTIONAL_CLASS`
- `AADT`
- `LENGTH`
- `GRADE`
- `LIGHTING`
- `CURVE`
- `LANEWIDTH`
- `SHOULDER`
- `MEDIAN`
- `RAIN`
- `ZERO_FLAG`
- `MEMB_URBAN`
- `URBAN`
- `INTERSECTION_DENSITY`
- `SPEED`
- `LANES`
- `B`
- `DURATION`
- `LINEAR_X1`
- `LINEAR_X2`
- `LINEAR_X3`

All cookbook examples below use columns from this bundled dataset.

## 4. Build The Main ExperimentBuilder

```python
from metacountregressor import ExperimentBuilder, load_example_crash_data

df = load_example_crash_data()

builder = ExperimentBuilder(
    df=df,
    id_col="ID",
    y_col="Y",
    offset_col="OFFSET",
    group_id_col="FACILITY_CLASS",
)
```

### Which arguments can be `None`

In `ExperimentBuilder(...)`:

- `id_col`
  Required. Do not pass `None`.
- `y_col`
  Required. Do not pass `None`.
- `offset_col`
  Optional. You can pass `None`.
- `group_id_col`
  Optional. You can pass `None`.

In `build_evaluator(...)` and `build_count_evaluator(...)`:

- `variables=None`
  Means use all candidate variables not reserved as IDs, outcomes, offsets, or groups.
- `fixed_override=None`
  Means no per-variable role restrictions.
- `membership_override=None`
  Means no special membership-role restrictions.
- `exclude=None`
  Means do not exclude extra variables.
- `default_roles=None`
  Means the package will choose defaults for that family.

In CMF helpers:

- `offset_col=None`
  Allowed.
- `group_id_col=None`
  Allowed.
- `variables=None`
  Allowed.

Helpful inspection methods:

```python
builder.describe()
builder.suggest_config(max_latent_classes=2)
print(builder.get_family_capabilities())
print(builder.get_search_argument_guide())
```

## 5. Main Search Arguments

Shared arguments you will change most often:

- `algo`
  Search driver. Use `sa`, `hc`, `de`, or `hs`.
- `R`
  Number of simulation draws.
- `max_iter`
  Search iterations.
- `max_latent_classes`
  Maximum number of latent classes.
- `variables`
  Search variables.
- `default_roles`
  Allowed structural roles.
- `fixed_override`
  Restrict roles for specific variables.
- `membership_override`
  Restrict membership-role behavior for specific variables.

Output helper:

```python
from metacountregressor import SearchOutputConfig

output_config = SearchOutputConfig(
    output_dir="results",
    experiment_name="count_search_demo",
    search_description="Search over count structures with latent classes",
)
```

If you pass `output_config=...` to `run(...)` or `run_search(...)`, the package saves a consistent JSON record of the run.

## 6. Role Codes

For the hierarchical search architecture:

| Code | Meaning |
| --- | --- |
| `0` | Excluded |
| `1` | Fixed |
| `2` | Random independent |
| `3` | Random correlated |
| `4` | Grouped random |
| `5` | Heterogeneity in means |
| `6` | Zero inflation |
| `7` | Membership only |
| `8` | Membership plus fixed outcome |

Supported random-parameter distributions:

- `normal`
- `lognormal`
- `triangular`
- `uniform`

## 7. Count Models

### 7.1 Count search

```python
evaluator = builder.build_count_evaluator(
    variables=[
        "AADT",
        "SPEED",
        "LANES",
        "CURVE",
        "LIGHTING",
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
    output_config=output_config,
)
```

### 7.2 Manual count model

```python
manual_spec = builder.make_manual_spec(
    fixed_terms=["AADT", "SPEED"],
    rdm_terms=["LANES:normal"],
    rdm_cor_terms=["CURVE:normal", "LIGHTING:lognormal"],
    hetro_in_means=["RAIN"],
    zi_terms=["ZERO_FLAG"],
    membership_terms=["MEMB_URBAN"],
    dispersion=1,
    latent_classes=2,
)

fit = builder.fit_manual_model(
    manual_spec=manual_spec,
    model="nb",
    R=200,
)
```

## 8. CMF Models

CMF models use:

```text
log(mu) = baseline block + local block * log(AADT)
```

The default CMF route transforms the design and then uses the same JAX hierarchical solver and search architecture as the count family.

### 8.1 CMF search

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
    default_roles=[0, 1, 2, 3, 4, 5, 6, 7, 8],
)

result = builder.run_search(
    cmf_search,
    algo="sa",
    max_iter=2000,
    seed=7,
    output_config=SearchOutputConfig(
        output_dir="results",
        experiment_name="cmf_search_demo",
        search_description="CMF search on the main JAX hierarchical architecture",
    ),
)
```

### 8.2 Manual CMF model

```python
from metacountregressor import CMFExperimentBuilder

cmf_builder = CMFExperimentBuilder(
    df=df,
    y_col="Y",
    aadt_col="AADT",
    baseline_vars=["GRADE", "LIGHTING"],
    local_vars=["LANEWIDTH", "MEDIAN"],
)

manual_cmf_spec = cmf_builder.make_manual_cmf_spec(
    baseline_fixed=["GRADE"],
    baseline_correlated=["LIGHTING"],
    local_random=["LANEWIDTH"],
    local_correlated=["MEDIAN"],
    hetro_in_means=["RAIN"],
    zi_terms=["ZERO_FLAG"],
    membership_terms=["MEMB_URBAN"],
    dispersion=1,
    latent_classes=2,
)

fit = cmf_builder.fit_manual_cmf_model(
    id_col="ID",
    offset_col="OFFSET",
    group_id_col="FACILITY_CLASS",
    manual_spec=manual_cmf_spec,
    model="nb",
    R=200,
)
```

### 8.3 Legacy GA-CMF route

```python
legacy_cmf = builder.build_evaluator(
    model_family="cmf",
    cmf_driver="ga",
    aadt_col="AADT",
    baseline_vars=["GRADE", "LIGHTING"],
    local_vars=["LANEWIDTH", "MEDIAN"],
)

legacy_result = builder.run_search(
    legacy_cmf,
    algo="ga",
    R=200,
    fit_final=True,
    final_R=500,
)
```

## 9. Duration Models

The default duration route now uses the main JAX hierarchical architecture with a lognormal family.

### 9.1 Duration search

```python
duration_search = builder.build_evaluator(
    model_family="duration",
    variables=["SHOULDER", "LANES", "RAIN", "MEMB_URBAN"],
    budget_col="B",
    mode="single",
    max_latent_classes=2,
    R=200,
    default_roles=[0, 1, 2, 3, 4, 5, 6, 7, 8],
)

duration_result = builder.run_search(
    duration_search,
    algo="sa",
    max_iter=1500,
    seed=11,
)
```

### 9.2 Manual duration model

```python
duration_spec = builder.make_manual_spec(
    fixed_terms=["SHOULDER"],
    rdm_terms=["LANES:normal"],
    rdm_cor_terms=["RAIN:normal", "MEMB_URBAN:normal"],
    hetro_in_means=["URBAN"],
    membership_terms=["INTERSECTION_DENSITY"],
    latent_classes=2,
)

duration_fit = builder.fit_manual_model(
    manual_spec=duration_spec,
    model="lognormal",
    R=200,
)
```

## 10. Linear Models

The default linear route now uses the main JAX hierarchical architecture with a Gaussian family.

### 10.1 Linear search

```python
linear_search = builder.build_evaluator(
    model_family="linear",
    variables=["LINEAR_X1", "LINEAR_X2", "LINEAR_X3", "MEMB_URBAN"],
    mode="single",
    max_latent_classes=2,
    R=200,
    default_roles=[0, 1, 2, 3, 4, 5, 6, 7, 8],
)

linear_result = builder.run_search(
    linear_search,
    algo="hs",
    max_iter=1500,
    seed=13,
)
```

### 10.2 Manual linear model

```python
linear_spec = builder.make_manual_spec(
    fixed_terms=["LINEAR_X1"],
    rdm_terms=["LINEAR_X2:normal"],
    rdm_cor_terms=["LINEAR_X3:normal", "MEMB_URBAN:normal"],
    hetro_in_means=["URBAN"],
    zi_terms=["ZERO_FLAG"],
    membership_terms=["INTERSECTION_DENSITY"],
    latent_classes=2,
)

linear_fit = builder.fit_manual_model(
    manual_spec=linear_spec,
    model="gaussian",
    R=200,
)
```

## 11. What Changing The Search Arguments Does

### Change the search algorithm

```python
builder.run(evaluator=evaluator, algo="sa", max_iter=2000, seed=1)
builder.run(evaluator=evaluator, algo="de", max_iter=2000, seed=1)
builder.run(evaluator=evaluator, algo="hs", max_iter=2000, seed=1)
```

- `sa`
  Good default for single-objective structure search.
- `de`
  Differential-evolution style search.
- `hs`
  Harmony-search style search.
- `hc`
  Routed through the annealing-style search code.

### Change the number of simulation draws

```python
evaluator = builder.build_count_evaluator(R=500)
```

Higher `R` means:

- slower estimation
- more stable simulated mixed-model integration

### Change latent-class complexity

```python
evaluator = builder.build_count_evaluator(
    variables=["AADT", "SPEED", "MEMB_URBAN"],
    max_latent_classes=3,
)
```

Higher `max_latent_classes` means:

- richer segmentation
- more parameters
- slower fitting
- heavier BIC penalty

### Restrict the allowed structures

```python
evaluator = builder.build_count_evaluator(
    variables=["AADT", "SPEED", "ZERO_FLAG"],
    default_roles=[0, 1, 2, 6],
)
```

That limits the search to:

- exclusion
- fixed effects
- random independent effects
- zero inflation

### Restrict specific variables

```python
evaluator = builder.build_count_evaluator(
    variables=["AADT", "SPEED", "MEMB_URBAN"],
    fixed_override={"AADT": [1]},
    membership_override={"MEMB_URBAN": [7, 8]},
)
```

## 12. Consistent Run Output

To save runs consistently:

```python
from metacountregressor import SearchOutputConfig

output_config = SearchOutputConfig(
    output_dir="results",
    experiment_name="latent_class_demo",
    search_description="Latent class experiment on bundled crash data",
)

result = builder.run_search(
    cmf_search,
    algo="sa",
    max_iter=1000,
    output_config=output_config,
)

print(result["saved_to"])
```

Each saved JSON file includes:

- experiment name
- search description
- family
- algorithm
- normalized run result payload

## 13. Latent-Class Cookbook Example: Recover Functional Class

This example is designed to capture an underlying characteristic in the data, `TRUE_FUNCTIONAL_CLASS`, without using that true label directly as an outcome predictor.

### 13.1 Build a latent-class count model

We do not include `TRUE_FUNCTIONAL_CLASS` in the model specification. Instead we let the latent classes explain the hidden segmentation, and we let observable membership variables drive class probabilities.

```python
latent_spec = builder.make_manual_spec(
    fixed_terms=["AADT", "SPEED", "LANES"],
    rdm_cor_terms=["CURVE:normal", "LIGHTING:normal"],
    hetro_in_means=["RAIN"],
    membership_terms=["URBAN", "INTERSECTION_DENSITY", "MEMB_URBAN"],
    dispersion=1,
    latent_classes=2,
)

latent_fit = builder.fit_manual_model(
    manual_spec=latent_spec,
    model="nb",
    R=200,
)
```

### 13.2 Compute estimated class probabilities

```python
class_probs = builder.compute_latent_class_probabilities(
    latent_fit,
    true_class_col="TRUE_FUNCTIONAL_CLASS",
)

print(class_probs.head())
```

This returns:

- `ID`
- `class_1_prob`
- `class_2_prob`
- `TRUE_FUNCTIONAL_CLASS`

### 13.3 Compare predicted latent class to the true class

```python
class_probs["predicted_class"] = (
    class_probs[["class_1_prob", "class_2_prob"]]
    .to_numpy()
    .argmax(axis=1)
)

comparison = class_probs[["ID", "TRUE_FUNCTIONAL_CLASS", "predicted_class"]]
print(comparison.head())
```

You can also compute a simple agreement rate:

```python
agreement = (
    comparison["TRUE_FUNCTIONAL_CLASS"].to_numpy()
    == comparison["predicted_class"].to_numpy()
).mean()

print("Agreement:", agreement)
```

This is the package cookbook pattern for checking whether the latent-class model is recovering a real hidden grouping.

## 14. Common Validation Errors

The package now raises clearer errors for:

- missing dataframe columns
- invalid family-specific arguments
- CMF models without `aadt_col`, `baseline_vars`, or `local_vars`
- CMF models with non-positive `AADT`
- latent-class probability requests on single-class fits

Example:

```python
builder.build_evaluator(
    model_family="duration",
    variables=["SHOULDER"],
    budget_col="B",
    not_a_real_arg=True,
)
```

That raises a direct `ValueError`.

## 15. Practical Modeling Sequence

For count work:

1. start with a count search
2. decide whether Poisson or NB is preferred
3. allow random parameters
4. test correlated random parameters
5. add heterogeneity in means
6. test zero inflation
7. test latent classes

For CMF work:

1. start with the default JAX CMF search
2. allow random and correlated CMF terms
3. add heterogeneity variables
4. add zero-inflation variables
5. add membership variables
6. use the legacy GA-CMF route only if you specifically need it

For duration work:

1. use the duration family with the JAX hierarchical search
2. move to manual lognormal fits when you want a fixed structure

For linear work:

1. use the linear family with the JAX hierarchical search
2. move to manual Gaussian fits when you want a fixed structure
