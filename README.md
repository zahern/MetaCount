# metacountregressor Cookbook

`metacountregressor` is a JAX-first package for hierarchical model fitting and metaheuristic structure search across:

- count models
- CMF models
- duration models
- linear models

This cookbook now uses the bundled Example 16-3 data from the linked CSV source and keeps the original source column names.

## 1. Install

```bash
python -m pip install -e .
python -m pip install jax jaxlib jaxopt
```

Quick import check:

```bash
python -c "from metacountregressor import __version__, load_example16_3_raw_data; print(__version__, load_example16_3_raw_data().shape)"
```

## 2. Example Data In The Package

The package now exposes the Example 16-3 data directly:

```python
from metacountregressor import load_example16_3_raw_data, load_example16_3_model_data

raw_df = load_example16_3_raw_data()
model_df = load_example16_3_model_data()
```

### 3.1 Raw data loader

`load_example16_3_raw_data()` returns the original CSV columns:

- `ID`
- `FREQ`
- `LENGTH`
- `INCLANES`
- `DECLANES`
- `WIDTH`
- `MIMEDSH`
- `MXMEDSH`
- `SPEED`
- `URB`
- `FC`
- `AADT`
- `SINGLE`
- `DOUBLE`
- `TRAIN`
- `PEAKHR`
- `GRADEBR`
- `MIGRADE`
- `MXGRADE`
- `MXGRDIFF`
- `TANGENT`
- `CURVES`
- `MINRAD`
- `ACCESS`
- `MEDWIDTH`
- `FRICTION`
- `ADTLANE`
- `SLOPE`
- `INTECHAG`
- `AVEPRE`
- `AVESNOW`

### 3.2 Model-ready loader

`load_example16_3_model_data()` preserves all source columns and adds:

- `OFFSET`
- `FC_ENCODED`
- `FC_LABEL`

Notes:

- `FC` remains the original source coding from the Example 16-3 data.
- `FC_ENCODED` is a clean ordered encoding of the observed `FC` categories for comparison experiments.
- `FC_LABEL` is a readable string form like `FC_1`, `FC_2`, `FC_5`.

## 3. Build The Main ExperimentBuilder

```python
from metacountregressor import ExperimentBuilder, load_example16_3_model_data

df = load_example16_3_model_data()

builder = ExperimentBuilder(
    df=df,
    id_col="ID",
    y_col="FREQ",
    offset_col="OFFSET",
    group_id_col="FC",
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

In `build_evaluator(...)`:

- `variables=None`
  Uses all candidate columns.
- `fixed_override=None`
  No variable-specific fixed-role restrictions.
- `membership_override=None`
  No variable-specific membership-role restrictions.
- `exclude=None`
  Do not exclude extra columns.
- `default_roles=None`
  Let the package choose family defaults.

In CMF helpers:

- `offset_col=None`
  Allowed.
- `group_id_col=None`
  Allowed.
- `variables=None`
  Allowed.

Helpful inspection:

```python
builder.describe()
builder.suggest_config(max_latent_classes=2)
print(builder.get_family_capabilities())
print(builder.get_search_argument_guide())
```

## 4. Main Search Arguments

Shared arguments:

- `algo`
  Use `sa`, `hc`, `de`, or `hs`.
- `R`
  Number of simulation draws.
- `max_iter`
  Search iterations.
- `max_latent_classes`
  Maximum latent classes allowed.
- `variables`
  Candidate search columns.
- `default_roles`
  Allowed structural roles.
- `fixed_override`
  Restrict roles for named variables.
- `membership_override`
  Restrict membership roles for named variables.

To save results consistently:

```python
from metacountregressor import SearchOutputConfig

output_config = SearchOutputConfig(
    output_dir="results",
    experiment_name="example16_3_count_search",
    search_description="Count model search on Example 16-3 data",
)
```

## 5. Role Codes

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

Random-parameter distributions:

- `normal`
- `lognormal`
- `triangular`
- `uniform`

## 6. Count Models

### 7.1 Count search

```python
evaluator = builder.build_count_evaluator(
    variables=[
        "AADT",
        "LENGTH",
        "SPEED",
        "CURVES",
        "TANGENT",
        "SLOPE",
        "ACCESS",
        "URB",
        "AVEPRE",
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
    fixed_terms=["AADT", "LENGTH", "SPEED"],
    rdm_terms=["CURVES:normal"],
    rdm_cor_terms=["TANGENT:normal", "SLOPE:lognormal"],
    hetro_in_means=["AVEPRE"],
    zi_terms=["ACCESS"],
    membership_terms=["URB"],
    dispersion=1,
    latent_classes=2,
)

fit = builder.fit_manual_model(
    manual_spec=manual_spec,
    model="nb",
    R=200,
)
```

## 7. CMF Models

CMF models use:

```text
log(mu) = baseline block + local block * log(AADT)
```

The default CMF route transforms the CMF design and then runs on the main JAX hierarchical architecture.

### 8.1 CMF search

```python
cmf_search = builder.build_evaluator(
    model_family="cmf",
    aadt_col="AADT",
    baseline_vars=["URB", "ACCESS", "GRADEBR"],
    local_vars=["CURVES", "SLOPE", "WIDTH"],
    variables=["AVEPRE", "AVESNOW", "FC_ENCODED"],
    mode="single",
    max_latent_classes=2,
    R=200,
    default_roles=[0, 1, 2, 3, 4, 5, 6, 7, 8],
)

cmf_result = builder.run_search(
    cmf_search,
    algo="sa",
    max_iter=2000,
    seed=7,
)
```

### 8.2 Manual CMF model

```python
from metacountregressor import CMFExperimentBuilder

cmf_builder = CMFExperimentBuilder(
    df=df,
    y_col="FREQ",
    aadt_col="AADT",
    baseline_vars=["URB", "ACCESS"],
    local_vars=["CURVES", "SLOPE"],
)

manual_cmf_spec = cmf_builder.make_manual_cmf_spec(
    baseline_fixed=["URB"],
    baseline_correlated=["ACCESS"],
    local_random=["CURVES"],
    local_correlated=["SLOPE"],
    hetro_in_means=["AVEPRE"],
    zi_terms=["INTECHAG"],
    membership_terms=["FC_ENCODED"],
    dispersion=1,
    latent_classes=2,
)

cmf_fit = cmf_builder.fit_manual_cmf_model(
    id_col="ID",
    offset_col="OFFSET",
    group_id_col="FC",
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
    baseline_vars=["URB", "ACCESS"],
    local_vars=["CURVES", "SLOPE"],
)
```

## 8. Duration Models

The default duration route now uses the main JAX hierarchical architecture with a lognormal family.

Use the model-ready duration loader:

```python
from metacountregressor import ExperimentBuilder, load_example_duration_data

duration_df = load_example_duration_data()
duration_builder = ExperimentBuilder(
    df=duration_df,
    id_col="ID",
    y_col="DURATION",
    offset_col=None,
    group_id_col="FC",
)
```

### 9.1 Duration search

```python
duration_search = duration_builder.build_evaluator(
    model_family="duration",
    variables=["WIDTH", "CURVES", "SLOPE", "URB", "FC_ENCODED"],
    budget_col="AADT",
    mode="single",
    max_latent_classes=2,
    R=200,
    default_roles=[0, 1, 2, 3, 4, 5, 6, 7, 8],
)
```

### 9.2 Manual duration model

```python
duration_spec = duration_builder.make_manual_spec(
    fixed_terms=["WIDTH"],
    rdm_terms=["CURVES:normal"],
    rdm_cor_terms=["SLOPE:normal", "URB:normal"],
    hetro_in_means=["AVEPRE"],
    membership_terms=["FC_ENCODED"],
    latent_classes=2,
)

duration_fit = duration_builder.fit_manual_model(
    manual_spec=duration_spec,
    model="lognormal",
    R=200,
)
```

## 9. Linear Models

The default linear route now uses the main JAX hierarchical architecture with a Gaussian family.

Use the model-ready linear loader:

```python
from metacountregressor import ExperimentBuilder, load_example_linear_data

linear_df = load_example_linear_data()
linear_builder = ExperimentBuilder(
    df=linear_df,
    id_col="ID",
    y_col="LINEAR_TARGET",
    offset_col=None,
    group_id_col="FC",
)
```

### 10.1 Linear search

```python
linear_search = linear_builder.build_evaluator(
    model_family="linear",
    variables=["WIDTH", "CURVES", "SLOPE", "URB", "FC_ENCODED"],
    mode="single",
    max_latent_classes=2,
    R=200,
    default_roles=[0, 1, 2, 3, 4, 5, 6, 7, 8],
)
```

### 10.2 Manual linear model

```python
linear_spec = linear_builder.make_manual_spec(
    fixed_terms=["WIDTH"],
    rdm_terms=["CURVES:normal"],
    rdm_cor_terms=["SLOPE:normal", "URB:normal"],
    hetro_in_means=["AVEPRE"],
    membership_terms=["FC_ENCODED"],
    latent_classes=2,
)

linear_fit = linear_builder.fit_manual_model(
    manual_spec=linear_spec,
    model="gaussian",
    R=200,
)
```

## 10. Platform-Speed Linear Mixed Effects Example

The package now includes a synthetic example designed for linear mixed-effects style experiments around vehicle speed relative to a platform.

Load it with:

```python
from metacountregressor import load_example_platform_speed_data, ExperimentBuilder

platform_df = load_example_platform_speed_data()

platform_builder = ExperimentBuilder(
    df=platform_df,
    id_col="PLATFORM_ID",
    y_col="RELATIVE_SPEED",
    offset_col=None,
    group_id_col="PLATFORM_TYPE",
)
```

Available columns include:

- `DIST_TO_PLATFORM`
- `VEHICLE_SPEED`
- `RELATIVE_SPEED`
- `POSTED_SPEED`
- `APPROACH_ACCEL`
- `PLATFORM_TYPE`
- `PLATFORM_HEIGHT`
- `PLATFORM_WIDTH`
- `AT_PLATFORM`

### 11.1 Linear mixed-effects style search

```python
platform_linear_search = platform_builder.build_evaluator(
    model_family="linear",
    variables=[
        "DIST_TO_PLATFORM",
        "POSTED_SPEED",
        "APPROACH_ACCEL",
        "PLATFORM_HEIGHT",
        "PLATFORM_WIDTH",
        "AT_PLATFORM",
    ],
    mode="single",
    max_latent_classes=2,
    R=200,
    default_roles=[0, 1, 2, 3, 4, 5, 7, 8],
)
```

Run the search:

```python
from metacountregressor import SearchOutputConfig

platform_linear_result = platform_builder.run_search(
    platform_linear_search,
    algo="sa",
    max_iter=1500,
    seed=21,
    output_config=SearchOutputConfig(
        output_dir="results",
        experiment_name="platform_linear_speed_search",
        search_description="Linear mixed-effects style search for speed relative to platform",
    ),
)

print(platform_linear_result["algorithm"])
print(platform_linear_result["best_score"])
print(platform_linear_result["saved_to"])
```

If you want to manually fit one chosen structure after the search:

```python
platform_linear_spec = platform_builder.make_manual_spec(
    fixed_terms=["DIST_TO_PLATFORM", "POSTED_SPEED"],
    rdm_terms=["APPROACH_ACCEL:normal"],
    rdm_cor_terms=["PLATFORM_HEIGHT:normal", "PLATFORM_WIDTH:normal"],
    hetro_in_means=["AT_PLATFORM"],
    membership_terms=["DIST_TO_PLATFORM"],
    latent_classes=2,
)

platform_linear_fit = platform_builder.fit_manual_model(
    manual_spec=platform_linear_spec,
    model="gaussian",
    R=200,
)
```

This is set up to model speed relative to the platform while allowing:

- random parameters
- correlated random parameters
- grouped effects
- heterogeneity in means
- latent classes

## 11. Duration Example: Time Until Another Vehicle Speeds Over The Platform

The package also includes a synthetic duration experiment for the time before another vehicle speeds over the platform.

Load it with:

```python
from metacountregressor import load_example_platform_gap_duration_data, ExperimentBuilder

gap_df = load_example_platform_gap_duration_data()

gap_builder = ExperimentBuilder(
    df=gap_df,
    id_col="PLATFORM_ID",
    y_col="DURATION_UNTIL_NEXT_SPEEDING",
    offset_col=None,
    group_id_col=None,
)
```

Available columns include:

- `DURATION_UNTIL_NEXT_SPEEDING`
- `PRECEDING_VEHICLE_SPEED`
- `FOLLOWING_VEHICLE_SPEED`
- `POSTED_SPEED`
- `PLATFORM_HEIGHT`
- `PLATFORM_WIDTH`
- `APPROACH_VOLUME`

### 12.1 Duration search

```python
gap_duration_search = gap_builder.build_evaluator(
    model_family="duration",
    variables=[
        "PRECEDING_VEHICLE_SPEED",
        "FOLLOWING_VEHICLE_SPEED",
        "POSTED_SPEED",
        "PLATFORM_HEIGHT",
        "PLATFORM_WIDTH",
        "APPROACH_VOLUME",
    ],
    budget_col="POSTED_SPEED",
    mode="single",
    max_latent_classes=2,
    R=200,
    default_roles=[0, 1, 2, 3, 4, 5, 7, 8],
)
```

Run the search:

```python
gap_duration_result = gap_builder.run_search(
    gap_duration_search,
    algo="sa",
    max_iter=1500,
    seed=22,
    output_config=SearchOutputConfig(
        output_dir="results",
        experiment_name="platform_gap_duration_search",
        search_description="Duration search for time until another vehicle speeds over the platform",
    ),
)

print(gap_duration_result["algorithm"])
print(gap_duration_result["best_score"])
print(gap_duration_result["saved_to"])
```

If you want to manually fit one chosen duration structure:

```python
gap_duration_spec = gap_builder.make_manual_spec(
    fixed_terms=["PRECEDING_VEHICLE_SPEED", "POSTED_SPEED"],
    rdm_terms=["FOLLOWING_VEHICLE_SPEED:normal"],
    rdm_cor_terms=["PLATFORM_HEIGHT:normal", "PLATFORM_WIDTH:normal"],
    hetro_in_means=["APPROACH_VOLUME"],
    latent_classes=2,
)

gap_duration_fit = gap_builder.fit_manual_model(
    manual_spec=gap_duration_spec,
    model="lognormal",
    R=200,
)
```

This uses the JAX hierarchical lognormal path and is intended for duration-before-speeding style analysis.

## 12. What Changing Search Arguments Does

### Change the search algorithm

```python
builder.run(evaluator=evaluator, algo="sa", max_iter=2000, seed=1)
builder.run(evaluator=evaluator, algo="de", max_iter=2000, seed=1)
builder.run(evaluator=evaluator, algo="hs", max_iter=2000, seed=1)
```

### Change simulation draws

```python
evaluator = builder.build_count_evaluator(R=500)
```

Higher `R` means:

- slower estimation
- more stable simulation-based fitting

### Restrict allowed structures

```python
evaluator = builder.build_count_evaluator(
    variables=["AADT", "SPEED", "ACCESS"],
    default_roles=[0, 1, 2, 6],
)
```

### Restrict specific variables

```python
evaluator = builder.build_count_evaluator(
    variables=["AADT", "SPEED", "URB"],
    fixed_override={"AADT": [1]},
    membership_override={"URB": [7, 8]},
)
```

## 13. Consistent Run Output

```python
from metacountregressor import SearchOutputConfig

output_config = SearchOutputConfig(
    output_dir="results",
    experiment_name="cmf_example16_3",
    search_description="CMF search on Example 16-3 data",
)

saved = builder.run_search(
    cmf_search,
    algo="sa",
    max_iter=1000,
    output_config=output_config,
)

print(saved["saved_to"])
```

Each saved JSON file stores:

- experiment name
- search description
- family
- algorithm
- normalized result payload

## 14. Latent-Class Example: Recover Functional Class

This example is designed to see whether a latent-class model can recover the hidden `FC` grouping pattern without using `FC` itself as a direct predictor in the outcome equation.

We keep:

- original truth column: `FC`
- comparison encoding: `FC_ENCODED`

We do not place `FC` or `FC_ENCODED` in the outcome equation. Instead we let membership variables explain latent class probabilities.

### 13.1 Fit a latent-class count model

```python
latent_spec = builder.make_manual_spec(
    fixed_terms=["AADT", "SPEED", "LENGTH"],
    rdm_cor_terms=["CURVES:normal", "SLOPE:normal"],
    hetro_in_means=["AVEPRE"],
    membership_terms=["URB", "ACCESS", "GRADEBR"],
    dispersion=1,
    latent_classes=2,
)

latent_fit = builder.fit_manual_model(
    manual_spec=latent_spec,
    model="nb",
    R=200,
)
```

### 13.2 Compute latent-class probabilities and compare to the true FC grouping

```python
class_probs = builder.compute_latent_class_probabilities(
    latent_fit,
    true_class_col="FC_ENCODED",
)

print(class_probs.head())
```

Returned columns include:

- `ID`
- `class_1_prob`
- `class_2_prob`
- `FC_ENCODED`

### 13.3 Compare predicted class with the encoded true class

```python
class_probs["predicted_class"] = (
    class_probs[["class_1_prob", "class_2_prob"]]
    .to_numpy()
    .argmax(axis=1)
)

agreement = (
    class_probs["predicted_class"].to_numpy()
    == class_probs["FC_ENCODED"].to_numpy()
).mean()

print("Agreement:", agreement)
```

This is the cookbook pattern for checking whether the latent-class structure is capturing the observed facility-class segmentation.

## 15. Common Validation Errors

The package now raises clearer errors for:

- missing columns
- invalid family-specific arguments
- CMF specifications missing `aadt_col`, `baseline_vars`, or `local_vars`
- CMF data with non-positive `AADT`
- latent-class probability requests on single-class fits

## 16. Summary

Use these loaders when you want the real Example 16-3 data inside the package:

- `load_example16_3_raw_data()`
- `load_example16_3_model_data()`
- `load_example_duration_data()`
- `load_example_linear_data()`

Use these builder patterns:

- count: `build_count_evaluator(...)`
- CMF: `build_evaluator(model_family="cmf", ...)`
- duration: `build_evaluator(model_family="duration", ...)`
- linear: `build_evaluator(model_family="linear", ...)`


## 17. Running on HPC Clusters

The package automatically detects walltime limits from HPC schedulers and integrates them into the metaheuristic search algorithms to prevent job timeouts.

### Automatic Walltime Detection

When running on PBS/Torque or SLURM clusters, the package automatically reads the walltime limit from environment variables:

- **PBS**: `PBS_WALLTIME` (format: HH:MM:SS)
- **SLURM**: `SLURM_TIME_LIMIT` (format: seconds or HH:MM:SS)

The walltime is automatically converted to seconds and used as a `max_time` parameter in the optimization algorithms, enabling early stopping before the job is killed by the scheduler.

### HPC Job Script Example

Here's an example PBS job script (`job_script.pbs`):

```bash
#!/bin/bash
#PBS -N metaheuristic_batch_job
#PBS -l nodes=1:ppn=4
#PBS -l walltime=04:00:00
#PBS -l mem=16gb
#PBS -j oe
#PBS -o output.log

module load python/3.8.0
cd $PBS_O_WORKDIR

python main_hpc_new.py --algo hs --max_iter 5000 --seed 42
```

### Running with Walltime Awareness

The main HPC script (`main_hpc_new.py`) automatically detects the walltime and passes it to the search algorithms:

```bash
# Walltime automatically detected from PBS_WALLTIME or SLURM_TIME_LIMIT
python main_hpc_new.py --algo sa --max_iter 10000 --seed 123
```

### Manual Walltime Override

You can also manually specify a maximum runtime in seconds:

```bash
python main_hpc_new.py --algo de --max_iter 5000 --max_time 7200 --seed 456
```

This is useful for testing or when the environment variables are not set correctly.

### Supported Algorithms with Walltime

All metaheuristic algorithms support walltime-based early stopping:

- **SA (Simulated Annealing)**: Stops when walltime is exceeded
- **DE (Differential Evolution)**: Respects time limits during population evolution
- **HS (Harmony Search)**: Terminates search when time limit reached

The algorithms will complete the current iteration and save results before stopping, ensuring no work is lost.
