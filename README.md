# metacountregressor

A JAX-first Python package for hierarchical model fitting and metaheuristic-driven model structure search.  Supports count, CMF, duration, and linear models with random parameters, latent classes, zero-inflation, and heterogeneity in means — all with one unified API.

---

## Table of Contents

- [Install](#install)
- [Notebooks — Start Here](#notebooks--start-here)
- [Quick Start](#quick-start)
- [What the package does](#what-the-package-does)
- [Data loaders](#data-loaders)
- [ExperimentBuilder API](#experimentbuilder-api)
- [ModelConstraints API](#modelconstraints-api)
- [Role codes](#role-codes)
- [Search algorithms](#search-algorithms)
- [Model families](#model-families)
  - [Count models](#count-models-poisson--negative-binomial)
  - [CMF models](#cmf-models)
  - [Duration models](#duration-models)
  - [Linear models](#linear-models)
- [Latent class models](#latent-class-models)
- [Output and saving results](#output-and-saving-results)
- [Help system](#help-system)
- [Running on HPC clusters](#running-on-hpc-clusters)

---

## Install

```bash
pip install metacountregressor
pip install jax jaxlib jaxopt   # JAX backend
```

Quick import check:

```bash
python -c "from metacountregressor import __version__, load_example16_3_raw_data; print(__version__, load_example16_3_raw_data().shape)"
```

---

## Notebooks — Start Here

The fastest way to learn the package is to open the notebooks in order.
Each one builds on the previous and uses the bundled Example 16-3 crash-frequency data
so you can run everything without sourcing your own dataset.

| # | Notebook | What you learn |
| --- | -------- | -------------- |
| 00 | [00_quickstart.ipynb](metacountregressor/metacountregressor/templates/00_quickstart.ipynb) | Install, load data, first search run in under 10 minutes |
| 01 | [01_crash_frequency_search.ipynb](metacountregressor/metacountregressor/templates/01_crash_frequency_search.ipynb) | Mixed Negative Binomial search — constraints, roles, re-fit |
| 02 | [02_latent_class_fc_validation.ipynb](metacountregressor/metacountregressor/templates/02_latent_class_fc_validation.ipynb) | 2-class LC model — fit, extract class probabilities, validate against FC |
| 03 | [03_cmf_aadt_search.ipynb](metacountregressor/metacountregressor/templates/03_cmf_aadt_search.ipynb) | CMF model — baseline + AADT-interaction structure search |
| 04 | [04_linear_speed_prediction.ipynb](metacountregressor/metacountregressor/templates/04_linear_speed_prediction.ipynb) | Gaussian linear model search (speed prediction) |
| 05 | [05_batch_script_tutorial.ipynb](metacountregressor/metacountregressor/templates/05_batch_script_tutorial.ipynb) | Batch scripts, parallel runs, PBS/SLURM HPC job templates |

> **Tip:** open with `jupyter lab metacountregressor/metacountregressor/templates/` to browse all notebooks together.

---

## Quick Start

```python
import numpy as np
from metacountregressor import (
    ExperimentBuilder,
    ModelConstraints,
    SearchOutputConfig,
    load_example16_3_model_data,
    get_help,
)

# ── 1. Load the bundled crash-frequency dataset ──────────────────────────────
df = load_example16_3_model_data()
exposure = df['LENGTH'] * df['AADT'] * 365 / 1e8
df['OFFSET'] = np.log(exposure.clip(lower=1e-9))

# ── 2. Build constraints ──────────────────────────────────────────────────────
c = (
    ModelConstraints()
    .force_include('OFFSET')
    .no_zi('LENGTH', 'CURVES', 'WIDTH', 'SLOPE')
    .no_random('URB')
    .allow_random('CURVES', distributions=['lognormal'])
)

# ── 3. Create the experiment ──────────────────────────────────────────────────
builder = ExperimentBuilder(df, id_col='ID', y_col='FREQ', offset_col='OFFSET')
builder.describe()              # print data summary
get_help('crash_frequency')     # print end-to-end workflow guide

# ── 4. Build the structure evaluator ─────────────────────────────────────────
evaluator = builder.build_evaluator(
    variables=['AADT', 'LENGTH', 'SPEED', 'CURVES', 'URB', 'AVEPRE'],
    constraints=c,
    default_roles=[0, 1, 2, 3, 5],
    max_latent_classes=1,
    R=200,
)

# ── 5. Run the search ─────────────────────────────────────────────────────────
result = builder.run(
    evaluator,
    algo='sa',          # 'sa' | 'de' | 'hs'
    max_iter=1000,
    seed=42,
    output_config=SearchOutputConfig(output_dir='results', experiment_name='demo'),
)
print('Best BIC:', result.best_score)
print('Saved to:', result.saved_to)

# ── 6. Re-fit with more draws ─────────────────────────────────────────────────
fit = builder.fit_manual_model(manual_spec=result.best_spec, model='nb', R=500)
print(fit)
```

---

## What the package does

`metacountregressor` solves two related problems:

1. **Structure search** — automatically discover which variables to include, whether each coefficient should be fixed or random, and whether the model needs latent classes, zero-inflation, or heterogeneity in means.  The search is driven by metaheuristic algorithms (SA, DE, HS) that minimise BIC.

2. **Model estimation** — fit the discovered (or manually specified) model structure using JAX-accelerated simulation-based maximum likelihood with Halton draws.

The same API handles crash-frequency count models, CMF (Crash Modification Factor) models, duration models, and linear (Gaussian) models.

---

## Data loaders

All loaders return a `pandas.DataFrame`.

```python
from metacountregressor import (
    load_example16_3_raw_data,      # Example 16-3: original 31 columns
    load_example16_3_model_data,    # + OFFSET, FC_ENCODED, FC_LABEL
    load_example_crash_data,        # alias for load_example16_3_model_data
    load_example_duration_data,     # synthetic duration target from Ex 16-3
    load_example_linear_data,       # synthetic linear target from Ex 16-3
    load_example_platform_speed_data,           # speed relative to platform
    load_example_platform_gap_duration_data,    # time until next speeding event
    load_example_panel_data,        # panel-structure example
)
```

### Example 16-3 columns

`load_example16_3_raw_data()` returns the original source columns:

| Group | Columns |
| ------- | ------- |
| Identifiers | `ID` |
| Outcome | `FREQ` |
| Geometry | `LENGTH`, `WIDTH`, `INCLANES`, `DECLANES`, `MEDWIDTH`, `MIMEDSH`, `MXMEDSH` |
| Speed / grade | `SPEED`, `MIGRADE`, `MXGRADE`, `MXGRDIFF`, `SLOPE` |
| Traffic | `AADT`, `SINGLE`, `DOUBLE`, `TRAIN`, `PEAKHR`, `ADTLANE` |
| Road class | `URB`, `FC`, `ACCESS`, `TANGENT`, `CURVES`, `MINRAD`, `GRADEBR` |
| Friction / weather | `FRICTION`, `INTECHAG`, `AVEPRE`, `AVESNOW` |

`load_example16_3_model_data()` adds `OFFSET`, `FC_ENCODED`, `FC_LABEL`.

---

## ExperimentBuilder API

```python
from metacountregressor import ExperimentBuilder

builder = ExperimentBuilder(
    df=df,
    id_col='ID',           # required — observation identifier
    y_col='FREQ',          # required — outcome variable
    offset_col='OFFSET',   # optional — log-exposure offset (count models)
    group_id_col='FC',     # optional — group/panel identifier
)
```

### Key methods

| Method | Purpose |
| -------- | ------- |
| `builder.describe()` | Print data summary: N, outcome stats, variable types |
| `builder.suggest_config(max_latent_classes=2)` | Print recommended ExperimentBuilder settings |
| `builder.build_evaluator(...)` | Build a structure evaluator (see below) |
| `builder.build_count_evaluator(...)` | Shortcut for count models |
| `builder.run(evaluator, algo, max_iter, seed, ...)` | Run metaheuristic search |
| `builder.run_search(evaluator, ...)` | Alias for `run()` |
| `builder.make_manual_spec(...)` | Build a model spec dict manually |
| `builder.fit_manual_model(manual_spec, model, R)` | Fit a manually specified structure |
| `builder.compute_latent_class_probabilities(fit, true_class_col)` | Get class membership probabilities |
| `ExperimentBuilder.get_family_capabilities()` | Static: list supported model families |
| `ExperimentBuilder.get_search_argument_guide()` | Static: full argument documentation |

### build_evaluator arguments

```python
evaluator = builder.build_evaluator(
    variables=['AADT', 'LENGTH', 'SPEED', 'CURVES'],   # candidate columns
    constraints=c,                                       # ModelConstraints object
    model_family='count',          # 'count' | 'cmf' | 'duration' | 'linear'
    default_roles=[0, 1, 2, 3, 5], # roles the search may assign
    max_latent_classes=2,          # 1 = standard, 2 = allow LC
    mode='single',                 # 'single' = minimise BIC
    R=200,                         # Halton simulation draws
    # CMF-only arguments:
    aadt_col='AADT',
    baseline_vars=['URB', 'ACCESS'],
    local_vars=['CURVES', 'WIDTH'],
    # Duration-only:
    budget_col='AADT',
)
```

---

## ModelConstraints API

`ModelConstraints` restricts which roles and distributions each variable may take.
All methods return `self` for chaining.

```python
from metacountregressor import ModelConstraints

c = (
    ModelConstraints()
    .force_include('OFFSET')                          # cannot be excluded
    .force_fixed('AADT')                              # only fixed or excluded
    .no_zi('LENGTH', 'CURVES', 'SLOPE', 'WIDTH')      # cannot be ZI term
    .no_random('URB', 'GRADEBR')                      # no random parameter
    .allow_random('CURVES', distributions=['lognormal'])  # restrict distribution
    .membership_only('FC_ENCODED')                    # drives class prob only
    .allow_membership('SPEED')                        # may also enter membership
    .outcome_only('AADT')                             # no membership role
    .exclude('YEAR', 'ID')                            # removed from search
    .set_roles('WIDTH', [0, 1, 2])                    # low-level override
)

print(c)           # display all constraints
c.summary()        # same as print(c)
```

Get detailed API documentation:

```python
from metacountregressor import get_help
get_help('constraints')
```

---

## Role codes

| Code | Name | Description |
| ---- | ---- | ----------- |
| `0` | Excluded | Variable not in the model |
| `1` | Fixed | Same coefficient for every observation |
| `2` | Random (ind.) | Individual random effect, independent draws |
| `3` | Random (corr.) | Individual random effect, correlated with others |
| `4` | Grouped | Group-level random effect (shared within group) |
| `5` | Heterogeneity | Explains variation in random-parameter means |
| `6` | Zero Inflation | Enters the zero-inflation probability equation |
| `7` | Membership only | Drives latent-class probability — not the outcome |
| `8` | Membership + Fixed | Drives class membership AND has class-specific outcome effect |

Random-parameter distributions: `normal`, `lognormal`, `triangular`, `uniform`.

```python
get_help('roles')   # full reference with examples
```

---

## Search algorithms

| Alias | Algorithm | Best for |
| ----- | --------- | -------- |
| `'sa'` | Simulated Annealing | Robust default — escapes local minima via cooling schedule |
| `'de'` | Differential Evolution | Thorough population-based search — use when SA converges early |
| `'hs'` | Harmony Search | Fast initial convergence — good for a quick first pass |

```python
# Run the same evaluator with different algorithms
result_sa = builder.run(evaluator, algo='sa', max_iter=2000, seed=42)
result_de = builder.run(evaluator, algo='de', max_iter=2000, seed=42)
result_hs = builder.run(evaluator, algo='hs', max_iter=2000, seed=42)
```

```python
get_help('metaheuristics')   # full parameter reference
```

---

## Model families

### Count models (Poisson / Negative Binomial)

```python
evaluator = builder.build_count_evaluator(
    variables=['AADT', 'LENGTH', 'SPEED', 'CURVES', 'URB', 'AVEPRE'],
    constraints=c,
    default_roles=[0, 1, 2, 3, 5],
    max_latent_classes=1,
    R=200,
)
result = builder.run(evaluator, algo='sa', max_iter=2000, seed=42)
fit = builder.fit_manual_model(manual_spec=result.best_spec, model='nb', R=500)
```

Manual spec:

```python
spec = builder.make_manual_spec(
    fixed_terms=['AADT', 'LENGTH', 'SPEED'],
    rdm_terms=['CURVES:normal'],
    rdm_cor_terms=['TANGENT:normal', 'SLOPE:lognormal'],
    hetro_in_means=['AVEPRE'],
    zi_terms=['ACCESS'],
    membership_terms=['URB'],
    dispersion=1,
    latent_classes=2,
)
fit = builder.fit_manual_model(manual_spec=spec, model='nb', R=200)
```

### CMF models

```python
from metacountregressor import CMFExperimentBuilder

cmf = CMFExperimentBuilder(
    df=df,
    y_col='FREQ',
    aadt_col='AADT',
    baseline_vars=['URB', 'ACCESS', 'GRADEBR', 'CURVES'],
    local_vars=['CURVES', 'WIDTH'],
)

# Route A: full JAX flexibility (random params, LC, ZI)
builder_jax, evaluator_jax, meta = cmf.build_jax_count_evaluator(
    id_col='ID', offset_col='OFFSET', constraints=c, max_latent_classes=1, R=200)
result = builder_jax.run(evaluator_jax, algo='sa', max_iter=500, seed=42)

# Route B: classic GA search (fast, two-component structure)
search = cmf.run_search(R=200)
fit = cmf.fit_best_model(search, final_R=500)
cmf.print_report(search, fit)
```

```python
get_help('cmf')   # full workflow guide
```

### Duration models

```python
from metacountregressor import load_example_duration_data

duration_df = load_example_duration_data()
duration_builder = ExperimentBuilder(
    df=duration_df, id_col='ID', y_col='DURATION', group_id_col='FC')

evaluator = duration_builder.build_evaluator(
    variables=['WIDTH', 'CURVES', 'SLOPE', 'URB', 'FC_ENCODED'],
    model_family='duration',
    default_roles=[0, 1, 2, 3],
    max_latent_classes=1, R=200,
)
result = duration_builder.run(evaluator, algo='sa', max_iter=500, seed=42)
fit = duration_builder.fit_manual_model(manual_spec=result.best_spec,
                                        model='lognormal', R=500)
```

### Linear models

```python
from metacountregressor import load_example_platform_speed_data

speed_df = load_example_platform_speed_data()
speed_builder = ExperimentBuilder(
    df=speed_df, id_col='PLATFORM_ID', y_col='SPEED', offset_col=None)

evaluator = speed_builder.build_evaluator(
    variables=['DIST_TO_PLATFORM', 'POSTED_SPEED', 'APPROACH_ACCEL',
               'PLATFORM_HEIGHT', 'PLATFORM_WIDTH'],
    model_family='linear',
    default_roles=[0, 1, 2, 3],   # no ZI for linear
    max_latent_classes=1, R=200,
)
result = speed_builder.run(evaluator, algo='sa', max_iter=500, seed=42)
fit = speed_builder.fit_manual_model(manual_spec=result.best_spec,
                                     model='gaussian', R=500)
```

---

## Latent class models

```python
# 1. Constrain FC_ENCODED to drive class membership only
c = (
    ModelConstraints()
    .membership_only('FC_ENCODED')
    .force_include('OFFSET')
    .no_zi('LENGTH', 'CURVES', 'WIDTH', 'SLOPE')
    .no_random('URB', 'GRADEBR')
)

# 2. Build LC evaluator (max_latent_classes=2, include roles 7 & 8)
evaluator = builder.build_evaluator(
    variables=['URB', 'ACCESS', 'GRADEBR', 'CURVES', 'LENGTH',
               'SPEED', 'WIDTH', 'SLOPE', 'AVEPRE', 'FC_ENCODED'],
    constraints=c,
    default_roles=[0, 1, 2, 3, 5, 7, 8],
    max_latent_classes=2,
    R=150,
)

# 3. Run search
result = builder.run(evaluator, algo='sa', max_iter=500, seed=1)

# 4. Manually fit a specific structure
spec = builder.make_manual_spec(
    fixed_terms=['AADT', 'SPEED', 'LENGTH'],
    rdm_cor_terms=['CURVES:normal', 'SLOPE:normal'],
    hetro_in_means=['AVEPRE'],
    membership_terms=['URB', 'ACCESS', 'GRADEBR'],
    dispersion=1, latent_classes=2,
)
fit = builder.fit_manual_model(manual_spec=spec, model='nb', R=200)

# 5. Extract class membership probabilities
class_probs = builder.compute_latent_class_probabilities(
    fit, true_class_col='FC_ENCODED')
print(class_probs.head())

# 6. Compare predicted class vs actual FC
class_probs['predicted'] = (
    class_probs[['class_1_prob', 'class_2_prob']].to_numpy().argmax(axis=1))
agreement = (class_probs['predicted'] == class_probs['FC_ENCODED']).mean()
print(f'Agreement with FC: {agreement:.1%}')
```

Pre-specified reference model:

```python
from metacountregressor import (
    load_book_latent_class_spec, describe_book_latent_class_spec)

describe_book_latent_class_spec()
spec = load_book_latent_class_spec()
fit = builder.fit_manual_model(manual_spec=spec, model='nb', R=200)
```

```python
get_help('latent_class')   # full workflow guide
```

---

## Output and saving results

```python
from metacountregressor import SearchOutputConfig

output_config = SearchOutputConfig(
    output_dir='results',
    experiment_name='example16_3_count',
    search_description='NB count model search on Example 16-3',
    save_json=True,
)

result = builder.run(evaluator, algo='sa', max_iter=2000,
                     output_config=output_config)
print('Saved to:', result.saved_to)
```

Each saved JSON contains: experiment name, description, model family, algorithm, best BIC, and the best structural specification.

Collect results from multiple runs:

```python
import json, pathlib

results = sorted(
    [json.load(open(f)) for f in pathlib.Path('results').glob('*.json')],
    key=lambda r: r.get('best_score', float('inf'))
)
print('Best BIC:', results[0]['best_score'])
print('Algorithm:', results[0]['algorithm'])
```

---

## Help system

The package includes a built-in interactive help system:

```python
from metacountregressor import get_help

get_help()                    # list all topics
get_help('roles')             # role code table + distributions
get_help('constraints')       # ModelConstraints API
get_help('metaheuristics')    # algorithm comparison and parameters
get_help('crash_frequency')   # count model workflow
get_help('latent_class')      # latent class workflow
get_help('cmf')               # CMF workflow
get_help('linear')            # linear model workflow
get_help('duration')          # duration model workflow
get_help('batch')             # batch script and HPC guide
```

---

## Running on HPC clusters

### Automatic walltime detection

On PBS/Torque or SLURM, the package reads the scheduler walltime automatically and uses it as a `max_time` limit — the search stops cleanly before the job is killed.

| Scheduler | Environment variable | Format |
| --------- | ------------------- | ------ |
| PBS/Torque | `PBS_WALLTIME` | `HH:MM:SS` |
| SLURM | `SLURM_TIME_LIMIT` | seconds or `HH:MM:SS` |

Set manually for local testing:

```python
result = builder.run(evaluator, algo='sa', max_iter=99999, max_time=3600)
```

### PBS job script

```bash
#!/bin/bash
#PBS -N metacount_sa
#PBS -l nodes=1:ppn=4
#PBS -l walltime=04:00:00
#PBS -l mem=16gb
#PBS -j oe
#PBS -o logs/sa_seed42.log

module load python/3.11
cd $PBS_O_WORKDIR
source venv/bin/activate

# Walltime auto-detected from PBS_WALLTIME
python run_experiment.py sa 42 200 99999
```

### SLURM job array

```bash
#!/bin/bash
#SBATCH --job-name=metacount
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --output=logs/%j.log
#SBATCH --array=1-10

module load python/3.11
source venv/bin/activate

python run_experiment.py sa $SLURM_ARRAY_TASK_ID 200 99999
```

See [05_batch_script_tutorial.ipynb](metacountregressor/metacountregressor/templates/05_batch_script_tutorial.ipynb) for a complete worked example including a reusable `run_experiment.py` template and result-collection scripts.

```python
get_help('batch')   # inline guide
```
