"""
help.py
-------
Interactive help system and pre-specified reference models for metacountregressor.

Usage
-----
>>> from metacountregressor import get_help
>>> get_help()                       # list all topics
>>> get_help('roles')                # role code reference
>>> get_help('constraints')          # ModelConstraints API
>>> get_help('metaheuristics')       # algorithm guide
>>> get_help('crash_frequency')      # count-model workflow
>>> get_help('latent_class')         # latent-class workflow
>>> get_help('cmf')                  # CMF model workflow
>>> get_help('linear')               # linear model workflow
>>> get_help('duration')             # duration model workflow
>>> get_help('batch')                # batch / HPC job guide
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Role / distribution constants (duplicated here to avoid circular imports)
# ---------------------------------------------------------------------------
_ROLE_TABLE = [
    ("0", "Excluded",          "Variable not in the model at all"),
    ("1", "Fixed",             "Same coefficient for every observation"),
    ("2", "Random (ind.)",     "Individual random effect — independent draws"),
    ("3", "Random (corr.)",    "Individual random effect — correlated with others"),
    ("4", "Grouped",           "Group-level random effect (shared within group)"),
    ("5", "Heterogeneity",     "Explains variation in random-parameter MEANS"),
    ("6", "Zero Inflation",    "Enters the ZI probability equation"),
    ("7", "Membership only",   "Drives latent-class probability — NOT the outcome"),
    ("8", "Membership + Fixed","Drives class membership AND has a class-specific outcome effect"),
]

_DIST_TABLE = [
    ("normal",     "Symmetric, unbounded. Default for unrestricted effects."),
    ("lognormal",  "Positive-only. Good for effects that must be positive."),
    ("triangular", "Bounded, symmetric. Useful when sign is constrained."),
    ("uniform",    "Flat prior over a range. Rarely used; try normal first."),
]

_ALGO_TABLE = [
    ("sa",  "Simulated Annealing",     "Probabilistic hill-climbing. Robust, good default. Escape local minima via temperature schedule."),
    ("de",  "Differential Evolution",  "Population-based mutation. Thorough exploration. Slower per-iteration but very stable."),
    ("hs",  "Harmony Search",          "Music-inspired. Rapid early convergence. Good when you want a quick first pass."),
]

# ---------------------------------------------------------------------------
# Topic text
# ---------------------------------------------------------------------------

_TOPICS: dict[str, str] = {}

_TOPICS["roles"] = """
ROLE CODE REFERENCE
===================
Each variable in the model is assigned one of these roles during the search.

  Code  Name                 Description
  ----  -------------------  --------------------------------------------------
  0     Excluded             Variable not in the model at all
  1     Fixed                Same coefficient for every observation
  2     Random (ind.)        Individual random effect — independent draws
  3     Random (corr.)       Individual random effect — correlated with others
  4     Grouped              Group-level random effect (shared within group)
  5     Heterogeneity        Explains variation in random-parameter MEANS
  6     Zero Inflation       Enters the ZI probability equation
  7     Membership only      Drives latent-class probability — NOT the outcome
  8     Membership + Fixed   Drives class membership AND outcome (class-specific coeff)

Notes:
  • Roles 7 and 8 only activate when max_latent_classes >= 2.
  • Roles 2, 3, 4 require choosing a distribution (see get_help('roles') for dists).
  • Role 5 (Heterogeneity-in-means) requires at least one other variable with role 2/3/4.

RANDOM-PARAMETER DISTRIBUTIONS
================================
  Distribution  Best used when ...
  ------------  --------------------------------------------------
  normal        Symmetric, unbounded. Default for unrestricted effects.
  lognormal     Positive-only. Good for effects that must be positive (e.g. AADT).
  triangular    Bounded, symmetric. Useful when sign is constrained.
  uniform       Flat prior over a range. Rarely used; try normal first.

QUICK EXAMPLES
==============
  default_roles=[0, 1, 2, 3]            # exclude, fixed, or random
  default_roles=[0, 1, 2, 3, 5]         # + heterogeneity in means
  default_roles=[0, 1, 2, 3, 5, 6]      # + zero inflation
  default_roles=[0, 1, 2, 3, 5, 7, 8]   # + latent-class membership
"""

_TOPICS["constraints"] = """
MODELCONSTRAINTS API
====================
ModelConstraints is a fluent builder that restricts which roles and
distributions each variable may take during the search.

All methods return 'self' for chaining.  Pass the constraints object to
build_evaluator(constraints=c, ...).

METHOD REFERENCE
----------------
  c.force_fixed('VAR')
      VAR may only be excluded (0) or fixed (1).
      Use for: forced AADT, mandatory exposure terms.

  c.force_include('VAR')
      VAR cannot be excluded (role 0 banned).
      Use for: exposure offsets that must always appear.

  c.no_zi('VAR1', 'VAR2', ...)
      VAR cannot enter the zero-inflation equation (role 6 banned).
      Use for: geometric road features where ZI is implausible.

  c.no_random('VAR1', 'VAR2', ...)
      VAR cannot have a random parameter (roles 2, 3, 4 banned).
      Use for: binary dummies, categorical indicators.

  c.allow_random('VAR', distributions=['normal', 'lognormal'])
      VAR MAY be a random parameter, restricted to listed distributions.
      Does NOT force it to be random — search can still choose fixed.

  c.membership_only('VAR')
      VAR can only drive latent-class membership (role 7) or be excluded.
      Has no effect when max_latent_classes=1.

  c.allow_membership('VAR')
      VAR may additionally enter the class-membership equation.

  c.outcome_only('VAR')
      VAR cannot enter class-membership equation (roles 7, 8 banned).

  c.exclude('VAR')
      Completely removes VAR from the search space.

  c.set_roles('VAR', [0, 1, 5])
      Low-level: directly set the allowed role codes for VAR.

  c.set_distributions('VAR', ['normal'])
      Low-level: directly set allowed distributions for VAR.

  c.summary()   OR   print(c)
      Display all current constraints.

EXAMPLE (chained)
-----------------
  from metacountregressor import ModelConstraints

  c = (
      ModelConstraints()
      .force_include('OFFSET')
      .no_zi('LENGTH', 'CURVES', 'SLOPE', 'WIDTH')
      .no_random('URB', 'GRADEBR')
      .allow_random('CURVES', distributions=['lognormal'])
      .membership_only('FC_ENCODED')
  )
  print(c)
"""

_TOPICS["metaheuristics"] = """
METAHEURISTIC ALGORITHM GUIDE
==============================
All three algorithms minimise BIC (or [BIC, RMSE] in multi-objective mode)
over the discrete decision space of model structures.

  Alias  Algorithm               Best for
  -----  ----------------------  --------------------------------------------------
  'sa'   Simulated Annealing     Robust default. Escapes local minima via a
                                 temperature schedule.  Good first choice.
  'de'   Differential Evolution  Population-based. Thorough exploration of the full
                                 structure space. Use when SA converges too early.
  'hs'   Harmony Search          Music-inspired. Fast initial convergence. Good for
                                 a quick first pass or tight iteration budgets.

SHARED PARAMETERS (pass to builder.run())
------------------------------------------
  algo         str     'sa' | 'de' | 'hs'
  max_iter     int     Number of candidate evaluations  (default 2000)
  seed         int     Random seed for reproducibility  (default 42)
  max_time     float   Wall-time limit in seconds (0 = no limit)

SA-SPECIFIC PARAMETERS (pass as **kwargs to builder.run())
-----------------------------------------------------------
  alpha            float   Cooling rate (0 < alpha < 1, default ≈ 0.97)
  STEPS_PER_TEMP   int     Evaluations per temperature level (default 10)
  INTL_ACPT        float   Initial acceptance probability (default 0.9)

DE-SPECIFIC PARAMETERS
-----------------------
  _pop_size          int     Population size (default 20)
  _crossover_perc    float   Crossover fraction (default 0.8)

HS-SPECIFIC PARAMETERS
-----------------------
  HMS    int     Harmony memory size (default 20)
  HMCR   float   Harmony memory consideration rate (default 0.9)
  PAR    float   Pitch adjustment rate (default 0.3)

CHOOSING max_iter
-----------------
  Quick check / sanity run:  max_iter=50–100
  Moderate search:           max_iter=500–1000
  Production search:         max_iter=2000–5000
  HPC overnight run:         max_iter=10000+  (set max_time from walltime)

HPC WALLTIME INTEGRATION
-------------------------
  On PBS/SLURM the package reads the walltime automatically:
    PBS:   PBS_WALLTIME  (HH:MM:SS)
    SLURM: SLURM_TIME_LIMIT  (seconds or HH:MM:SS)
  Set max_time manually for local testing:
    result = builder.run(evaluator, algo='sa', max_iter=99999, max_time=3600)
"""

_TOPICS["crash_frequency"] = """
CRASH FREQUENCY (COUNT) MODEL WORKFLOW
=======================================
Count models (Poisson or Negative Binomial) are the standard model type
for crash-frequency data.  The outcome is a non-negative integer count.

STEP 1 — LOAD DATA
-------------------
  from metacountregressor import load_example16_3_model_data, ExperimentBuilder

  df = load_example16_3_model_data()
  # Columns: ID, FREQ, LENGTH, AADT, URB, FC, SPEED, CURVES, ...
  # Also adds: OFFSET, FC_ENCODED, FC_LABEL

STEP 2 — BUILD EXPOSURE OFFSET (if not already in data)
---------------------------------------------------------
  import numpy as np
  exposure = df['LENGTH'] * df['AADT'] * 365 / 1e8
  df['OFFSET'] = np.log(exposure.clip(lower=1e-9))

STEP 3 — CREATE ExperimentBuilder
-----------------------------------
  builder = ExperimentBuilder(
      df=df,
      id_col='ID',
      y_col='FREQ',
      offset_col='OFFSET',   # or None
      group_id_col='FC',     # or None
  )
  builder.describe()                    # print data summary
  builder.suggest_config()              # get recommended settings

STEP 4 — SET CONSTRAINTS
--------------------------
  from metacountregressor import ModelConstraints

  c = (
      ModelConstraints()
      .force_include('OFFSET')
      .no_zi('LENGTH', 'CURVES', 'WIDTH', 'SLOPE')
      .no_random('URB')
      .allow_random('CURVES', distributions=['lognormal'])
  )
  print(c)

STEP 5 — BUILD EVALUATOR
--------------------------
  evaluator = builder.build_evaluator(
      variables=['AADT', 'LENGTH', 'SPEED', 'CURVES', 'URB', 'AVEPRE'],
      constraints=c,
      default_roles=[0, 1, 2, 3, 5],   # roles the search may use
      max_latent_classes=1,             # 1 = standard, 2 = latent class
      mode='single',                    # 'single' = minimise BIC
      R=200,                            # Halton draws (200–500 typical)
  )

STEP 6 — RUN SEARCH
---------------------
  result = builder.run(evaluator, algo='sa', max_iter=1000, seed=42)
  print('Best BIC:', result.best_score)
  print('Best structure:', result.best_solution)

STEP 7 — RE-FIT BEST MODEL
----------------------------
  fit = builder.fit_manual_model(
      manual_spec=result.best_spec,
      model='nb',    # 'nb' = Negative Binomial,  'poisson' = Poisson
      R=500,         # more draws for accurate standard errors
  )
  print(fit)

See also: get_help('latent_class'), get_help('constraints'), get_help('roles')
"""

_TOPICS["latent_class"] = """
LATENT CLASS MODEL WORKFLOW
============================
Latent class (LC) models split the population into C unobserved subgroups,
each with its own parameter vector.  The class a site belongs to is not
known — it is estimated from the data.

WHEN TO USE LATENT CLASSES
---------------------------
  • You suspect distinct risk sub-populations (e.g. urban vs rural corridors)
  • Standard mixed models have poor fit or large over-dispersion
  • You want to test whether a known grouping (FC, road type) explains heterogeneity

KEY ROLES FOR LATENT-CLASS MODELS
-----------------------------------
  Role 7  Membership only      Variable drives CLASS PROBABILITY but not the outcome
  Role 8  Membership + Fixed   Variable drives class probability AND has class-specific coeff

STEP 1 — CONSTRAINTS FOR LC SEARCH
-------------------------------------
  from metacountregressor import ModelConstraints

  c = (
      ModelConstraints()
      # Let FC drive class membership only (not the outcome equation)
      .membership_only('FC_ENCODED')
      .force_include('OFFSET')
      .no_zi('LENGTH', 'CURVES', 'WIDTH', 'SLOPE')
      .no_random('URB', 'GRADEBR')
  )

STEP 2 — BUILD LC EVALUATOR
-----------------------------
  evaluator = builder.build_evaluator(
      variables=['URB', 'ACCESS', 'GRADEBR', 'CURVES', 'LENGTH',
                 'SPEED', 'WIDTH', 'SLOPE', 'AVEPRE', 'FC_ENCODED'],
      constraints=c,
      default_roles=[0, 1, 2, 3, 5, 7, 8],   # include membership roles!
      max_latent_classes=2,                    # allow up to 2 classes
      R=150,
  )

STEP 3 — RUN SEARCH
---------------------
  result = builder.run(evaluator, algo='sa', max_iter=500, seed=1)

STEP 4 — MANUAL FIT WITH SPECIFIC STRUCTURE
---------------------------------------------
  latent_spec = builder.make_manual_spec(
      fixed_terms=['AADT', 'SPEED', 'LENGTH'],
      rdm_cor_terms=['CURVES:normal', 'SLOPE:normal'],
      hetro_in_means=['AVEPRE'],
      membership_terms=['URB', 'ACCESS', 'GRADEBR'],
      dispersion=1,
      latent_classes=2,
  )
  fit = builder.fit_manual_model(manual_spec=latent_spec, model='nb', R=200)

STEP 5 — EXTRACT CLASS PROBABILITIES
--------------------------------------
  class_probs = builder.compute_latent_class_probabilities(
      fit,
      true_class_col='FC_ENCODED',    # optional: for comparison
  )
  print(class_probs.head())
  # Columns: ID, class_1_prob, class_2_prob, FC_ENCODED

STEP 6 — COMPARE PREDICTED CLASS VS TRUE CLASS
------------------------------------------------
  import pandas as pd
  class_probs['predicted_class'] = (
      class_probs[['class_1_prob', 'class_2_prob']].to_numpy().argmax(axis=1)
  )
  agreement = (class_probs['predicted_class'] == class_probs['FC_ENCODED']).mean()
  print(f'Agreement with FC: {agreement:.1%}')

COMPARING 1-CLASS VS 2-CLASS BIC
----------------------------------
  Fit both and compare BIC directly.  Lower BIC = better fit.
  BIC penalty for extra class ≈ K * log(N) where K = new parameters.
  Rule of thumb: accept 2-class if ΔBIC > 10.

See also: get_help('roles'), get_help('constraints'), get_help('crash_frequency')
"""

_TOPICS["cmf"] = """
CRASH MODIFICATION FACTOR (CMF) MODEL WORKFLOW
================================================
CMF models separate the AADT (traffic volume) effect from site characteristics.

MODEL STRUCTURE
---------------
  log(mu_i) = [alpha_0 + SUM_k alpha_k * X_ki]            (Baseline block)
            + [beta_0  + SUM_k beta_k  * X_ki] * log(AADT_i)  (Local block)

  CMF for baseline predictor X_k:  exp(alpha_k)
    e.g. alpha_k = 0.15  →  CMF = exp(0.15) = 1.16  (+16% crashes)
    e.g. alpha_k = -0.30 →  CMF = exp(-0.30) = 0.74  (-26% crashes)

  CMF for AADT-interaction predictor (at mean AADT):
    CMF = AADT_mean ^ beta_k

TWO ROUTES
-----------
  1. JAX full-flexibility route (recommended)
     - Full role system 0-8, latent classes, random parameters, ZI
     - Use CMFExperimentBuilder.build_jax_count_evaluator()

  2. Classic GA CMF route
     - Original two-component AADT structure only; fast; no LC
     - Use CMFExperimentBuilder.run_search()

STEP 1 — CREATE CMFExperimentBuilder
--------------------------------------
  from metacountregressor import CMFExperimentBuilder, load_example16_3_model_data

  df = load_example16_3_model_data()
  cmf = CMFExperimentBuilder(
      df=df,
      y_col='FREQ',
      aadt_col='AADT',
      baseline_vars=['URB', 'ACCESS', 'GRADEBR', 'CURVES'],   # Component A
      local_vars=['CURVES', 'WIDTH'],                           # Component B
  )

STEP 2A — JAX SEARCH (full flexibility)
-----------------------------------------
  from metacountregressor import ModelConstraints

  c = (
      ModelConstraints()
      .no_zi('CURVES', 'WIDTH', 'GRADEBR')
      .no_random('URB', 'GRADEBR')
      .allow_random('CURVES', distributions=['lognormal'])
  )

  builder_jax, evaluator_jax, meta = cmf.build_jax_count_evaluator(
      id_col='ID',
      offset_col='OFFSET',
      constraints=c,
      max_latent_classes=1,
      R=200,
  )
  result = builder_jax.run(evaluator_jax, algo='sa', max_iter=500, seed=42)
  print('Best BIC:', result.best_score)

STEP 2B — CLASSIC GA SEARCH
------------------------------
  search_result = cmf.run_search(R=200)
  fit_result = cmf.fit_best_model(search_result, final_R=500)
  cmf.print_report(search_result, fit_result)

STEP 3 — MANUAL CMF FIT
--------------------------
  manual_spec = cmf.make_manual_cmf_spec(
      baseline_fixed=['URB'],
      baseline_correlated=['ACCESS'],
      local_random=['CURVES'],
      local_correlated=['SLOPE'],
      hetro_in_means=['AVEPRE'],
      membership_terms=['FC_ENCODED'],
      dispersion=1,
      latent_classes=2,
  )
  fit = cmf.fit_manual_cmf_model(id_col='ID', offset_col='OFFSET',
                                  manual_spec=manual_spec, model='nb', R=200)
  print(fit)

See also: get_help('roles'), get_help('constraints'), get_help('latent_class')
"""

_TOPICS["linear"] = """
LINEAR (GAUSSIAN) MODEL WORKFLOW
==================================
Linear models are for real-valued continuous outcomes (speed, gap time, etc.).

KEY DIFFERENCES FROM COUNT MODELS
------------------------------------
  • No exposure offset needed (offset_col=None)
  • Zero-inflation (role 6) is NOT applicable — omit from default_roles
  • The dispersion parameter is residual variance, not count over-dispersion
  • model='gaussian' in fit_manual_model()

STEP 1 — LOAD DATA
-------------------
  from metacountregressor import load_example_platform_speed_data, ExperimentBuilder

  df = load_example_platform_speed_data()
  # Columns: PLATFORM_ID, SPEED, DIST_TO_PLATFORM, POSTED_SPEED,
  #          APPROACH_ACCEL, PLATFORM_HEIGHT, PLATFORM_WIDTH, AT_PLATFORM

STEP 2 — BUILD EXPERIMENT
---------------------------
  builder = ExperimentBuilder(
      df=df,
      id_col='PLATFORM_ID',
      y_col='SPEED',
      offset_col=None,           # no offset for linear models
      group_id_col='PLATFORM_TYPE',
  )
  builder.describe()

STEP 3 — CONSTRAINTS
----------------------
  from metacountregressor import ModelConstraints

  c = (
      ModelConstraints()
      # ZI is not meaningful for continuous outcomes
      .no_zi('DIST_TO_PLATFORM', 'POSTED_SPEED', 'APPROACH_ACCEL',
             'PLATFORM_HEIGHT', 'PLATFORM_WIDTH', 'AT_PLATFORM')
  )

STEP 4 — BUILD EVALUATOR
--------------------------
  evaluator = builder.build_evaluator(
      variables=['DIST_TO_PLATFORM', 'POSTED_SPEED', 'APPROACH_ACCEL',
                 'PLATFORM_HEIGHT', 'PLATFORM_WIDTH'],
      constraints=c,
      model_family='linear',
      default_roles=[0, 1, 2, 3],   # NO role 6 (ZI)
      max_latent_classes=1,
      R=200,
  )

STEP 5 — RUN SEARCH + RE-FIT
------------------------------
  result = builder.run(evaluator, algo='sa', max_iter=500, seed=42)
  fit = builder.fit_manual_model(
      manual_spec=result.best_spec, model='gaussian', R=500)
  print(fit)

See also: get_help('duration'), get_help('roles'), get_help('constraints')
"""

_TOPICS["duration"] = """
DURATION (LOGNORMAL) MODEL WORKFLOW
=====================================
Duration models are for positive, right-skewed outcomes: travel time,
time-to-event, gap duration, incident duration.

KEY DIFFERENCES FROM COUNT MODELS
------------------------------------
  • No exposure offset needed (offset_col=None)
  • Zero-inflation (role 6) is NOT applicable
  • model='lognormal' in fit_manual_model()
  • Outcome must be strictly positive

STEP 1 — LOAD DATA
-------------------
  from metacountregressor import load_example_duration_data, ExperimentBuilder

  df = load_example_duration_data()
  # Columns: ID, DURATION, FC, AADT, LENGTH, CURVES, SLOPE, WIDTH, URB, ...

STEP 2 — BUILD EXPERIMENT
---------------------------
  builder = ExperimentBuilder(
      df=df,
      id_col='ID',
      y_col='DURATION',
      offset_col=None,
      group_id_col='FC',
  )
  builder.describe()

STEP 3 — BUILD EVALUATOR
--------------------------
  evaluator = builder.build_evaluator(
      variables=['WIDTH', 'CURVES', 'SLOPE', 'URB', 'FC_ENCODED'],
      model_family='duration',
      budget_col='AADT',          # optional: for budget-penalty objective
      default_roles=[0, 1, 2, 3],
      max_latent_classes=1,
      R=200,
  )

STEP 4 — RUN SEARCH + RE-FIT
------------------------------
  result = builder.run(evaluator, algo='sa', max_iter=500, seed=42)
  fit = builder.fit_manual_model(
      manual_spec=result.best_spec, model='lognormal', R=500)
  print(fit)

PLATFORM GAP DURATION EXAMPLE
-------------------------------
  from metacountregressor import load_example_platform_gap_duration_data

  gap_df = load_example_platform_gap_duration_data()
  # Columns: PLATFORM_ID, DURATION_UNTIL_NEXT_SPEEDING,
  #          PRECEDING_VEHICLE_SPEED, FOLLOWING_VEHICLE_SPEED,
  #          POSTED_SPEED, PLATFORM_HEIGHT, PLATFORM_WIDTH, APPROACH_VOLUME

See also: get_help('linear'), get_help('roles'), get_help('constraints')
"""

_TOPICS["batch"] = """
BATCH SCRIPT & HPC JOB GUIDE
==============================
Running metacountregressor on a cluster or in a batch script.

PYTHON BATCH SCRIPT PATTERN
-----------------------------
Create a file, e.g.  run_experiment.py:

  import sys
  from metacountregressor import (
      ExperimentBuilder, ModelConstraints, SearchOutputConfig,
      load_example16_3_model_data
  )
  import numpy as np

  # --- Configuration (edit these) ---
  ALGO   = sys.argv[1] if len(sys.argv) > 1 else 'sa'
  SEED   = int(sys.argv[2]) if len(sys.argv) > 2 else 42
  R      = int(sys.argv[3]) if len(sys.argv) > 3 else 200
  ITER   = int(sys.argv[4]) if len(sys.argv) > 4 else 2000

  df = load_example16_3_model_data()
  exposure = df['LENGTH'] * df['AADT'] * 365 / 1e8
  df['OFFSET'] = np.log(exposure.clip(lower=1e-9))

  builder = ExperimentBuilder(df, id_col='ID', y_col='FREQ', offset_col='OFFSET')

  evaluator = builder.build_evaluator(
      variables=['AADT','LENGTH','SPEED','CURVES','URB','AVEPRE'],
      default_roles=[0, 1, 2, 3, 5],
      max_latent_classes=1, R=R,
  )

  result = builder.run(
      evaluator, algo=ALGO, max_iter=ITER, seed=SEED,
      output_config=SearchOutputConfig(
          output_dir='results',
          experiment_name=f'search_{ALGO}_seed{SEED}',
      ),
  )
  print('Done. BIC:', result.best_score, '| saved to:', result.saved_to)

Run it:
  python run_experiment.py sa 42 200 2000
  python run_experiment.py de 7  300 5000

PARALLEL RUNS (shell loop)
----------------------------
  for seed in 1 2 3 4 5; do
      python run_experiment.py sa $seed 200 2000 &
  done
  wait
  echo "All done"

PBS/TORQUE JOB SCRIPT
----------------------
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

  python run_experiment.py sa 42 200 99999
  # max_time is set automatically from PBS_WALLTIME

Submit:  qsub job_sa.pbs
Check:   qstat -u $USER

SLURM JOB SCRIPT
-----------------
  #!/bin/bash
  #SBATCH --job-name=metacount_sa
  #SBATCH --nodes=1
  #SBATCH --ntasks=4
  #SBATCH --time=04:00:00
  #SBATCH --mem=16G
  #SBATCH --output=logs/sa_%j.log

  module load python/3.11
  cd $SLURM_SUBMIT_DIR
  source venv/bin/activate

  python run_experiment.py sa $SLURM_ARRAY_TASK_ID 200 99999

Submit array:  sbatch --array=1-10 job_sa.slurm
Check:         squeue -u $USER

WALLTIME AWARENESS
-------------------
The package auto-detects walltime from PBS_WALLTIME and SLURM_TIME_LIMIT.
To set manually:
  result = builder.run(evaluator, algo='sa', max_iter=99999, max_time=3600)
  # Stops cleanly after 3600 seconds and saves results.

COLLECTING RESULTS
-------------------
Each run writes a JSON file to output_dir/.  Collect them:

  import json, pathlib
  results = []
  for f in pathlib.Path('results').glob('*.json'):
      with open(f) as fh:
          results.append(json.load(fh))

  # Sort by BIC (lower = better)
  results.sort(key=lambda r: r.get('best_score', float('inf')))
  print('Best BIC:', results[0]['best_score'])
  print('Algorithm:', results[0]['algorithm'])

See notebook 05_batch_script_tutorial.ipynb for a full worked example.
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_help(topic: str | None = None) -> None:
    """
    Print an interactive usage guide for *topic*.

    Call with no arguments (or topic=None) to list all available topics.

    Parameters
    ----------
    topic : str, optional
        One of: 'roles', 'constraints', 'metaheuristics', 'crash_frequency',
        'latent_class', 'cmf', 'linear', 'duration', 'batch'.
        Case-insensitive.

    Examples
    --------
    >>> from metacountregressor import get_help
    >>> get_help()
    >>> get_help('roles')
    >>> get_help('latent_class')
    """
    if topic is None:
        _print_index()
        return

    key = topic.lower().strip().replace(" ", "_").replace("-", "_")
    # Allow short aliases
    _aliases = {
        "count": "crash_frequency",
        "nb": "crash_frequency",
        "poisson": "crash_frequency",
        "lc": "latent_class",
        "latentclass": "latent_class",
        "constraint": "constraints",
        "algo": "metaheuristics",
        "algorithms": "metaheuristics",
        "sa": "metaheuristics",
        "de": "metaheuristics",
        "hs": "metaheuristics",
        "hpc": "batch",
        "hpc_batch": "batch",
        "gaussian": "linear",
        "lognormal": "duration",
    }
    key = _aliases.get(key, key)

    if key not in _TOPICS:
        print(f"Unknown topic: {topic!r}")
        print(f"Available topics: {', '.join(sorted(_TOPICS))}")
        return

    print(_TOPICS[key])


def _print_index() -> None:
    print("""
METACOUNTREGRESSOR — INTERACTIVE HELP SYSTEM
=============================================
Call get_help('<topic>') for detailed guidance on any topic below.

  TOPIC                  DESCRIPTION
  ---------------------  -------------------------------------------------------
  'roles'                Role code table and random-parameter distributions
  'constraints'          ModelConstraints API — all methods with examples
  'metaheuristics'       Algorithm comparison (SA / DE / HS) and parameters
  'crash_frequency'      End-to-end count model workflow (Poisson / NB)
  'latent_class'         Latent class NB model — fitting and validation
  'cmf'                  CMF model — baseline + AADT-interaction workflow
  'linear'               Gaussian linear model workflow
  'duration'             Lognormal duration model workflow
  'batch'                Batch script and HPC job guide (PBS / SLURM)

QUICK START
-----------
  from metacountregressor import (
      ExperimentBuilder, ModelConstraints, SearchOutputConfig,
      load_example16_3_model_data, get_help
  )
  df = load_example16_3_model_data()
  builder = ExperimentBuilder(df, id_col='ID', y_col='FREQ', offset_col='OFFSET')
  builder.describe()
  get_help('crash_frequency')

NOTEBOOKS
---------
  00_quickstart.ipynb               — Install, load data, first run
  01_crash_frequency_search.ipynb   — Mixed NB search on Example 16-3
  02_latent_class_fc_validation.ipynb — LC model and FC recovery
  03_cmf_aadt_search.ipynb          — CMF model search
  04_linear_speed_prediction.ipynb  — Linear model search
  05_batch_script_tutorial.ipynb    — Batch jobs and HPC
""")


# ---------------------------------------------------------------------------
# Pre-specified "book" model specs for Example 16-3
# ---------------------------------------------------------------------------

def load_book_latent_class_spec() -> dict:
    """
    Return the structural specification for a representative 2-class
    Negative Binomial latent class model on the Example 16-3 crash-frequency
    data.

    This spec is drawn from the mixed-model crash-frequency literature and
    serves as a reference point for the FC validation experiment (notebook 02).

    Returns
    -------
    dict
        A spec dict compatible with ``ExperimentBuilder.fit_manual_model()``.
        Keys: fixed_terms, rdm_terms, rdm_cor_terms, grouped_terms,
        hetro_in_means, zi_terms, membership_terms, dispersion, latent_classes.

    Examples
    --------
    >>> from metacountregressor import load_book_latent_class_spec
    >>> spec = load_book_latent_class_spec()
    >>> fit = builder.fit_manual_model(manual_spec=spec, model='nb', R=200)
    """
    return {
        "fixed_terms": ["AADT", "LENGTH", "SPEED"],
        "rdm_terms": [],
        "rdm_cor_terms": ["CURVES:normal", "SLOPE:normal"],
        "grouped_terms": [],
        "hetro_in_means": ["AVEPRE"],
        "zi_terms": ["ACCESS"],
        "membership_terms": ["URB", "GRADEBR"],
        "dispersion": 1,
        "latent_classes": 2,
    }


def describe_book_latent_class_spec() -> None:
    """
    Print a human-readable description of the book 2-class NB specification.

    Examples
    --------
    >>> from metacountregressor import describe_book_latent_class_spec
    >>> describe_book_latent_class_spec()
    """
    spec = load_book_latent_class_spec()
    print("""
BOOK LATENT CLASS SPECIFICATION — Example 16-3
===============================================
This is a 2-class Negative Binomial model representative of the
crash-frequency mixed-model literature.

Outcome equation (per class):
  Fixed terms:       {fixed}
  Correlated random: {rdm_cor}  (normal distribution)
  Heterogeneity:     {hetro}    (explains variance in random means)
  Zero inflation:    {zi}

Class membership equation (logit):
  Membership vars:   {mem}

Dispersion:         {disp}  (1 = NB over-dispersion estimated)
Latent classes:     {lc}

To fit this model:
  from metacountregressor import load_book_latent_class_spec
  spec = load_book_latent_class_spec()
  fit = builder.fit_manual_model(manual_spec=spec, model='nb', R=200)
""".format(
        fixed=", ".join(spec["fixed_terms"]),
        rdm_cor=", ".join(spec["rdm_cor_terms"]),
        hetro=", ".join(spec["hetro_in_means"]),
        zi=", ".join(spec["zi_terms"]) or "(none)",
        mem=", ".join(spec["membership_terms"]),
        disp=spec["dispersion"],
        lc=spec["latent_classes"],
    ))


def load_book_cmf_spec() -> dict:
    """
    Return a reference CMF structural specification for Example 16-3.

    Returns
    -------
    dict
        Keys: baseline_fixed, baseline_random, local_fixed, local_random,
        hetro_in_means, membership_terms, dispersion, latent_classes.

    Examples
    --------
    >>> from metacountregressor import load_book_cmf_spec
    >>> spec = load_book_cmf_spec()
    >>> manual_spec = cmf.make_manual_cmf_spec(**spec)
    """
    return {
        "baseline_fixed": ["URB", "ACCESS"],
        "baseline_random": ["GRADEBR"],
        "local_fixed": ["CURVES"],
        "local_random": [],
        "hetro_in_means": ["AVEPRE"],
        "membership_terms": ["FC_ENCODED"],
        "dispersion": 1,
        "latent_classes": 1,
    }


def describe_book_cmf_spec() -> None:
    """
    Print a human-readable description of the reference CMF specification.

    Examples
    --------
    >>> from metacountregressor import describe_book_cmf_spec
    >>> describe_book_cmf_spec()
    """
    spec = load_book_cmf_spec()
    print("""
REFERENCE CMF SPECIFICATION — Example 16-3
===========================================
Model structure:
  log(mu_i) = [alpha_0 + baseline_block]
            + [beta_0  + local_block] * log(AADT_i)

Baseline block (Component A — independent of AADT):
  Fixed:          {bf}
  Random (corr.): {br}  (correlated normal)

Local block (Component B — multiplied by log(AADT)):
  Fixed:          {lf}
  Random (corr.): {lr}  (correlated normal)

Heterogeneity in means: {hetro}
Membership terms:       {mem}
Dispersion:  {disp}  |  Latent classes: {lc}

Interpreting CMF values:
  alpha_k = 0.15  →  CMF = exp(0.15) = 1.16  (+16% crashes)
  alpha_k = -0.30 →  CMF = exp(-0.30) = 0.74  (-26% crashes)
  beta_k at mean AADT=10000 →  CMF = 10000^beta_k

To fit this model:
  from metacountregressor import load_book_cmf_spec
  spec = load_book_cmf_spec()
  manual_spec = cmf.make_manual_cmf_spec(**spec)
  fit = cmf.fit_manual_cmf_model(id_col='ID', manual_spec=manual_spec,
                                  model='nb', R=200)
""".format(
        bf=", ".join(spec["baseline_fixed"]) or "(none)",
        br=", ".join(spec["baseline_random"]) or "(none)",
        lf=", ".join(spec["local_fixed"]) or "(none)",
        lr=", ".join(spec["local_random"]) or "(none)",
        hetro=", ".join(spec["hetro_in_means"]) or "(none)",
        mem=", ".join(spec["membership_terms"]) or "(none)",
        disp=spec["dispersion"],
        lc=spec["latent_classes"],
    ))
