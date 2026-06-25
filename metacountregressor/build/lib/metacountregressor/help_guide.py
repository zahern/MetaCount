"""
help_guide.py
-------------
Inline usage guide for the metacountregressor package.

Usage
-----
>>> from metacountregressor import get_help
>>> get_help()              # topic index
>>> get_help("data")        # loading data
>>> get_help("constraints") # ModelConstraints API
>>> get_help("latent_class")
>>> get_help("cmf")
>>> get_help("linear")

Download template notebooks
---------------------------
>>> from metacountregressor import get_templates
>>> get_templates()               # copies to current directory
>>> get_templates("/path/to/dir") # copies to specific directory
"""

from __future__ import annotations

import os
import shutil


# ---------------------------------------------------------------------------
# Topic registry
# ---------------------------------------------------------------------------

_TOPICS: dict[str, str] = {}


def _topic(key: str):
    """Decorator - register a function's return value as a help topic."""
    def decorator(fn):
        _TOPICS[key] = fn()
        return fn
    return decorator


# ---------------------------------------------------------------------------
# Topics
# ---------------------------------------------------------------------------

@_topic("data")
def _data_topic() -> str:
    return """
+======================================================================+
|  LOADING DATA                                                        |
+======================================================================+

Quick-start with the bundled Example 16-3 dataset
--------------------------------------------------
  from metacountregressor import (
      load_example16_3_model_data,   # recommended - includes OFFSET, FC_ENCODED
      load_example16_3_raw_data,     # original CSV columns only
  )
  df = load_example16_3_model_data()

Other bundled datasets
----------------------
  load_example_crash_data()             # generic crash count data
  load_example_linear_data()            # continuous outcome (speed)
  load_example_platform_speed_data()    # platform speed observations
  load_example_duration_data()          # time-to-event
  load_example_panel_data()             # repeated-measures panel

Loading your own data
---------------------
  import pandas as pd
  df = pd.read_csv("path/to/your_data.csv")

Required columns
----------------
  Column          Role          Notes
  --------------  ------------  -------------------------------------
  ID / site_id    Identifier    One unique value per observation unit
  FREQ / Y        Outcome       Non-negative integer (count models)
  OFFSET          Exposure      log(exposure) - use 0 if no exposure
  GROUP_ID        (optional)    For grouped / panel random effects

Creating an exposure offset
---------------------------
  # vehicle-miles travelled exposure for crash frequency:
  df["OFFSET"] = np.log(df["LENGTH"] * df["AADT"] * 365 / 1e8)
  df["OFFSET"] = df["OFFSET"].clip(lower=0)   # guard against log(0)

  # if no exposure, use zero (model has an intercept-like baseline):
  df["OFFSET"] = 0

See also: get_help("variables"), get_help("crash_frequency")
"""


@_topic("variables")
def _variables_topic() -> str:
    return """
+======================================================================+
|  SELECTING VARIABLES AND ROLE CODES                                  |
+======================================================================+

Pass a list of column names to build_evaluator().  The search algorithm
will decide the role (structural position) of each variable.

  variables = ["URB", "ACCESS", "GRADEBR", "CURVES", "LENGTH", "SPEED"]
  evaluator = builder.build_evaluator(variables=variables, ...)

Role codes (0-8)
----------------
  Code  Name                  Description
  ----  --------------------  -----------------------------------------
  0     Excluded              Not in model
  1     Fixed                 Constant coefficient across all sites
  2     Random Independent    Site-level draws, independent
  3     Random Correlated     Site-level draws, joint covariance matrix
  4     Grouped               Group-level shared draws
  5     Heterogeneity         Explains variation in random-effect means
  6     Zero Inflation        Enters the zero-probability sub-equation
  7     Membership only       Drives latent class probability only
  8     Membership + Fixed    Drives class probability AND has a fixed
                              effect in the outcome equation

Distribution options (for roles 2, 3, 4)
-----------------------------------------
  "normal"      Symmetric, unbounded           (default)
  "lognormal"   Strictly positive effect
  "triangular"  Bounded, symmetric
  "uniform"     Flat / no prior shape

Controlling which roles each variable may take
-----------------------------------------------
  Use ModelConstraints (recommended - see get_help("constraints"))
  or pass raw dicts:

  fixed_override = {
      "OFFSET": [1],          # must be fixed (never excluded)
      "SPEED":  [0, 1, 6],    # excluded, fixed, or ZI only
  }
  membership_override = {
      "FC_ENCODED": [0, 7],   # excluded or membership only
  }
  evaluator = builder.build_evaluator(
      variables=variables,
      fixed_override=fixed_override,
      membership_override=membership_override,
  )

Structural decisions you can hard-code
---------------------------------------
  - No zero-inflation at all: omit role 6 from default_roles
      evaluator = builder.build_evaluator(
          default_roles=[0, 1, 2, 3, 4, 5],   # no 6
          ...
      )
  - Only fixed and random (no ZI, no membership):
      default_roles=[0, 1, 2, 3]
  - Latent classes with membership search:
      max_latent_classes=2, default_roles=[0, 1, 2, 3, 5, 7, 8]

See also: get_help("constraints")
"""


@_topic("constraints")
def _constraints_topic() -> str:
    return """
+======================================================================+
|  MODELCONSTRAINTS API                                                |
+======================================================================+

ModelConstraints is a fluent builder that lets you express structural
restrictions on variables in plain English before passing them to
build_evaluator().

Basic usage
-----------
  from metacountregressor import ModelConstraints

  c = ModelConstraints()
  c.force_fixed("AADT")           # must be fixed (role 1) or excluded
  c.no_zi("SPEED", "WIDTH")       # cannot be zero-inflation terms
  c.membership_only("FC_ENCODED") # drives class membership only (role 7)
  c.allow_random("CURVES", distributions=["normal", "lognormal"])
  c.no_random("URB")              # binary dummy - no taste variation
  c.force_include("OFFSET")       # cannot be dropped from model
  c.exclude("YEAR", "ID")         # always excluded from search

  # Inspect the resolved constraints:
  print(c)

  # Pass to evaluator:
  evaluator = builder.build_evaluator(constraints=c, variables=[...])

All methods return self - chain them:
--------------------------------------
  c = (ModelConstraints()
       .force_fixed("AADT")
       .no_zi("SPEED", "WIDTH", "GRADEBR")
       .membership_only("FC_ENCODED")
       .allow_random("CURVES", "LENGTH"))

Method reference
----------------
  .force_fixed(*vars)                   Roles: {0, 1} only
  .force_include(*vars)                 Role 0 banned (must appear)
  .no_zi(*vars)                         Role 6 banned
  .no_random(*vars)                     Roles 2, 3, 4 banned
  .allow_random(*vars, distributions)   Roles 2-4 allowed, opt. dists
  .membership_only(*vars)               Roles: {0, 7} only
  .allow_membership(*vars)              Add roles 7, 8 to existing set
  .outcome_only(*vars)                  Roles 7, 8 banned
  .exclude(*vars)                       Remove from variable list
  .set_roles(var, [roles])              Low-level direct role override
  .set_distributions(var, [dists])      Low-level distribution override
  .to_evaluator_kwargs()                Convert to build_evaluator() dict

Typical constraint patterns
----------------------------
  # Crash frequency: no ZI for road geometry vars, FC drives membership
  c = (ModelConstraints()
       .force_include("OFFSET")
       .no_zi("LENGTH", "GRADEBR", "CURVES", "WIDTH", "SLOPE")
       .membership_only("FC_ENCODED")
       .allow_random("CURVES", distributions=["lognormal"])
       .exclude("ID", "FREQ"))

  # Speed model: linear family, no ZI anywhere
  c = (ModelConstraints()
       .no_zi("SPEED", "INCLANES", "WIDTH")
       .no_random("URB"))

  # CMF: AADT always fixed, no ZI
  c = (ModelConstraints()
       .force_fixed("AADT")
       .no_zi("LENGTH", "CURVES", "WIDTH", "GRADEBR"))

See also: get_help("crash_frequency"), get_help("latent_class")
"""


@_topic("metaheuristics")
def _metaheuristics_topic() -> str:
    return """
+======================================================================+
|  METAHEURISTICS: SA, DE, HS                                          |
+======================================================================+

The search engine supports three main algorithms:

  "sa"  Simulated Annealing      (single-objective; robust default)
  "de"  Differential Evolution   (multi-objective / Pareto-style)
  "hs"  Harmony Search           (multi-objective / Pareto-style)

Quick start (single-objective BIC)
----------------------------------
  result = builder.run(
      evaluator=evaluator,
      algo="sa",
      max_iter=1500,
      seed=42,
      mutation_rate=0.3,
      step_size=1,
      min_changes=1,
      max_changes=3,
      alpha=0.995,
  )

Common tuning knobs
-------------------
  max_iter      Total evaluations / generations
  n_starts      SA restarts (higher = more global exploration)
  mutation_rate Chance of modifying each decision element
  step_size     Size of local neighborhood move
  alpha         SA cooling parameter (closer to 1 = slower cooling)
  seed          Reproducibility

What the result dict contains
-----------------------------
  result["best_solution"]   Best decision vector
  result["best_score"]      Best objective value (typically BIC)
  result["scores"]          Objective values over sampled candidates
  result["solutions"]       Candidate decision vectors evaluated

Compatibility note
------------------
  Older examples may use keys "best_fitness", "best_decision", or
  "history". Current API uses "best_score", "best_solution", and
  "scores".

See also: get_help("variables"), get_help("constraints")
"""


@_topic("crash_frequency")
def _crash_frequency_topic() -> str:
    return """
+======================================================================+
|  CRASH FREQUENCY SEARCH - END TO END                                 |
+======================================================================+

1. Load data
------------
  from metacountregressor import load_example16_3_model_data, ExperimentBuilder
  import numpy as np

  df = load_example16_3_model_data()

  # Create exposure offset (vehicle-miles x 10^-8)
  df["OFFSET"] = np.log(
      (df["LENGTH"] * df["AADT"] * 365 / 1e8).clip(lower=1e-9)
  )

2. Select variables and constraints
------------------------------------
  from metacountregressor import ModelConstraints

  variables = [
      "URB", "ACCESS", "GRADEBR", "CURVES", "LENGTH",
      "SPEED", "WIDTH", "SLOPE", "AVEPRE",
  ]

  c = (ModelConstraints()
       .force_include("OFFSET")          # keep offset in
       .no_zi("LENGTH", "CURVES",        # geometry vars can't be ZI
               "GRADEBR", "WIDTH")
       .allow_random("CURVES",           # taste variation in curvature
                     distributions=["lognormal"]))

3. Build and run
----------------
  builder = ExperimentBuilder(
      df=df, id_col="ID", y_col="FREQ", offset_col="OFFSET"
  )
  builder.describe()   # prints data summary and role guide

  evaluator = builder.build_evaluator(
      variables=variables,
      constraints=c,
      default_roles=[0, 1, 2, 3, 5],    # no ZI by default
      max_latent_classes=1,
      mode="single",
      R=200,
  )

  result = builder.run(
      evaluator=evaluator,
      algo="sa",       # simulated annealing (try "de" or "hs" too)
      max_iter=50,
      seed=42,
  )

4. Inspect results
------------------
  print("Best BIC:", result.best_score)
  print("Best structure:", result.best_solution)

  # Re-fit the best model with more draws for final standard errors:
  fit = builder.fit_manual_model(
      manual_spec=result.best_spec,
      model="nb",
      R=500,
  )

Tips
----
  - Start with max_iter=10 for a smoke test, then increase to 100+
  - Use algo="de" (differential evolution) for wider search
  - model="nb" (Negative Binomial) handles over-dispersion common in crashes
  - Add mode="multi" for Pareto search on [BIC, test-RMSE]

See also: get_help("variables"), get_help("latent_class")
"""


@_topic("latent_class")
def _latent_class_topic() -> str:
    return """
+======================================================================+
|  LATENT CLASS SEARCH AND FC VALIDATION                               |
+======================================================================+

Latent class models assume the population is composed of C unobserved
sub-groups with different crash-risk parameters.  Membership variables
shift the probability of belonging to each class.

Basic latent class search (2 classes)
--------------------------------------
  evaluator = builder.build_evaluator(
      variables=variables,
      constraints=c,
      max_latent_classes=2,
      default_roles=[0, 1, 2, 3, 5, 7, 8],  # includes membership roles
      mode="single",
      R=150,
  )
  result = builder.run(evaluator=evaluator, algo="sa", max_iter=30)

Hypothesis: functional class drives latent sub-populations
-----------------------------------------------------------
  Functional class (FC) codes the road type (freeway, arterial, local...).
  If FC explains why sites cluster into different risk profiles, the
  latent classes should align with FC categories.

  from metacountregressor import ModelConstraints

  c = (ModelConstraints()
       .membership_only("FC_ENCODED")    # FC only drives class membership
       .force_include("OFFSET")
       .no_zi("LENGTH", "CURVES", "GRADEBR"))

  evaluator = builder.build_evaluator(
      variables=variables,
      constraints=c,
      max_latent_classes=2,
      R=150,
  )
  result = builder.run(evaluator=evaluator, algo="sa", max_iter=30)

Validating classes against actual FC labels
-------------------------------------------
  import pandas as pd
  from scipy.stats import chi2_contingency

  # Extract posterior class probabilities
  probs = builder.get_class_probabilities(result)   # shape (N, C)
  df["pred_class"] = probs.argmax(axis=1) + 1       # 1-indexed

  # Cross-tabulate predicted class vs actual FC
  ct = pd.crosstab(df["FC_LABEL"], df["pred_class"],
                   rownames=["Functional Class"],
                   colnames=["Predicted Class"])
  print(ct)
  print(ct.div(ct.sum(axis=1), axis=0).round(2))   # row proportions

  # Chi-square test of association
  chi2, p, dof, _ = chi2_contingency(ct)
  print(f"Chi-square={chi2:.2f}, p={p:.4f}, df={dof}")

Pre-fitted book specification
------------------------------
  from metacountregressor import (
      load_book_latent_class_spec,
      describe_book_latent_class_spec,
  )
  describe_book_latent_class_spec()
  spec = load_book_latent_class_spec()
  fit  = builder.fit_manual_model(manual_spec=spec, model="nb", R=200)

Role codes relevant to LC models
----------------------------------
  7  Membership only     - variable shifts Pr(class=c) but not outcome
  8  Membership + Fixed  - shifts Pr(class=c) AND has outcome effect
  When latent_classes=1, roles 7 and 8 collapse to 0 and 1 (ignored).

See also: get_help("variables"), get_help("constraints")
"""


@_topic("cmf")
def _cmf_topic() -> str:
    return """
+======================================================================+
|  CMF SEARCH - AADT AS MAIN TERM                                      |
+======================================================================+

CMF (Crash Modification Factor) models use AADT as the primary scaling
term.  The two-component model structure is:

  log(mu_i) = [alpha_0 + sum_k alpha_k * X_ki]           (Component A)
            + [beta_0  + sum_k beta_k  * X_ki] * log(AADT_i)  (Component B)

  CMF interpretation:
    Component A: CMF_k = exp(alpha_k)          (direct multiplier)
    Component B: CMF_k = AADT_mean ^ beta_k    (evaluated at mean AADT)

1. Build the CMF experiment
----------------------------
  from metacountregressor import (
      load_example16_3_model_data,
      CMFExperimentBuilder,
      ModelConstraints,
  )

  df = load_example16_3_model_data()

  # AADT must be strictly positive (it is log-transformed internally)
  cmf = CMFExperimentBuilder(
      df=df,
      y_col="FREQ",
      aadt_col="AADT",
      baseline_vars=["URB", "ACCESS", "GRADEBR", "CURVES"],
      local_vars=["CURVES", "WIDTH"],    # interact with log(AADT)
  )

2. Full-flexibility JAX search (recommended)
---------------------------------------------
  # log(AADT) is automatically forced to role=1 (fixed, always included)
  # All other variables can take any role 0-8

  c = (ModelConstraints()
       .no_zi("CURVES", "WIDTH", "GRADEBR", "ACCESS")
       .allow_random("CURVES", distributions=["lognormal"]))

  builder, evaluator, metadata = cmf.build_jax_count_evaluator(
      id_col="ID",
      offset_col="OFFSET",
      constraints=c,
      max_latent_classes=1,
      R=200,
  )
  result = builder.run(evaluator=evaluator, algo="sa", max_iter=50)

3. Classic GA CMF search (original two-component)
--------------------------------------------------
  search = cmf.run_search(R=200)
  fit    = cmf.fit_best_model(search, final_R=500)
  cmf.print_report(search, fit)

4. Manual CMF model from book specification
--------------------------------------------
  from metacountregressor import load_book_cmf_spec
  spec      = load_book_cmf_spec()
  man_spec  = cmf.make_manual_cmf_spec(
      baseline_fixed=spec["baseline_fixed"],
      local_fixed=spec["local_fixed"],
      baseline_random=spec["baseline_random"],
  )
  fit = cmf.fit_manual_cmf_model(id_col="ID", manual_spec=man_spec, R=200)

See also: get_help("constraints"), get_help("crash_frequency")
"""


@_topic("linear")
def _linear_topic() -> str:
    return """
+======================================================================+
|  LINEAR MODEL SEARCH (CONTINUOUS OUTCOMES)                           |
+======================================================================+

Use ExperimentBuilder with model_family="linear" for continuous outcomes
such as operating speed, gap duration, or travel time.

1. Load data
------------
  from metacountregressor import (
      load_example_platform_speed_data,
      ExperimentBuilder,
      ModelConstraints,
  )

  df = load_example_platform_speed_data()

2. Select variables and constraints
------------------------------------
  variables = ["INCLANES", "DECLANES", "WIDTH", "URB", "SPEED_LIMIT"]

  c = (ModelConstraints()
       .no_zi("INCLANES", "DECLANES", "WIDTH")   # no ZI for linear
       .no_random("URB"))                          # binary dummy

3. Build and run
----------------
  builder = ExperimentBuilder(
      df=df,
      id_col="ID",
      y_col="SPEED",   # continuous outcome
  )

  evaluator = builder.build_evaluator(
      variables=variables,
      constraints=c,
      model_family="linear",       # switches to Gaussian likelihood
      default_roles=[0, 1, 2, 3],  # no ZI for linear models
      max_latent_classes=1,
      R=200,
  )

  result = builder.run(evaluator=evaluator, algo="sa", max_iter=50)

4. Latent class linear model
------------------------------
  evaluator = builder.build_evaluator(
      variables=variables,
      constraints=c,
      model_family="linear",
      max_latent_classes=2,
      default_roles=[0, 1, 2, 3, 7, 8],
      R=200,
  )

Linear vs Count model family
------------------------------
  model_family="count"    Poisson / NB (default)  - non-negative integers
  model_family="linear"   Gaussian                - real-valued outcomes
  model_family="duration" Lognormal               - positive, skewed

Notes
-----
  - Zero-inflation (role 6) is not meaningful for linear models.
    Use no_zi() in ModelConstraints or omit role 6 from default_roles.
  - The dispersion parameter in a linear model is the residual variance.
  - All other structural features (random params, LC, membership) work
    identically to count models.

See also: get_help("constraints"), get_help("variables")
"""


# ---------------------------------------------------------------------------
# Main get_help() entry point
# ---------------------------------------------------------------------------

_INDEX = """
+======================================================================+
|  metacountregressor - Usage Guide                                    |
+======================================================================+
|  get_help(topic)  ->  detailed guide for each topic                  |
+======================================================================+
|  Topics:                                                             |
|    "data"             Loading data, required columns, offsets        |
|    "variables"        Variable selection and role codes (0-8)        |
|    "constraints"      ModelConstraints API - intuitive restrictions  |
|    "metaheuristics"   SA/DE/HS algorithms and tuning                |
|    "crash_frequency"  End-to-end crash frequency model search        |
|    "latent_class"     LC search, membership vars, FC validation      |
|    "cmf"              CMF with AADT as main term                     |
|    "linear"           Linear model for continuous outcomes           |
+======================================================================+
|  Template notebooks:                                                 |
|    get_templates()    Copy .ipynb templates to current directory     |
+======================================================================+
|  Pre-fitted specifications:                                          |
|    load_book_latent_class_spec()   2-class NB LC  (Example 16-3)    |
|    load_book_nb_baseline_spec()    Single-class NB baseline          |
|    load_book_cmf_spec()            CMF two-component AADT model      |
|    list_book_specifications()      Print all available specs         |
+======================================================================+
"""


_ALIASES: dict[str, str] = {
  "count": "crash_frequency",
  "nb": "crash_frequency",
  "poisson": "crash_frequency",
  "lc": "latent_class",
  "latentclass": "latent_class",
  "constraint": "constraints",
  "role": "variables",
  "roles": "variables",
  "algo": "metaheuristics",
  "algorithms": "metaheuristics",
  "sa": "metaheuristics",
  "de": "metaheuristics",
  "hs": "metaheuristics",
}


def get_help(topic: str | None = None) -> None:
    """
    Print a structured usage guide for the metacountregressor package.

    Parameters
    ----------
    topic : str, optional
      One of: "data", "variables", "constraints", "metaheuristics",
      "crash_frequency", "latent_class", "cmf", "linear".
        If *None* (default), prints the topic index.

    Examples
    --------
    >>> from metacountregressor import get_help
    >>> get_help()
    >>> get_help("constraints")
    >>> get_help("latent_class")
    """
    if topic is None:
        print(_INDEX)
        return

    key = topic.lower().strip().replace("-", "_").replace(" ", "_")
    key = _ALIASES.get(key, key)
    text = _TOPICS.get(key)
    if text is None:
        available = ", ".join(f'"{k}"' for k in sorted(_TOPICS))
        raise ValueError(
            f"Unknown help topic '{topic}'. "
            f"Available topics: {available}"
        )
    print(text)


# ---------------------------------------------------------------------------
# Template download
# ---------------------------------------------------------------------------

def get_templates(dest_dir: str = ".") -> None:
    """
    Copy the bundled Jupyter notebook templates to *dest_dir*.

    Templates included
    ------------------
    01_crash_frequency_search.ipynb
    02_latent_class_fc_validation.ipynb
    03_cmf_aadt_search.ipynb
    04_linear_speed_prediction.ipynb

    Parameters
    ----------
    dest_dir : str
        Destination directory.  Created if it does not exist.
        Defaults to the current working directory.

    Example
    -------
    >>> from metacountregressor import get_templates
    >>> get_templates()                  # copies to current dir
    >>> get_templates("~/my_project")    # copies to specific dir
    """
    dest = os.path.expanduser(dest_dir)
    os.makedirs(dest, exist_ok=True)

    # templates/ lives next to this file inside the installed package
    template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")

    if not os.path.isdir(template_dir):
        raise FileNotFoundError(
            f"Template directory not found at {template_dir!r}. "
            "Ensure metacountregressor is properly installed."
        )

    copied = []
    for fname in sorted(os.listdir(template_dir)):
        if fname.endswith(".ipynb") or fname.endswith(".py"):
            src = os.path.join(template_dir, fname)
            dst = os.path.join(dest, fname)
            shutil.copy2(src, dst)
            copied.append(fname)

    if copied:
        print(f"Copied {len(copied)} template(s) to {os.path.abspath(dest)}:")
        for f in copied:
            print(f"  - {f}")
    else:
        print(f"No templates found in {template_dir!r}.")
