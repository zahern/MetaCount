# ======================================================================
# latent_class_experiment_runner.py
# ======================================================================
#
# Standalone runner for Latent Class Structure Search
# Compatible with your ExperimentBuilder framework
#
# Usage:
#   python latent_class_experiment_runner.py
#
# ======================================================================

import pandas as pd
import numpy as np
from experiment_package import ExperimentBuilder


# ======================================================================
# 1. PRINT FULL GUIDE
# ======================================================================

def print_latent_class_guide():
    print("""
====================================================================
 LATENT CLASS STRUCTURE SEARCH - COMPLETE GUIDE
====================================================================

This framework searches over:

 - Variable roles (fixed / random / grouped / etc.)
 - Distributions
 - Dispersion (Poisson vs NB)
 - Number of latent classes
 - Membership variables (optional)

------------------------------------------------------------
DECISION VECTOR STRUCTURE
------------------------------------------------------------
[ roles(D) | dists(D) | dispersion_bit | latent_class_code ]

latent_classes = (latent_class_code % max_latent_classes) + 1

------------------------------------------------------------
ROLE CODES
------------------------------------------------------------
0  Excluded
1  Fixed
2  Random Independent
3  Random Correlated
4  Grouped
5  Heterogeneity in means
6  Zero Inflation
7  Membership only
8  Membership + Fixed outcome

------------------------------------------------------------
MINIMAL LATENT CLASS SEARCH
------------------------------------------------------------
evaluator = builder.build_evaluator(
    mode="single",
    max_latent_classes=2
)

------------------------------------------------------------
WITH MEMBERSHIP VARIABLES
------------------------------------------------------------
evaluator = builder.build_evaluator(
    mode="single",
    max_latent_classes=2,
    membership_override={
        "URB": [0, 7],
        "INCOME": [0, 7, 8]
    }
)

------------------------------------------------------------
MULTI-OBJECTIVE SEARCH
------------------------------------------------------------
mode="multi"  -> minimizes:
   [BIC, Test RMSE]

algo="de" or "hs" -> NSGA-II
algo="sa"         -> Simulated Annealing

------------------------------------------------------------
RECOMMENDED SETTINGS
------------------------------------------------------------
First LC experiment:
  max_latent_classes=2
  R=150
  mode="single"

More aggressive:
  max_latent_classes=3
  R=300
  mode="multi"

====================================================================
""")


def print_latent_class_setup_instructions():
    print("""
------------------------------------------------------------
QUICK SETUP CHECKLIST
------------------------------------------------------------
1. Load your dataset
   df = pd.read_csv("path/to/your_file.csv")

2. Make sure your dataframe has:
   - one outcome/count column
   - one unique row identifier column
   - an offset/exposure column if your model needs one
   - optional grouping column for grouped/random structures

3. Rename or create the required modeling columns
   Example:
       df.rename(columns={"your_count_column": "Y"}, inplace=True)
       df["OFFSET"] = 0

4. Build the experiment
   builder = ExperimentBuilder(
       df=df,
       id_col="ID",
       y_col="Y",
       offset_col="OFFSET",
       group_id_col="GROUP_COL"   # optional
   )

5. Decide what should be tested
   - Variables not listed in membership_override can be used in outcome roles
   - Variables in membership_override control latent class membership options
   - Set max_latent_classes to control how many classes are searched

6. Build the evaluator and run
   evaluator = builder.build_evaluator(
       mode="single",
       max_latent_classes=2,
       R=150
   )

   result = builder.run(
       evaluator=evaluator,
       algo="sa",
       max_iter=10,
       seed=0
   )

Common starting point:
   - Start with max_latent_classes=2
   - Use a small max_iter to smoke test the pipeline
   - Add membership_override only after the basic run works
------------------------------------------------------------
""")


def print_generalized_crash_frequency_setup():
    print("""
------------------------------------------------------------
GENERALISED CRASH FREQUENCY SETUP
------------------------------------------------------------
Use this pattern when adapting the latent class search to a new
crash-frequency dataset.

1. Prepare data
   data_path = "data/your_crash_data.csv"
   df = pd.read_csv(data_path)

2. Map your columns into the standard runner inputs
   config = {
       "id_col": "ID",
       "count_col": "FREQ",
       "exposure_col": None,          # or "EXPOSURE"
       "group_id_col": "FC",          # optional, enables grouped roles
       "latent_classes": 2,
       "draws": 150,
       "algo": "sa",
       "max_iter": 10,
       "seed": 0,
   }

3. Create/rename the model columns
   - count_col becomes Y
   - exposure_col, if supplied, is copied into OFFSET
   - if no exposure is supplied, OFFSET defaults to 0

4. Change the searchable structure as needed
   - membership_override:
       decides which variables can explain class membership
       example: {"URB": [0, 7], "SPEED": [0, 7, 8]}
   - latent_classes:
       controls the maximum number of classes searched
   - group_id_col:
       include for grouped/shared effects, omit otherwise
   - draws (R):
       higher values are slower but more stable

5. Minimal reusable call
   result = run_generalized_crash_frequency_experiment(
       data_path="data/your_crash_data.csv",
       id_col="ID",
       count_col="FREQ",
       exposure_col="EXPOSURE",
       group_id_col="FC",
       membership_override={"URB": [0, 7], "SPEED": [0, 7, 8]},
       max_latent_classes=2,
       R=150,
       algo="sa",
       max_iter=10,
       seed=0,
   )

This lets you change the dataset, identifiers, exposure handling,
membership variables, search size, and optimizer settings without
rewriting the experiment logic.
------------------------------------------------------------
""")


# ======================================================================
# 2. EXAMPLE LATENT CLASS EXPERIMENT
# ======================================================================

def prepare_crash_frequency_data(
    data_path,
    count_col,
    exposure_col=None,
    exposure_fn=None,
):
    df = pd.read_csv(data_path)
    df = df.copy()

    if count_col != "Y":
        df.rename(columns={count_col: "Y"}, inplace=True)

    if exposure_fn is not None:
        exposure = exposure_fn(df)
    elif exposure_col is not None:
        exposure = df[exposure_col]
    else:
        df["OFFSET"] = 0
        return df

    exposure = pd.to_numeric(exposure, errors="coerce").to_numpy(dtype=float, copy=False)
    df["OFFSET"] = np.where(exposure > 0.0, np.log(exposure), 0.0)

    return df


def run_generalized_crash_frequency_experiment(
    data_path,
    id_col,
    count_col,
    exposure_col=None,
    exposure_fn=None,
    group_id_col=None,
    membership_override=None,
    max_latent_classes=2,
    R=150,
    algo="sa",
    max_iter=10,
    seed=0,
):
    df = prepare_crash_frequency_data(
        data_path=data_path,
        count_col=count_col,
        exposure_col=exposure_col,
        exposure_fn=exposure_fn,
    )

    builder = ExperimentBuilder(
        df=df,
        id_col=id_col,
        y_col="Y",
        offset_col="OFFSET",
        group_id_col=group_id_col
    )

    evaluator = builder.build_evaluator(
        mode="single",
        max_latent_classes=max_latent_classes,
        R=R,
        membership_override=membership_override
    )

    return builder.run(
        evaluator=evaluator,
        algo=algo,
        max_iter=max_iter,
        seed=seed
    )


def run_example_latent_class_experiment():

    print("\n==============================")
    print(" RUNNING LATENT CLASS SEARCH ")
    print("==============================\n")

    result = run_generalized_crash_frequency_experiment(
        data_path="data/Ex-16-3.csv",
        id_col="ID",
        count_col="FREQ",
        exposure_fn=lambda df: df["LENGTH"] * df["AADT"] * 365 / 100000000,
        group_id_col="FC",
        membership_override={
            "URB": [0, 7],
            "SPEED": [0, 7, 8]
        },
        max_latent_classes=2,
        R=150,
        algo="sa",
        max_iter=10,
        seed=0
    )

    print("\nExperiment finished.")
    return result


def run_fc_latent_class_experiment():

    print("\n==============================================")
    print(" RUNNING FC-DRIVEN LATENT CLASS SEARCH ")
    print("==============================================\n")

    df = prepare_crash_frequency_data(
        data_path="data/Ex-16-3.csv",
        count_col="FREQ",
        exposure_fn=lambda df: df["LENGTH"] * df["AADT"] * 365 / 100000000,
    )

    fc_levels = sorted(df["FC"].dropna().unique())
    fc_dummies = pd.get_dummies(df["FC"], prefix="FC", drop_first=True, dtype=int)
    df = pd.concat([df, fc_dummies], axis=1)

    builder = ExperimentBuilder(
        df=df,
        id_col="ID",
        y_col="Y",
        offset_col="OFFSET"
    )

    evaluator = builder.build_evaluator(
        mode="single",
        max_latent_classes=len(fc_levels),
        R=80,
        membership_override={col: [7] for col in fc_dummies.columns}
    )

    result = builder.run(
        evaluator=evaluator,
        algo="sa",
        max_iter=10,
        seed=1
    )

    print("\nFC latent class experiment finished.")
    return result


# ======================================================================
# 3. MAIN
# ======================================================================

if __name__ == "__main__":
    print_latent_class_guide()
    print_latent_class_setup_instructions()
    print_generalized_crash_frequency_setup()

    # Comment this out if you only want the guide
    run_example_latent_class_experiment()
    run_fc_latent_class_experiment()
