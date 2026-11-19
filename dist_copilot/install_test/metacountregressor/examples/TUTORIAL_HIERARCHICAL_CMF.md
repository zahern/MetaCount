# Tutorial: Hierarchical CMF models, and how to fit one manually

This is a practical companion to `manual_hierarchical_cmf_tutorial.py` in this
folder, and to `paper/Article Multilevel modelling in traffic safety
analysis.docx`. It covers three things:

1. What "pooling" and "partial pooling" mean, and why they matter for CMF/SPF development.
2. How to fit a hierarchical CMF model **manually** — one Python script, no search, no HPC.
3. How to submit a model **automatically** (structure search + PBS), for when you don't already know which variables/roles to use.

## 1. Pooling, in one page

Road-safety data is *structured*: crashes are nested within segments, functional
classes, or road types. There are three ways to handle that structure:

| Approach | What it does | Failure mode |
| --- | --- | --- |
| **No pooling** | Fit each group separately | Unstable for small groups (few crashes per segment) |
| **Complete pooling** | Ignore grouping, fit one model on everyone | Biased if groups actually differ (Simpson's-paradox risk) |
| **Partial pooling** (hierarchical) | Shrink each group's estimate toward the population mean, weighted by how much data supports it | The compromise — provably lower MSE than either extreme |

The shrinkage weight is `λ = n_j·τ² / (n_j·τ² + σ²)`: a well-sampled segment
leans on its own data; a sparse one leans on the population average. This is
mathematically identical to Empirical Bayes, which is why partial pooling
integrates naturally with the EB methods already standard in road-safety
practice.

**In `metacountregressor`, partial pooling = random parameters** (`rdm_terms`
in a manual spec, or role 2/3 in a structure search). A random parameter gives
each observation its own draw around a population mean + SD — that SD *is*
the between-group heterogeneity the paper is describing.

## 2. Fitting a hierarchical CMF model manually

You don't need the metaheuristic structure search to build a hierarchical CMF
model — if you already know (or want to hand-pick) which variables matter,
`ExperimentBuilder` fits any specification directly:

```python
import pandas as pd
from metacountregressor.experiment_package import ExperimentBuilder

df = pd.read_csv("my_crash_data.csv")
df["_id"] = range(len(df))

builder = ExperimentBuilder(df=df, id_col="_id", y_col="CRASHES", offset_col="log_exposure")

spec = builder.make_manual_spec(
    fixed_terms=["LNAADT", "WIDTH", "GRADEBR"],     # role 1: same coefficient for everyone
    rdm_terms=["CURVES:normal"],                    # role 2: partial pooling (random parameter)
    dispersion=1,                                    # 1 = Negative Binomial, 0 = Poisson
)

fit = builder.fit_manual_model(spec, model="nb", print_report=True)
```

That's the whole thing — no search, no cluster, runs in seconds to minutes
depending on data size. `fit["result"].params`, `fit["summary"]`, and
`fit["spec"]` give you everything downstream (coefficients, BIC/AIC, standard
errors).

### Varying AADT elasticity (the paper's core idea)

The traditional HSM formulation gives AADT a *fixed* elasticity (via an
offset with coefficient forced to 1). To let elasticity vary with context
(paper Eq. 38–43: `elasticity = β_AADT + γ·x`), just build the interaction
column yourself and fit it as an ordinary term:

```python
df["slope_x_lnaadt"] = df["SLOPE"] * df["LNAADT"]

spec = builder.make_manual_spec(
    fixed_terms=["LNAADT", "WIDTH", "GRADEBR", "slope_x_lnaadt"],
)
```

The coefficient on `slope_x_lnaadt` **is** γ from the paper — it tells you how
much AADT's elasticity shifts on flat vs. graded segments. No two-stage
SPF-then-CMF calibration required; both come out of the same one-stage fit.

### Reading off a CMF

Because the model is log-linear, a CMF for a `delta`-unit change in variable
`x` is just `exp(β · delta)` — but **watch the scaling**: `metacountregressor`
standardizes continuous predictors internally before fitting, so
`fit["result"].params` are per-standard-deviation, not per raw unit. Convert
back first:

```python
import numpy as np

names = list(fit["spec"].fixed_names)
beta_std = fit["result"].params[names.index("WIDTH")]

scaler = fit["data"]["scaler"]          # {colname: (mean, sd)}
_, sd = scaler["WIDTH"]
beta_original = beta_std / sd

cmf = np.exp(beta_original * 1.0)       # CMF for +1 foot of lane width
```

This exactly reproduces the "Estimate" column metacountregressor prints in
its own model summary — that's the same back-transform happening internally.
`manual_hierarchical_cmf_tutorial.py` wraps this in a `derive_cmf()` helper.

## 3. Submitting a model

**Manual (this tutorial):** just run the script.

```bash
python metacountregressor/examples/manual_hierarchical_cmf_tutorial.py
```

No PBS, no queue, no HPC account needed — it's a normal local Python process.
Use this path when you already know which variables you want, or you're
iterating quickly on a specification.

**Automated (structure search):** when you *don't* know which variables/roles
belong in the model, `generate_washington_hierarchical_cmf_assets.py` (bundled
inside the installed package) runs a metaheuristic search (SA/harmony search)
over variable inclusion, role (fixed vs. AADT-varying), and family
(Poisson/NB), then derives CMFs from the winning specification automatically.
It's a standalone script, not a `-m`-importable module — locate it via the
installed package and run it directly:

```bash
SCRIPT=$(python -c "import metacountregressor, os; print(os.path.join(os.path.dirname(metacountregressor.__file__), 'scripts', 'generate_washington_hierarchical_cmf_assets.py'))")
python "$SCRIPT" \
    --input my_crash_data.csv --output-dir results/my_run \
    --y-col CRASHES --aadt-col AADT --offset-col log_exposure \
    --search-iter 300 --families both
```

Or, simplest of all, let `run_metacount_analysis.py` find and invoke it for
you across all your configured datasets/experiments (see below).

For a full multi-dataset run on an HPC cluster via PBS, see
`run_metacount.pbs` / `run_metacount_analysis.py` in your project directory —
they wrap the same script across datasets/experiments and handle environment
setup (conda env, JAX CPU/GPU platform, walltime) for you. The search-based
`exp3_cmf_random_params` experiment in that pipeline now performs genuine
partial pooling too (independent random parameters via `ExperimentBuilder`,
the same mechanism shown above), rather than the placeholder it used to fall
back to.

## Further reading

- `paper/Article Multilevel modelling in traffic safety analysis.docx` — the
  full derivation (pooling variance/MSE proofs, Empirical Bayes connection,
  varying-coefficient SPF, worked Washington/QLD examples).
- `ExperimentBuilder.__doc__` (`experiment_package.py`) — the general-purpose
  quick-start for building any manual spec, including latent classes,
  grouped/correlated random effects, and zero-inflation.
