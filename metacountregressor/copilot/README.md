# Experiment copilot

An LLM-orchestrated front end for `metacountregressor`.

The design principle is:

> **The LLM proposes; the package disposes.**

The LLM is used for language understanding and modelling suggestions.  It
does **not** execute code, estimate models, or write directly into the
specification.  Every proposal is turned into a typed `ExperimentSpec`,
validated against the profiled data and the package's constraint objects, and
only then sent to the estimator.

## Why this works

The workflow is a deterministic state machine with an optional LLM at the
edges:

```
data path
  -> profile columns (types, missingness, cardinality, candidates)
  -> response / family / ID / offset / group
  -> variable screen (univariate + domain prior) and user pruning
  -> tiered constraint hierarchy
  -> objective / driver / budget
  -> ExperimentSpec
  -> ExperimentBuilder / CMFExperimentBuilder
  -> search
  -> diagnostics -> proposed constraint patch -> rerun
```

The LLM only fills the slots.  A rule-based advisor implements the same
interface, so the package is usable with no API key, no network, and no
non-determinism.  This is important for reproducibility and for regulated
environments.

## Install

The copilot is shipped **inside** `metacountregressor` (version 1.0.216+).
One install provides both the estimator and the CLI:

```powershell
pip install -U metacountregressor
metacount-copilot
```

Python API:

```python
from metacountregressor import run_cli
run_cli()
```

or, equivalently:

```python
from metacountregressor.copilot import run_cli
run_cli()
```

From a source checkout:

```powershell
cd C:\Users\ahernz\source\MetaCount
python -m metacountregressor.copilot
```

No separate `metacount-copilot` package is required.

## Command line

Interactive wizard:

```powershell
python -m metacountregressor.copilot
```

Non-interactive quickstart:

```powershell
python -m metacountregressor.copilot --auto --data crashes.csv `
    --response FREQ --family count --id-col ID --offset-col OFFSET `
    --algo sa --max-iter 300 --R 50 --run --diagnose
```

From Python, `run_cli()` starts the same wizard:

```python
from metacountregressor.copilot import run_cli
run_cli()
```

Inside the wizard:

```
/start PATH        load data and begin
/run               estimate the current spec
/diagnose          inspect the result and propose constraint patches
/apply             apply the diagnosis patch
/rerun             apply and run again
/spec              print the validated ExperimentSpec JSON
/save PATH         save the spec
/load PATH         restore a saved spec
/help              command reference
```

## Conversation

```python
from metacountregressor.copilot import ExperimentCopilot

cp = ExperimentCopilot()                 # rule-based advisor by default
turn = cp.start("data/crashes.csv")
print(turn.question, turn.options)

while not turn.done:
    print(turn.question, turn.options, turn.examples)
    answer = input("> ")
    turn = cp.answer(answer)

spec = cp.build_spec()
result = cp.run()
diagnosis = cp.diagnose()
print(diagnosis["summary"])
for issue in diagnosis["issues"]:
    print(issue["severity"], issue["code"], issue["message"])

# Apply the deterministic patches (e.g. no-ZI if zeros are rare,
# relax structural roles after non-convergence) and rerun.
spec2 = cp.apply_diagnosis()
result2 = cp.run()
```

The turns are:

1. response column (inferred count candidates)
2. model family (`count`, `cmf`)
3. ID column (optional)
4. offset column (optional; already logged)
5. grouping column (optional)
6. AADT column (CMF only)
7. variable catalogue (LLM/rule screen; user can exclude or add back)
8. baseline vs local CMF blocks (CMF only)
9. constraint hierarchy (accept or natural-language edit)
10. objective, driver, budget, draws

## Constraint hierarchy

The proposal is explicitly tiered:

| Tier | Examples | Purpose |
| --- | --- | --- |
| hard | exclude ID / response / offset / group key | structural validity |
| structural | binary: no random, no ZI; geometry: no ZI | identifiability |
| plausibility | continuous: normal / lognormal / triangular | behavioural prior |
| search | mutual exclusion for near-collinear pairs | numerical stability |

`runner.build_constraints()` translates the tiers into the package's
`ModelConstraints` fluent API.  Unknown methods or variable names are rejected
before estimation.

## LLM configuration

The optional client is a minimal OpenAI-compatible chat client using the
standard library:

```powershell
$env:METACOUNT_LLM_BASE_URL = "http://localhost:11434/v1"   # Ollama / LM Studio / vLLM
$env:METACOUNT_LLM_API_KEY  = "not-needed"
$env:METACOUNT_LLM_MODEL    = "qwen2.5:14b"
```

```python
from metacountregressor.copilot import ExperimentCopilot, OpenAICompatibleAdvisor

cp = ExperimentCopilot(advisor=OpenAICompatibleAdvisor())
```

The LLM is prompted to return strict JSON.  Responses are parsed, filtered to
known columns, and validated.  If the LLM is unavailable or returns malformed
JSON, the copilot falls back to the rule-based advisor.

## Diagnostics and the feedback loop

After a search, `diagnose_result()` runs deterministic checks:

* no finite BIC / non-convergence
* over-parameterisation relative to sample size
* zero-inflation selected when zeros are rare
* (LLM advisor, optional) coefficient signs, identifiability, separation

Each issue carries a `proposed_patch`.  The deterministic patches are:

* `loosen_roles` - remove structural/plausibility tiers and rerun
* `prune_weak_terms` - remove random-role allowances
* `constrain` - append a new search-tier constraint (e.g. `no_zi`)

`apply_diagnosis()` returns a new `ExperimentSpec`; the original is not
mutated.

## What is guaranteed, and what is not

There is **no guarantee** that an LLM will produce a good specification, or
even a valid one.  What is guaranteed by this design is:

* no hallucinated column can reach the estimator (it is checked against the
  profile);
* no invalid constraint method can reach `ModelConstraints`;
* the full setup is serialisable (`ExperimentSpec.to_dict()`) and reproducible;
* the workflow still runs with the rule-based advisor when no LLM is
  configured.

## Current scope

* `count` and `cmf` families are wired to the package evaluators.
* Search drivers exposed through the generic runner: `sa`, `ga`, `hs`, `de`.
* The Delphos RL driver lives in `metacountregressor.copilot.delphos` and can
  be run explicitly; it is intentionally not hidden behind the generic
  `ExperimentBuilder.run()`.


