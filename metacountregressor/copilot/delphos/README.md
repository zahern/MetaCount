# Delphos-style RL specification search for count models

This is a standalone prototype that adapts the Delphos MDP / DeepSet-Q
framework (Nova, Hess & van Cranenburgh, arXiv:2506.06410 and
arXiv:2609.18441) to the `metacountregressor` model-specification problem.

It is an *alternative search driver*: the model evaluator, likelihoods,
parameter packing, and data handling all remain `metacountregressor`'s.

## State

A state is a set of modelling terms plus global family decisions.  Each term
records

```
(variable, transformation, heterogeneity role, distribution, block)
```

and the global state records

```
count family (Poisson / NB), dispersion bit, number of latent classes
```

The role vocabulary is the package's role system: excluded, fixed, random
independent, random correlated, grouped, heterogeneity-in-means,
zero-inflation, membership-only, membership+fixed.  `block` distinguishes
the baseline and log-AADT-scaled local components of a hierarchical CMF.

Transformations are implemented as mutually exclusive evaluator columns
(`identity`, `log1p`, `sqrt`), so the state can switch a variable's
functional form without changing the underlying model code.

## Actions

* `set_role(variable, role)`
* `set_distribution(variable, distribution)` (only for random roles)
* `set_transform(variable, transform)`
* `toggle_dispersion`
* `set_latent_classes(k)`
* `terminate`

Actions are masked for feasibility (no no-ops, no immediate reversals, no
membership roles when there is one class, no grouped effects without a group
column).  Terminating submits the current specification to the evaluator.

## Reward

The terminal reward is the Delphos-style per-observation improvement in
log-likelihood over the null model:

```
r = tanh((LL - LL_null) / N)
```

The experiment driver switches to a linear clipped version
(`clip((LL - LL_null) / N, -5, 5)`) because the null-to-good-model gap on
Example 16-3 is about 2.6 per observation, which saturates `tanh` and hides
differences between good candidates.  Failed or invalid specifications get
`-1`.

The terminal reward is propagated to every decision in the episode with
`gamma ** (L - l) * R` (Eq. 13 of Delphos), and the Q-network is trained on
those Monte-Carlo returns.

## Agent

`DeepSetQAgent` encodes each term with a shared `phi` MLP, sum-pools the term
embeddings, concatenates a context vector (dispersion, latent classes,
has-random, has-ZI, step fraction, incumbent reward, active fraction,
baseline BIC), and outputs one Q-value per action.  JAX is used for the
network; no PyTorch dependency is required.

## Run

```powershell
python -m metacountregressor.copilot.delphos.experiment --task both --episodes 60 --fits 60 `
    --R 10 --max-latent-classes 1 --max-steps 10 --out results/delphos_count
```

Results are written as CSV/JSON plus a learning-curve PNG.

## Fixing the cold-start caveat

The single-task agent needs many episodes.  Two remedies are implemented:

### 1. Multitask pretraining

`metacountregressor.copilot.delphos.pretrain.MultiTaskTrainer` trains one shared DeepSet-Q policy
across related tasks with the same variable catalogue, then fine-tunes on the
target:

```python
from metacountregressor.copilot.delphos.experiment import build_bootstrap_count_tasks
from metacountregressor.copilot.delphos.env import DelphosCountEnv
from metacountregressor.copilot.delphos.pretrain import MultiTaskTrainer, PretrainConfig

tasks = build_bootstrap_count_tasks(n_tasks=3, R=5, max_latent_classes=1)
envs = []
for task in tasks:
    _, evaluator = task.make_evaluator()
    envs.append(DelphosCountEnv(evaluator, task.space, start_mode="mixed"))

trainer = MultiTaskTrainer(envs, config=PretrainConfig(n_episodes=60))
trainer.fit()
trainer.fine_tune(target_env, n_episodes=20)
```

This mirrors the second Delphos paper: shared experience -> higher sample
efficiency on a new task.  The current trainer requires a shared variable
catalogue; datasets with different schemas need a concept-mapping layer.

### 2. Expert-demonstration warm-start

When only one dataset is available, seed the replay buffer with an
SA/random/AGDS best model expressed as an action trajectory:

```python
from metacountregressor.copilot.delphos.demonstrations import (
    state_from_mapping, add_demonstration, warm_start_with_experts,
)

target = state_from_mapping(space, {
    "SPEED": {"role": "random_cor", "distribution": "normal"},
    "GRADEBR": {"role": "fixed"},
    "WIDTH": {"role": "random_cor", "transform": "log1p",
              "distribution": "neglognormal"},
}, dispersion=1)

add_demonstration(agent, env, target, repeats=2, train_updates=8)
# then continue with normal DelphosSearch fine-tuning
```

This is behaviour cloning on the discounted terminal return, followed by
normal RL.  It gives the agent a good trajectory immediately instead of
spending episodes rediscovering it.

## Showcase scripts

See `../examples/`:

* `showcase_delphos.py` - state/action/reward + demonstration warm-start.
* `showcase_pretraining.py` - multitask pretraining vs cold start.
* `showcase_copilot.py` - conversational setup + optional RL run.
* `showcase_cli.ps1` - command-line wizard.

## Note on the installed package

This build of `metacountregressor` starts `CountModel.fit()` from tiny random
noise.  On NB models that can walk to a degenerate dispersion and a
log-likelihood far below Poisson.  The experiment driver patches the default
to `use_prefit=True` (a Poisson prefit) and a 400-iteration L-BFGS budget.
Both RL and the metaheuristic baselines use the same patched estimator, so
the comparison is fair.

## Files

```
state.py      state / catalogue / decision-vector encode-decode
env.py        action catalogue, masking, estimation environment, reward
agent.py      DeepSet-Q network, replay buffer, Adam
search.py     episode loop, MC credit assignment, inference rollouts
experiment.py task builders, SA/random baselines, CLI, reporting
```

