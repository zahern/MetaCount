"""Benchmark Delphos-style RL against the metaheuristic drivers.

Two tasks are provided:

``count``
    Heterogeneous crash-frequency model on Example 16-3.  The catalogue
    includes transformations (identity / log1p / sqrt), fixed and random
    parameters, random-parameter distributions, zero inflation, Poisson vs
    NB, and (optionally) latent classes.

``cmf``
    Hierarchical Crash Modification Factor model: Component A (baseline)
    plus Component B (effects scaled by log AADT), using the package's
    ``CMFExperimentBuilder``.

The script is intentionally standalone: it uses the installed
``metacountregressor`` API and does not modify the package source.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from metacountregressor import (
    CMFExperimentBuilder,
    ExperimentBuilder,
    ModelConstraints,
    load_example16_3_model_data,
)

from .env import DelphosCountEnv, EvalResult
from .search import DelphosSearch, SearchConfig
from .state import (
    DISTRIBUTIONS,
    ROLE_CODES,
    ROLE_NAMES,
    CountModelState,
    ModelTerm,
    SearchSpace,
    VariableSpec,
)


# ---------------------------------------------------------------------------
# Work around a bad default in this metacountregressor build:
# CountModel.fit() starts L-BFGS from tiny random noise.  For NB models that
# frequently walks to a degenerate dispersion and an LL far below Poisson.
# A Poisson prefit (`use_prefit=True`) fixes it.  The evaluator calls
# model.fit() with no arguments, so patch the default for this experiment.
# ---------------------------------------------------------------------------
_PATCHED = False


def patch_count_model_initialization() -> None:
    global _PATCHED
    if _PATCHED:
        return
    try:
        from metacountregressor.main_hpc import CountModel
    except Exception:
        return
    original = CountModel.fit

    def fit_with_prefit(self, *args, **kwargs):
        if not args:
            kwargs.setdefault("use_prefit", True)
            # The default 2,000-iteration L-BFGS budget is unnecessary for
            # ranking specifications with these small models; 400 gives the
            # same BIC at a fraction of the cost.
            kwargs.setdefault("lbfgs_maxiter", 400)
            kwargs.setdefault("lbfgs_tol", 1e-5)
        return original(self, *args, **kwargs)

    CountModel.fit = fit_with_prefit
    # The package clears the JAX compilation cache every 50 new ModelSpecs.
    # That is sensible for multi-hour HPC searches but pure overhead for a
    # short RL/SA comparison; keep the compilation cache for the run.
    try:
        import metacountregressor.main_hpc as _mh

        _mh._spec_cache_threshold = 10_000
    except Exception:
        pass
    _PATCHED = True


@dataclass
class Task:
    name: str
    space: SearchSpace
    make_evaluator: Callable[[], Tuple[object, object]]
    df: pd.DataFrame
    meta: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Task construction
# ---------------------------------------------------------------------------
def _transformed_count_frame(
    df: pd.DataFrame,
    base_vars: Sequence[str],
    transforms: Dict[str, Sequence[str]],
) -> Tuple[pd.DataFrame, Tuple[VariableSpec, ...], List[List[str]]]:
    out = df.copy()
    specs: List[VariableSpec] = []
    mutex: List[List[str]] = []
    binary = {"URB", "ACCESS", "GRADEBR", "SINGLE", "DOUBLE", "TRAIN", "PEAKHR"}

    for var in base_vars:
        cols: List[Tuple[str, str]] = []
        for transform in transforms.get(var, ("identity",)):
            col = f"{var}__{transform}"
            if transform == "identity":
                out[col] = df[var].astype(float)
            elif transform == "log1p":
                out[col] = np.log1p(np.clip(df[var].astype(float), 0.0, None))
            elif transform == "sqrt":
                out[col] = np.sqrt(np.clip(df[var].astype(float), 0.0, None))
            elif transform == "square":
                out[col] = np.square(df[var].astype(float))
            else:
                raise ValueError(f"unknown transform {transform!r}")
            cols.append((transform, col))
        if len(cols) > 1:
            mutex.append([c for _, c in cols])
        specs.append(
            VariableSpec(
                name=var,
                transform_columns=tuple(cols),
                allowed_roles=tuple(ROLE_CODES),
                allowed_distributions=DISTRIBUTIONS,
                block="outcome",
                forced_role=None,
                kind="binary" if var in binary else "continuous",
            )
        )
    return out, tuple(specs), mutex


def build_count_task(
    *,
    R: int = 50,
    max_latent_classes: int = 2,
    seed: int = 0,
    df: Optional[pd.DataFrame] = None,
    name: str = "count",
) -> Task:
    patch_count_model_initialization()
    df = load_example16_3_model_data() if df is None else df.copy()
    base_vars = ["URB", "CURVES", "SPEED", "ACCESS", "GRADEBR", "WIDTH", "AVEPRE"]
    transforms = {
        "URB": ("identity",),
        "ACCESS": ("identity",),
        "GRADEBR": ("identity",),
        "CURVES": ("identity", "log1p", "sqrt"),
        "SPEED": ("identity", "log1p"),
        "WIDTH": ("identity", "log1p"),
        "AVEPRE": ("identity", "log1p", "sqrt"),
    }
    expanded, specs, mutex = _transformed_count_frame(df, base_vars, transforms)
    columns = [col for spec in specs for _, col in spec.transform_columns]

    # Binary dummies must not receive random parameters or zero-inflation
    # equations; geometry terms are not plausible ZI drivers either.
    column_constraints = ModelConstraints()
    for spec in specs:
        for _, col in spec.transform_columns:
            if spec.kind == "binary":
                column_constraints.no_random(col).no_zi(col)
            if spec.name == "CURVES":
                column_constraints.no_zi(col)

    builder = ExperimentBuilder(
        df=expanded,
        id_col="ID",
        y_col="FREQ",
        offset_col="OFFSET",
    )

    def make_evaluator():
        ev = builder.build_count_evaluator(
            variables=columns,
            mutual_exclusion=mutex,
            max_latent_classes=max_latent_classes,
            R=R,
            default_roles=(
                [0, 1, 2, 3, 5, 6, 7, 8]
                if max_latent_classes > 1
                else [0, 1, 2, 3, 5, 6]
            ),
            constraints=column_constraints,
            mode="single",
        )
        return builder, ev

    # Build one evaluator to read the resolved role/distribution sets.
    _, evaluator = make_evaluator()
    resolved_specs = []
    for spec in specs:
        primary = spec.transform_columns[0][1]
        allowed = tuple(
            ROLE_NAMES[int(r)] for r in evaluator.allowed_roles.get(primary, [])
        )
        # No group_id_col was supplied, so grouped random effects cannot be
        # estimated; drop them from the catalogue rather than wasting fits.
        allowed = tuple(r for r in allowed if r != "grouped")
        if spec.kind == "binary":
            # Heterogeneity-in-means has no meaning for a binary dummy with
            # no random parameter in the model.
            allowed = tuple(r for r in allowed if r != "hetero")
        dists = tuple(evaluator.allowed_distributions.get(primary, DISTRIBUTIONS))
        resolved_specs.append(
            VariableSpec(
                name=spec.name,
                transform_columns=spec.transform_columns,
                allowed_roles=allowed or ("excluded",),
                allowed_distributions=dists,
                block=spec.block,
                forced_role=spec.forced_role,
                kind=spec.kind,
            )
        )

    space = SearchSpace(
        tuple(resolved_specs),
        max_latent_classes=max_latent_classes,
        allow_dispersion=True,
        family="count",
        task=name,
    )
    return Task(
        name=name,
        space=space,
        make_evaluator=make_evaluator,
        df=expanded,
        meta={"columns": columns, "R": R, "max_latent_classes": max_latent_classes},
    )


def build_bootstrap_count_tasks(
    n_tasks: int = 3,
    *,
    sample_frac: float = 0.7,
    R: int = 10,
    max_latent_classes: int = 1,
    seed: int = 0,
) -> List[Task]:
    """Related tasks with the same schema for multitask pretraining.

    Each task is a bootstrap resample of Example 16-3 with fresh site IDs.
    The variable catalogue and action space are identical, which is exactly
    the setting where a shared DeepSet-Q policy can transfer.
    """
    patch_count_model_initialization()
    base = load_example16_3_model_data()
    rng = np.random.default_rng(seed)
    n = max(20, int(len(base) * float(sample_frac)))
    tasks = []
    for k in range(int(n_tasks)):
        idx = rng.choice(len(base), size=n, replace=True)
        df = base.iloc[idx].reset_index(drop=True).copy()
        # Bootstrap duplicates would otherwise be treated as repeated
        # measures of the same site; give each row a fresh panel ID.
        df["ID"] = np.arange(len(df))
        tasks.append(
            build_count_task(
                df=df,
                R=R,
                max_latent_classes=max_latent_classes,
                seed=seed + k,
                name=f"bootstrap_{k}",
            )
        )
    return tasks


def build_cmf_task(
    *,
    R: int = 50,
    max_latent_classes: int = 2,
    seed: int = 0,
) -> Task:
    patch_count_model_initialization()
    df = load_example16_3_model_data()
    baseline_vars = ["URB", "ACCESS", "GRADEBR", "CURVES"]
    local_vars = ["CURVES", "WIDTH"]
    cmf = CMFExperimentBuilder(
        df=df,
        y_col="FREQ",
        aadt_col="AADT",
        baseline_vars=baseline_vars,
        local_vars=local_vars,
    )

    constraints = (
        ModelConstraints()
        .no_zi("CURVES", "WIDTH", "GRADEBR", "ACCESS", "URB")
        .no_random("URB", "GRADEBR")
        .allow_random("CURVES", distributions=["lognormal", "normal"])
    )

    def make_evaluator():
        builder, ev, _meta = cmf.build_jax_count_evaluator(
            id_col="ID",
            offset_col="OFFSET",
            constraints=constraints,
            max_latent_classes=max_latent_classes,
            R=R,
        )
        return builder, ev

    _, evaluator, meta = cmf.build_jax_count_evaluator(
        id_col="ID",
        offset_col="OFFSET",
        constraints=constraints,
        max_latent_classes=max_latent_classes,
        R=R,
    )
    baseline_set = set(baseline_vars)
    interaction_to_var = {
        col: var for var, col in zip(local_vars, meta["interaction_cols"])
    }
    binary = {"URB", "ACCESS", "GRADEBR"}

    specs: List[VariableSpec] = []
    for col in evaluator.vars:
        if col == meta["log_aadt_col"]:
            name, block, forced = "logAADT", "baseline", "fixed"
        elif col in interaction_to_var:
            name = f"{interaction_to_var[col]}@logAADT"
            block, forced = "local", None
        else:
            name, block, forced = col, "baseline", None
        allowed = tuple(
            ROLE_NAMES[int(r)] for r in evaluator.allowed_roles.get(col, [])
        )
        # No group_id_col -> grouped random effects are not estimable.
        allowed = tuple(r for r in allowed if r != "grouped")
        # The CMF constraint API maps a name shared by a baseline and a local
        # term to the local interaction.  Re-apply the intended baseline
        # restrictions here so the state catalogue matches the model.
        if block == "baseline" and name in ("URB", "GRADEBR"):
            allowed = tuple(
                r for r in allowed
                if r not in ("random_ind", "random_cor", "zi", "grouped", "hetero")
            )
        if block == "baseline" and name == "ACCESS":
            allowed = tuple(r for r in allowed if r not in ("grouped", "hetero"))
        if block == "baseline" and name == "CURVES":
            allowed = tuple(r for r in allowed if r not in ("zi", "grouped"))
        if block == "local" and name == "CURVES@logAADT":
            allowed = tuple(
                r for r in allowed
                if r in ("excluded", "fixed", "random_ind", "random_cor", "hetero")
            )
        specs.append(
            VariableSpec(
                name=name,
                transform_columns=(("identity", col),),
                allowed_roles=allowed or ("excluded",),
                allowed_distributions=tuple(
                    evaluator.allowed_distributions.get(col, DISTRIBUTIONS)
                ),
                block=block,
                forced_role=forced,
                kind="binary" if name in binary else "continuous",
            )
        )

    space = SearchSpace(
        tuple(specs),
        max_latent_classes=max_latent_classes,
        allow_dispersion=True,
        family="cmf",
        task="example16_3_hierarchical_cmf",
    )
    return Task(
        name="cmf",
        space=space,
        make_evaluator=make_evaluator,
        df=df,
        meta={
            "R": R,
            "max_latent_classes": max_latent_classes,
            "baseline_vars": baseline_vars,
            "local_vars": local_vars,
            "cmf_metadata": meta,
        },
    )


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------
def run_sa(task: Task, n_fits: int, seed: int = 0, verbose: bool = False) -> Dict:
    builder, evaluator = task.make_evaluator()
    t0 = time.time()
    buf = io.StringIO()
    init_decision = task.space.to_decision(task.space.default_state())
    if task.space.max_latent_classes > 1:
        # SA's decision vector also carries the per-class masks.  Our state
        # uses the shorter (2D+2) form; pad the masks with zeros ("all
        # classes") so MultiStartSA receives the dimension it expects.
        init_decision = np.pad(
            init_decision, (0, task.space.n_columns), constant_values=0.0
        )
    init = [init_decision]
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        result = builder.run(
            evaluator=evaluator,
            algo="sa",
            max_iter=int(n_fits),
            n_starts=1,
            seed=seed,
            refit=False,
            init_solutions=init,
        )
    elapsed = time.time() - t0
    best_solution = np.asarray(result.get("best_solution", []), dtype=float)
    best_state = (
        task.space.from_decision(best_solution) if best_solution.size else None
    )
    best_score = result.get("best_score")
    return {
        "method": "sa",
        "best_bic": float(best_score) if best_score is not None else 1e12,
        "best_solution": best_solution.tolist(),
        "best_state": best_state.pretty() if best_state else None,
        "best_family": best_state.derived_family() if best_state else None,
        "n_active": best_state.n_active if best_state else None,
        "evaluations": int(len(getattr(evaluator, "cache", {}))),
        "elapsed": elapsed,
        "log_tail": buf.getvalue().splitlines()[-12:],
    }


def _random_state(space: SearchSpace, rng: np.random.Generator) -> CountModelState:
    terms = []
    for spec in space.variables:
        if spec.forced_role is not None:
            active = True
            role = spec.forced_role
        else:
            active = rng.random() < 0.55
            roles = [r for r in spec.allowed_roles if r != "excluded"]
            role = str(rng.choice(roles)) if active and roles else "excluded"
        transform = str(
            rng.choice([t for t, _ in spec.transform_columns])
        )
        dist = str(rng.choice(spec.allowed_distributions))
        terms.append(
            ModelTerm(
                variable=spec.name,
                transform=transform,
                role=role,
                distribution=dist,
                block=spec.block,
                active=role != "excluded" or spec.forced_role is not None,
            )
        )
    disp = int(rng.integers(0, 2)) if space.allow_dispersion else 0
    lc = int(rng.integers(1, max(1, space.max_latent_classes) + 1))
    return space.normalize(
        CountModelState(
            terms=tuple(terms),
            dispersion=disp,
            latent_classes=lc,
            family=space.family,
            task=space.task,
        )
    )


def run_random(
    task: Task, n_fits: int, seed: int = 0, max_attempts: Optional[int] = None
) -> Dict:
    _, evaluator = task.make_evaluator()
    env = DelphosCountEnv(evaluator, task.space)
    env._baseline_result()
    rng = np.random.default_rng(seed)
    max_attempts = int(max_attempts or (n_fits * 50 + 500))
    t0 = time.time()
    # Every method gets the same linear-additive incumbent.
    env.evaluate_state(task.space.default_state())
    attempts = 0
    while len(env._eval_cache) < n_fits and attempts < max_attempts:
        attempts += 1
        state = _random_state(task.space, rng)
        key = tuple(task.space.to_decision(state).tolist())
        if key in env._eval_cache:
            continue
        env.evaluate_state(state)
    elapsed = time.time() - t0
    best_state, best_eval = env.best()
    return {
        "method": "random",
        "best_bic": float(best_eval.bic) if best_eval else 1e12,
        "best_state": best_state.pretty() if best_state else None,
        "best_family": best_state.derived_family() if best_state else None,
        "n_active": best_eval.n_active if best_eval else None,
        "evaluations": len(env._eval_cache),
        "attempts": attempts,
        "elapsed": elapsed,
    }


def run_rl(
    task: Task,
    n_episodes: int,
    seed: int = 0,
    verbose: bool = True,
    agent_config: Optional[dict] = None,
    max_steps: Optional[int] = None,
) -> Dict:
    _, evaluator = task.make_evaluator()
    # Linear LL-improvement reward: the null -> good-model LL gap on this
    # dataset is ~2.6 per observation, so tanh would saturate and hide the
    # differences between good candidates.
    env = DelphosCountEnv(
        evaluator,
        task.space,
        max_steps=max_steps,
        reward_mode="linear",
        start_mode="mixed",
        seed=seed,
    )
    from .agent import AgentConfig

    # A slightly larger net and a higher exploration floor than the paper's
    # long-horizon default: this benchmark only gets 60-80 episodes.
    defaults = dict(
        hidden=128,
        embedding=64,
        learning_rate=3e-3,
        batch_size=64,
        warmup_episodes=8,
        updates_per_episode=8,
        epsilon_start=1.0,
        epsilon_end=0.20,
        epsilon_decay_frac=0.7,
    )
    defaults.update(agent_config or {})
    cfg = SearchConfig(
        n_episodes=int(n_episodes),
        seed=seed,
        verbose=verbose,
        agent=AgentConfig(**defaults),
    )
    search = DelphosSearch(env, cfg)
    t0 = time.time()
    history = search.run(n_episodes=n_episodes, verbose=verbose)
    elapsed = time.time() - t0
    best_state, best_eval = env.best()
    return {
        "method": "delphos_rl",
        "best_bic": float(best_eval.bic) if best_eval else 1e12,
        "best_state": best_state.pretty() if best_state else None,
        "best_family": best_state.derived_family() if best_state else None,
        "n_active": best_eval.n_active if best_eval else None,
        "evaluations": len(env._eval_cache),
        "episodes": n_episodes,
        "elapsed": elapsed,
        "history": [asdict(r) for r in history],
        "agent": asdict(cfg.agent),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def summarize(results: Sequence[Dict]) -> pd.DataFrame:
    rows = []
    best = min((r["best_bic"] for r in results), default=float("nan"))
    for r in results:
        rows.append(
            {
                "method": r["method"],
                "best_bic": r["best_bic"],
                "gap_to_best": r["best_bic"] - best,
                "family": r.get("best_family"),
                "n_active": r.get("n_active"),
                "evaluations": r.get("evaluations"),
                "elapsed_s": round(r.get("elapsed", float("nan")), 2),
                "bic_per_eval": (
                    r["best_bic"] / max(1, r.get("evaluations", 1))
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("best_bic")


def plot_learning_curve(rl_result: Dict, path: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    hist = rl_result.get("history") or []
    if not hist:
        return
    eps = [h["episode"] for h in hist]
    best = [
        h["best_bic"] if np.isfinite(h["best_bic"]) else np.nan
        for h in hist
    ]
    rew = [h["reward"] for h in hist]
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(eps, best, color="#1f77b4", lw=1.8, label="best BIC so far")
    ax1.set_xlabel("episode")
    ax1.set_ylabel("best BIC", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax2 = ax1.twinx()
    ax2.plot(eps, rew, color="#d62728", alpha=0.45, lw=0.9, label="episode reward")
    ax2.set_ylabel("terminal reward", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax1.set_title("Delphos-style RL specification search")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["count", "cmf", "both"], default="both")
    parser.add_argument("--episodes", type=int, default=120)
    parser.add_argument("--fits", type=int, default=120)
    parser.add_argument("--R", type=int, default=50)
    parser.add_argument("--max-latent-classes", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="results/delphos")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    tasks = []
    if args.task in ("count", "both"):
        tasks.append(
            build_count_task(
                R=args.R,
                max_latent_classes=args.max_latent_classes,
                seed=args.seed,
            )
        )
    if args.task in ("cmf", "both"):
        tasks.append(
            build_cmf_task(
                R=args.R,
                max_latent_classes=args.max_latent_classes,
                seed=args.seed,
            )
        )

    all_results = {}
    for task in tasks:
        print("\n" + "=" * 78)
        print(f"TASK: {task.name} | {task.space.n_variables} variables | "
              f"{task.space.n_columns} columns | max_latent_classes="
              f"{task.space.max_latent_classes}")
        print("=" * 78)

        results = []
        print(f"\n[1/3] Delphos RL ({args.episodes} episodes)")
        rl = run_rl(
            task,
            n_episodes=args.episodes,
            seed=args.seed,
            verbose=not args.quiet,
            max_steps=args.max_steps,
        )
        results.append(rl)

        print(f"\n[2/3] Simulated annealing ({args.fits} iterations)")
        sa = run_sa(task, n_fits=args.fits, seed=args.seed)
        results.append(sa)

        print(f"\n[3/3] Random search ({args.fits} unique fits)")
        rnd = run_random(task, n_fits=args.fits, seed=args.seed)
        results.append(rnd)

        table = summarize(results)
        print("\n" + table.to_string(index=False))
        all_results[task.name] = results
        table.to_csv(os.path.join(args.out, f"{task.name}_summary.csv"), index=False)
        with open(os.path.join(args.out, f"{task.name}_results.json"), "w") as f:
            json.dump(results, f, indent=2, default=float)
        plot_learning_curve(
            rl, os.path.join(args.out, f"{task.name}_rl_learning_curve.png")
        )

    with open(os.path.join(args.out, "all_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"\nSaved results to {os.path.abspath(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

