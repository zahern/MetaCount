"""Deterministic execution layer: validate a spec, build and run the search."""

from __future__ import annotations

import contextlib
import io
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from metacountregressor import (
    CMFExperimentBuilder,
    ExperimentBuilder,
    ModelConstraints,
    SearchOutputConfig,
    load_example16_3_model_data,
)

from .profile import load_data
from .schema import ConstraintTier, ExperimentSpec, VariableDecision

_ROLE_CODES = {
    "excluded": 0,
    "fixed": 1,
    "random_ind": 2,
    "random_cor": 3,
    "grouped": 4,
    "hetero": 5,
    "zi": 6,
    "membership": 7,
    "membership_fixed": 8,
}


def _validate_variables(spec: ExperimentSpec, df: pd.DataFrame) -> None:
    missing = [v for v in spec.included_variables if v not in df.columns]
    if missing:
        raise ValueError(
            f"LLM/spec proposed variables that are not in the data: {missing}"
        )
    if spec.response not in df.columns:
        raise ValueError(f"response {spec.response!r} is not in the data")
    for name in (spec.id_col, spec.offset_col, spec.group_id_col):
        if name and name not in df.columns:
            raise ValueError(f"column {name!r} is not in the data")


def build_constraints(spec: ExperimentSpec) -> ModelConstraints:
    """Turn the tiered proposal into a validated ModelConstraints object."""
    constraints = ModelConstraints()
    known = {v.name for v in spec.variables}

    baseline_vars = set(spec.cmf.get("baseline_vars", []) or [])
    local_vars = list(spec.cmf.get("local_vars", []) or [])

    def _safe(name: str) -> str:
        return re.sub(r"[^0-9A-Za-z_]+", "_", str(name)).strip("_") or "var"

    def _evaluator_name(name: str) -> str:
        """Map friendly CMF upper/lower labels to evaluator columns."""
        if spec.family == "cmf":
            if name in baseline_vars:
                return f"__cmf_baseline__{_safe(name)}"
            for local in local_vars:
                if name == f"{local}@logAADT":
                    return f"__cmf_local__{_safe(local)}"
        return name

    # Variable-level proposals are the base layer; tier rules below can only
    # restrict them further.
    for v in spec.variables:
        name = _evaluator_name(v.name)
        if v.include and v.allowed_roles:
            codes = [_ROLE_CODES[r] for r in v.allowed_roles if r in _ROLE_CODES]
            if codes:
                constraints.set_roles(name, codes)
        if not v.include:
            constraints.exclude(name)

    for tier in spec.constraints:
        if not isinstance(tier, ConstraintTier):
            continue
        for rule in tier.rules:
            method = str(rule.get("method", "")).strip()
            variables = [
                _evaluator_name(str(v))
                for v in rule.get("variables", [])
                if str(v) in known
            ]
            if method == "mutual_exclusion":
                groups = [
                    [
                        _evaluator_name(str(v))
                        for v in group
                        if str(v) in known
                    ]
                    for group in rule.get("groups", [])
                ]
                groups = [g for g in groups if len(g) >= 2]
                if groups:
                    constraints.mutual_exclusion(*groups)
                continue
            if not variables:
                continue
            if method == "allow_random":
                constraints.allow_random(
                    *variables,
                    distributions=rule.get("distributions"),
                )
            elif method == "exclude":
                constraints.exclude(*variables)
            elif method in {
                "force_fixed",
                "force_include",
                "no_zi",
                "no_random",
                "membership_only",
                "allow_membership",
                "outcome_only",
            }:
                getattr(constraints, method)(*variables)
            elif method == "set_roles":
                for v in variables:
                    roles = rule.get("roles", [])
                    codes = [_ROLE_CODES[r] for r in roles if r in _ROLE_CODES]
                    if codes:
                        constraints.set_roles(v, codes)
            else:
                raise ValueError(f"unsupported constraint method {method!r}")

    return constraints


def build_experiment(spec: ExperimentSpec):
    """Return ``(df, builder, evaluator, metadata)`` for a validated spec."""
    if spec.family not in {"count", "cmf"}:
        raise NotImplementedError(
            f"the copilot runner currently wires count and cmf; "
            f"{spec.family!r} needs its family_search adapter"
        )
    df = load_data(spec.data_path)
    _validate_variables(spec, df)
    constraints = build_constraints(spec)
    variables = [v.name for v in spec.variables if v.include]

    if spec.family == "cmf":
        aadt = spec.cmf.get("aadt_col")
        if not aadt or aadt not in df.columns:
            raise ValueError("CMF experiments need cmf['aadt_col'] present in the data")
        baseline = list(spec.cmf.get("baseline_vars", []))
        local = list(spec.cmf.get("local_vars", []))
        if not baseline or not local:
            raise ValueError("CMF experiments need baseline_vars and local_vars")
        # Give the upper-level terms unique column names.  Otherwise a
        # variable that appears in both blocks (e.g. CURVES) collides with
        # the lower-level __cmf_local__CURVES interaction in the CMF
        # constraint translation, and "no random CURVES" would silently
        # switch off the local random effect instead of the upper one.
        baseline_cols = []
        for var in baseline:
            col = f"__cmf_baseline__{re.sub(r'[^0-9A-Za-z_]+', '_', str(var)).strip('_') or 'var'}"
            df[col] = df[var]
            baseline_cols.append(col)
        cmf = CMFExperimentBuilder(
            df=df,
            y_col=spec.response,
            aadt_col=aadt,
            baseline_vars=baseline_cols,
            local_vars=local,
        )
        builder, evaluator, meta = cmf.build_jax_count_evaluator(
            id_col=spec.id_col or "ID",
            offset_col=spec.offset_col,
            constraints=constraints,
            max_latent_classes=spec.max_latent_classes,
            R=spec.R,
        )
        return df, builder, evaluator, {"cmf": meta}

    builder = ExperimentBuilder(
        df=df,
        id_col=spec.id_col,
        y_col=spec.response,
        offset_col=spec.offset_col,
        group_id_col=spec.group_id_col,
    )
    evaluator = builder.build_count_evaluator(
        variables=variables,
        constraints=constraints,
        max_latent_classes=spec.max_latent_classes,
        R=spec.R,
        default_roles=(
            [0, 1, 2, 3, 5, 6, 7, 8]
            if spec.max_latent_classes > 1
            else [0, 1, 2, 3, 5, 6]
        ),
        mode="single",
    )
    return df, builder, evaluator, {}


def run_search(
    spec: ExperimentSpec,
    *,
    verbose: bool = False,
    refit: bool = False,
) -> Dict[str, Any]:
    """Build and run a search, returning a compact result for diagnosis."""
    if spec.algo == "delphos_rl":
        return _run_delphos(spec, verbose=verbose)
    df, builder, evaluator, meta = build_experiment(spec)
    output_config = SearchOutputConfig(
        output_dir=spec.output_dir,
        experiment_name=f"copilot_{spec.family}",
        search_description=(
            f"LLM-orchestrated {spec.family} search; objective={spec.objective}"
        ),
    )
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        result = builder.run(
            evaluator=evaluator,
            algo=spec.algo,
            max_iter=int(spec.max_iter),
            seed=int(spec.seed),
            refit=refit,
            output_config=output_config,
            **({"n_starts": 1} if spec.algo in {"sa", "hc", "ga"} else {}),
        )
    best_solution = np.asarray(result.get("best_solution", []), dtype=float)
    best_spec = (
        evaluator.build_spec(best_solution) if best_solution.size else None
    )
    summary = _summarise_spec(best_spec)
    return {
        "algorithm": spec.algo,
        "objective": spec.objective,
        "best_score": float(result.get("best_score", 1e12)),
        "best_solution": best_solution.tolist(),
        "best_spec": best_spec,
        "n_active": summary["n_active"],
        "family": summary["family"],
        "n_obs": int(len(df)),
        "evaluations": int(len(getattr(evaluator, "cache", {}))),
        "search_result": result,
        "log_tail": buf.getvalue().splitlines()[-20:],
    }


def _run_delphos(spec: ExperimentSpec, *, verbose: bool = False) -> Dict[str, Any]:
    """Run the DeepSet-Q specification agent on a validated copilot spec."""
    import time

    from .delphos.agent import AgentConfig
    from .delphos.env import DelphosCountEnv
    from .delphos.search import DelphosSearch, SearchConfig
    from .delphos.state import ROLE_NAMES, SearchSpace, VariableSpec

    df, builder, evaluator, meta = build_experiment(spec)
    cmf_meta = meta.get("cmf", {}) if meta else {}
    specs = []
    for col in evaluator.vars:
        allowed = tuple(
            ROLE_NAMES[int(r)]
            for r in evaluator.allowed_roles.get(col, [])
            if int(r) in ROLE_NAMES
        )
        forced = (
            allowed[0]
            if len(allowed) == 1 and allowed[0] != "excluded"
            else None
        )
        block = "outcome"
        if spec.family == "cmf":
            if col == cmf_meta.get("log_aadt_col"):
                block = "baseline"
            elif col in (cmf_meta.get("interaction_cols") or []):
                block = "local"
            else:
                block = "baseline"
        specs.append(
            VariableSpec(
                name=col,
                transform_columns=(("identity", col),),
                allowed_roles=allowed or ("excluded",),
                allowed_distributions=tuple(
                    evaluator.allowed_distributions.get(col, ("normal",))
                ),
                block=block,
                forced_role=forced,
            )
        )
    space = SearchSpace(
        tuple(specs),
        max_latent_classes=spec.max_latent_classes,
        allow_dispersion=True,
        family=spec.family,
        task="copilot",
    )
    env = DelphosCountEnv(
        evaluator,
        space,
        max_steps=max(8, 2 * space.n_variables + 2),
        reward_mode="linear",
        start_mode="mixed",
        seed=spec.seed,
    )
    cfg = SearchConfig(
        n_episodes=max(1, int(spec.max_iter)),
        seed=spec.seed,
        verbose=verbose,
        agent=AgentConfig(),
    )
    t0 = time.time()
    history = DelphosSearch(env, cfg).run(verbose=verbose)
    best_state, best_eval = env.best()
    best_solution = (
        space.to_decision(best_state) if best_state is not None else np.zeros(0)
    )
    best_spec = evaluator.build_spec(best_solution) if best_solution.size else None
    if best_spec is not None and spec.family == "cmf":
        best_spec = {**best_spec, "family": "cmf", "metadata": dict(cmf_meta)}
    summary = _summarise_spec(best_spec)
    return {
        "algorithm": "delphos_rl",
        "objective": spec.objective,
        "best_score": float(best_eval.bic) if best_eval else 1e12,
        "best_solution": best_solution.tolist(),
        "best_spec": best_spec,
        "n_active": summary["n_active"],
        "family": summary["family"],
        "n_obs": int(len(df)),
        "evaluations": int(len(env._eval_cache)),
        "episodes": len(history),
        "elapsed": time.time() - t0,
        "search_result": {"history": [h.__dict__ for h in history]},
        "log_tail": [],
    }


def _summarise_spec(best_spec: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not best_spec:
        return {"n_active": 0, "family": "unknown"}
    fixed = best_spec.get("fixed_terms") or []
    rdm = best_spec.get("rdm_terms") or []
    rdm_cor = best_spec.get("rdm_cor_terms") or []
    grouped = best_spec.get("grouped_terms") or []
    zi = best_spec.get("zi_terms") or []
    membership = best_spec.get("membership_terms") or []
    lc = int(best_spec.get("latent_classes", 1))
    disp = int(best_spec.get("dispersion", 0))
    n_active = len(fixed) + len(rdm) + len(rdm_cor) + len(grouped) + len(zi) + len(membership)
    family = "nb" if disp else "poisson"
    if lc > 1:
        family = f"lc{lc}_{family}"
    if rdm or rdm_cor or grouped:
        family = f"rp_{family}"
    if zi:
        family = f"zi_{family}"
    if "cmf" in best_spec.get("family", ""):
        family = f"cmf_{family}"
    return {"n_active": n_active, "family": family}


def diagnose_result(
    spec: ExperimentSpec,
    result: Dict[str, Any],
    advisor=None,
) -> Dict[str, Any]:
    """Combine deterministic checks with an optional LLM diagnosis."""
    issues: List[Dict[str, Any]] = []
    score = result.get("best_score")
    if score is None or not np.isfinite(float(score)) or float(score) >= 1e11:
        issues.append(
            {
                "severity": "error",
                "code": "no_convergence",
                "message": "No finite BIC found; relax constraints or raise the budget.",
                "evidence": {"best_score": score},
                "proposed_patch": {"method": "loosen_roles"},
            }
        )
    best_spec = result.get("best_spec") or {}
    n_active = int(result.get("n_active") or 0)
    n_obs = int(result.get("n_obs") or 0)
    if n_obs and n_active > max(5, n_obs // 8):
        issues.append(
            {
                "severity": "warning",
                "code": "overparameterised",
                "message": (
                    f"{n_active} active terms for {n_obs} observations; "
                    "prune weak terms or tighten the random-parameter allowance."
                ),
                "evidence": {"n_active": n_active, "n_obs": n_obs},
                "proposed_patch": {"method": "prune_weak_terms"},
            }
        )
    zi_terms = best_spec.get("zi_terms") or []
    if zi_terms and n_obs:
        zero_frac = 0.0
        try:
            df = load_data(spec.data_path)
            zero_frac = float((df[spec.response] == 0).mean())
        except Exception:
            pass
        if zero_frac < 0.03:
            issues.append(
                {
                    "severity": "warning",
                    "code": "implausible_zi",
                    "message": (
                        f"Zero-inflation terms {zi_terms} selected, but only "
                        f"{zero_frac:.1%} of responses are zeros."
                    ),
                    "evidence": {"zero_fraction": zero_frac, "zi_terms": zi_terms},
                    "proposed_patch": {
                        "method": "constrain",
                        "rules": [{"method": "no_zi", "variables": zi_terms}],
                    },
                }
            )

    if advisor is not None:
        try:
            llm = advisor.diagnose(
                _profile_for(spec), spec, _compact_result(result)
            )
            for issue in llm.get("issues", []):
                if isinstance(issue, dict):
                    issues.append(issue)
        except Exception:
            pass

    return {
        "summary": _summary_text(issues),
        "issues": issues,
        "result": _compact_result(result),
        "next_spec": _apply_issues(spec, issues),
    }


def _profile_for(spec: ExperimentSpec):
    from .profile import profile_dataframe

    return profile_dataframe(load_data(spec.data_path), spec.data_path)


def _compact_result(result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        k: result.get(k)
        for k in (
            "algorithm",
            "objective",
            "best_score",
            "n_active",
            "family",
            "n_obs",
            "evaluations",
        )
    }


def _summary_text(issues: List[Dict[str, Any]]) -> str:
    if not issues:
        return "No material problems detected in this search."
    counts = {}
    for issue in issues:
        counts[issue.get("severity", "info")] = counts.get(issue.get("severity", "info"), 0) + 1
    return "Diagnosis: " + ", ".join(f"{k}={v}" for k, v in counts.items())


def _apply_issues(spec: ExperimentSpec, issues: List[Dict[str, Any]]) -> ExperimentSpec:
    """Return a copy of the spec with deterministic patches applied."""
    import copy
    from dataclasses import replace

    new = copy.deepcopy(spec)
    for issue in issues:
        patch = issue.get("proposed_patch") or {}
        method = patch.get("method")
        if method == "loosen_roles":
            new.constraints = [
                t for t in new.constraints if t.tier not in ("structural", "plausibility")
            ]
        elif method == "prune_weak_terms":
            # Conservative default: disable random roles for the least
            # supported terms, keeping fixed effects.
            for v in new.variables:
                if v.allowed_roles:
                    v.allowed_roles = tuple(
                        r for r in v.allowed_roles if not r.startswith("random")
                    ) or v.allowed_roles
        elif method == "constrain":
            rules = patch.get("rules", [])
            if rules:
                new.constraints.append(
                    ConstraintTier(
                        tier="search",
                        rationale="Applied from diagnosis",
                        rules=list(rules),
                    )
                )
    return new
