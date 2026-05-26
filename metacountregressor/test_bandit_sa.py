"""
test_bandit_sa.py
=================
Benchmarks BanditGuidedSA against vanilla Simulated Annealing on a
synthetic panel count dataset with a KNOWN data-generating process.

True model
----------
    log(mu_it) = -1.5 + beta1_i * x1_it + 0.6*x2_it - 0.5*x3_it
    beta1_i    ~ N(0.8, 0.4^2)    -- x1 has a RANDOM slope (role 2)
    alpha      = 0.5               -- NB2 dispersion
    n1, n2    = zero true effect   -- noise variables

Both algorithms search over:
    roles  = [0, 1, 2, 3, 5]   (excl / fixed / rnd-ind / rnd-cor / hetero)
    models = NB2

Usage
-----
    .venv/Scripts/python metacountregressor/test_bandit_sa.py
    .venv/Scripts/python metacountregressor/test_bandit_sa.py --iters 300 --R 50
    .venv/Scripts/python metacountregressor/test_bandit_sa.py --no-sa    # skip SA
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

# Suppress library noise (not UserWarning — we want those)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*Empty bitcode.*")

# ── path so script runs from repo root or metacountregressor/ directly ──────
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
for _p in [_here, _root]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ════════════════════════════════════════════════════════════════════════
# 1.  Data-Generating Process
# ════════════════════════════════════════════════════════════════════════

#  True fixed effects
TRUE_INT  = -1.5
TRUE_COEF = {"x1": 0.8, "x2": 0.6, "x3": -0.5}
TRUE_SD   = {"x1": 0.4}          # random slope on x1 (site-level)
NB2_ALPHA = 0.5
NOISE_VARS = ["n1", "n2"]


def generate_dgp(n_sites: int = 200, n_periods: int = 3, seed: int = 42) -> pd.DataFrame:
    """
    NB2 panel data with:
      - 3 predictors (x1 random slope, x2/x3 fixed)
      - 2 pure noise vars (n1, n2) with zero effect

    x1 has a SITE-LEVEL RANDOM SLOPE beta1_i ~ N(0.8, 0.4^2).
    This means the search should find role 2 (Rnd-Ind) for x1,
    role 1 (Fixed) for x2/x3, and role 0 (Excluded) for n1/n2.
    """
    rng   = np.random.default_rng(seed)
    N     = n_sites * n_periods
    ids   = np.repeat(np.arange(n_sites), n_periods)

    x1 = rng.standard_normal(N)
    x2 = rng.standard_normal(N)
    x3 = rng.standard_normal(N)
    n1 = rng.standard_normal(N)
    n2 = rng.standard_normal(N)

    # Random slope: one draw per site, repeated across periods
    beta1_i = rng.normal(TRUE_COEF["x1"], TRUE_SD["x1"], n_sites)
    beta1   = np.repeat(beta1_i, n_periods)

    log_mu = TRUE_INT + beta1*x1 + TRUE_COEF["x2"]*x2 + TRUE_COEF["x3"]*x3
    mu     = np.exp(log_mu)
    r      = 1.0 / NB2_ALPHA
    p      = r / (r + mu)
    y      = rng.negative_binomial(r, p)

    df = pd.DataFrame({"ID": ids, "CRASHES": y,
                       "x1": x1, "x2": x2, "x3": x3,
                       "n1": n1, "n2": n2})

    m, v = y.mean(), y.var()
    print("=" * 68)
    print("  SYNTHETIC DGP")
    print("=" * 68)
    print(f"  {n_sites} sites x {n_periods} periods = {N} obs")
    print(f"  x1: random slope  mean={TRUE_COEF['x1']:.1f}  sd={TRUE_SD['x1']:.1f}  [role 2]")
    print(f"  x2: fixed coef    {TRUE_COEF['x2']:+.1f}                               [role 1]")
    print(f"  x3: fixed coef    {TRUE_COEF['x3']:+.1f}                               [role 1]")
    print(f"  n1, n2: zero effect                                      [role 0]")
    print(f"  y: mean={m:.3f}  var/mean={v/m:.2f}  zeros={100*(y==0).mean():.1f}%")
    print()
    return df


# ════════════════════════════════════════════════════════════════════════
# 2.  Build parameter names from spec  (matches build_base_index order)
# ════════════════════════════════════════════════════════════════════════

def build_param_names(spec) -> list[str]:
    """
    Reconstruct parameter names in the same order as build_base_index:
      1. fixed params    (Kf)             — intercept + fixed vars
      2. zi params       (Kzi)            — zero-inflation vars
      3. cor means       (Kr_cor)         — correlated random means
      4. chol factors    (Kr_cor*(K+1)/2) — lower-triangular Cholesky
      5. ind means       (Kr_ind)         — independent random means
      6. ind sds         (Kr_ind)         — independent random SDs
      7. grouped means   (Kg)
      8. grouped sds     (Kg)
      9. hetero params   (Kh * K_rnd)
     10. dispersion      (1 if NB)
    """
    names: list[str] = []

    # 1. Fixed effects (includes __INTERCEPT__)
    for nm in getattr(spec, "fixed_names", ()):
        display = "intercept" if nm == "__INTERCEPT__" else nm
        names.append(display)

    # 2. Zero-inflation
    for nm in getattr(spec, "zi_names", ()):
        names.append(f"zi:{nm}")

    # 3. Correlated random — means first
    cor_names = list(getattr(spec, "random_cor_names", ()))
    for nm in cor_names:
        names.append(f"{nm}:mean_cor")

    # 4. Cholesky factors (lower-triangular, column-major)
    K_cor = len(cor_names)
    for j in range(K_cor):
        for i in range(j, K_cor):
            names.append(f"chol[{cor_names[i]},{cor_names[j]}]")

    # 5. Independent random — means
    ind_names = list(getattr(spec, "random_ind_names", ()))
    for nm in ind_names:
        names.append(f"{nm}:mean")

    # 6. Independent random — SDs (log-scale in optimiser, labelled :lsd)
    for nm in ind_names:
        names.append(f"{nm}:lsd")

    # 7-8. Grouped random
    grp_names = list(getattr(spec, "grouped_names", ()))
    for nm in grp_names:
        names.append(f"{nm}:grp_mean")
    for nm in grp_names:
        names.append(f"{nm}:grp_lsd")

    # 9. Heterogeneity-in-means
    K_rnd_total = getattr(spec, "Kr_cor", 0) + getattr(spec, "Kr_ind", 0)
    all_rnd = cor_names + ind_names
    for h_nm in getattr(spec, "hetro_names", ()):
        for rnd_nm in all_rnd[:K_rnd_total]:
            names.append(f"{h_nm}:h_{rnd_nm}")

    # 10. Dispersion / scale
    mdl = getattr(spec, "model", "nb")
    if mdl == "nb":
        names.append("alpha (dispersion)")
    elif mdl in ("lognormal", "gaussian", "tobit", "weibull", "loglogistic"):
        names.append("sigma")

    return names


# ════════════════════════════════════════════════════════════════════════
# 3.  Full model print  (coefficients, SE, t-stat, CI, significance)
# ════════════════════════════════════════════════════════════════════════

def print_full_model(
    builder,
    evaluator,
    solution:   np.ndarray,
    label:      str = "MODEL",
    model:      str = "nb",
    R:          int = 50,
) -> dict:
    """
    Re-fit the best solution and print a full coefficient table:
      estimate | SE | t-stat | sig | 95% CI
    Uses diagonal Hessian t-stats (O(p) JAX jvp passes, no full Hessian).
    Returns the fit dict enriched with bic_final, n_sig, n_insig.
    """
    from functools import partial

    try:
        from adaptive_search import _fast_tstats
        from main_hpc_lc_patch import mixed_model_loglik
    except ImportError:
        from metacountregressor.adaptive_search import _fast_tstats
        from metacountregressor.main_hpc_lc_patch import mixed_model_loglik

    import jax.numpy as jnp

    print("\n" + "=" * 68)
    print(f"  FULL MODEL REPORT  —  {label}")
    print("=" * 68)

    spec_dict = evaluator.build_spec(solution)
    if spec_dict is None:
        print("  [ERROR] build_spec returned None — invalid solution")
        return {}

    # ── Role summary ───────────────────────────────────────────────
    role_names = {0:"Excl",1:"Fixed",2:"Rnd-Ind",3:"Rnd-Cor",
                  4:"Grouped",5:"Hetero",6:"ZI",7:"Membership",8:"Mem+Fixed"}
    TRUE_VARS  = set(TRUE_COEF.keys())

    print("\n  Variable assignments:")
    print(f"  {'Variable':<8}  {'Role':<12}  {'DGP truth'}")
    print(f"  {'-'*8}  {'-'*12}  {'-'*20}")
    for i, v in enumerate(evaluator.vars):
        role = int(solution[i])
        if v in TRUE_VARS:
            truth = f"mean={TRUE_COEF[v]:+.1f}"
            if v in TRUE_SD:
                truth += f"  sd={TRUE_SD[v]:.1f}  [RANDOM]"
            else:
                truth += "  [FIXED]"
        else:
            truth = "0.0  [NOISE]"
        print(f"  {v:<8}  {role_names.get(role,'?'):<12}  {truth}")
    print()

    # ── Fit ────────────────────────────────────────────────────────
    try:
        fit = builder.fit_manual_model(
            manual_spec  = spec_dict,
            model        = model,
            R            = R,
            print_report = False,
        )
    except Exception as exc:
        print(f"  [ERROR] fit_manual_model failed: {exc}")
        return {}

    result  = fit.get("result")
    spec    = fit.get("spec")
    data    = fit.get("data")
    summary = fit.get("summary", {})

    if result is None or spec is None or data is None:
        print("  [ERROR] incomplete fit result")
        return fit

    params_np = np.asarray(result.params, dtype=float)
    params_j  = jnp.asarray(params_np)
    obj       = partial(mixed_model_loglik, data=data, spec=spec)

    # ── Standard errors via diagonal Hessian ───────────────────────
    print(f"  Computing SEs for {len(params_np)} parameters "
          f"(O(p) diagonal Hessian)...", end=" ", flush=True)
    t0 = time.perf_counter()
    try:
        t_stats, se = _fast_tstats(params_j, obj)
    except Exception as exc:
        print(f"FAILED ({exc})")
        t_stats = np.full_like(params_np, np.nan)
        se      = np.full_like(params_np, np.nan)
    else:
        print(f"{(time.perf_counter()-t0)*1000:.0f} ms")

    # ── Build parameter names from spec layout ─────────────────────
    param_names = build_param_names(spec)
    # Pad any tail if spec reporting is incomplete
    while len(param_names) < len(params_np):
        param_names.append(f"extra_param_{len(param_names)}")

    # ── Coefficient table ──────────────────────────────────────────
    def stars(t: float) -> str:
        if not np.isfinite(t): return "   "
        a = abs(t)
        if a >= 3.09: return "***"
        if a >= 1.96: return "** "
        if a >= 1.64: return "*  "
        return "   "

    hdr = (f"  {'Parameter':<32}  {'Estimate':>9}  {'SE':>8}  "
           f"{'t-stat':>8}  Sig  {'95% CI':^27}")
    print(f"\n{hdr}")
    print("  " + "-" * 99)

    n_sig = n_insig = 0
    for i, nm in enumerate(param_names):
        if i >= len(params_np):
            break
        est  = float(params_np[i])
        s    = float(se[i])      if i < len(se)      and np.isfinite(se[i])      else np.nan
        ts   = float(t_stats[i]) if i < len(t_stats) and np.isfinite(t_stats[i]) else np.nan
        ci_lo = est - 1.96*s if np.isfinite(s) else np.nan
        ci_hi = est + 1.96*s if np.isfinite(s) else np.nan

        s_str    = f"{s:8.4f}"    if np.isfinite(s)    else f"{'nan':>8}"
        ts_str   = f"{ts:+8.2f}"  if np.isfinite(ts)   else f"{'nan':>8}"
        lo_str   = f"{ci_lo:+.3f}" if np.isfinite(ci_lo) else "     nan"
        hi_str   = f"{ci_hi:+.3f}" if np.isfinite(ci_hi) else "    nan"

        print(f"  {nm:<32}  {est:+9.4f}  {s_str}  {ts_str}  {stars(ts)}  "
              f"[{lo_str}, {hi_str}]")

        skip = nm in ("intercept", "alpha (dispersion)", "sigma")
        if not skip:
            if np.isfinite(ts) and abs(ts) >= 1.64:
                n_sig += 1
            else:
                n_insig += 1

    print("  " + "-" * 99)
    print("  Sig: *** |t|>=3.09   ** |t|>=1.96   * |t|>=1.64\n")

    # ── Fit statistics ─────────────────────────────────────────────
    # summary keys from print_summary() are 'loglik', 'bic', 'aic', 'num_parm', 'n_obs'
    ll  = float(summary.get("loglik", np.nan))
    bic = float(summary.get("bic",    np.nan))
    aic = float(summary.get("aic",    np.nan))
    n   = int(data["y"].shape[0])
    k   = len(params_np)

    # Recompute from objective if summary didn't populate them
    if not np.isfinite(ll):
        try:
            ll  = float(-obj(params_j))
        except Exception:
            ll = np.nan
    if not np.isfinite(bic) and np.isfinite(ll):
        bic = k * np.log(n) - 2.0 * ll
    if not np.isfinite(aic) and np.isfinite(ll):
        aic = 2 * k - 2.0 * ll

    print(f"  Fit statistics")
    print(f"    N (obs)          : {n}")
    print(f"    k (params)       : {k}")
    print(f"    Log-likelihood   : {ll:+.3f}")
    print(f"    BIC              : {bic:.3f}")
    print(f"    AIC              : {aic:.3f}")
    print(f"    Significant params (|t|>=1.64) : {n_sig}")
    print(f"    Insignificant    (|t|< 1.64)   : {n_insig}")
    print()

    fit["bic_final"]   = float(bic)
    fit["ll_final"]    = float(ll)
    fit["n_sig"]       = n_sig
    fit["n_insig"]     = n_insig
    fit["param_names"] = param_names
    fit["t_stats"]     = t_stats
    return fit


# ════════════════════════════════════════════════════════════════════════
# 4.  Evaluator factory  (fresh instance each time — clears cache)
# ════════════════════════════════════════════════════════════════════════

def make_evaluator(builder, R: int):
    """
    Single-class evaluator that allows roles 0,1,2,3,5 (fixed + random).
    Returns a fresh StructureEvaluatorLC with empty caches.
    """
    return builder.build_evaluator(
        variables          = ["x1", "x2", "x3", "n1", "n2"],
        max_latent_classes = 1,
        R                  = R,
        mode               = "single",
        # default_roles = [0,1,2,3,5] is the default for max_latent_classes=1
    )


# ════════════════════════════════════════════════════════════════════════
# 5a. Role-space audit  (run before search to confirm random roles present)
# ════════════════════════════════════════════════════════════════════════

ROLE_LABELS = {
    0: "Excl",
    1: "Fixed",
    2: "Rnd-Ind",
    3: "Rnd-Cor",
    4: "Grouped",
    5: "Hetero",
    6: "ZI",
    7: "Membership",
    8: "Mem+Fixed",
}
RANDOM_ROLES = {2, 3, 4, 5}   # roles that involve random / mixed effects


def audit_search_space(evaluator) -> None:
    """
    Print the allowed roles for every variable and confirm that random-
    parameter roles (2=Rnd-Ind, 3=Rnd-Cor, 5=Hetero) are reachable.

    This is the definitive check that the search CAN propose random params.
    """
    print("\n" + "=" * 68)
    print("  SEARCH SPACE AUDIT  —  allowed roles per variable")
    print("=" * 68)
    print(f"  {'Variable':<8}  {'Allowed roles (code: label)'}")
    print(f"  {'-'*8}  {'-'*50}")

    any_random = False
    for v in evaluator.vars:
        roles = evaluator.allowed_roles.get(v, [0])
        role_str = "  ".join(f"{r}:{ROLE_LABELS.get(r,r)}" for r in sorted(roles))
        has_rnd  = bool(set(roles) & RANDOM_ROLES)
        flag     = "  [random reachable]" if has_rnd else ""
        print(f"  {v:<8}  {role_str}{flag}")
        if has_rnd:
            any_random = True

    print()
    if any_random:
        print("  [CONFIRMED] At least one variable has random-effect roles in its")
        print("  allowed set.  The bandit UCB loop and initial solution both sample")
        print("  from this set, so random parameters WILL be proposed and evaluated.")
    else:
        print("  [WARNING] No variable has random-effect roles — search is fixed-only.")
        print("  Pass default_roles=[0,1,2,3,5] or use fixed_override to allow roles 2/3.")
    print()


# ════════════════════════════════════════════════════════════════════════
# 5b. Post-search role-type breakdown  (shows what was actually evaluated)
# ════════════════════════════════════════════════════════════════════════

def report_role_type_coverage(bandit_sa, var_names: list[str]) -> None:
    """
    Summarise how many UCB arm updates fell in each role category.
    Confirms that random-effect arms were not just available but USED.
    """
    print("\n" + "=" * 68)
    print("  ROLE-TYPE COVERAGE  —  what the bandit actually evaluated")
    print("=" * 68)

    n_vars  = len(var_names)
    n_roles = bandit_sa.bandit.n_roles

    # Collect updates per role category across all variables
    role_updates: dict[int, int] = {r: 0 for r in range(n_roles)}
    role_evals: dict[int, list[str]] = {r: [] for r in range(n_roles)}

    for vi in range(n_vars):
        for role in range(n_roles):
            arm     = bandit_sa.bandit._arm_idx(vi, role)
            n_upd   = int(np.trace(bandit_sa.bandit.A[arm])) - bandit_sa.bandit.N_FEATURES
            if n_upd > 0:
                role_updates[role] += n_upd
                role_evals[role].append(f"{var_names[vi]}({n_upd})")

    print(f"\n  {'Role':<12}  {'Label':<12}  {'Total evals':>11}  Variables evaluated")
    print(f"  {'-'*12}  {'-'*12}  {'-'*11}  {'-'*35}")
    for role in sorted(role_updates):
        n     = role_updates[role]
        label = ROLE_LABELS.get(role, str(role))
        tag   = "  <-- RANDOM" if role in RANDOM_ROLES else ""
        vars_str = ", ".join(role_evals[role]) if role_evals[role] else "(none)"
        print(f"  {role:<12}  {label:<12}  {n:>11}  {vars_str}{tag}")

    total_random = sum(role_updates[r] for r in RANDOM_ROLES)
    total_all    = sum(role_updates.values())
    pct = 100 * total_random / max(total_all, 1)

    print(f"\n  Total evaluations : {total_all}")
    print(f"  Random-role evals : {total_random}  ({pct:.1f}% of all arm updates)")
    if total_random > 0:
        print("  [CONFIRMED] Random-parameter models were proposed and fitted.")
    else:
        print("  [NOTE] No random-parameter arms updated yet — run more iterations.")
    print()


# ════════════════════════════════════════════════════════════════════════
# 5.  Run BanditGuidedSA
# ════════════════════════════════════════════════════════════════════════

def run_bandit(builder, evaluator, max_iter: int, R: int, seed: int = 0) -> dict:
    try:
        from adaptive_search import BanditGuidedSA
    except ImportError:
        from metacountregressor.adaptive_search import BanditGuidedSA

    print("\n" + "=" * 68)
    print(f"  BANDIT-GUIDED SA   ({max_iter} iterations,  R={R})")
    print("=" * 68)
    print("  Reward = -delta_BIC - n_insig * 5.0")
    print("  Penalty per insignificant variable (|t|<1.0) added to BIC")
    print()

    # ── Confirm random roles are reachable before starting ────────
    audit_search_space(evaluator)

    sa = BanditGuidedSA(
        builder      = builder,
        evaluator    = evaluator,
        t_start      = 5.0,
        t_end        = 0.01,
        alpha        = 0.99,
        bandit_alpha = 1.5,
        blend        = 0.4,
        t_stat_min   = 1.0,
        extreme_coef = 15.0,
        use_checker  = True,
        model        = "nb",
        R            = R,
    )

    t0 = time.perf_counter()
    res = sa.run(max_iter=max_iter, seed=seed,
                 print_every=max(1, max_iter // 10))
    res["elapsed"]   = time.perf_counter() - t0
    res["algorithm"] = "BanditGuidedSA"
    res["bandit_sa"] = sa

    # ── Show what role types were actually evaluated ───────────────
    report_role_type_coverage(sa, list(evaluator.vars))

    print(f"  Done in {res['elapsed']:.1f}s  "
          f"({res['elapsed']/max_iter*1000:.0f} ms/iter)")
    return res


# ════════════════════════════════════════════════════════════════════════
# 6.  Run vanilla SA
# ════════════════════════════════════════════════════════════════════════

def run_vanilla_sa(builder, evaluator, max_iter: int, seed: int = 0) -> dict:
    print("\n" + "=" * 68)
    print(f"  VANILLA SA   ({max_iter} iterations,  R matches evaluator)")
    print("=" * 68)

    t0  = time.perf_counter()
    res = builder.run(evaluator=evaluator, algo="sa",
                      max_iter=max_iter, seed=seed)
    res["elapsed"]   = time.perf_counter() - t0
    res["algorithm"] = "VanillaSA"
    print(f"\n  Done in {res['elapsed']:.1f}s  "
          f"({res['elapsed']/max_iter*1000:.0f} ms/iter)")
    return res


# ════════════════════════════════════════════════════════════════════════
# 7.  Side-by-side comparison table
# ════════════════════════════════════════════════════════════════════════

def print_comparison(results: list[dict], var_names: list[str]) -> None:
    TRUE_VARS  = set(TRUE_COEF.keys())
    NOISE_SET  = set(NOISE_VARS)
    role_names = {0:"Excl",1:"Fixed",2:"Rnd-Ind",3:"Rnd-Cor",
                  4:"Grouped",5:"Hetero"}

    print("\n" + "=" * 68)
    print("  SEARCH COMPARISON")
    print("=" * 68)

    rows = []
    for r in results:
        best_sol = r["best_solution"]
        D = len(var_names)
        active = {var_names[i]: role_names.get(int(best_sol[i]), str(int(best_sol[i])))
                  for i in range(D) if int(best_sol[i]) != 0}
        true_found  = [f"{v}({rl})" for v, rl in active.items() if v in TRUE_VARS]
        noise_found = [f"{v}({rl})" for v, rl in active.items() if v in NOISE_SET]
        bic_s = r.get("best_score", np.nan)
        bic_r = r.get("bic_final", bic_s)
        rows.append({
            "Algorithm":     r["algorithm"],
            "BIC (search)":  f"{bic_s:.1f}",
            "BIC (refit)":   f"{bic_r:.1f}" if np.isfinite(float(bic_r)) else "N/A",
            "LL":            f"{r.get('ll_final', np.nan):.1f}",
            "True vars":     ", ".join(true_found) if true_found else "(none)",
            "Noise in model":  ", ".join(noise_found) if noise_found else "(none -- good)",
            "n_sig":         str(r.get("n_sig", "?")),
            "Time (s)":      f"{r.get('elapsed', 0):.0f}",
        })

    # Print
    keys   = list(rows[0].keys())
    widths = {k: max(len(k), max(len(str(row[k])) for row in rows)) for k in keys}
    header = "  " + "  ".join(f"{k:<{widths[k]}}" for k in keys)
    sep    = "  " + "  ".join("-" * widths[k] for k in keys)
    print(header)
    print(sep)
    for row in rows:
        print("  " + "  ".join(f"{str(row[k]):<{widths[k]}}" for k in keys))

    # Winner by refit BIC
    bic_vals = [float(r.get("bic_final", r.get("best_score", np.inf)))
                for r in results]
    idx = int(np.argmin(bic_vals))
    print(f"\n  ** Winner by BIC: {results[idx]['algorithm']}  "
          f"BIC={bic_vals[idx]:.2f} **")
    if len(results) > 1:
        others = [b for i, b in enumerate(bic_vals) if i != idx]
        print(f"     Delta BIC vs next best: {min(others)-bic_vals[idx]:+.2f}")
    print()


# ════════════════════════════════════════════════════════════════════════
# 8.  Main
# ════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters",  type=int, default=200,
                        help="Iterations per algorithm  (default 200)")
    parser.add_argument("--sites",  type=int, default=200)
    parser.add_argument("--R",      type=int, default=30,
                        help="Halton draws (default 30 — use 100+ for publication)")
    parser.add_argument("--seed",   type=int, default=0)
    parser.add_argument("--no-sa",  action="store_true",
                        help="Skip the vanilla SA comparison")
    args = parser.parse_args()

    print("\n" + "=" * 68)
    print("  BANDIT-GUIDED SA  vs  VANILLA SA  —  BENCHMARK")
    print("=" * 68)
    print(f"  iters={args.iters}  sites={args.sites}  R={args.R}  seed={args.seed}")
    print()

    # ── Data ────────────────────────────────────────────────────────
    df = generate_dgp(n_sites=args.sites, n_periods=3, seed=42)

    # ── Import ──────────────────────────────────────────────────────
    print("  Importing MetaCount...", end=" ", flush=True)
    import main_hpc_lc_patch as _   # patches main_hpc in-place
    from experiment_package import ExperimentBuilder
    print("OK\n")

    builder = ExperimentBuilder(df, id_col="ID", y_col="CRASHES")

    # ── Pre-fit checks ───────────────────────────────────────────────
    builder.validate_before_fit(variables=["x1","x2","x3","n1","n2"])
    ok = builder.smoke_test(fixed_terms=["x1","x2"], model="nb",
                            R=args.R, latent_classes=1)
    assert ok, "Smoke test FAILED — fitting pipeline is broken"
    print("  Smoke test: PASS\n")

    all_results = []

    # ═══════════════════════════════════════════════════════════════
    # RUN 1 — BanditGuidedSA
    # ═══════════════════════════════════════════════════════════════
    ev_b = make_evaluator(builder, R=args.R)
    res_b = run_bandit(builder, ev_b, max_iter=args.iters,
                       R=args.R, seed=args.seed)

    # Full model for bandit best solution
    fit_b = print_full_model(
        builder, ev_b,
        solution = res_b["best_solution"],
        label    = (f"BanditGuidedSA  [search BIC={res_b['best_score']:.1f}]"),
        model    = "nb",
        R        = args.R,
    )
    res_b.update({k: fit_b.get(k) for k in
                  ("bic_final","ll_final","n_sig","n_insig","param_names","t_stats")
                  if k in fit_b})
    all_results.append(res_b)

    # ═══════════════════════════════════════════════════════════════
    # RUN 2 — Vanilla SA  (skippable with --no-sa)
    # ═══════════════════════════════════════════════════════════════
    if not args.no_sa:
        ev_sa = make_evaluator(builder, R=args.R)
        res_sa = run_vanilla_sa(builder, ev_sa,
                                max_iter=args.iters, seed=args.seed)

        # Full model for SA best solution
        fit_sa = print_full_model(
            builder, ev_sa,
            solution = res_sa["best_solution"],
            label    = (f"VanillaSA  [search BIC={res_sa['best_score']:.1f}]"),
            model    = "nb",
            R        = args.R,
        )
        res_sa.update({k: fit_sa.get(k) for k in
                       ("bic_final","ll_final","n_sig","n_insig","param_names","t_stats")
                       if k in fit_sa})
        all_results.append(res_sa)

    # ═══════════════════════════════════════════════════════════════
    # Comparison table
    # ═══════════════════════════════════════════════════════════════
    var_names = list(ev_b.vars)
    print_comparison(all_results, var_names)

    # ── Role memory (Thompson priors) ───────────────────────────────
    sa_obj = res_b.get("bandit_sa")
    if sa_obj is not None:
        print("  Role memory — Thompson prior failure rates")
        print("  (noise vars should have higher failure rate than true vars)\n")
        rpt = sa_obj.role_memory.report(var_names)
        if len(rpt):
            print(rpt.head(10).to_string(index=False))
        else:
            print("  (no data — run more iterations)")

        print("\n  Bandit top arms — UCB posterior quality")
        print("  (true vars should rank above noise vars)\n")
        top = sa_obj.bandit.report_top_arms(var_names, top_n=10)
        print(top.to_string(index=False))
        print()

    print("  DONE\n")


if __name__ == "__main__":
    main()
