"""SparseEA-AGDS specification search for the two-tier AADT CMF model.

The search perturbs, per candidate variable:
  * whether it is in the model                  (include gene, maskable)
  * which tier it lives in                      (1 = upper/intercept, 2 = lower/AADT slope)
  * whether it is a random parameter and the
    mixing distribution                         (0 fixed, 1 normal, 2 uniform,
                                                 3 triangular, 4 lognormal; maskable)
and globally the count model (Poisson/NB) and the correlation between the
random terms (rho gene).  The winner is refitted and turned into a CMF
catalogue with the AADT attribution of the lower-tier terms.
"""

import random
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.stats import chi2

from . import GA_CMF_AADT_JAX as G
from .metaheuristics import SparseEA_AGDS

DISTS = ("normal", "uniform", "triangular", "lognormal")
RHOS = (0.0, 0.3, 0.6)
MODELS = ("poisson", "nb")


def _is_binary(series) -> bool:
    """True when the column only ever takes 0 or 1."""
    values = pd.unique(np.asarray(series, dtype=float))
    values = values[np.isfinite(values)]
    return len(values) > 0 and bool(np.isin(values, (0.0, 1.0)).all())


def derive_forbidden_pairs(data, variables, corr_threshold=None):
    """Candidate pairs that must not be selected together, derived from the data.

    Two ways a pair can be unidentifiable side by side:

    * **nesting.**  For two 0/1 indicators, if every row with ``A = 1`` also has
      ``B = 1`` (and not conversely), ``A``'s effect is already inside ``B``'s:
      the pair spans the same rows and the coefficient on ``A`` has nothing of
      its own to explain.  This is the `Curve` inside `Curve50` case.
    * **near-duplication** (opt-in via ``corr_threshold``).  ``|r|`` above the
      threshold is the continuous analogue; it is off by default because
      correlated regressors are not automatically redundant, whereas nesting
      is a structural fact about the design.

    Derived rather than named, so it follows the data: a variable renamed, or
    recomputed on a different sample, cannot silently keep a stale exclusion.

    Returns a list of dicts ``{"a", "b", "reason", "detail"}`` where ``a`` is
    the nested term for ``reason="nested"`` (``a`` implies ``b``).
    """
    from itertools import combinations, permutations

    cols = [v for v in variables if v in data.columns]
    if not cols:
        return []
    X = data[cols].astype(float)
    binary = [c for c in cols if _is_binary(X[c])]

    found: dict[tuple[str, str], dict] = {}
    for a, b in permutations(binary, 2):
        av, bv = X[a] > 0, X[b] > 0
        n_a, n_b = int(av.sum()), int(bv.sum())
        # n_a >= n_b is excluded: equal supports are a duplicate, not a nesting,
        # and the reverse direction is caught on the swapped permutation.  A
        # column that is always on (n_b == len(X)) vacuously contains every
        # indicator, which says nothing about the design, so it cannot be the
        # outer term either.
        if n_a == 0 or n_b == 0 or n_a == len(X) or n_b == len(X):
            continue
        if n_a >= n_b:
            continue
        if int((av & ~bv).sum()) == 0:
            key = tuple(sorted((a, b)))
            found[key] = {
                "a": a, "b": b, "reason": "nested",
                "detail": (f"all {n_a} rows with {a}=1 also have {b}=1, so {a} "
                           f"is nested in {b}"),
            }
    for a, b in permutations(binary, 2):
        av, bv = X[a] > 0, X[b] > 0
        n_a, n_b = int(av.sum()), int(bv.sum())
        if n_a == 0 or n_b == 0 or n_a != n_b:
            continue
        if bool((av == bv).all()):
            key = tuple(sorted((a, b)))
            found.setdefault(key, {
                "a": a, "b": b, "reason": "duplicate",
                "detail": f"{a} and {b} are the same indicator on every row",
            })

    if corr_threshold is not None:
        corr = X.corr().abs()
        for a, b in combinations(cols, 2):
            r = corr.loc[a, b]
            if not np.isfinite(r) or r < float(corr_threshold):
                continue
            key = tuple(sorted((a, b)))
            found.setdefault(key, {
                "a": a, "b": b, "reason": "correlated",
                "detail": f"|corr({a}, {b})| = {r:.3f}",
            })

    return [found[k] for k in sorted(found)]


def normalise_forbidden(forbidden):
    """Turn dicts or ``(a, b)`` tuples into unordered frozensets."""
    out = []
    for item in forbidden or ():
        if isinstance(item, dict):
            pair = (item.get("a"), item.get("b"))
        else:
            pair = tuple(item)[:2]
        if len(pair) != 2 or pair[0] is None or pair[1] is None:
            raise ValueError(f"cannot read a forbidden pair from {item!r}")
        out.append(frozenset((str(pair[0]), str(pair[1]))))
    return out


def find_violation(spec, forbidden):
    """First forbidden pair present in a specification, or None.

    ``spec`` is the usual ``{"upper": [...], "lower": [...]}`` mapping; a term
    counts as selected whichever tier it sits in, because nesting makes the pair
    unidentifiable regardless of the tier split.
    """
    selected = set(map(str, spec.get("upper", []))) | set(map(str, spec.get("lower", [])))
    for pair in normalise_forbidden(forbidden):
        if pair <= selected:
            return tuple(sorted(pair))
    return None


class CMFAGDSObjective:
    """Adapter exposing the interface SparseEA_AGDS expects."""

    is_multi = False
    maximize = False

    def __init__(self, data, variables, max_terms=8, min_lower=0, R=100, n_hops=1,
                 allow_models=MODELS, allow_dists=DISTS, allow_rho=RHOS,
                 integration="halton", seed=0, instance_name="run_agds_cmf",
                 max_time=1e9, forbid_together=None):
        self.data = data
        self.variables = list(variables)
        self.K = len(self.variables)
        self.max_terms = max_terms
        # A lower-tier term makes the AADT elasticity covariate-dependent, i.e.
        # a genuinely hierarchical CMF.  With min_lower = 0 the search may
        # return a plain random-parameters model instead; set it to 1 to keep
        # the search inside the hierarchical class.
        self.min_lower = int(min_lower)
        self.R = R
        self.n_hops = n_hops
        self.integration = integration
        self.allow_models = tuple(allow_models)
        self.allow_dists = tuple(allow_dists)
        self.allow_rho = tuple(allow_rho)
        self._obj_1 = "bic"
        self._obj_2 = None
        self.instance_name = instance_name
        self._seed = seed
        self._max_time = max_time
        self.algorithm = "agds"
        self.Last_Sol = None
        # Pairs that must not be co-selected (nested indicators and the like).
        # Enforced as a hard constraint in get_fitness, alongside max_terms and
        # min_lower: a violating layout is never scored, so the search cannot
        # report one as its winner.
        self.forbid_together = normalise_forbidden(forbid_together)
        self.cache = {}
        K = self.K
        dist_vals = [0] + list(range(1, len(self.allow_dists) + 1))
        self._discrete_values = (
            [[0, 1] for _ in range(K)]                    # include
            + [[1, 2] for _ in range(K)]                  # tier
            + [list(dist_vals) for _ in range(K)]         # fixed / random+dist
            + [list(range(1, len(self.allow_models) + 1))]  # model
            + [list(range(len(self.allow_rho)))]          # rho index
        )

    # --- interface used by SparseEA_AGDS ---------------------------------
    def _get_obj1(self):
        return self._obj_1

    def _get_obj2(self):
        return self._obj_2

    def get_num_parameters(self):
        return len(self._discrete_values)

    def is_discrete(self, i):
        return True

    def get_value(self, i, j=None):
        return random.choice(self._discrete_values[i])

    def get_max_time(self):
        return self._max_time

    def get_termination_iter(self):
        return 10 ** 9

    def use_random_seed(self):
        return True

    def set_random_seed(self):
        random.seed(self._seed)
        np.random.seed(self._seed)

    def get_instance_name(self):
        return self.instance_name

    # --- decoding ---------------------------------------------------------
    def decode(self, layout):
        K = self.K
        inc = [int(layout[i]) for i in range(K)]
        tier = [int(layout[K + i]) for i in range(K)]
        dst = [int(layout[2 * K + i]) for i in range(K)]
        model = self.allow_models[int(layout[3 * K]) - 1]
        rho = self.allow_rho[int(layout[3 * K + 1])]
        base_m = np.array([inc[i] and tier[i] == 1 for i in range(K)], float)
        loc_m = np.array([inc[i] and tier[i] == 2 for i in range(K)], float)
        rand = [d != 0 for d in dst]
        names = [self.allow_dists[d - 1] if d else "normal" for d in dst]
        return base_m, loc_m, rand, names, model, rho

    def spec(self, layout):
        base_m, loc_m, rand, names, model, rho = self.decode(layout)
        pick = lambda m: [v for v, k in zip(self.variables, m) if k]
        pr = lambda m: [r for r, k in zip(rand, m) if k]
        pn = lambda m: [n for n, k in zip(names, m) if k]
        return {"upper": pick(base_m), "lower": pick(loc_m),
                "rand_upper": pr(base_m), "rand_lower": pr(loc_m),
                "dist_upper": pn(base_m), "dist_lower": pn(loc_m),
                "model": model, "rho": rho if sum(
                    pr(base_m)) + sum(pr(loc_m)) > 1 else 0.0}

    def get_fitness(self, layout, multi=False, max_routine=2):
        key = tuple(int(v) for v in layout)
        sp = self.spec(layout)
        n_terms = len(sp["upper"]) + len(sp["lower"])
        key = (tuple(sp["upper"]), tuple(sp["lower"]),
               tuple(sp["rand_upper"]), tuple(sp["rand_lower"]),
               tuple(d for d, r in zip(sp["dist_upper"], sp["rand_upper"]) if r),
               tuple(d for d, r in zip(sp["dist_lower"], sp["rand_lower"]) if r),
               sp["model"], sp["rho"])
        if key in self.cache:
            return {"layout": list(layout), "bic": self.cache[key]}
        n_lower = len(sp["lower"])
        if n_terms == 0 or n_terms > self.max_terms or n_lower < self.min_lower:
            self.cache[key] = 1e12
            return {"layout": list(layout), "bic": 1e12}
        if find_violation(sp, self.forbid_together):
            self.cache[key] = 1e12
            return {"layout": list(layout), "bic": 1e12}
        base_m, loc_m, rand, names, model, rho = self.decode(layout)
        K = self.K
        sol = np.concatenate([base_m, loc_m])
        # evaluate_model takes the same candidate list for both tiers; the
        # masks decide which tier each variable is in.
        dists_all = list(names) + list(names)
        try:
            fit, _ = G.evaluate_model(
                sol, self.data, self.variables, self.variables,
                model=model, rand_baseline_all=list(rand),
                rand_local_all=list(rand), R=self.R,
                integration=self.integration, n_hops=self.n_hops,
                dists_all=dists_all, rho=sp["rho"])
            fit = float(fit)
        except Exception:
            fit = 1e12
        if not np.isfinite(fit):
            fit = 1e12
        self.cache[key] = fit
        self.Last_Sol = {"layout": list(layout), "bic": fit}
        return {"layout": list(layout), "bic": fit}


def run_agds_search(data, variables, seeds=(0,), pop_size=12, generations=10,
                    **objective_kwargs):
    """Run SparseEA-AGDS once per seed; returns a list of result dicts sorted
    by fitness (best first)."""
    out = []
    for s in seeds:
        obj = CMFAGDSObjective(data, variables, seed=s,
                               instance_name=f"run_agds_cmf_{s}",
                               **objective_kwargs)
        agds = SparseEA_AGDS(obj, _pop_size=pop_size, _max_iter=generations)
        res = agds.run()
        layout, fit = res[0], res[1]
        out.append({"seed": s, "fitness": float(fit), "layout": list(layout),
                    "spec": obj.spec(layout), "n_evaluated": len(obj.cache)})
    return sorted(out, key=lambda r: r["fitness"])


def aadt_attribution(params, selected_baseline, selected_local, rand_baseline,
                     rand_local, data, model="poisson",
                     levels=(0.05, 0.25, 0.5, 0.75, 0.95)):
    """CMF of each lower-tier term at AADT quantiles (CMF = AADT**beta for a
    unit change) plus the mean elasticity of crashes with respect to AADT."""
    labels = G._param_labels(selected_baseline, selected_local,
                             rand_baseline, rand_local, model)
    p = dict(zip(labels, np.asarray(params, float)))
    aadt = data["AADT"].to_numpy(float)
    qs = np.quantile(aadt, levels)
    rows = []
    for v in selected_local:
        for q, a in zip(levels, qs):
            rows.append({"term": v, "quantile": q, "aadt": a,
                         "cmf": float(a ** p[f"beta[{v}]"])})
    elast = p["beta0"] + sum(
        p[f"beta[{v}]"] * float(data[v].mean()) for v in selected_local)
    return pd.DataFrame(rows), float(elast)


def boundary_lrt_pvalue(lr):
    """p-value for H0: sigma = 0 against H1: sigma > 0, one variance component.

    The null sits on the boundary of the parameter space (sigma >= 0), so the
    likelihood-ratio statistic is not chi-squared with one degree of freedom: it
    follows the 50:50 mixture 0.5*chi2_0 + 0.5*chi2_1 (Chernoff 1954;
    Self & Liang 1987; Stram & Lee 1994).  In practice that is half the naive
    chi2_1 p-value, and p = 0.5 when the restricted fit is no worse - a nominal
    chi2_1 tail probability would overstate the evidence for heterogeneity.
    """
    lr = float(lr)
    if not np.isfinite(lr) or lr <= 0:
        return 0.5
    return float(0.5 * chi2.sf(lr, 1))


def _sigma_column(spec, tier, k):
    """Draw column and parameter label of one random term.

    Draw order is baseline random terms first, then local ones, so a term's
    column in ``draws`` is its position among the random terms of that order.
    """
    ru = spec["rand_upper"]
    key = "alpha" if tier == "upper" else "beta"
    if tier == "upper":
        col = sum(1 for f in ru[:k] if f)
    else:
        col = sum(ru) + sum(1 for f in spec["rand_lower"][:k] if f)
    return col, f"sigma_{key}[{spec[tier][k]}]"


def _restricted_ll_common_draws(fit, spec, tier, k, R, integration="halton"):
    """Refit with one sigma pinned at zero, on the full model's own draws.

    The restricted model has one fewer random dimension, so fitting it from
    scratch would generate a different Halton set and the two likelihoods would
    differ partly by simulation noise.  Dropping that term's draw column instead
    keeps every other draw identical, which makes the models exactly nested and
    the LR a genuine likelihood-ratio statistic for the simulated likelihood.
    Returns the restricted negative log-likelihood at its optimum.
    """
    labels = G._param_labels(spec["upper"], spec["lower"],
                             spec["rand_upper"], spec["rand_lower"],
                             spec["model"])
    col, sigma_label = _sigma_column(spec, tier, k)
    drop = labels.index(sigma_label)
    keep = [i for i in range(len(labels)) if i != drop]

    ru, rl = list(spec["rand_upper"]), list(spec["rand_lower"])
    (ru if tier == "upper" else rl)[k] = False

    draws = np.delete(np.asarray(fit["draws"]), col, axis=2)
    bounds = [b for i, b in enumerate(G._sigma_bounds(
        spec["upper"], spec["lower"], spec["rand_upper"],
        spec["rand_lower"], spec["model"])) if i != drop]

    obj = G.make_objective(fit["y"], fit["A"], fit["B"], fit["L"],
                           tuple(ru), tuple(rl), draws, R, spec["model"],
                           fit["offset"], None, integration, 7)
    p0 = np.asarray(fit["params"], dtype=float)[keep]
    res = G.run_optimizer_bounded(obj, p0, bounds, method="SLSQP", maxiter=2000)
    best = np.asarray(res.params, dtype=float)
    f = float(obj(G.jnp.array(best)))
    try:
        xp, fp = G._newton_polish(obj, best, bounds)
        if np.isfinite(fp) and fp < f:
            f = float(fp)
    except Exception:                                   # pragma: no cover
        pass
    return f


def sigma_boundary_lrt(data, spec, R=300, integration="halton", ll_full=None,
                       fit=None):
    """One-at-a-time LR test of each random term against sigma = 0.

    Refits the same model with one random term turned fixed (its sigma pinned at
    zero) and compares the simulated likelihoods.  ``fit`` carries the full
    model's ``y``/``A``/``B``/``L``/``draws``/``params``/``offset`` (as returned
    in ``refit_and_report(...).fit_parts``); the restricted model then uses the
    full model's draws with that term's column removed, so the comparison is
    exactly nested rather than partly simulation noise.  Without ``fit`` the
    full model is fitted here first.  Quadrature integration has no draw matrix
    to share, so it falls back to an independent restricted fit.

    Returns ``{(tier, term): {lr, p, ll_restricted}}``; the p-value uses the
    boundary mixture, see ``boundary_lrt_pvalue``.  This is the formal
    counterpart of the sigma/SE screen, which is only a heuristic because the
    Wald interval of a boundary parameter is not trustworthy near zero.
    """
    common = (integration == "halton" and fit is not None
              and fit.get("draws") is not None)
    if fit is None:
        full = refit_and_report(data, spec, R=R, integration=integration)
        ll_full = full.ll if ll_full is None else ll_full
        fit = full.fit_parts
        common = integration == "halton"

    if common and ll_full is None:
        obj = G.make_objective(fit["y"], fit["A"], fit["B"], fit["L"],
                               tuple(spec["rand_upper"]),
                               tuple(spec["rand_lower"]), fit["draws"], R,
                               spec["model"], fit["offset"], None,
                               integration, 7)
        ll_full = -float(obj(G.jnp.array(np.asarray(fit["params"], float))))

    out = {}
    for tier in ("upper", "lower"):
        flags_key = f"rand_{tier}"
        for k, (v, is_rand) in enumerate(zip(spec[tier], spec[flags_key])):
            if not is_rand:
                continue
            if common:
                neg_ll = _restricted_ll_common_draws(fit, spec, tier, k, R,
                                                     integration)
            else:
                restricted = dict(spec)
                restricted["rand_upper"] = list(spec["rand_upper"])
                restricted["rand_lower"] = list(spec["rand_lower"])
                restricted[f"rand_{tier}"] = [
                    False if i == k else f
                    for i, f in enumerate(restricted[f"rand_{tier}"])]
                neg_ll = -refit_and_report(data, restricted, R=R,
                                           integration=integration).ll
            lr = 2.0 * (float(ll_full) + float(neg_ll))
            out[(tier, v)] = {"lr": lr, "p": boundary_lrt_pvalue(lr),
                              "ll_restricted": -float(neg_ll),
                              "common_draws": bool(common)}
    return out


def refit_and_report(data, spec, R=300, integration="halton", lrt=False,
                     forbid_together=None):
    """Final fit of a winning specification with inference, identification
    check, CMF catalogue and AADT attribution.

    Pass ``forbid_together`` to refuse a specification that co-selects a pair
    the design cannot identify separately — the search never produces one, but a
    specification recorded before the constraint existed can still be refitted
    here, and that should fail loudly rather than fit.
    """
    bad = find_violation(spec, forbid_together) if forbid_together else None
    if bad:
        raise ValueError(
            f"specification selects {bad[0]} and {bad[1]} together; that pair "
            "is unidentifiable side by side (see derive_forbidden_pairs)")
    from .cmf_catalogue import build_cmf_catalogue
    ru, rl = spec["rand_upper"], spec["rand_lower"]
    ordered = [d for d, r in zip(spec["dist_upper"], ru) if r] + \
              [d for d, r in zip(spec["dist_lower"], rl) if r]
    res, y, A, B, L, draws, _, off = G.fit_final_model(
        data, spec["upper"], spec["lower"], ru, rl, model=spec["model"],
        R=R, integration=integration, dists=ordered, rho=spec["rho"])
    cov = G.observed_covariance(res, y, A, B, L, ru, rl, draws,
                                spec["model"], off, integration)
    # A fitted sigma can come back negative because the likelihood is even in
    # sigma: with draws u and -u the model is the same, though a *fixed* finite
    # draw set is not exactly symmetric, so the sign must NOT be flipped here —
    # that would move the parameters off the optimum of the very likelihood the
    # draws define.  The sign is simply not identified; the catalogue reports the
    # magnitude, and the reported fit stays exactly where the optimiser left it.
    labels = G._param_labels(spec["upper"], spec["lower"], ru, rl,
                             spec["model"])
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    ident = G.identification_report(res, y, A, B, L, spec["upper"],
                                    spec["lower"], ru, rl, draws,
                                    spec["model"], off, integration)
    cat = build_cmf_catalogue(res.params, se, data, spec["upper"],
                              spec["lower"], ru, rl, spec["model"],
                              dists=ordered)
    cat["lrt"] = np.nan
    cat["lrt_p"] = np.nan
    attr, elast = aadt_attribution(res.params, spec["upper"], spec["lower"],
                                   ru, rl, data, spec["model"])

    # Fit statistics on the SAME draws the refit used, so the searched model
    # and any comparison model can be reported on one scale.
    ll = float("nan")
    bic = float("nan")
    n_params = int(G.count_params(ru, rl, spec["model"]))
    try:
        obj = G.make_objective(y, A, B, L, ru, rl, draws, R, spec["model"],
                               off, None, integration, 7)
        ll = -float(obj(G.jnp.array(res.params)))
        if np.isfinite(ll):
            bic = float(-2.0 * ll + n_params * np.log(len(y)))
    except Exception as exc:  # pragma: no cover - diagnostics only
        print(f"  [refit] fit statistics unavailable: {exc}")

    lrt_tests = {}
    if lrt and (any(ru) or any(rl)) and np.isfinite(ll):
        lrt_tests = sigma_boundary_lrt(
            data, spec, R=R, integration=integration, ll_full=ll,
            fit={"y": y, "A": A, "B": B, "L": L, "draws": draws,
                 "params": np.asarray(res.params, dtype=float), "offset": off})
        for (tier, term), t in lrt_tests.items():
            m = (cat["tier"] == tier) & (cat["term"] == term)
            cat.loc[m, "lrt"] = t["lr"]
            cat.loc[m, "lrt_p"] = t["p"]

    return SimpleNamespace(result=res, cov=cov, se=se, identification=ident,
                           catalogue=cat, attribution=attr, ll=ll, bic=bic,
                           lrt=lrt_tests,
                           n_params=n_params, n_obs=int(len(y)),
                           labels=labels, mean_aadt_elasticity=elast,
                           fit_parts={"y": y, "A": A, "B": B, "L": L,
                                      "draws": draws,
                                      "params": np.asarray(res.params, dtype=float),
                                      "offset": off})
