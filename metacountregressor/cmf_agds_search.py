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

from . import GA_CMF_AADT_JAX as G
from .metaheuristics import SparseEA_AGDS

DISTS = ("normal", "uniform", "triangular", "lognormal")
RHOS = (0.0, 0.3, 0.6)
MODELS = ("poisson", "nb")


class CMFAGDSObjective:
    """Adapter exposing the interface SparseEA_AGDS expects."""

    is_multi = False
    maximize = False

    def __init__(self, data, variables, max_terms=8, min_lower=0, R=100, n_hops=1,
                 allow_models=MODELS, allow_dists=DISTS, allow_rho=RHOS,
                 integration="halton", seed=0, instance_name="run_agds_cmf",
                 max_time=1e9):
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


def refit_and_report(data, spec, R=300, integration="halton"):
    """Final fit of a winning specification with inference, identification
    check, CMF catalogue and AADT attribution."""
    from .cmf_catalogue import build_cmf_catalogue
    ru, rl = spec["rand_upper"], spec["rand_lower"]
    ordered = [d for d, r in zip(spec["dist_upper"], ru) if r] + \
              [d for d, r in zip(spec["dist_lower"], rl) if r]
    res, y, A, B, L, draws, _, off = G.fit_final_model(
        data, spec["upper"], spec["lower"], ru, rl, model=spec["model"],
        R=R, integration=integration, dists=ordered, rho=spec["rho"])
    cov = G.observed_covariance(res, y, A, B, L, ru, rl, draws,
                                spec["model"], off, integration)
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    ident = G.identification_report(res, y, A, B, L, spec["upper"],
                                    spec["lower"], ru, rl, draws,
                                    spec["model"], off, integration)
    cat = build_cmf_catalogue(res.params, se, data, spec["upper"],
                              spec["lower"], ru, rl, spec["model"])
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

    return SimpleNamespace(result=res, cov=cov, se=se, identification=ident,
                           catalogue=cat, attribution=attr, ll=ll, bic=bic,
                           n_params=n_params, n_obs=int(len(y)),
                           mean_aadt_elasticity=elast)
