"""
zeke_pavement_pipeline.py
========================================================================
Pipeline orchestrator + metacountregressor-compatible plugin surface.

This module wires the new estimation/prediction modules into the *same*
search contract that ``zeke_pavement_search.py`` already uses with
``metacountregressor``:

    ExperimentBuilder(df, id_col, y_col)     -> builder
    ModelConstraints()                       -> constraints (allow/no_random)
    builder.build_*_evaluator(constraints)   -> evaluator
    evaluator.evaluate(structure) -> (score, details)
    evaluator.cache                         -> resume/persist hook

Here we provide an analoguesurface for the *continuous* pavement program
(cluster form + Markov + hazard) so that:

  (a) the SA loop in ``zeke_pavement_search.py`` can swap
      ``PavementDeteriorationEvaluator`` in for the MCR evaluator without
      changing the search logic, and
  (b) ``MetacountRegressorBridge`` shows how MCR's fitted latent-class
      structure can *seed* this evaluator (the "pick up and embed" path).

Decision vector (mirrors MCR's gene)
------------------------------------
structure = {
  "K"             : int number of clusters,
  "membership"    : (n_seg,) int cluster id per segment,
  "var_mask"      : (K, p) bool which continuous vars active per cluster,
  "form"          : (K,) int functional-form id (0..11),
  "lam"           : (K, p) float estimable power params,
  "shape"         : (K,) dict of age-shape params,
  "R"             : int number of Markov condition states,
  "theta"         : (K, R-1) state thresholds,
  "markov_on"     : (K,) bool transition modelling active,
  "hazard_on"     : (K,) bool hazard modelling active,
  "hazard_family" : str,
  "temporal_models": (K,) str per-cluster error structure (e_k),
  "gp_tree"       : (K,) optional GP expression tree (form 11 only),
}
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field

from .pavement_forms import N_FORMS, fit_form
from .pavement_markov import (discretize, fit_transitions, forward_propagate,
                                markov_bic, composite_bic)
from .pavement_hazard import fit_hazard, FAMILIES

# Per-cluster error structures selectable as a decision layer. "ols" keeps
# the classical level-scale score; the others are scored on the conditional
# sample (t >= 2, identical N, log-PSI levels) so they are like-for-like.
TEMPORAL_MODELS = ("ols", "ar1", "ar2", "random_walk", "nur")


# ---------------------------------------------------------------------------
# Constraints (mirrors metacountregressor.ModelConstraints)
# ---------------------------------------------------------------------------
class Constraints:
    """Which decisions are allowed. Drop-in analogue of MCR's ModelConstraints."""
    def __init__(self):
        self.allowed_forms = list(range(N_FORMS))
        self.allow_lambda = True
        self.allow_markov = True
        self.allow_hazard = True
        self.allow_temporal = True
        self.allowed_temporal = list(TEMPORAL_MODELS)
        self.allowed_hazard_families = list(FAMILIES)
        self.max_clusters = 4
        self.min_cluster_size = 50

    def allow_form(self, fid):
        if fid not in self.allowed_forms:
            self.allowed_forms.append(fid)
        return self

    def no_form(self, fid):
        self.allowed_forms = [f for f in self.allowed_forms if f != fid]
        return self

    def __repr__(self):
        return (f"Constraints(forms={self.allowed_forms}, "
                f"lambda={self.allow_lambda}, markov={self.allow_markov}, "
                f"hazard={self.allow_hazard})")


# ---------------------------------------------------------------------------
# Evaluator (mirrors MCR's evaluator with .evaluate() + .cache)
# ---------------------------------------------------------------------------
@dataclass
class EvalResult:
    bic_tot: float
    bic_reg: float
    bic_mk: float
    bic_haz: float
    fits: dict = field(default_factory=dict)
    transitions: dict = field(default_factory=dict)
    hazards: dict = field(default_factory=dict)
    temporal: dict = field(default_factory=dict)


class PavementDeteriorationEvaluator:
    """Evaluate one full decision structure -> composite BIC_tot."""
    def __init__(self, df, id_col, psi_col, cont_vars, cat_vars,
                 constraints: Constraints, time_col="age",
                 treatment_col=None):
        self.df = df
        self.id_col = id_col
        self.psi_col = psi_col
        self.cont_vars = cont_vars
        self.cat_vars = cat_vars
        self.constraints = constraints
        self.time_col = time_col
        self.treatment_col = treatment_col
        self.cache = {}   # resume/persist hook (as in the MCR evaluator)
        # Segment-id order that the `membership` decision vector is aligned to.
        self.seg_ids = np.sort(df[id_col].unique())

    # -- helpers ----------------------------------------------------------
    def _cluster_data(self, cid, membership):
        # membership is aligned to self.seg_ids (one entry per segment)
        seg_ids = self.seg_ids[np.asarray(membership) == cid]
        return self.df[self.df[self.id_col].isin(seg_ids)].copy()

    def _cat_dummies(self, df):
        cols = []
        for c in self.cat_vars:
            if c in df.columns:
                d = pd_get_dummies(df[c])
                cols.append(d)
        if not cols:
            return np.zeros((len(df), 0))
        return np.hstack(cols)

    def evaluate(self, structure) -> EvalResult:
        key = _hash_structure(structure)
        if key in self.cache:
            return self.cache[key]

        K = int(structure["K"])
        membership = np.asarray(structure["membership"])
        cont = self.df[self.cont_vars].to_numpy(float)
        p = len(self.cont_vars)
        bic_reg = bic_mk = bic_haz = 0.0
        fits, transitions, hazards, temporal = {}, {}, {}, {}
        temporal_codes = structure.get("temporal_models")
        total_n = 0

        for k in range(K):
            df_k = self._cluster_data(k, membership)
            if len(df_k) < self.constraints.min_cluster_size:
                res = EvalResult(np.inf, np.inf, np.inf, np.inf)
                self.cache[key] = res
                return res
            y = df_k[self.psi_col].to_numpy(float)
            Xc = df_k[self.cont_vars].to_numpy(float)
            mask = np.asarray(structure["var_mask"][k], bool)
            form = int(structure["form"][k])
            lam = np.asarray(structure["lam"][k], float) if "lam" in structure else np.ones(p)
            shape = structure.get("shape", [None] * K)[k]
            cat = self._cat_dummies(df_k)
            gp_call = None
            if form == 11 and structure.get("gp_tree") is not None:
                from .pavement_gp import tree_to_lambda
                gp_call = tree_to_lambda(structure["gp_tree"][k], p)

            # (1) regression / functional form
            fit = fit_form(Xc, y, self.cont_vars, form, mask,
                           cat_dummies=cat, lam0=lam, shape0=shape,
                           gp_callable=gp_call)
            n_k = len(y)
            total_n += n_k
            if temporal_codes is None:
                k_reg = n_k * np.log(fit["rss"] / n_k) \
                    + fit["n_params"] * np.log(n_k)
            else:
                model = str(temporal_codes[k])
                t_out = self._conditional_regression(df_k, y, fit, model)
                if t_out is None:
                    res = EvalResult(np.inf, np.inf, np.inf, np.inf)
                    self.cache[key] = res
                    return res
                rss_c, n_c, k_c, t_fit = t_out
                k_reg = n_c * np.log(rss_c / n_c) + k_c * np.log(n_c)
                temporal[k] = {"model": model, "rss_cond": rss_c,
                               "n_cond": n_c, "n_params": k_c, "fit": t_fit}
            bic_reg += k_reg
            fits[k] = fit

            # (2) Markov layer (if active)
            if bool(structure.get("markov_on", [False] * K)[k]):
                R = int(structure["R"])
                theta = structure["theta"][k]
                states = discretize(df_k[self.psi_col].to_numpy(float), theta)
                Xd = fit["design"]
                tr = fit_transitions(states, Xd, n_states=R,
                                     monotone=True,
                                     has_treatment=self.treatment_col is not None,
                                     treatment=(df_k[self.treatment_col].to_numpy(float)
                                                if self.treatment_col else None))
                bic_mk += markov_bic(tr["loglik"], tr["n_params"], n_k)
                transitions[k] = tr

            # (3) Hazard / survival layer (if active)
            if bool(structure.get("hazard_on", [False] * K)[k]):
                t = df_k[self.time_col].to_numpy(float)
                # event = first interval where a treatment is recorded, else censor
                if self.treatment_col:
                    event = df_k[self.treatment_col].to_numpy(float) > 0
                else:
                    event = np.zeros(len(df_k), int)
                hz = fit_hazard(Xc, self.cont_vars, form,
                                np.asarray(structure["lam"][k]),
                                t, event, family=structure.get("hazard_family", "weibull"),
                                cat_dummies=cat, active_mask=mask)
                bic_haz += hz["bic"]
                hazards[k] = hz

        bic_tot = composite_bic(bic_reg, bic_mk, bic_haz)
        res = EvalResult(bic_tot, bic_reg, bic_mk, bic_haz,
                         fits=fits, transitions=transitions, hazards=hazards,
                         temporal=temporal)
        self.cache[key] = res
        return res

    def _conditional_regression(self, df_k, y, form_fit, model):
        """Like-for-like conditional-sample regression score (t >= 2).

        The functional-form design is carried into the temporal fitters as
        plain columns, so every error structure is estimated on the same
        design and scored on log-PSI levels with the identical conditional
        sample. ``ols`` keeps the form fit itself (same design, same beta),
        just re-scored on the conditional sample so the BIC scale matches
        the temporal structures.
        """
        import pandas as pd

        from .pavement_clr import (fit_cluster_ar1, fit_cluster_ar2,
                                   fit_cluster_random_walk, fit_cluster_nur,
                                   _conditional_rss)
        models = {
            "ar1": fit_cluster_ar1,
            "ar2": fit_cluster_ar2,
            "random_walk": fit_cluster_random_walk,
            "nur": fit_cluster_nur,
        }
        model = str(model)
        if model != "ols" and model not in models:
            return None
        Xd = np.asarray(form_fit.get("design"), dtype=float)
        if Xd.ndim != 2 or Xd.shape[0] != len(df_k):
            return None

        df_k = df_k.reset_index(drop=True)
        order = np.lexsort((df_k[self.time_col].to_numpy(float),
                            df_k[self.id_col].to_numpy()))
        df_s = df_k.iloc[order].reset_index(drop=True)
        cols = [f"d{i}" for i in range(Xd.shape[1])]
        df_t = pd.DataFrame(Xd[order], columns=cols)
        df_t.insert(0, self.id_col, df_s[self.id_col].to_numpy())
        df_t.insert(1, self.psi_col, np.asarray(y, float)[order])

        if model == "ols":
            beta = np.asarray(form_fit.get("beta"), dtype=float)
            X_int = np.hstack([np.ones((len(df_t), 1)), Xd[order]])
            if X_int.shape[1] != len(beta):
                return None
            resid = df_t[self.psi_col].to_numpy(float) - X_int @ beta
            t_fit = {"residuals": resid,
                     "coefficients": beta,
                     "n_params_total": int(form_fit.get("n_params",
                                                        X_int.shape[1]))}
        else:
            t_fit = models[model](df_t, self.psi_col, cols, set(),
                                  self.id_col)
            if t_fit is None:
                return None

        rss_c, n_c = _conditional_rss(t_fit, model, df_t, self.id_col)
        if not np.isfinite(rss_c) or n_c <= 0:
            return None
        k_c = int(t_fit.get("n_params_total",
                            len(t_fit.get("coefficients", []))))
        if n_c <= k_c:
            return None
        return float(rss_c), int(n_c), k_c, t_fit


# ---------------------------------------------------------------------------
# metacountregressor bridge -- the "pick up and embed" path
# ---------------------------------------------------------------------------
class MetacountRegressorBridge:
    """Adapt an MCR-fitted latent-class structure into our decision vector.

    metacountregressor fits a *count* outcome (round(psi*10)) with Poisson/NB
    and discovers K latent classes + per-class variable roles. We reuse its
    CLASS ASSIGNMENTS and per-class ACTIVE VARIABLES as a warm start for the
    continuous pavement program, then let the SA search refine forms / lambda /
    Markov / hazard on top. This keeps MCR as the cluster discoverer and adds
    the pavement-specific layers without forking MCR internals.
    """
    @staticmethod
    def from_mcr(mcr_class_assignment, mcr_active_vars_per_class,
                 cont_vars, defaults=None):
        """Build a starting ``structure`` dict from MCR outputs.

        mcr_class_assignment : (n_seg,) int cluster id (0-based)
        mcr_active_vars_per_class : list (K,) of list of active var names
        """
        K = int(np.max(mcr_class_assignment)) + 1
        p = len(cont_vars)
        var_mask = np.zeros((K, p), bool)
        for k in range(K):
            for v in mcr_active_vars_per_class[k]:
                if v in cont_vars:
                    var_mask[k, cont_vars.index(v)] = True
        structure = {
            "K": K,
            "membership": np.asarray(mcr_class_assignment, int),
            "var_mask": var_mask,
            "form": np.zeros(K, int),             # start at power (form 0)
            "lam": np.ones((K, p)),
            "shape": [None] * K,
            "R": 4,
            "theta": [[1.5, 3.0, 4.0]] * K,
            "markov_on": [True] * K,
            "hazard_on": [True] * K,
            "hazard_family": "weibull",
            "temporal_models": ["ols"] * K,
            "gp_tree": None,
        }
        if defaults:
            structure.update(defaults)
        return structure


# ---------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------
def _hash_structure(structure):
    import json, hashlib
    def default(o):
        if isinstance(o, (np.ndarray,)):
            return o.tolist()
        if isinstance(o, dict):
            return {k: default(v) for k, v in o.items()}
        return str(o)
    s = json.dumps(structure, default=default, sort_keys=True)
    return hashlib.md5(s.encode()).hexdigest()[:16]


def _pd_get_dummies(series):
    import pandas as pd
    return pd.get_dummies(series.astype(str), drop_first=True).to_numpy(float)


# alias used inside evaluator
def pd_get_dummies(series):
    return _pd_get_dummies(series)


def run_pipeline(df, id_col, psi_col, cont_vars, cat_vars, structure,
                 constraints=None, time_col="age", treatment_col=None):
    """One-shot evaluate of a structure (used by tests / the SA loop)."""
    constraints = constraints or Constraints()
    ev = PavementDeteriorationEvaluator(
        df, id_col, psi_col, cont_vars, cat_vars, constraints,
        time_col=time_col, treatment_col=treatment_col)
    return ev.evaluate(structure)


if __name__ == "__main__":
    import pandas as pd
    rng = np.random.default_rng(4)
    n_seg = 120
    rows = []
    for s in range(n_seg):
        k = 0 if s < 60 else 1
        for t in range(rng.integers(3, 8)):
            rows.append({
                "sample_id": s, "psi": max(0.5, 5 - 0.2 * t - 0.05 * k * t + rng.normal(0, 0.2)),
                "age": float(t + 1), "aadt": rng.uniform(200, 4000),
                "sys_id": rng.choice(["A", "B"]),
            })
    df = pd.DataFrame(rows)
    cont = ["age", "aadt"]; cat = ["sys_id"]
    seg_ids = np.sort(df["sample_id"].unique())
    # per-segment membership (aligned to sorted unique sample_id)
    membership = (seg_ids < 60).astype(int)
    structure = {
        "K": 2, "membership": membership,
        "var_mask": np.array([[True, True], [True, True]]),
        "form": np.array([0, 0]), "lam": np.ones((2, 2)), "shape": [None, None],
        "R": 4, "theta": [[1.5, 3.0, 4.0], [1.5, 3.0, 4.0]],
        "markov_on": [True, True], "hazard_on": [True, True],
        "hazard_family": "weibull", "gp_tree": None,
    }
    res = run_pipeline(df, "sample_id", "psi", cont, cat, structure,
                       treatment_col=None)
    print(f"pipeline: bic_tot={res.bic_tot:.3f} "
          f"(reg={res.bic_reg:.3f}, mk={res.bic_mk:.3f}, haz={res.bic_haz:.3f})")
    print("OK: zeke_pavement_pipeline self-test passed")
