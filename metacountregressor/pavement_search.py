"""pavement_search.py
========================================================================
Multi-objective structure search for the pavement deterioration programme.

Engines
-------
* ``engine="agds"``   SparseEA-AGDS via :func:`sparse_ea_agds` (default;
  adaptive downhill search, same family as the count-model pipeline).
* ``engine="nsga2"``  :class:`PavementNSGA2` subclassing the package's
  ``NSGA2Engine`` (non-dominated sort, crowding distance, hypervolume).

Genome
------
``[form_1..form_K | var_mask (K*p bits) | markov_on (K bits) | hazard_family]``

Objectives (minimised)
----------------------
1. composite BIC of the clusterwise pipeline (regression layer, optionally
   + Markov + hazard layers when ``enable_markov/enable_hazard``).
2. hold-out RMSE on **log(PSI)** over the 2011--2012 test panel
   (continuous in the genes; falls back to mean active parameter count
   without a ``test_df``).

Cluster membership is fixed input --- typically discovered by the
count-model search / ``MetacountRegressorBridge``.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

try:
    from .pavement_pipeline import Constraints, PavementDeteriorationEvaluator
    from .pavement_forms import N_FORMS, build_design
    from .pavement_hazard import FAMILIES
except ImportError:  # flat import (script inside package dir)
    from pavement_pipeline import Constraints, PavementDeteriorationEvaluator
    from pavement_forms import N_FORMS, build_design
    from pavement_hazard import FAMILIES

try:
    from .Solvers_METAJAX import NSGA2Engine
except ImportError:
    from Solvers_METAJAX import NSGA2Engine

try:
    from .metaheuristics import sparse_ea_agds
except ImportError:
    from metaheuristics import sparse_ea_agds

__all__ = [
    "PavementMultiObjectiveEvaluator",
    "PavementNSGA2",
    "PavementSparseAGDSObjective",
    "run_pavement_multiobjective_search",
    "run_pavement_sparse_agds",
]

_BIG = 1e12


def _genome_high(mo):
    # GP form (11) needs an expression tree the GA does not carry -> exclude
    hi = np.empty(mo.dimension, dtype=int)
    hi[mo.form_slice] = N_FORMS - 2
    hi[mo.mask_slice] = 1
    hi[mo.markov_slice] = 1
    hi[mo.haz_idx] = len(mo.hazard_families) - 1
    return hi


# ══════════════════════════════════════════════════════════════════════
class PavementMultiObjectiveEvaluator:
    """Adapt the pavement pipeline to a two-objective vector fitness."""

    def __init__(self, df, id_col, psi_col, cont_vars, cat_vars, membership,
                 constraints: Optional[Constraints] = None, test_df=None,
                 time_col: str = "age", treatment_col: Optional[str] = None,
                 hazard_default: str = "weibull",
                 enable_markov: bool = False, enable_hazard: bool = False):
        self.enable_markov = bool(enable_markov)
        self.enable_hazard = bool(enable_hazard)
        self.base = PavementDeteriorationEvaluator(
            df, id_col, psi_col, cont_vars, cat_vars,
            constraints or Constraints(),
            time_col=time_col, treatment_col=treatment_col)
        self.test_df = test_df
        self.membership = np.asarray(membership, int)
        self.K = int(self.membership.max()) + 1
        self.p = len(cont_vars)
        self.hazard_families = list(FAMILIES)
        if hazard_default in self.hazard_families:
            self.hazard_families.remove(hazard_default)
            self.hazard_families.insert(0, hazard_default)

        self.form_slice = slice(0, self.K)
        m0 = self.K
        self.mask_slice = slice(m0, m0 + self.K * self.p)
        mk0 = m0 + self.K * self.p
        self.markov_slice = slice(mk0, mk0 + self.K)
        self.haz_idx = mk0 + self.K
        self.dimension = self.haz_idx + 1

        # NSGA2Engine compatibility aliases
        self.vars = list(range(self.dimension))
        self.max_latent_classes = 1

        self._cache: Dict[tuple, np.ndarray] = {}
        self.last_eval_error = None
        enable_mk, enable_hz = self.enable_markov, self.enable_hazard
        self._aux: Dict[tuple, dict] = {}

        self._train_frames = {}
        self._cat_levels = {}
        for k in range(self.K):
            tr_ids = self.base.seg_ids[self.membership == k]
            tr = self.base.df[self.base.df[self.base.id_col].isin(tr_ids)]
            self._train_frames[k] = tr
            levels = {}
            for c in self.base.cat_vars:
                if c in tr.columns:
                    levels[c] = sorted(tr[c].astype(str).unique())
            self._cat_levels[k] = levels

    # ------------------------------------------------------------------
    def decode(self, vec) -> Dict[str, Any]:
        v = np.asarray(vec, int)
        return {
            "K": self.K,
            "membership": self.membership,
            "var_mask": v[self.mask_slice].reshape(self.K, self.p).astype(bool),
            "form": v[self.form_slice] % N_FORMS,
            "lam": np.ones((self.K, self.p)),
            "shape": [None] * self.K,
            "R": 4,
            "theta": [[1.5, 3.0, 4.0]] * self.K,
            "markov_on": (v[self.markov_slice].astype(bool).tolist()
                          if self.enable_markov else [False] * self.K),
            "hazard_on": ([True] * self.K if self.enable_hazard
                          else [False] * self.K),
            "hazard_family":
                self.hazard_families[int(v[self.haz_idx])
                                     % len(self.hazard_families)],
            "gp_tree": None,
        }

    def fitness(self, vec):
        key = tuple(np.asarray(vec, int).tolist())
        if key in self._cache:
            return self._cache[key]
        st = self.decode(key)
        try:
            res = self.base.evaluate(st)
        except Exception as exc:
            self.last_eval_error = f"{type(exc).__name__}: {exc}"
            res = None
        # Objective-1 is the REGRESSION-LAYER BIC only: unlike the composite
        # criterion it stays on one scale no matter which optional layers
        # (Markov / hazard) a structure enables, so frontier points remain
        # comparable with single-layer tables elsewhere in the paper.
        invalid = (res is None or not np.isfinite(res.bic_reg)
                   or abs(res.bic_reg) >= _BIG)
        if invalid:
            obj = np.array([_BIG, _BIG])
        else:
            self._aux[key] = {
                "bic_regression": float(res.bic_reg),
                "bic_markov": (float(res.bic_mk)
                               if self.enable_markov else None),
                "bic_hazard": (float(res.bic_haz)
                               if self.enable_hazard else None),
                "n_params": int(sum(f.get("n_params", 0)
                                    for f in res.fits.values())),
            }
            obj1 = float(res.bic_reg)
            if self.test_df is not None:
                obj2 = self._holdout_logrmse(st, res)
                obj2 = obj2 if np.isfinite(obj2) else _BIG
            else:
                n_par = [f.get("n_params", 0) for f in res.fits.values()]
                obj2 = float(np.mean(n_par)) if n_par else _BIG
            obj = np.array([obj1, float(obj2)])
        self._cache[key] = obj
        return obj

    def _designs(self, st, fit, k, d_te):
        """Train/hold-out numeric designs (+ drop-first dummy blocks) that
        match widths by construction."""
        mask = st["var_mask"][k]
        names_a = [self.base.cont_vars[j] for j in range(self.p) if mask[j]]
        lam_full = np.asarray(st["lam"][k], float)
        form_id = int(st["form"][k])
        lam_a = (lam_full[mask] if form_id == 6 else np.ones(int(mask.sum())))
        shape = st["shape"][k]
        Xc_tr = self._train_frames[k][self.base.cont_vars].to_numpy(float)
        num_tr = build_design(Xc_tr[:, mask], names_a, form_id, lam_a,
                              shape=shape)
        d_tr = self._train_frames[k]
        blocks_tr, blocks_te = [], []
        Xc_te = d_te[self.base.cont_vars].to_numpy(float)
        num_te = build_design(Xc_te[:, mask], names_a, form_id, lam_a,
                              shape=shape)
        for c in self.base.cat_vars:
            levels = self._cat_levels[k].get(c) or []
            if c not in d_tr.columns or len(levels) < 2:
                continue
            b_tr = pd.get_dummies(pd.Categorical(
                d_tr[c].astype(str), categories=levels)).to_numpy(float)
            b_te = pd.get_dummies(pd.Categorical(
                d_te[c].astype(str), categories=levels)).to_numpy(float)
            blocks_tr.append(b_tr[:, 1:])
            blocks_te.append(b_te[:, 1:])
        X_tr = np.hstack([num_tr] + blocks_tr)
        X_te = np.hstack([num_te] + blocks_te)
        y_tr = d_tr[self.base.psi_col].to_numpy(float)
        return X_tr, X_te, y_tr

    def front_records(self, vecs, objs):
        """Decode a front back into human-readable rows (with the layer
        BIC decomposition captured during evaluation)."""
        rows = []
        for v, o in zip(vecs, objs):
            key = tuple(np.asarray(v, int).tolist())
            st = self.decode(key)
            a = self._aux.get(key, {})
            mask = np.asarray(st["var_mask"], bool)
            act = [[self.base.cont_vars[j]
                    for j in range(self.p) if mask[k][j]]
                   for k in range(self.K)]
            rows.append({
                "genome": "|".join(map(str, key)),
                "bic_regression": float(o[0]),
                "log_holdout_rmse": float(o[1]),
                "bic_markov": a.get("bic_markov"),
                "bic_hazard": a.get("bic_hazard"),
                "forms": [int(f) for f in st["form"]],
                "active_vars": act,
                "markov_on": list(st["markov_on"]),
                "hazard_family": st["hazard_family"],
            })
        return rows

    def _holdout_logrmse(self, st, res) -> float:
        """RMSE on log(PSI) over the hold-out panel, refitting OLS on the
        training panel with this module's own design construction."""
        import os as _os
        dbg = _os.environ.get("PAVEMENT_SEARCH_DEBUG")
        sub = self.test_df[self.test_df[self.base.id_col].isin(
            self.base.seg_ids)]
        sq_err, n_all = 0.0, 0
        for k in range(self.K):
            tr_ids = self.base.seg_ids[self.membership == k]
            d_te = sub[np.isin(sub[self.base.id_col].to_numpy(), tr_ids)]
            if len(d_te) == 0:
                continue
            try:
                X_tr, X_te, y_tr = self._designs(st, res.fits[k], k, d_te)
                beta, *_ = np.linalg.lstsq(X_tr, y_tr, rcond=None)
                pred = np.clip(X_te @ beta, -30.0, 30.0)
                y_log = np.log(np.clip(d_te[self.base.psi_col]
                                       .to_numpy(float), 1e-3, None))
                sq_err += float(np.sum((pred - y_log) ** 2))
                n_all += len(y_log)
                if dbg:
                    print(f"    [holdout] K{k} X={X_tr.shape}->{X_te.shape} "
                          f"pred=[{pred.min():.2f},{pred.max():.2f}]")
            except Exception as exc:
                if dbg:
                    print(f"    [holdout] K{k} FAILED: {exc!r}")
                return float("nan")
        if n_all == 0:
            if dbg:
                print("    [holdout] no hold-out rows matched")
            return float("nan")
        return float(np.sqrt(sq_err / n_all))


# ══════════════════════════════════════════════════════════════════════
class _PavementOperator:
    """Uniform crossover + point mutation over the integer genome."""

    def __init__(self, evaluator, mut_rate=0.35):
        self.evaluator = evaluator
        self.mut_rate = mut_rate

    def generate(self, pop, i, gen, max_iter):
        hi = _genome_high(self.evaluator)
        a = pop[np.random.randint(len(pop))]
        b = pop[np.random.randint(len(pop))]
        child = np.where(np.random.rand(len(hi)) < 0.5, a, b)
        mut = np.random.rand(len(hi)) < self.mut_rate
        child[mut] = np.random.randint(0, hi[mut] + 1)
        return child


class PavementNSGA2(NSGA2Engine):
    """NSGA-II specialised for the pavement genome."""

    def _initialise_start_pop(self):
        hi = _genome_high(self.evaluator)
        pop = [np.array([np.random.randint(0, h + 1) for h in hi], dtype=int)
               for _ in range(self.pop_size)]
        return np.array(pop, dtype=int)

    def repair(self, solution):
        e = self.evaluator
        v = np.clip(np.asarray(solution, int), 0, _genome_high(e))
        mask = v[e.mask_slice].reshape(e.K, e.p)
        for k in range(e.K):
            if not mask[k].any():
                mask[k, np.random.randint(e.p)] = 1
        v[e.mask_slice] = mask.reshape(-1)
        return v


class PavementSparseAGDSObjective:
    """Adapter exposing the pavement genome to SparseEA_AGDS."""

    is_multi = True
    algorithm = None
    instance_name = "pavement_agds"

    def __init__(self, mo_evaluator, max_time_s: Optional[float] = None,
                 seed: int = 0):
        self.mo = mo_evaluator
        self._max_time = max_time_s
        self._seed = seed
        self._obj_1 = "composite_bic"
        self._obj_2 = ("log_holdout_rmse"
                       if mo_evaluator.test_df is not None
                       else "mean_n_params")
        self._discrete_values = [list(range(h + 1))
                                 for h in _genome_high(mo_evaluator)]
        self.Last_Sol = None

    def get_num_parameters(self):
        return self.mo.dimension

    def get_value(self, i):
        return int(np.random.choice(self._discrete_values[int(i)]))

    def is_discrete(self, i):
        return True

    def use_random_seed(self):
        return True

    def set_random_seed(self):
        np.random.seed(self._seed)

    def get_max_time(self):
        return self._max_time if self._max_time else float("inf")

    def get_fitness(self, layout, multi=False, max_routine=2):
        v = np.clip(np.asarray(layout, float), 0, None).astype(int)
        v = np.minimum(v, _genome_high(self.mo))
        obj = self.mo.fitness(v)
        self.Last_Sol = list(v)
        if multi:
            return float(obj[0]), float(obj[1])
        return float(obj[0])

    def _get_obj1(self):
        return self._obj_1

    def _get_obj2(self):
        return self._obj_2

    def modify_initial_fit(self, model_nature):  # pragma: no cover
        raise RuntimeError("not supported - skipped by the engine")

    def reconstruct_vector(self, a):  # pragma: no cover
        return list(a)


def _nondominated_from_cache(mo):
    vecs = np.array([np.asarray(k, int) for k in mo._cache])
    objs = np.array([mo._cache[k] for k in mo._cache], dtype=float)
    keep = []
    for i in range(len(objs)):
        dominated = np.any(
            np.all(objs <= objs[i], axis=1) & np.any(objs < objs[i], axis=1))
        if not dominated:
            keep.append(i)
    return vecs[keep], objs[keep]


def run_pavement_sparse_agds(
    df, id_col, psi_col, cont_vars, cat_vars, membership,
    test_df=None, constraints=None,
    generations: int = 20, seed: int = 0, max_time_s: Optional[float] = None,
    time_col: str = "age", treatment_col: Optional[str] = None,
    enable_markov: bool = False, enable_hazard: bool = False,
):
    mo = PavementMultiObjectiveEvaluator(
        df, id_col, psi_col, cont_vars, cat_vars, membership,
        constraints=constraints, test_df=test_df,
        time_col=time_col, treatment_col=treatment_col,
        enable_markov=enable_markov, enable_hazard=enable_hazard)
    objective = PavementSparseAGDSObjective(
        mo, max_time_s=max_time_s, seed=seed)
    sparse_ea_agds(
        objective,
        MAX_ITERATIONS=int(generations),
        _max_iter=int(generations),
        _max_time=max_time_s,
        verbose=False,
    )
    vecs, objs = _nondominated_from_cache(mo)
    structures = [mo.decode(v) for v in vecs]
    return {"front": structures, "objectives": objs,
            "all_objectives": objs, "engine": "sparseea-agds",
            "evaluator": mo, "aux": mo._aux,
            "front_records": mo.front_records(vecs, objs)}


def run_pavement_multiobjective_search(
    df, id_col, psi_col, cont_vars, cat_vars, membership,
    test_df=None, constraints=None,
    pop_size: int = 24, generations: int = 20, seed: int = 0,
    time_col: str = "age", treatment_col: Optional[str] = None,
    engine: str = "agds", max_time_s: Optional[float] = None,
    enable_markov: bool = False, enable_hazard: bool = False,
):
    """Multi-objective pavement structure search (default SparseEA-AGDS).

    Returns dict: Pareto ``front`` structures, ``objectives`` (F,2)
    [composite BIC, log-scale hold-out RMSE], ``engine``, ``evaluator``.
    """
    np.random.seed(seed)
    if str(engine).lower() in {"agds", "sparseea", "sparseea-agds"}:
        return run_pavement_sparse_agds(
            df, id_col, psi_col, cont_vars, cat_vars, membership,
            test_df=test_df, constraints=constraints,
            generations=generations, seed=seed, max_time_s=max_time_s,
            time_col=time_col, treatment_col=treatment_col,
            enable_markov=enable_markov, enable_hazard=enable_hazard)

    mo = PavementMultiObjectiveEvaluator(
        df, id_col, psi_col, cont_vars, cat_vars, membership,
        constraints=constraints, test_df=test_df,
        time_col=time_col, treatment_col=treatment_col,
        enable_markov=enable_markov, enable_hazard=enable_hazard)
    op = _PavementOperator(mo)
    # NSGA2Engine interprets max_iter as a TOTAL evaluation budget:
    #   generations = max_iter // pop_size
    engine_obj = PavementNSGA2(evaluator=mo, operator=op,
                               dimension=mo.dimension,
                               pop_size=pop_size,
                               max_iter=max(1, pop_size * generations))
    pop, scores = engine_obj.optimize()
    scores = np.atleast_2d(scores)

    keep = [i for i in range(len(scores))
            if not np.any(np.all(scores <= scores[i], axis=1)
                          & np.any(scores < scores[i], axis=1))]
    fvecs = np.array([pop[i] for i in keep], dtype=int)
    structures = [mo.decode(v) for v in fvecs]
    return {"front": structures, "objectives": scores[keep],
            "all_objectives": scores, "engine": "nsga2",
            "evaluator": mo, "aux": mo._aux,
            "front_records": mo.front_records(fvecs, scores[keep])}
