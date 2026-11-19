"""Structure questions for attaching random parameters to the two-tier CMF.

The hierarchical model is  mu = exp(alpha0 + Z_u alpha) * AADT^(beta0 + Z_l beta).
A random parameter can sit on the upper tier (varies the crash level) or on the
lower tier (varies the AADT elasticity).  For every selected term this module
answers three questions with likelihood-ratio tests against the all-fixed model:

  Q1. Does the term need to be random at all?
  Q2. If so, does the heterogeneity attach to the upper or the lower tier?
  Q3. Is the random effect identified (sigma away from the boundary)?

A variance component sits on the boundary of its parameter space under H0, so
p-values use the 50:50 chi-bar-square mixture 0.5 * P(chi2_1 > LR).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import jax.numpy as jnp
from scipy.stats import chi2

from .GA_CMF_AADT_JAX import (
    _aghq_loglik, _ghq_draws, fit_final_model, generate_draws, make_objective,
)


def _loglik(data, sel_b, sel_l, rand_b, rand_l, model, R, integration,
            ghq_points, seed):
    res, y, a, b, l, draws, _std, off = fit_final_model(
        data, sel_b, sel_l, rand_b, rand_l, model=model, use_halton=True,
        R=R, integration=integration, ghq_points=ghq_points, seed=seed)
    params = jnp.array(res.params)
    n_rand = sum(rand_b) + sum(rand_l)
    if integration == 'aghq' and n_rand > 0:
        nll = _aghq_loglik(params, y, a, b, l, tuple(rand_b), tuple(rand_l),
                           model, off, ghq_points)
    else:
        lw = None
        if integration == 'ghq' and n_rand > 0:
            draws, lw = _ghq_draws(len(y), n_rand, ghq_points)
        elif n_rand > 0:
            draws = generate_draws(len(y), R, n_rand, True, seed=seed)
        obj = make_objective(y, a, b, l, tuple(rand_b), tuple(rand_l), draws,
                             R, model, off, lw, integration, ghq_points)
        nll = obj(params)
    return -float(nll), len(np.asarray(res.params)), len(y)


def structure_diagnostics(data, selected_baseline, selected_local,
                          model='poisson', R=200, integration='halton',
                          ghq_points=7, seed=42, alpha=0.05):
    """Test where each selected term should carry a random parameter.

    Returns a DataFrame with one row per (term, tier) that has the LR statistic,
    boundary-corrected p-value and BIC change versus the all-fixed model, plus
    a ``recommendation`` column ('fixed', 'upper' or 'lower') per term.
    """
    sel_b, sel_l = list(selected_baseline), list(selected_local)
    fixed_b, fixed_l = [False] * len(sel_b), [False] * len(sel_l)
    ll0, k0, n = _loglik(data, sel_b, sel_l, fixed_b, fixed_l, model, R,
                         integration, ghq_points, seed)
    bic0 = -2.0 * ll0 + k0 * np.log(n)

    rows = []
    for tier, names in (('upper', sel_b), ('lower', sel_l)):
        for k, v in enumerate(names):
            rb, rl = list(fixed_b), list(fixed_l)
            (rb if tier == 'upper' else rl)[k] = True
            ll1, k1, _ = _loglik(data, sel_b, sel_l, rb, rl, model, R,
                                 integration, ghq_points, seed)
            lr = max(2.0 * (ll1 - ll0), 0.0)
            rows.append({
                'term': v, 'tier': tier, 'LR': lr,
                'p_value': 0.5 * chi2.sf(lr, 1),
                'dBIC': (-2.0 * ll1 + k1 * np.log(n)) - bic0,
            })
    out = pd.DataFrame(rows)

    rec = {}
    for term, g in out.groupby('term'):
        ok = g[(g['p_value'] < alpha) & (g['dBIC'] < 0)]
        rec[term] = 'fixed' if ok.empty else ok.loc[ok['LR'].idxmax(), 'tier']
    out['recommendation'] = out['term'].map(rec)
    return out
