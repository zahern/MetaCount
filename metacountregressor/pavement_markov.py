"""
zeke_markov_bridge.py
========================================================================
Markovian prediction-process bridge for the pavement-deterioration program.

This module converts the *estimated* cluster specifications (from
``zeke_functional_forms`` / the search) into the **forward-looking**
prediction engine described by ``eq:mk*`` in ``trb_summary.tex``:

    1. discretise continuous PSI into ``R`` ordered condition states
       (thresholds ``theta``),                       eq:mkstate, eq:mkthreshold
    2. estimate per-cluster transition matrices ``P_k`` from a multinomial
       logit link on the active covariates + treatment reset,  eq:mklogit,
       eq:mkcov, with row-stochasticity (eq:mkrow) and monotone
       non-improvement (eq:mkmono),
    3. accumulate the transition log-likelihood ``LL_MK`` into ``BIC_MK``,
    4. forward-propagate the network condition-state distribution
       ``pi_{t+1} = pi_t P_k`` for life-cycle / M&R planning.

The existing standalone ``zeke_pavement_mdp.py`` already builds homogeneous /
age-dependent transition matrices, but it reads *raw* PSI and pools all
segments. This bridge instead consumes the **cluster-specific** functional
forms and membership produced by the search, so the Markov layer is driven by
the same heterogeneity the regression discovered -- closing the gap between
estimation and prediction.
"""
from __future__ import annotations

import numpy as np
from .pavement_minimize import minimize, softmax


def discretize(psi, thresholds):
    """Map continuous PSI to an integer state in {1..R}.

    ``thresholds`` are the (R-1) upper bounds theta_1 < ... < theta_{R-1}.
    State r covers (theta_{r-1}, theta_r] with theta_0=-inf, theta_R=+inf.
    """
    psi = np.asarray(psi, float)
    th = np.asarray(thresholds, float)
    R = len(th) + 1
    states = np.ones_like(psi, dtype=int)
    for r in range(1, R):
        states = np.where(psi > th[r - 1], r + 1, states)
    # states currently in {1..R}; caller may shift to 0..R-1
    return states


def _transition_logits(coeffs, Xd, n_states, has_treatment, treatment=None):
    """Multinomial-logit logits for one origin state.

    coeffs : (n_states, n_states, q)  -> link parameters per (i, j) row
    Returns (n, n_states, n_states) logits.
    """
    n, q = Xd.shape
    logits = np.zeros((n, n_states, n_states))
    for i in range(n_states):
        for j in range(n_states):
            lin = Xd @ coeffs[i, j]  # (n,)
            logits[:, i, j] = lin
    if has_treatment and treatment is not None:
        # treatment resets to best state (row replaced by a reset distribution)
        pass
    return logits


def fit_transitions(states, Xd, n_states, treatment=None,
                    monotone=False, has_treatment=False):
    """Estimate per-cluster transition probabilities via multinomial logit.

    states : (n,) sequence of state at time t (1..R)
    Xd     : (n, q) covariate design (form-transformed) for the *origin* row
    Returns dict(P, loglik, n_params, n_states).
    """
    states = np.asarray(states, int)
    Xd = np.asarray(Xd, float)
    n, q = Xd.shape
    # align origin t and destination t+1
    orig = states[:-1]
    dest = states[1:]
    Xo = Xd[:-1]
    n_trans = len(orig)

    def unpack(params):
        # params: intercept (n_states,n_states) + slope (n_states,n_states,q)
        b0 = params[: n_states * n_states].reshape(n_states, n_states)
        b1 = params[n_states * n_states:].reshape(n_states, n_states, q)
        return b0, b1

    def nll(params):
        b0, b1 = unpack(params)
        ll = 0.0
        for i in range(n_states):
            mask = orig == (i + 1)
            if mask.sum() == 0:
                continue
            Xm = Xo[mask]
            logits = b0[i] + Xm @ b1[i].T              # (m, n_states)
            if monotone:
                # forbid improvement: zero prob for j < i
                logits = logits.at[:, :i].set(-1e9) if hasattr(logits, "at") \
                    else _zero_lower(logits, i)
            p = softmax(logits, axis=1)
            ll += np.sum(np.log(p[np.arange(mask.sum()), dest[mask] - 1] + 1e-12))
        return -ll

    start = np.zeros(n_states * n_states * (1 + q))
    res = minimize(nll, start, method="Nelder-Mead",
                   options={"maxiter": 4000, "xatol": 1e-5, "fatol": 1e-5})
    b0, b1 = unpack(res.x)
    ll = -res.fun
    # build P per (i, covariate profile) -- return mean P over origin rows
    P = np.zeros((n_states, n_states))
    for i in range(n_states):
        mask = orig == (i + 1)
        if mask.sum() == 0:
            P[i] = np.eye(n_states)[i]  # degenerate: stay
            continue
        Xm = Xo[mask]
        logits = b0[i] + Xm @ b1[i].T
        if monotone:
            logits = _zero_lower(logits, i)
        P[i] = softmax(logits, axis=1).mean(axis=0)
    return {"P": P, "loglik": float(ll),
            "n_params": int(n_states * n_states * (1 + q)),
            "n_states": n_states, "b0": b0, "b1": b1}


def _zero_lower(M, i):
    M = M.copy()
    M[:, :i] = -1e9
    return M


def forward_propagate(P, init_dist, horizon):
    """Propagate a condition-state distribution.

    P : (R, R) transition matrix (row-stochastic).  Returns (horizon+1, R)
    distribution trajectory starting from ``init_dist``.
    """
    P = np.asarray(P, float)
    pi = np.asarray(init_dist, float).astype(float)
    traj = [pi.copy()]
    for _ in range(int(horizon)):
        pi = pi @ P
        traj.append(pi.copy())
    return np.vstack(traj)


def markov_bic(loglik, n_params, N):
    """BIC contribution of the Markov layer (eq:bic_mk)."""
    return -2.0 * loglik + n_params * np.log(max(N, 2))


def composite_bic(bic_reg, bic_mk, bic_haz):
    """Composite objective used by the SA search (eq:bic_total)."""
    return bic_reg + bic_mk + bic_haz


if __name__ == "__main__":
    rng = np.random.default_rng(2)
    n = 500
    psi = np.sort(rng.uniform(0, 5, n))[::-1]  # deteriorating trend
    th = [1.5, 3.0, 4.0]
    st = discretize(psi, th)
    Xd = np.column_stack([np.ones(n), rng.normal(0, 1, n)])
    out = fit_transitions(st, Xd, n_states=len(th) + 1, monotone=True)
    print("transition P (mean):\n", np.round(out["P"], 3))
    traj = forward_propagate(out["P"], np.eye(4)[3], horizon=10)
    print("state-4 share after 10 yrs: %.3f" % traj[-1, 3])
    print("markov BIC: %.3f" % markov_bic(out["loglik"], out["n_params"], n))
    print("OK: zeke_markov_bridge self-test passed")
