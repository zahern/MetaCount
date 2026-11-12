"""
zeke_hazard_survival.py
========================================================================
Hazard / survival component of the pavement-deterioration program.

This is the **event-oriented** half of the mathematical program
(``eq:hazard``, ``eq:survival`` in ``trb_summary.tex``). Where the cluster
regression answers "what is PSI *now*?" the hazard model answers
"*when* will the segment require treatment / reach a failure state?".

Families implemented (parametric, with covariate effects on the baseline):
    * Exponential   h(t|x) = h0 * exp(x'beta)
    * Weibull       h(t|x) = (p / scale) (t/scale)^(p-1) * exp(x'beta)
    * Gompertz      h(t|x) = h0 * exp(g*t) * exp(x'beta)
    * Log-logistic  h(t|x) = (p/scale)(t/scale)^(p-1) / (1+(t/scale)^p) * exp(x'beta)

Each model is fit by maximum likelihood (``scipy.optimize.minimize``) using a
clean per-cluster covariate design produced by ``zeke_functional_forms``.
The negative log-likelihood and the parameter count feed the Markov/hazard
BIC term ``BIC_MK`` in the composite objective ``BIC_tot``.

Metacountregressor embed note
-----------------------------
metacountregressor does not currently model time-to-event. This module is
self-contained and exposes ``fit_hazard(...)`` / ``HazardModel`` so that an
MCR post-estimation hook can call it on the *same* covariate design that MCR
already builds, returning a hazard layer that MCR can append to its output.
See ``SCOPING_metacountregressor_embed.md``.
"""
from __future__ import annotations

import numpy as np
from .pavement_minimize import minimize, expit

FAMILIES = ("exponential", "weibull", "gompertz", "loglogistic")


def _design_matrix(Xc, names, form, lam_vec, cat_dummies=None, shape=None):
    """Thin wrapper so this module can reuse the form transforms."""
    from .pavement_forms import build_design
    return build_design(np.asarray(Xc, float), names, form,
                        np.asarray(lam_vec, float) if lam_vec is not None else None,
                        shape=shape, cat_dummies=cat_dummies)


def _build_covariate_design(Xc, names, form, lam_vec, active_mask,
                            cat_dummies, shape):
    """Build the covariate design exactly as used in fitting.

    Raw ``Xc`` (n, p) + feature ``names`` are reduced to the active columns
    and transformed by the chosen functional form, then categorical dummies
    are appended. This is the single source of truth shared by ``fit_hazard``
    and ``HazardModel.predict_*`` so that predictions never need a
    pre-transformed matrix and never mismatch the fitted parameter count.
    """
    Xc = np.asarray(Xc, float)
    p = len(names)
    if active_mask is None:
        active_mask = np.ones(p, bool)
    Xa = Xc[:, active_mask]
    names_a = [names[j] for j in range(p) if active_mask[j]]
    lam_a = np.asarray(lam_vec)[active_mask] if lam_vec is not None else None
    # Align categorical dummies to the prediction rows: if they were stored as a
    # per-cluster mean (1 row) or a different row count, broadcast the mean.
    if cat_dummies is not None and np.asarray(cat_dummies).ndim == 2 \
            and np.asarray(cat_dummies).shape[1] > 0:
        cd = np.asarray(cat_dummies, float)
        if cd.shape[0] != Xa.shape[0]:
            cd = np.tile(cd.mean(0, keepdims=True), (Xa.shape[0], 1))
        cat_dummies = cd
    return _design_matrix(Xa, names_a, form, lam_a,
                          cat_dummies=cat_dummies, shape=shape)


def _neg_log_likelihood(params, family, t, event, Xd):
    """Negative log-likelihood for one parametric family.

    params layout:
        [h0_or_logscale, (p or g if applicable), beta...]
    Xd : (n, q) covariate design (already transformed by the form)
    """
    n, q = Xd.shape
    # Clip the linear predictor to avoid exp() overflow / NaN during MLE.
    def _lin(beta):
        return np.clip(Xd @ beta, -30.0, 30.0)
    if family == "exponential":
        h0 = np.exp(np.clip(params[0], -30.0, 30.0))
        beta = params[1:]
        lin = _lin(beta)
        h = h0 * np.exp(lin)
        ll = np.sum(event * np.log(h + 1e-12) - h * t)
    elif family == "weibull":
        log_scale = params[0]
        p = np.exp(np.clip(params[1], -10.0, 10.0))   # shape > 0
        beta = params[2:]
        scale = np.exp(np.clip(log_scale, -10.0, 10.0))
        lin = _lin(beta)
        z = np.clip(t / scale, 1e-9, 1e9)
        # Log-space evaluation: h and S are computed from log terms so large
        # p / z combinations cannot overflow intermediate powers.
        log_h = (np.log(p) - p * np.log(scale)
                 + (p - 1.0) * np.log(z) + lin)
        log_h = np.clip(log_h, -700.0, 700.0)
        h = np.exp(log_h)
        log_cum = np.clip(p * np.log(z) + lin, -30.0, 30.0)
        logS = -np.exp(log_cum)
        ll = np.sum(event * np.log(h + 1e-12) + logS)
    elif family == "gompertz":
        h0 = np.exp(np.clip(params[0], -30.0, 30.0))
        g = np.clip(params[1], -10.0, 10.0)
        beta = params[2:]
        lin = _lin(beta)
        h = h0 * np.exp(g * t) * np.exp(lin)
        if abs(g) > 1e-6:
            logS = -np.clip((h0 / g) * (np.exp(g * t) - 1.0) * np.exp(lin), -30.0, 30.0)
        else:
            logS = -np.clip(h0 * t * np.exp(lin), -30.0, 30.0)
        ll = np.sum(event * np.log(h + 1e-12) + logS)
    elif family == "loglogistic":
        log_scale = params[0]
        p = np.exp(params[1])
        beta = params[2:]
        scale = np.exp(np.clip(log_scale, -10.0, 10.0))
        lin = np.clip(Xd @ beta, -30.0, 30.0)
        z = np.clip(t / scale, 1e-9, 1e9)
        h = (p / scale) * z ** (p - 1) * np.exp(lin) / (1.0 + z ** p)
        logS = -np.clip(np.log1p(z ** p * np.exp(lin)), -30.0, 30.0)
        ll = np.sum(event * np.log(h + 1e-12) + logS)
    else:
        raise ValueError(family)
    return -ll


def fit_hazard(Xc, names, form, lam_vec, t, event,
               family="weibull", cat_dummies=None, shape=None,
               active_mask=None):
    """Fit a parametric hazard model for one cluster.

    Parameters
    ----------
    Xc : (n, p) raw continuous covariates (segment-year rows)
    t  : (n,) time (e.g. age or years-since-last-treatment)
    event : (n,) 1 if a treatment/failure occurred in that interval
    active_mask : (p,) which continuous vars enter the hazard link

    Returns
    -------
    dict(hazard_model, beta, n_params, loglik, bic, family,
         hazard_fn, survival_fn)
    """
    Xc = np.asarray(Xc, float); t = np.asarray(t, float); event = np.asarray(event, int)
    p = len(names)
    if active_mask is None:
        active_mask = np.ones(p, bool)
    Xa = Xc[:, active_mask]
    na = Xa.shape[1]
    names_a = [names[j] for j in range(p) if active_mask[j]]
    Xd = _build_covariate_design(Xc, names, form, lam_vec, active_mask,
                                cat_dummies, shape)
    # Xd is the covariate design only (no intercept column); the baseline /
    # shape parameter is params[0] (and params[1] for shape families).
    q = Xd.shape[1]
    n_params = q + (2 if family in ("weibull", "loglogistic", "gompertz") else 1)

    beta0 = np.zeros(q)
    if family in ("weibull", "loglogistic", "gompertz"):
        x0 = np.concatenate([[np.log(t.mean()), 0.0], beta0])
    else:
        x0 = np.concatenate([[0.0], beta0])

    res = minimize(_neg_log_likelihood, x0,
                   args=(family, t, event, Xd),
                   method="Nelder-Mead",
                   options={"maxiter": 5000, "xatol": 1e-6, "fatol": 1e-6})
    params = res.x
    ll = -res.fun
    n = len(t)
    bic = -2.0 * ll + n_params * np.log(n)

    def hazard_fn(tt, Xnew):
        Xnew = np.atleast_2d(np.asarray(Xnew, float))
        m = Xnew.shape[0]
        tt = np.asarray(tt, float)
        out = np.empty((len(tt), m))
        for ti, tval in enumerate(tt):
            out[ti] = _hazard_from(params, family,
                                  np.full(m, tval), Xnew)
        return out

    def survival_fn(tt, Xnew):
        Xnew = np.atleast_2d(np.asarray(Xnew, float))
        m = Xnew.shape[0]
        tt = np.asarray(tt, float)
        out = np.empty((len(tt), m))
        for ti, tval in enumerate(tt):
            out[ti] = _survival_from(params, family,
                                    np.full(m, tval), Xnew)
        return out

    return {
        "family": family,
        "params": params,
        "n_params": int(n_params),
        "loglik": float(ll),
        "bic": float(bic),
        "beta": params[1:],
        "hazard_fn": hazard_fn,
        "survival_fn": survival_fn,
    }


def _hazard_from(params, family, t, Xd):
    if family == "exponential":
        return np.exp(params[0]) * np.exp(Xd @ params[1:])
    if family == "weibull":
        p = np.exp(params[1]); scale = np.exp(params[0])
        z = t / scale
        return (p / scale) * z ** (p - 1) * np.exp(Xd @ params[2:])
    if family == "gompertz":
        return np.exp(params[0]) * np.exp(params[1] * t) * np.exp(Xd @ params[2:])
    if family == "loglogistic":
        p = np.exp(params[1]); scale = np.exp(params[0]); z = t / scale
        return (p / scale) * z ** (p - 1) * np.exp(Xd @ params[2:]) / (1.0 + z ** p)


def _survival_from(params, family, t, Xd):
    if family == "exponential":
        return np.exp(-np.exp(params[0]) * t * np.exp(Xd @ params[1:]))
    if family == "weibull":
        p = np.exp(params[1]); scale = np.exp(params[0]); z = t / scale
        return np.exp(-(z ** p) * np.exp(Xd @ params[2:]))
    if family == "gompertz":
        g = params[1]; h0 = np.exp(params[0]); lin = Xd @ params[2:]
        return np.exp(-(h0 / g) * (np.exp(g * t) - 1.0) * np.exp(lin)) if abs(g) > 1e-6 \
            else np.exp(-h0 * t * np.exp(lin))
    if family == "loglogistic":
        p = np.exp(params[1]); scale = np.exp(params[0]); z = t / scale
        return 1.0 / (1.0 + z ** p * np.exp(Xd @ params[2:]))


class HazardModel:
    """Convenience wrapper around ``fit_hazard`` for one cluster."""
    def __init__(self, family="weibull"):
        self.family = family
        self.fitted = None
        self._spec = None

    def fit(self, Xc, names, form, lam_vec, t, event, **kw):
        self.fitted = fit_hazard(Xc, names, form, lam_vec, t, event,
                                 family=self.family, **kw)
        # store the design spec so predict_* can rebuild the covariate design
        self._spec = {
            "names": names, "form": form, "lam_vec": lam_vec,
            "active_mask": kw.get("active_mask"),
            "cat_dummies": kw.get("cat_dummies"),
            "shape": kw.get("shape"),
        }
        return self

    def _design(self, Xnew):
        s = self._spec
        return _build_covariate_design(Xnew, s["names"], s["form"], s["lam_vec"],
                                       s["active_mask"], s["cat_dummies"], s["shape"])

    def predict_hazard(self, t, Xnew):
        return self.fitted["hazard_fn"](t, self._design(Xnew))

    def predict_survival(self, t, Xnew):
        return self.fitted["survival_fn"](t, self._design(Xnew))

    def time_to_event(self, p_target, Xnew, t_grid=None):
        """Years until survival drops to ``p_target`` (treatment horizon)."""
        if t_grid is None:
            t_grid = np.linspace(0.1, 60, 600)
        S = self.predict_survival(t_grid, Xnew)   # shape (T, m)
        S = np.atleast_2d(S)
        out = []
        for j in range(S.shape[1]):
            idx = np.where(S[:, j] <= p_target)[0]
            out.append(float(t_grid[idx[0]]) if len(idx) else np.inf)
        return out[0] if len(out) == 1 else np.array(out)

    @property
    def bic(self):
        return self.fitted["bic"] if self.fitted else np.inf


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    n = 400
    age = rng.uniform(1, 30, n)
    aadt = rng.uniform(100, 5000, n)
    Xc = np.column_stack([age, aadt])
    names = ["age", "aadt"]
    # Synthetic Weibull events: higher aadt -> higher hazard
    true_beta = np.array([0.0, 0.4])
    h = 0.05 * np.exp(true_beta[1] * (aadt / 1000.0))
    t_event = (-np.log(rng.uniform(0, 1, n)) / h) ** (1 / 1.5) * 5.0
    t_obs = np.minimum(t_event, age)
    event = (t_event <= age).astype(int)
    out = fit_hazard(Xc, names, 0, None, t_obs, event, family="weibull",
                     active_mask=np.array([False, True]))
    print(f"weibull fit: ll={-out['loglik']:.3f} bic={out['bic']:.3f} "
          f"beta={np.round(out['beta'], 3)}")
    hm = HazardModel("gompertz").fit(Xc, names, 0, None, t_obs, event,
                                     active_mask=np.array([False, True]))
    tte = hm.time_to_event(0.5, Xc[:3])
    print(f"gompertz fit: bic={hm.bic:.3f} "
          f"TTE@0.5={float(np.asarray(tte).ravel()[0]):.2f} yrs")
    print("OK: zeke_hazard_survival self-test passed")
