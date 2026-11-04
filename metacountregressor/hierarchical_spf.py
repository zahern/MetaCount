"""
hierarchical_spf.py
===================
Hierarchical safety performance functions with crash-modification-function
(CM-Function) sub-models, following Chen & Persaud (2014), "Methodology to
develop crash modification functions for road safety treatments with fully
specified and hierarchical models", Accident Analysis and Prevention 70,
131-139 (Eq. 13), extended with random parameters.

Model
-----
First-level SPF with M traffic-volume terms (e.g. major and minor AADT)::

    mu_it = A_it * prod_m AADT_m,it ^ B_m,it          (x exp(offset))

Sub-level CM-Functions::

    ln A   = ln(alpha0) + sum_k alpha_k x_k                 (Component A)
    B_m    = b0_m * exp( sum_j beta_mj z_mj )   form="exp"  (paper Eq. 13)
    B_m    = b0_m +      sum_j beta_mj z_mj     form="linear"

Count model: NB2 with Var = mu + alpha mu^2 (SAS NLMIXED convention used in
the paper) or Poisson.

Random parameters ("all flavours")
----------------------------------
Any structural coefficient -- ``ln_alpha0``, ``alpha[x]``, ``b0[AADT]``,
``beta[AADT|z]`` -- may be random::

    theta_ij = theta + sigma_g u_j + (L v_i)_theta

* ``"site"``  : site-level (panel) random parameter, constant across a
  site's years; several site-level parameters may be correlated
  (``correlated=True`` -> Cholesky L).
* ``"group"`` : group-level random parameter shared by all sites in a group
  (e.g. corridor, functional class, city district).
* ``"both"``  : random at the group AND site level (two-level hierarchy).

For ``form="exp"`` the base elasticity is parameterised on the log scale
(``b0 = exp(log b0)``), so a random ``b0`` is lognormal and keeps its sign.

Estimation is maximum simulated likelihood with the integrals nested
correctly: site draws are integrated within each site (product over its
years), group draws outside, across the group's sites::

    L_j = (1/R) sum_r prod_{i in j} (1/S) sum_s prod_t f(y_it | u_jr, v_is)

Outer draws are processed sequentially with gradient checkpointing, so
memory is O(rows x inner draws), not O(rows x R x S).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln, logsumexp

jax.config.update("jax_enable_x64", True)

__all__ = ["HierarchicalSPFSpec", "HierarchicalSPF", "fit_hierarchical_spf"]

_LEVELS = ("site", "group", "both")


@dataclass
class HierarchicalSPFSpec:
    """Structure of a hierarchical SPF.

    Parameters
    ----------
    aadt_cols : traffic-volume columns entering as AADT^B (raw, > 0).
    a_terms   : covariates of Component A (ln A = ln alpha0 + sum alpha_k x_k).
    b_terms   : {aadt_col: [covariates of that AADT's exponent sub-model]}.
    exponent_form : "exp" (paper) or "linear".
    random    : {parameter label: "site" | "group" | "both"}.
    correlated: correlate the site-level random parameters (Cholesky).
    family    : "nb2" or "poisson".
    site_col  : panel id (rows of one site share site-level draws). If None,
                every row is its own site.
    group_col : group id for group-level random parameters.
    offset_col: optional log-exposure column added to ln mu.
    """
    aadt_cols: List[str]
    a_terms: List[str] = field(default_factory=list)
    b_terms: Dict[str, List[str]] = field(default_factory=dict)
    exponent_form: str = "exp"
    random: Dict[str, str] = field(default_factory=dict)
    correlated: bool = False
    family: str = "nb2"
    site_col: Optional[str] = None
    group_col: Optional[str] = None
    offset_col: Optional[str] = None

    # ------------------------------------------------------------------
    def labels(self) -> List[str]:
        out = ["ln_alpha0"] + [f"alpha[{x}]" for x in self.a_terms]
        for m in self.aadt_cols:
            out.append(f"b0[{m}]")
            out += [f"beta[{m}|{z}]" for z in self.b_terms.get(m, [])]
        return out

    def validate(self, df: pd.DataFrame) -> None:
        if self.exponent_form not in ("exp", "linear"):
            raise ValueError("exponent_form must be 'exp' or 'linear'")
        if self.family not in ("nb2", "poisson"):
            raise ValueError("family must be 'nb2' or 'poisson'")
        if not self.aadt_cols:
            raise ValueError("at least one AADT column is required")
        unknown = set(self.b_terms) - set(self.aadt_cols)
        if unknown:
            raise ValueError(f"b_terms keys not in aadt_cols: {sorted(unknown)}")
        cols = list(self.aadt_cols) + list(self.a_terms) + \
            [z for zs in self.b_terms.values() for z in zs]
        for c in [self.site_col, self.group_col, self.offset_col]:
            if c is not None:
                cols.append(c)
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"columns not in data: {missing}")
        if (df[list(self.aadt_cols)].to_numpy(float) <= 0).any():
            raise ValueError("AADT columns must be strictly positive")
        labels = set(self.labels())
        for lab, lvl in self.random.items():
            if lab not in labels:
                raise ValueError(f"random parameter {lab!r} is not a model "
                                 f"parameter; choose from {sorted(labels)}")
            if lvl not in _LEVELS:
                raise ValueError(f"level for {lab!r} must be one of {_LEVELS}")
            if lvl in ("group", "both") and self.group_col is None:
                raise ValueError(f"{lab!r} is random at the group level but "
                                 "group_col is not set")
        if self.group_col is not None and self.site_col is not None:
            g_per_site = df.groupby(self.site_col)[self.group_col].nunique()
            if (g_per_site > 1).any():
                raise ValueError("each site must belong to exactly one group")


# ----------------------------------------------------------------------
# draws
# ----------------------------------------------------------------------
def _normal_draws(n_units: int, k: int, r: int, seed: int) -> np.ndarray:
    """(n_units, k, r) scrambled-Halton standard normals (independent
    scrambling per unit block avoids correlated draws across units)."""
    if k == 0:
        return np.zeros((n_units, 0, r))
    from scipy.stats import qmc, norm
    sampler = qmc.Halton(d=k, scramble=True, seed=seed)
    u = sampler.random(n_units * r)
    u = np.clip(u, 1e-10, 1 - 1e-10)
    z = norm.ppf(u).reshape(n_units, r, k).transpose(0, 2, 1)
    # randomly permute draw order per unit to break the sequence structure
    rng = np.random.default_rng(seed + 1)
    for i in range(n_units):
        z[i] = z[i][:, rng.permutation(r)]
    return z


# ----------------------------------------------------------------------
# model
# ----------------------------------------------------------------------
class HierarchicalSPF:
    """Fit a :class:`HierarchicalSPFSpec` by (simulated) maximum likelihood."""

    def __init__(self, spec: HierarchicalSPFSpec, R: int = 200,
                 inner_draws: int = 100, seed: int = 0):
        self.spec = spec
        self.R = int(R)
        self.S = int(inner_draws)
        self.seed = int(seed)
        self.result_ = None

    # -- data ----------------------------------------------------------
    def _prepare(self, df: pd.DataFrame, y_col: str):
        sp = self.spec
        sp.validate(df)
        df = df.reset_index(drop=True)
        y = df[y_col].to_numpy(float)
        if np.any(y < 0) or np.any(~np.isfinite(y)):
            raise ValueError("response must be finite and non-negative")
        labels = sp.labels()
        P = len(labels)
        N = len(df)
        # design: ln mu = sum_p f_p(coef_p) ... built from these pieces
        Xa = df[sp.a_terms].to_numpy(float) if sp.a_terms else np.zeros((N, 0))
        lnaadt = np.log(df[sp.aadt_cols].to_numpy(float))
        Zb = [df[sp.b_terms.get(m, [])].to_numpy(float) if sp.b_terms.get(m)
              else np.zeros((N, 0)) for m in sp.aadt_cols]
        offset = (df[sp.offset_col].to_numpy(float) if sp.offset_col
                  else np.zeros(N))
        if sp.site_col is not None:
            site_codes, site_idx = np.unique(df[sp.site_col].to_numpy(),
                                             return_inverse=True)
        else:
            site_codes, site_idx = np.arange(N), np.arange(N)
        I = len(site_codes)
        if sp.group_col is not None:
            g_of_row = df[sp.group_col].to_numpy()
            g_codes, g_idx_row = np.unique(g_of_row, return_inverse=True)
            g_idx_site = np.zeros(I, int)
            g_idx_site[site_idx] = g_idx_row
        else:
            g_codes, g_idx_row = np.array([0]), np.zeros(N, int)
            g_idx_site = np.zeros(I, int)
        site_rand = [labels.index(l) for l, v in sp.random.items()
                     if v in ("site", "both")]
        group_rand = [labels.index(l) for l, v in sp.random.items()
                      if v in ("group", "both")]
        self._d = dict(
            y=jnp.asarray(y), Xa=jnp.asarray(Xa), lnaadt=jnp.asarray(lnaadt),
            Zb=[jnp.asarray(z) for z in Zb], offset=jnp.asarray(offset),
            site_idx=jnp.asarray(site_idx), g_idx_site=jnp.asarray(g_idx_site),
            g_idx_row=jnp.asarray(g_idx_row),
        )
        self._labels, self._P, self._N, self._I = labels, P, N, I
        self._G = len(g_codes)
        self._site_rand, self._group_rand = site_rand, group_rand
        self._y_lgamma = jnp.asarray(gammaln(y + 1.0))
        Ks, Kg = len(site_rand), len(group_rand)
        S = self.S if (Ks and Kg) else self.R
        self._Ssite = S
        self._Vs = jnp.asarray(_normal_draws(I, Ks, S, self.seed + 11))
        self._Ug = jnp.asarray(_normal_draws(self._G, Kg, self.R, self.seed + 29))
        return y

    # -- parameter layout ------------------------------------------------
    def _layout(self):
        P, Ks, Kg = self._P, len(self._site_rand), len(self._group_rand)
        lay = {"theta": (0, P)}
        i = P
        if Ks:
            n = Ks * (Ks + 1) // 2 if self.spec.correlated else Ks
            lay["site"] = (i, i + n); i += n
        if Kg:
            lay["group"] = (i, i + Kg); i += Kg
        if self.spec.family == "nb2":
            lay["ln_disp"] = (i, i + 1); i += 1
        lay["total"] = i
        return lay

    def param_names(self):
        lay = self._layout()
        names = list(self._labels)
        sr = [self._labels[k] for k in self._site_rand]
        if "site" in lay:
            if self.spec.correlated:
                for a in range(len(sr)):
                    for b in range(a + 1):
                        names.append(f"site_chol[{sr[a]},{sr[b]}]" if a != b
                                     else f"site_sd[{sr[a]}]")
            else:
                names += [f"site_sd[{s}]" for s in sr]
        if "group" in lay:
            names += [f"group_sd[{self._labels[k]}]" for k in self._group_rand]
        if "ln_disp" in lay:
            names.append("ln_alpha_disp")
        return names

    def _site_chol(self, p):
        lay = self._layout()
        Ks = len(self._site_rand)
        a, b = lay["site"]
        raw = p[a:b]
        if not self.spec.correlated:
            return jnp.diag(jnp.exp(raw))
        L = jnp.zeros((Ks, Ks))
        L = L.at[jnp.tril_indices(Ks)].set(raw)
        d = jnp.diag_indices(Ks)
        return L.at[d].set(jnp.exp(L[d]))

    # -- likelihood pieces ----------------------------------------------
    def _ln_mu(self, coef):
        """coef: (N, P) per-row coefficients -> (N,) ln mu."""
        d, sp, lab = self._d, self.spec, self._labels
        ln_mu = coef[:, 0] + d["offset"]
        j = 1
        for k in range(len(sp.a_terms)):
            ln_mu = ln_mu + coef[:, j] * d["Xa"][:, k]
            j += 1
        for m in range(len(sp.aadt_cols)):
            b0 = coef[:, j]; j += 1
            zs = d["Zb"][m]
            lin = jnp.zeros_like(b0)
            for q in range(zs.shape[1]):
                lin = lin + coef[:, j] * zs[:, q]
                j += 1
            if sp.exponent_form == "exp":
                B = jnp.exp(b0 + lin)          # b0 holds log b0
            else:
                B = b0 + lin
            ln_mu = ln_mu + B * d["lnaadt"][:, m]
        return jnp.clip(ln_mu, -30.0, 30.0)

    def _row_ll(self, ln_mu, ln_disp):
        y = self._d["y"]
        if self.spec.family == "poisson":
            return y * ln_mu - jnp.exp(ln_mu) - self._y_lgamma
        a = jnp.exp(jnp.clip(ln_disp, -20.0, 10.0))
        inv = 1.0 / a
        # NB2:  Var = mu + a mu^2  (log-space, stable)
        log1p_amu = jnp.logaddexp(0.0, jnp.log(a) + ln_mu)
        return (gammaln(y + inv) - gammaln(inv) - self._y_lgamma
                - inv * log1p_amu + y * (jnp.log(a) + ln_mu - log1p_amu))

    def negloglik(self, p):
        lay = self._layout()
        d = self._d
        theta = p[lay["theta"][0]:lay["theta"][1]]
        ln_disp = p[lay["ln_disp"][0]] if "ln_disp" in lay else 0.0
        N, P, I, G = self._N, self._P, self._I, self._G
        Ks, Kg = len(self._site_rand), len(self._group_rand)
        base = jnp.broadcast_to(theta, (N, P))
        site_idx = d["site_idx"]

        if Ks == 0 and Kg == 0:
            return -jnp.sum(self._row_ll(self._ln_mu(base), ln_disp))

        sel_s = jnp.zeros((Ks, P)).at[jnp.arange(Ks), jnp.array(self._site_rand, int)].set(1.0) \
            if Ks else None
        sel_g = jnp.zeros((Kg, P)).at[jnp.arange(Kg), jnp.array(self._group_rand, int)].set(1.0) \
            if Kg else None

        def site_ll_given(coef_rows, v_sites):
            """Per-site log of (1/S) sum_s prod_t f, given site draws
            v_sites (I, Ks, S); coef_rows (N, P) w/o site part."""
            L = self._site_chol(p)                                  # (Ks,Ks)
            dev = jnp.einsum("ab,ibs->ias", L, v_sites)             # (I,Ks,S)
            dev_rows = dev[site_idx]                                # (N,Ks,S)
            def one_s(s):
                c = coef_rows + dev_rows[:, :, s] @ sel_s
                return jax.ops.segment_sum(
                    self._row_ll(self._ln_mu(c), ln_disp), site_idx,
                    num_segments=I)                                 # (I,)
            per = jax.lax.map(jax.checkpoint(one_s),
                              jnp.arange(v_sites.shape[-1]))        # (S,I)
            return logsumexp(per, axis=0) - jnp.log(per.shape[0])  # (I,)

        if Kg == 0:
            return -jnp.sum(site_ll_given(base, self._Vs))

        sd_g = jnp.exp(p[lay["group"][0]:lay["group"][1]])         # (Kg,)
        U = self._Ug                                                # (G,Kg,R)
        g_row = d["g_idx_row"]

        def one_r(r):
            dev_g = (sd_g[None, :] * U[:, :, r])[g_row] @ sel_g     # (N,P)
            coef = base + dev_g
            if Ks:
                ll_site = site_ll_given(coef, self._Vs)             # (I,)
            else:
                ll_site = jax.ops.segment_sum(
                    self._row_ll(self._ln_mu(coef), ln_disp), site_idx,
                    num_segments=I)
            return jax.ops.segment_sum(ll_site, d["g_idx_site"],
                                       num_segments=G)              # (G,)
        per_r = jax.lax.map(jax.checkpoint(one_r), jnp.arange(U.shape[-1]))
        ll_g = logsumexp(per_r, axis=0) - jnp.log(per_r.shape[0])
        return -jnp.sum(ll_g)

    # -- fitting ----------------------------------------------------------
    def _start(self, y):
        sp = self.spec
        lay = self._layout()
        p = np.zeros(lay["total"])
        b0 = 0.5
        mlna = np.asarray(self._d["lnaadt"]).mean(axis=0)
        p[0] = np.log(max(y.mean(), 1e-3)) - b0 * mlna.sum()
        j = 1 + len(sp.a_terms)
        for m in sp.aadt_cols:
            p[j] = np.log(b0) if sp.exponent_form == "exp" else b0
            j += 1 + len(sp.b_terms.get(m, []))
        if "site" in lay:
            a, b = lay["site"]
            p[a:b] = 0.0
            if sp.correlated:
                Ks = len(self._site_rand)
                raw = np.zeros((Ks, Ks)); raw[np.diag_indices(Ks)] = np.log(0.1)
                p[a:b] = raw[np.tril_indices(Ks)]
            else:
                p[a:b] = np.log(0.1)
        if "group" in lay:
            a, b = lay["group"]; p[a:b] = np.log(0.1)
        if "ln_disp" in lay:
            p[lay["ln_disp"][0]] = np.log(0.5)
        return p

    def _optimize(self, fun, p0, maxiter):
        from scipy.optimize import minimize
        vg = jax.jit(jax.value_and_grad(fun))
        def f(x):
            v, g = vg(jnp.asarray(x))
            v = float(v); g = np.asarray(g, float)
            if not np.isfinite(v):
                return 1e300, np.zeros_like(g)
            return v, g
        res = minimize(f, p0, jac=True, method="L-BFGS-B",
                       options={"maxiter": maxiter, "gtol": 1e-7, "ftol": 1e-12})
        return res

    def _newton_polish(self, p, f, hess, iters=20, tol=1e-6):
        """Damped Newton steps after L-BFGS. ln(alpha0) and the AADT
        elasticities are nearly collinear (ln AADT has a large mean), so
        first-order methods stall with a visible gradient; Newton removes
        it. Steps are accepted only if the objective decreases."""
        vg = jax.jit(jax.value_and_grad(self.negloglik))
        for _ in range(iters):
            _, g = vg(jnp.asarray(p))
            g = np.asarray(g, float)
            if np.max(np.abs(g)) < tol:
                break
            H = np.asarray(hess(jnp.asarray(p)))
            H = 0.5 * (H + H.T)
            w, V = np.linalg.eigh(H)
            if not np.all(np.isfinite(w)):
                break
            w = np.maximum(w, 1e-8 * max(1.0, np.max(np.abs(w))))
            step = V @ ((V.T @ g) / w)
            t, improved = 1.0, False
            for _ in range(30):
                cand = p - t * step
                fc = float(vg(jnp.asarray(cand))[0])
                if np.isfinite(fc) and fc <= f:
                    p, f, improved = cand, fc, True
                    break
                t *= 0.5
            if not improved:
                break
        return p, f

    def fit(self, df: pd.DataFrame, y_col: str, maxiter: int = 2000,
            start: Optional[np.ndarray] = None):
        y = self._prepare(df, y_col)
        lay = self._layout()
        p0 = self._start(y) if start is None else np.asarray(start, float)
        if start is None and (self._site_rand or self._group_rand):
            # warm start: fixed-parameter model on the same data
            fixed = HierarchicalSPF(
                HierarchicalSPFSpec(**{**self.spec.__dict__, "random": {}}),
                R=self.R, inner_draws=self.S, seed=self.seed)
            fixed.fit(df, y_col, maxiter=maxiter)
            p0[:self._P] = fixed.result_["params"][:self._P]
            if "ln_disp" in lay:
                p0[lay["ln_disp"][0]] = fixed.result_["params"][-1]
        res = self._optimize(self.negloglik, p0, maxiter)
        p_hat = np.asarray(res.x, float)
        hess = jax.jit(jax.hessian(self.negloglik))
        p_hat, f_hat = self._newton_polish(p_hat, float(res.fun), hess)
        res.fun = f_hat
        H = np.asarray(hess(jnp.asarray(p_hat)))
        H = 0.5 * (H + H.T)
        try:
            cov = np.linalg.inv(H)
            se = np.sqrt(np.where(np.diag(cov) > 0, np.diag(cov), np.nan))
        except np.linalg.LinAlgError:
            cov, se = None, np.full(len(p_hat), np.nan)
        ll = -float(res.fun)
        k = len(p_hat)
        n = self._N
        g = np.asarray(jax.grad(self.negloglik)(jnp.asarray(p_hat)))
        self.result_ = dict(params=p_hat, se=se, cov=cov, loglik=ll, k=k,
                            n_obs=n, aic=2 * k - 2 * ll,
                            bic=k * np.log(n) - 2 * ll,
                            converged=bool(res.success),
                            max_abs_grad=float(np.max(np.abs(g))),
                            message=str(res.message))
        return self

    # -- reporting --------------------------------------------------------
    def summary(self) -> pd.DataFrame:
        """Estimates in the paper's units.

        ``b0[...]`` is reported on the natural scale for the exp form (delta
        method from log b0); random-parameter scales as SDs; the NB2
        dispersion as alpha (Var = mu + alpha mu^2).
        """
        from scipy.stats import norm as _norm
        r = self.result_
        names = self.param_names()
        est, se = r["params"].copy(), r["se"].copy()
        out_names = []
        for i, nm in enumerate(names):
            if (nm.startswith("b0[") and self.spec.exponent_form == "exp") or \
                    nm.startswith(("site_sd[", "group_sd[")) or nm == "ln_alpha_disp":
                v = np.exp(est[i]); se[i] = v * se[i]; est[i] = v
                nm = "dispersion_alpha" if nm == "ln_alpha_disp" else nm
            out_names.append(nm)
        z = est / se
        df = pd.DataFrame({"parameter": out_names, "estimate": est,
                           "std_err": se, "t_value": z,
                           "p_value": 2 * (1 - _norm.cdf(np.abs(z)))})
        df.attrs.update({k: r[k] for k in ("loglik", "aic", "bic", "k", "n_obs",
                                           "converged", "max_abs_grad")})
        return df

    def site_covariance(self) -> pd.DataFrame:
        """Implied covariance of the site-level random parameters (L L')."""
        if not self._site_rand:
            return pd.DataFrame()
        L = np.asarray(self._site_chol(jnp.asarray(self.result_["params"])))
        names = [self._labels[k] for k in self._site_rand]
        return pd.DataFrame(L @ L.T, index=names, columns=names)

    def cm_function(self, term: str, values) -> np.ndarray:
        """CMF for a Component-A covariate: exp(alpha * value) (paper Table 10)."""
        i = self._labels.index(f"alpha[{term}]")
        return np.exp(self.result_["params"][i] * np.asarray(values, float))


def fit_hierarchical_spf(df: pd.DataFrame, y_col: str, spec: HierarchicalSPFSpec,
                         R: int = 200, inner_draws: int = 100, seed: int = 0,
                         maxiter: int = 2000) -> HierarchicalSPF:
    """Convenience wrapper: build, fit and return a :class:`HierarchicalSPF`."""
    return HierarchicalSPF(spec, R=R, inner_draws=inner_draws,
                           seed=seed).fit(df, y_col, maxiter=maxiter)
