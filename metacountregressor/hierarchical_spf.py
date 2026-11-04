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

__all__ = ["HierarchicalSPFSpec", "HierarchicalSPF", "fit_hierarchical_spf",
           "HierarchicalSPFSearch", "SearchResult"]

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


def _gh_product(k: int, n: int):
    """Gauss-Hermite product rule for E[f(v)], v ~ N(0, I_k).
    Returns nodes (k, n**k) and log-weights (n**k,) summing to 1."""
    x, w = np.polynomial.hermite.hermgauss(n)
    z, lw = np.sqrt(2.0) * x, np.log(w / np.sqrt(np.pi))
    grids = np.meshgrid(*([z] * k), indexing="ij")
    wgrids = np.meshgrid(*([lw] * k), indexing="ij")
    nodes = np.stack([g.ravel() for g in grids])
    logw = np.sum(np.stack([g.ravel() for g in wgrids]), axis=0)
    return jnp.asarray(nodes), jnp.asarray(logw)


# ----------------------------------------------------------------------
# model
# ----------------------------------------------------------------------
class HierarchicalSPF:
    """Fit a :class:`HierarchicalSPFSpec` by (simulated) maximum likelihood."""

    def __init__(self, spec: HierarchicalSPFSpec, R: int = 200,
                 inner_draws: int = 100, seed: int = 0,
                 site_integration: str = "auto", group_integration: str = "simulation",
                 quad_nodes: Optional[int] = None, draw_chunk: int = 64):
        """
        site_integration : "auto" | "quadrature" | "simulation"
            How the site-level random parameters are integrated. "auto"
            uses Gauss-Hermite product quadrature when there are at most
            two site-level random parameters (the integrand is a product
            over only a site's few years, so 12-20 nodes per dimension are
            as accurate as hundreds of draws), else Halton simulation with
            ``inner_draws`` (or ``R`` without group effects) draws.
        group_integration : "simulation" | "quadrature"
            Group-level integral. Simulation (``R`` draws) is the safe
            default: the integrand is a product over every site in the
            group, so it is sharply peaked and plain Gauss-Hermite needs
            many nodes. "quadrature" is allowed for one group-level
            random parameter.
        quad_nodes : nodes per dimension (default 16 for 1-D, 8 for 2-D).
        draw_chunk : draws evaluated together (vectorised) per step.
        """
        self.spec = spec
        self.R = int(R)
        self.S = int(inner_draws)
        self.seed = int(seed)
        self.site_integration = site_integration
        self.group_integration = group_integration
        self.quad_nodes = quad_nodes
        self.draw_chunk = int(draw_chunk)
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

        # Linear-predictor designs, so the per-draw work is only the random
        # deviations:  ln A = XA @ coef,  expo_m = E_m @ coef.
        XA = np.zeros((N, P)); XA[:, 0] = 1.0
        E = np.zeros((len(sp.aadt_cols), N, P))
        j = 1
        for k in range(len(sp.a_terms)):
            XA[:, j] = Xa[:, k]; j += 1
        for mi, m in enumerate(sp.aadt_cols):
            E[mi, :, j] = 1.0; j += 1
            for q in range(Zb[mi].shape[1]):
                E[mi, :, j] = Zb[mi][:, q]; j += 1
        self._XA, self._E = jnp.asarray(XA), jnp.asarray(E)
        sr, gr = np.array(site_rand, int), np.array(group_rand, int)
        self._XA_s, self._E_s = jnp.asarray(XA[:, sr]), jnp.asarray(E[:, :, sr])
        self._XA_g, self._E_g = jnp.asarray(XA[:, gr]), jnp.asarray(E[:, :, gr])

        # site-level integration nodes
        mode = self.site_integration
        if mode == "auto":
            mode = "quadrature" if 0 < Ks <= 2 else "simulation"
        if mode == "quadrature" and Ks > 3:
            raise ValueError("quadrature supports at most 3 site-level random parameters")
        self._site_mode = mode if Ks else None
        if Ks and mode == "quadrature":
            n = self.quad_nodes or {1: 16, 2: 8, 3: 6}[Ks]
            self._Vq, self._lwq = _gh_product(Ks, n)
            self._Ssite = self._Vq.shape[1]
        elif Ks:
            S = self.S if Kg else self.R
            self._Ssite = S
            self._Vs = jnp.asarray(_normal_draws(I, Ks, S, self.seed + 11))
        # group-level integration nodes
        gmode = self.group_integration if Kg else None
        if gmode == "quadrature" and Kg > 1:
            raise ValueError("group quadrature supports one group-level random parameter")
        self._group_mode = gmode
        if Kg and gmode == "quadrature":
            self._Uq, self._lwg = _gh_product(Kg, self.quad_nodes or 24)
        elif Kg:
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
    def _ll_rows(self, lnA, expo, ln_disp, const):
        """Row log-likelihoods for any trailing draw shape.
        lnA: (N, ...), expo: (M, N, ...), const: (N,) draw-free NB terms."""
        lnaadt = self._d["lnaadt"]                                   # (N, M)
        extra = (None,) * (lnA.ndim - 1)
        B = jnp.exp(expo) if self.spec.exponent_form == "exp" else expo
        ln_mu = lnA + jnp.sum(B * lnaadt.T[(slice(None), slice(None)) + extra], axis=0)
        ln_mu = jnp.clip(ln_mu, -30.0, 30.0)
        y = self._d["y"][(slice(None),) + extra]
        if self.spec.family == "poisson":
            return y * ln_mu - jnp.exp(ln_mu) + const[(slice(None),) + extra]
        la = jnp.clip(ln_disp, -20.0, 10.0)
        log1p_amu = jnp.logaddexp(0.0, la + ln_mu)
        return const[(slice(None),) + extra] - jnp.exp(-la) * log1p_amu + \
            y * (la + ln_mu - log1p_amu)

    def negloglik(self, p):
        lay = self._layout()
        d = self._d
        theta = p[lay["theta"][0]:lay["theta"][1]]
        ln_disp = p[lay["ln_disp"][0]] if "ln_disp" in lay else 0.0
        I, G = self._I, self._G
        Ks, Kg = len(self._site_rand), len(self._group_rand)
        site_idx = d["site_idx"]
        y = d["y"]
        # draw-invariant pieces, computed once per evaluation
        if self.spec.family == "poisson":
            const = -self._y_lgamma
        else:
            inv = jnp.exp(-jnp.clip(ln_disp, -20.0, 10.0))
            const = gammaln(y + inv) - gammaln(inv) - self._y_lgamma
        lnA0 = self._XA @ theta + d["offset"]                        # (N,)
        expo0 = jnp.einsum("mnp,p->mn", self._E, theta)              # (M,N)

        if Ks == 0 and Kg == 0:
            return -jnp.sum(self._ll_rows(lnA0, expo0, ln_disp, const))

        L = self._site_chol(p) if Ks else None

        def site_loglik(lnA, expo):
            """(I,) log of the site-level integral given (N,) lnA, (M,N) expo."""
            if self._site_mode == "quadrature":
                dev = L @ self._Vq                                     # (Ks,S)
                lnA_s = lnA[:, None] + self._XA_s @ dev                # (N,S)
                expo_s = expo[..., None] + jnp.einsum("mnk,ks->mns", self._E_s, dev)
                ll = self._ll_rows(lnA_s, expo_s, ln_disp, const)      # (N,S)
                per = jax.ops.segment_sum(ll, site_idx, num_segments=I)  # (I,S)
                return logsumexp(per + self._lwq[None, :], axis=1)
            V = self._Vs                                               # (I,Ks,S)
            S = V.shape[-1]
            c = max(1, min(self.draw_chunk, S))
            n_chunks = -(-S // c)
            pad = n_chunks * c - S
            Vp = jnp.concatenate([V, jnp.zeros(V.shape[:2] + (pad,))], axis=-1) if pad else V
            Vc = Vp.reshape(I, Ks, n_chunks, c).transpose(2, 0, 1, 3)  # (C,I,Ks,c)

            def chunk(Vk):
                dev = jnp.einsum("ab,ibs->ias", L, Vk)[site_idx]       # (N,Ks,c)
                lnA_s = lnA[:, None] + jnp.einsum("nk,nks->ns", self._XA_s, dev)
                expo_s = expo[..., None] + jnp.einsum("mnk,nks->mns", self._E_s, dev)
                ll = self._ll_rows(lnA_s, expo_s, ln_disp, const)
                return jax.ops.segment_sum(ll, site_idx, num_segments=I)  # (I,c)
            per = jax.lax.map(jax.checkpoint(chunk), Vc)               # (C,I,c)
            per = per.transpose(1, 0, 2).reshape(I, -1)[:, :S]
            return logsumexp(per, axis=1) - jnp.log(S)

        if Kg == 0:
            return -jnp.sum(site_loglik(lnA0, expo0))

        sd_g = jnp.exp(p[lay["group"][0]:lay["group"][1]])            # (Kg,)
        g_row = d["g_idx_row"]
        if self._group_mode == "quadrature":
            nodes, lw = self._Uq[0], self._lwg                       # (Q,), (Q,)
            draws = jnp.broadcast_to(nodes[None, None, :], (G, 1, nodes.shape[0]))
        else:
            draws, lw = self._Ug, None                               # (G,Kg,R)

        def one_r(u_r):                                               # (G,Kg)
            dev_g = (sd_g[None, :] * u_r)[g_row]                      # (N,Kg)
            lnA = lnA0 + jnp.sum(self._XA_g * dev_g, axis=1)
            expo = expo0 + jnp.einsum("mnk,nk->mn", self._E_g, dev_g)
            if Ks:
                ll_site = site_loglik(lnA, expo)
            else:
                ll_site = jax.ops.segment_sum(
                    self._ll_rows(lnA, expo, ln_disp, const), site_idx, num_segments=I)
            return jax.ops.segment_sum(ll_site, d["g_idx_site"], num_segments=G)
        per_r = jax.lax.map(jax.checkpoint(one_r), draws.transpose(2, 0, 1))  # (R,G)
        if lw is None:
            ll_g = logsumexp(per_r, axis=0) - jnp.log(per_r.shape[0])
        else:
            ll_g = logsumexp(per_r + lw[:, None], axis=0)
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
            start: Optional[np.ndarray] = None, compute_se: bool = True):
        y = self._prepare(df, y_col)
        lay = self._layout()
        p0 = self._start(y) if start is None else np.asarray(start, float)
        if start is None and (self._site_rand or self._group_rand):
            # warm start: fixed-parameter model on the same data
            fixed = HierarchicalSPF(
                HierarchicalSPFSpec(**{**self.spec.__dict__, "random": {}}),
                R=self.R, inner_draws=self.S, seed=self.seed)
            fixed.fit(df, y_col, maxiter=maxiter, compute_se=False)
            p0[:self._P] = fixed.result_["params"][:self._P]
            if "ln_disp" in lay:
                p0[lay["ln_disp"][0]] = fixed.result_["params"][-1]
        res = self._optimize(self.negloglik, p0, maxiter)
        p_hat = np.asarray(res.x, float)
        cov, se = None, np.full(len(p_hat), np.nan)
        is_random = bool(self._site_rand or self._group_rand)
        if compute_se or not is_random:
            hess = jax.jit(jax.hessian(self.negloglik))
            if not is_random:
                # Newton polish (ln a0 / elasticity collinearity). Only for
                # fixed models: each Hessian of a simulated likelihood costs
                # several gradient evaluations, and the polish could need 20.
                p_hat, f_hat = self._newton_polish(p_hat, float(res.fun), hess)
                res.fun = f_hat
            if compute_se:
                H = np.asarray(hess(jnp.asarray(p_hat)))
                H = 0.5 * (H + H.T)
                try:
                    cov = np.linalg.inv(H)
                    se = np.sqrt(np.where(np.diag(cov) > 0, np.diag(cov), np.nan))
                except np.linalg.LinAlgError:
                    cov = None
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



# ======================================================================
# Specification search
# ======================================================================
@dataclass
class SearchResult:
    """Outcome of :class:`HierarchicalSPFSearch`.

    ``best_spec`` / ``best_fit`` are the selected structure and its final
    refit (with standard errors and the full draw count); ``leaderboard``
    lists every evaluated structure, best first.
    """
    best_spec: HierarchicalSPFSpec
    best_fit: HierarchicalSPF
    best_score: float
    leaderboard: pd.DataFrame
    history: List[float]

    def summary(self) -> pd.DataFrame:
        return self.best_fit.summary()

    def describe(self) -> str:
        sp = self.best_spec
        lines = ["ln mu = ln(alpha0)"
                 + "".join(f" + alpha[{x}]*{x}" for x in sp.a_terms)]
        for m in sp.aadt_cols:
            zs = sp.b_terms.get(m, [])
            if sp.exponent_form == "exp":
                expo = f"b0[{m}]" + (" * exp(" + " + ".join(f"beta[{m}|{z}]*{z}" for z in zs) + ")" if zs else "")
            else:
                expo = f"b0[{m}]" + "".join(f" + beta[{m}|{z}]*{z}" for z in zs)
            lines.append(f"        + ({expo}) * ln({m})")
        if sp.random:
            lines.append("random: " + ", ".join(f"{k} ({v})" for k, v in sp.random.items()))
        return "\n".join(lines)


class HierarchicalSPFSearch:
    """Search the component structure of a hierarchical SPF / CMF model.

    Decision per candidate variable: excluded, in Component A (a CMF
    ``exp(alpha x)``), or in the exponent sub-model of one AADT term (a
    CM-Function that reshapes the traffic elasticity, Chen & Persaud 2014
    Eq. 13). Decision per included coefficient (and per base elasticity
    ``b0[AADT]``): fixed, or random at one of ``random_levels``. Optionally
    the exponent form (``"exp"`` / ``"linear"``).

    Scoring is BIC (or AIC) of the maximum (simulated) likelihood fit. With
    ``significance=0.10`` a structure is only admissible if every
    non-intercept coefficient is significant at that level (the paper's
    rule); inadmissible structures are penalised rather than discarded so
    the search can move through them.

    Search: simulated annealing over single-change moves (add / drop /
    move between components / change random level / change form), then a
    greedy best-improvement pass from the incumbent, then a final refit with
    ``final_R`` draws and standard errors.
    """

    def __init__(self, df: pd.DataFrame, y_col: str, aadt_cols: List[str],
                 candidates: List[str], site_col: Optional[str] = None,
                 group_col: Optional[str] = None, offset_col: Optional[str] = None,
                 family: str = "nb2", exponent_forms=("exp",),
                 allow_a: Optional[List[str]] = None,
                 allow_b: Optional[Dict[str, List[str]]] = None,
                 force_a: Optional[List[str]] = None,
                 random_levels=("site",), randomisable: Optional[List[str]] = None,
                 max_random: int = 2, max_terms: Optional[int] = None,
                 criterion: str = "bic", significance: Optional[float] = None,
                 R: int = 100, inner_draws: int = 32, final_R: int = 300,
                 final_inner_draws: int = 100, maxiter: int = 800,
                 verbose: bool = True):
        self.df, self.y_col = df, y_col
        self.aadt_cols = list(aadt_cols)
        self.candidates = list(candidates)
        self.site_col, self.group_col, self.offset_col = site_col, group_col, offset_col
        self.family = family
        self.forms = tuple(exponent_forms)
        self.allow_a = set(self.candidates if allow_a is None else allow_a)
        self.allow_b = {m: set(self.candidates if allow_b is None else allow_b.get(m, []))
                        for m in self.aadt_cols}
        self.force_a = set(force_a or [])
        levels = tuple(random_levels or ())
        if any(l in ("group", "both") for l in levels) and group_col is None:
            raise ValueError("group-level random parameters need group_col")
        if any(l not in _LEVELS for l in levels):
            raise ValueError(f"random_levels must be drawn from {_LEVELS}")
        self.levels = levels
        self.randomisable = None if randomisable is None else set(randomisable)
        self.max_random = int(max_random)
        self.max_terms = max_terms
        if criterion not in ("bic", "aic"):
            raise ValueError("criterion must be 'bic' or 'aic'")
        self.criterion = criterion
        self.significance = significance
        self.R, self.S, self.final_R, self.final_S = R, inner_draws, final_R, final_inner_draws
        self.maxiter = maxiter
        self.verbose = verbose
        self._cache: Dict[tuple, tuple] = {}

    # -- state <-> spec ---------------------------------------------------
    # state = (form, roles tuple, random tuple-of-(label, level) sorted)
    #   role: 0 excluded, 1 Component A, 2+m exponent of aadt_cols[m]
    def _spec(self, state) -> HierarchicalSPFSpec:
        form, roles, rand = state
        a = [v for v, r in zip(self.candidates, roles) if r == 1]
        b = {m: [v for v, r in zip(self.candidates, roles) if r == 2 + j]
             for j, m in enumerate(self.aadt_cols)}
        return HierarchicalSPFSpec(
            aadt_cols=self.aadt_cols, a_terms=a, b_terms={m: z for m, z in b.items() if z},
            exponent_form=form, random=dict(rand), correlated=False,
            family=self.family, site_col=self.site_col, group_col=self.group_col,
            offset_col=self.offset_col)

    def _role_options(self, v):
        opts = [0]
        if v in self.allow_a:
            opts.append(1)
        opts += [2 + j for j, m in enumerate(self.aadt_cols) if v in self.allow_b[m]]
        if v in self.force_a:
            opts = [1]
        return opts

    def _labels_of(self, state):
        return set(self._spec(state).labels())

    def _random_candidates(self, state):
        labs = [l for l in self._spec(state).labels() if l != "ln_alpha0"] + ["ln_alpha0"]
        if self.randomisable is not None:
            labs = [l for l in labs if l in self.randomisable or
                    any(l == f"alpha[{x}]" or l.startswith(f"beta[") and l.endswith(f"|{x}]")
                        for x in self.randomisable)]
        return labs

    def _clean(self, state):
        form, roles, rand = state
        labs = self._labels_of((form, roles, ()))
        rand = tuple(sorted((k, v) for k, v in rand if k in labs))[: self.max_random]
        return (form, tuple(roles), rand)

    # -- scoring ----------------------------------------------------------
    def evaluate(self, state):
        state = self._clean(state)
        if state in self._cache:
            return self._cache[state][0]
        sp = self._spec(state)
        n_terms = len(sp.a_terms) + sum(len(z) for z in sp.b_terms.values())
        if self.max_terms is not None and n_terms > self.max_terms:
            self._cache[state] = (np.inf, None)
            return np.inf
        try:
            m = HierarchicalSPF(sp, R=self.R, inner_draws=self.S)
            m.fit(self.df, self.y_col, maxiter=self.maxiter,
                  compute_se=self.significance is not None)
            r = m.result_
            score = r["bic"] if self.criterion == "bic" else r["aic"]
            if not np.isfinite(score):
                raise FloatingPointError
            if self.significance is not None:
                from scipy.stats import norm as _norm
                crit = _norm.ppf(1 - self.significance / 2)
                names = m.param_names()
                t = np.abs(r["params"] / r["se"])
                bad = [n for n, tv in zip(names, t)
                       if n.startswith(("alpha[", "beta[")) and not (tv >= crit)]
                # each non-significant coefficient costs a large penalty
                score = score + 1e3 * len(bad)
        except Exception:
            score, m = np.inf, None
        self._cache[state] = (float(score), m)
        if self.verbose:
            print(f"  [{len(self._cache):3d}] {self.criterion.upper()}={score:12.2f}  "
                  f"{self._short(state)}")
        return float(score)

    def _short(self, state):
        sp = self._spec(state)
        parts = [f"A={sp.a_terms}"] + [f"B[{m}]={z}" for m, z in sp.b_terms.items()]
        if sp.random:
            parts.append(f"rand={sp.random}")
        if len(self.forms) > 1:
            parts.append(sp.exponent_form)
        return " ".join(parts)

    # -- neighbourhood ----------------------------------------------------
    def _neighbours(self, state):
        form, roles, rand = state
        out = []
        for i, v in enumerate(self.candidates):
            for r in self._role_options(v):
                if r != roles[i]:
                    nr = list(roles); nr[i] = r
                    out.append((form, tuple(nr), rand))
        if self.levels:
            cur = dict(rand)
            for lab in self._random_candidates(state):
                for lvl in (None,) + self.levels:
                    if cur.get(lab) == lvl:
                        continue
                    nd = dict(cur)
                    if lvl is None:
                        nd.pop(lab, None)
                    else:
                        if lab not in nd and len(nd) >= self.max_random:
                            continue
                        nd[lab] = lvl
                    out.append((form, roles, tuple(sorted(nd.items()))))
        for f in self.forms:
            if f != form:
                out.append((f, roles, rand))
        return [self._clean(s) for s in out]

    # -- driver -----------------------------------------------------------
    def run(self, max_iter: int = 60, seed: int = 0, T0: Optional[float] = None,
            cooling: float = 0.93, greedy_polish: bool = True,
            init: Optional[dict] = None) -> SearchResult:
        rng = np.random.default_rng(seed)
        roles = [1 if v in self.force_a else 0 for v in self.candidates]
        if init:
            for v, r in init.items():
                roles[self.candidates.index(v)] = r
        cur = self._clean((self.forms[0], tuple(roles), ()))
        f_cur = self.evaluate(cur)
        best, f_best = cur, f_cur
        T = T0 if T0 is not None else max(10.0, 0.002 * abs(f_cur) if np.isfinite(f_cur) else 10.0)
        history = [f_best]
        for it in range(max_iter):
            nbrs = self._neighbours(cur)
            if not nbrs:
                break
            cand = nbrs[int(rng.integers(len(nbrs)))]
            f_c = self.evaluate(cand)
            if np.isfinite(f_c) and (f_c < f_cur or rng.random() < np.exp(-(f_c - f_cur) / max(T, 1e-9))):
                cur, f_cur = cand, f_c
                if f_c < f_best:
                    best, f_best = cand, f_c
            T *= cooling
            history.append(f_best)
        if greedy_polish:
            improved = True
            while improved:
                improved = False
                for cand in self._neighbours(best):
                    f_c = self.evaluate(cand)
                    if f_c < f_best - 1e-9:
                        best, f_best, improved = cand, f_c, True
                history.append(f_best)
        # final refit with more draws and standard errors
        spec = self._spec(best)
        final = HierarchicalSPF(spec, R=self.final_R, inner_draws=self.final_S)
        final.fit(self.df, self.y_col, maxiter=max(self.maxiter, 2000))
        rows = [dict(score=v[0], structure=self._short(k)) for k, v in self._cache.items()]
        board = pd.DataFrame(rows).sort_values("score").reset_index(drop=True)
        return SearchResult(spec, final, float(f_best), board, history)
