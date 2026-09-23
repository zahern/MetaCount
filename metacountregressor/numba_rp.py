"""Numba count engine: fixed + RP (independent/correlated/grouped) + ZI + GSE + LC.

Single-class parameter layout (flat vector, mirrors
``main_hpc.build_base_index`` order)::

    [beta_f (Kf),
     beta_zi (Kzi, iff zero-inflated),
     mean_cor (Kc), chol (Kc*(Kc+1)//2, tril row-major, exp on diagonal),
     mean_ind (Kr), sd_raw (Kr, softplus),
     mean_g (Kg), sd_raw_g (Kg, softplus),
     gamma (Kh*Ktot), gamma_var (Kv*Ktot),
     gamma_gse (Ktot, iff GSE scores supplied),
     alpha_raw? (1 iff NB)]

``Ktot = Kc + Kr`` (correlated block first, as in JAX shift slicing).
Hetero shifts are panel-means (mean over P, pads included); the
correlated Cholesky diagonals carry the global-mean variance shift, as in
``main_hpc.build_eta``. The correlated path uses the JAX convention
``transform_draws(z_corr, mean, zeros)`` i.e. fixed scale ``ln 2``.

Deviations from JAX (deliberate, documented):
  * NB dispersion is ``softplus`` everywhere, including the ZI zero-mass
    ``f0`` (JAX's ZI branch uses ``exp``, inconsistent with its own
    ``nb2_loglik``).
  * Grouped coefficients carry no hetero mean/var shifts (JAX's grouped
    var-shift block is dimensionally incoherent; means are unshifted there).
  * ``gse_scores`` rows must follow ``df`` sorted by ``id_col``.

Latent-class layout: ``[theta_1..theta_C, gamma_1..gamma_{C-1}]`` with
jagged per-class outcome blocks (reduced ``Kf_c`` when ``class_fixed``
subsets are given) and ``(1 + len(cols_c))`` membership params per
non-reference class (intercept + slopes), matching ``main_hpc_lc_patch``.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm, qmc

try:
    from numba import njit, prange
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "engine='numba-rp' requires: pip install 'metacountregressor[numba]'"
    ) from exc

DIST_CODES = {"normal": 0, "lognormal": 1, "triangular": 2, "uniform": 3}
CODE_DISTS = {v: k for k, v in DIST_CODES.items()}
_SQRT2 = math.sqrt(2.0)
_LN2 = 0.6931471805599453  # softplus(0): correlated-path fixed scale (JAX convention)


@njit(cache=True)
def _softplus(value):
    if value > 30.0:
        return value
    if value < -30.0:
        return math.exp(value)
    return math.log1p(math.exp(value))


@njit(cache=True)
def _sigmoid(value):
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    e = math.exp(value)
    return e / (1.0 + e)


@njit(cache=True)
def _norm_cdf(z):
    return 0.5 * (1.0 + math.erf(z / _SQRT2))


@njit(cache=True)
def _logaddexp(left, right):
    if left >= right:
        return left + math.log1p(math.exp(right - left))
    return right + math.log1p(math.exp(left - right))


@njit(cache=True, parallel=True)
def _rp_ll_ind_core(params, Xf, Xi, Xc, Xg, Xh_m, Xhv_m, Gm, Xzi,
                    y, mask, offset, draws_i, draws_c, draws_g, gid,
                    dist_i, dist_c, dist_g, has_zi, is_nb, out):
    """Per-individual log-likelihoods (parallel over N, streaming over R).

    All float64 contiguous; ``gid`` int64 (N,); dist codes int64; ``out``
    (N,) receives ``ll_ind`` (logsumexp over R minus log R). Caller sums
    (possibly weighted) for the NLL. Shapes: Xf (N,P,Kf), Xi (N,P,Kr),
    Xc (N,P,Kc), Xg (N,P,Kg), Xh_m (N,Kh), Xhv_m (N,Kv), Gm (N,Ktot-or-0),
    Xzi (N,P,Kzi), y/mask/offset (N,P), draws_* (N,K,R) / (G,Kg,R).
    """
    N = Xf.shape[0]
    P = Xf.shape[1]
    Kf = Xf.shape[2]
    Kr = Xi.shape[2]
    Kc = Xc.shape[2]
    Kg = Xg.shape[2]
    Kh = Xh_m.shape[1]
    Kv = Xhv_m.shape[1]
    Ks = Gm.shape[1]
    Kz = Xzi.shape[2]
    Ktot = Kc + Kr
    if Ktot + Kg == 0:
        R = 1
    elif Kr > 0:
        R = draws_i.shape[2]
    elif Kc > 0:
        R = draws_c.shape[2]
    else:
        R = draws_g.shape[2]

    idx = Kf
    zi_off = idx
    idx += Kz
    mean_cor_off = idx
    idx += Kc
    chol_off = idx
    idx += Kc * (Kc + 1) // 2
    mean_ind_off = idx
    idx += Kr
    sd_off = idx
    idx += Kr
    mean_g_off = idx
    idx += Kg
    sd_g_off = idx
    idx += Kg
    gamma_off = idx
    idx += Kh * Ktot
    gamma_var_off = idx
    idx += Kv * Ktot
    gse_off = idx
    idx += Ktot if Ks > 0 else 0
    alpha_raw = 0.0
    if is_nb:
        alpha_raw = params[idx]

    alpha = 0.0
    inv_alpha = 0.0
    log_inv_alpha = 0.0
    if is_nb:
        alpha = _softplus(alpha_raw)
        inv_alpha = 1.0 / alpha
        log_inv_alpha = math.log(inv_alpha)

    # Correlated-path Cholesky diagonal variance shift (global mean over N).
    diag_shift = np.empty(Kc, dtype=np.float64)
    for k in range(Kc):
        s = 0.0
        for n in range(N):
            inner = 0.0
            for j in range(Kv):
                inner += Xhv_m[n, j] * params[gamma_var_off + j * Ktot + k]
            s += inner
        diag_shift[k] = s / N if N > 0 else 0.0

    logR = math.log(float(R)) if R > 1 else 0.0
    for n in prange(N):
        run_max = -1e300
        run_sum = 0.0
        for r in range(R):
            ll_sum = 0.0
            for p in range(P):
                if mask[n, p] == 0.0:
                    continue
                pi = 0.0
                if has_zi:
                    s = 0.0
                    for k in range(Kz):
                        s += Xzi[n, p, k] * params[zi_off + k]
                    pi = _sigmoid(s)
                eta = offset[n, p]
                for k in range(Kf):
                    eta += Xf[n, p, k] * params[k]
                for k in range(Kc):
                    m = params[mean_cor_off + k]
                    for j in range(Kh):
                        m += Xh_m[n, j] * params[gamma_off + j * Ktot + k]
                    if Ks > 0:
                        m += Gm[n, k] * params[gse_off + k]
                    z = 0.0
                    for j in range(k + 1):
                        flat = k * (k + 1) // 2 + j
                        if j == k:
                            z += math.exp(params[chol_off + flat] + diag_shift[k]) * draws_c[n, j, r]
                        else:
                            z += params[chol_off + flat] * draws_c[n, j, r]
                    code = dist_c[k]
                    if code == 1:
                        b = math.exp(m + _LN2 * z)
                    elif code == 2 or code == 3:
                        u = _norm_cdf(z)
                        b = m + _LN2 * (2.0 * u - 1.0)
                    else:
                        b = m + _LN2 * z
                    eta += Xc[n, p, k] * b
                for k in range(Kr):
                    kk = Kc + k
                    m = params[mean_ind_off + k]
                    s_raw = params[sd_off + k]
                    for j in range(Kh):
                        m += Xh_m[n, j] * params[gamma_off + j * Ktot + kk]
                    for j in range(Kv):
                        s_raw += Xhv_m[n, j] * params[gamma_var_off + j * Ktot + kk]
                    if Ks > 0:
                        m += Gm[n, kk] * params[gse_off + kk]
                    s = _softplus(s_raw)
                    z = draws_i[n, k, r]
                    code = dist_i[k]
                    if code == 1:
                        b = math.exp(m + s * z)
                    elif code == 2 or code == 3:
                        u = _norm_cdf(z)
                        b = m + s * (2.0 * u - 1.0)
                    else:
                        b = m + s * z
                    eta += Xi[n, p, k] * b
                if Kg > 0:
                    g = gid[n]
                    for k in range(Kg):
                        m = params[mean_g_off + k]
                        s = _softplus(params[sd_g_off + k])
                        z = draws_g[g, k, r]
                        code = dist_g[k]
                        if code == 1:
                            b = math.exp(m + s * z)
                        elif code == 2 or code == 3:
                            u = _norm_cdf(z)
                            b = m + s * (2.0 * u - 1.0)
                        else:
                            b = m + s * z
                        eta += Xg[n, p, k] * b
                if eta > 25.0:
                    eta = 25.0
                elif eta < -25.0:
                    eta = -25.0
                ynp = y[n, p]
                if is_nb:
                    log_denom = _logaddexp(log_inv_alpha, eta)
                    ll_c = (
                        math.lgamma(ynp + inv_alpha)
                        - math.lgamma(inv_alpha)
                        - math.lgamma(ynp + 1.0)
                        + inv_alpha * (log_inv_alpha - log_denom)
                        + ynp * (eta - log_denom)
                    )
                    if has_zi:
                        f0 = math.exp(inv_alpha * (log_inv_alpha - log_denom))
                    else:
                        f0 = 0.0
                else:
                    mu = math.exp(eta)
                    ll_c = ynp * eta - mu - math.lgamma(ynp + 1.0)
                    if has_zi:
                        f0 = math.exp(-mu)
                    else:
                        f0 = 0.0
                if has_zi:
                    if ynp == 0.0:
                        ll_sum += math.log(pi + (1.0 - pi) * f0 + 1e-12)
                    else:
                        ll_sum += math.log(1.0 - pi + 1e-12) + ll_c
                else:
                    ll_sum += ll_c
            if R > 1:
                if r == 0:
                    run_max = ll_sum
                    run_sum = 1.0
                elif ll_sum > run_max:
                    run_sum = run_sum * math.exp(run_max - ll_sum) + 1.0
                    run_max = ll_sum
                else:
                    run_sum += math.exp(ll_sum - run_max)
            else:
                run_max = ll_sum
                run_sum = 1.0
        if R > 1:
            out[n] = run_max + math.log(run_sum) - logR
        else:
            out[n] = run_max


def _core_sum(params, arr, is_nb, has_zi=False, weights=None, out=None):
    """NLL from the parallel core (sums, optionally weighted, ``out``)."""
    (Xf, Xi, Xc, Xg, Xh_m, Xhv_m, Gm, Xzi, y, mask, offset,
     draws_i, draws_c, draws_g, gid, dist_i, dist_c, dist_g) = arr
    params = np.ascontiguousarray(params, dtype=np.float64)
    if out is None:
        out = np.empty(Xf.shape[0], dtype=np.float64)
    _rp_ll_ind_core(params, Xf, Xi, Xc, Xg, Xh_m, Xhv_m, Gm, Xzi, y, mask,
                    offset, draws_i, draws_c, draws_g, gid,
                    dist_i, dist_c, dist_g, bool(has_zi), bool(is_nb), out)
    if weights is not None:
        return float(-np.dot(np.asarray(weights, dtype=float), out))
    return float(-np.sum(out))


def _rp_full_nll(params, Xf, Xr_ind, Xr_cor, Xh_mean, Xhv_mean, y, mask,
                 offset, draws_ind, draws_cor, dist_ind, dist_cor, is_nb):
    """Extended likelihood entry point (independent + correlated + hetero).

    ``Xh_mean``/``Xhv_mean`` are (N,Kh)/(N,Kv) panel means. Delegates to the
    parallel core with empty grouped/ZI/GSE blocks.
    """
    N, P = Xf.shape[0], Xf.shape[1]
    arr = (np.ascontiguousarray(Xf), np.ascontiguousarray(Xr_ind),
           np.ascontiguousarray(Xr_cor),
           np.zeros((N, P, 0), dtype=np.float64),
           np.ascontiguousarray(Xh_mean), np.ascontiguousarray(Xhv_mean),
           np.zeros((N, 0), dtype=np.float64),
           np.zeros((N, P, 0), dtype=np.float64),
           np.ascontiguousarray(y), np.ascontiguousarray(mask),
           np.ascontiguousarray(offset), np.ascontiguousarray(draws_ind),
           np.ascontiguousarray(draws_cor),
           np.zeros((1, 0, draws_ind.shape[2] if Xr_ind.shape[2] > 0 else 1),
                    dtype=np.float64),
           np.zeros(N, dtype=np.int64),
           np.ascontiguousarray(dist_ind, dtype=np.int64),
           np.ascontiguousarray(dist_cor, dtype=np.int64),
           np.zeros((0,), dtype=np.int64))
    return _core_sum(params, arr, is_nb)


def _rp_nll(params, Xf, Xr, Xh, Xh_var, y, mask, offset,
            draws, dist_codes, is_nb):
    """Backward-compatible independent-only entry point (Phase-1 signature)."""
    N = Xf.shape[0]
    Kh = Xh.shape[2]
    Kv = Xh_var.shape[2]
    if Kh > 0:
        Xh_mean = np.ascontiguousarray(Xh.mean(axis=1))
    else:
        Xh_mean = np.zeros((N, 0), dtype=np.float64)
    if Kv > 0:
        Xhv_mean = np.ascontiguousarray(Xh_var.mean(axis=1))
    else:
        Xhv_mean = np.zeros((N, 0), dtype=np.float64)
    return _rp_full_nll(params, Xf, Xr,
                        np.zeros((N, Xf.shape[1], 0), dtype=np.float64),
                        Xh_mean, Xhv_mean, y, mask, offset, draws,
                        np.zeros((N, 0, draws.shape[2] if Xr.shape[2] > 0 else 1),
                                 dtype=np.float64),
                        np.ascontiguousarray(dist_codes, dtype=np.int64),
                        np.zeros((0,), dtype=np.int64), is_nb)


# ---------------------------------------------------------------------------
# Python-side helpers (draws, panel, spec)
# ---------------------------------------------------------------------------

def _decode_distribution(code, allowed):
    allowed = list(allowed)
    if len(allowed) == 1:
        return allowed[0]
    return allowed[int(code) % len(allowed)]


def generate_normal_draws(N, Kr, R, seed=42, method="halton"):
    """Return (N, Kr, R) float64 standard-normal draws (matches JAX layout)."""
    if Kr <= 0:
        return np.zeros((N, 0, R), dtype=np.float64)
    total = int(N) * int(R)
    if str(method).lower() == "sobol":
        import math as _math

        sampler = qmc.Sobol(d=int(Kr), scramble=True, seed=int(seed))
        k = _math.ceil(_math.log2(max(total, 2)))
        try:
            u = sampler.random_base2(k)[:total]
        except (AttributeError, TypeError):
            u = sampler.random(total)
    else:
        sampler = qmc.Halton(d=int(Kr), scramble=False, seed=int(seed))
        sampler.fast_forward(50)
        u = sampler.random(total)
    u = np.clip(np.asarray(u, dtype=np.float64), 1e-12, 1 - 1e-12)
    z = norm.ppf(u).reshape(int(N), int(R), int(Kr)).swapaxes(1, 2)
    return np.ascontiguousarray(z, dtype=np.float64)


def _num_col(df, col, n):
    if col not in df.columns:
        return np.zeros(n, dtype=np.float64)
    v = pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    return np.where(np.isfinite(v), v, 0.0)


def _log_softmax_rows(logits):
    m = logits.max(axis=1, keepdims=True)
    return logits - m - np.log(np.exp(logits - m).sum(axis=1, keepdims=True))


def numeric_hessian_se(objective, params, h=1e-5):
    """Ridge-regularised SEs from a central-difference Hessian (JAX-mirror).

    Returns ``(se, cov, hess)``; inversion follows
    ``main_hpc.compute_standard_errors`` (eigendecomposition, ridge
    ``clip(max|ev|*1e-6, 1e-12, 1e-4)``).
    """
    x = np.asarray(params, dtype=float).copy()
    P = x.size
    f0 = float(objective(x))
    if not np.isfinite(f0):
        raise ValueError("objective not finite at params")
    H = np.zeros((P, P), dtype=float)
    fp = np.empty(P)
    fm = np.empty(P)
    for i in range(P):
        d = np.zeros(P)
        d[i] = h
        fp[i] = float(objective(x + d))
        fm[i] = float(objective(x - d))
    H[np.diag_indices(P)] = (fp - 2.0 * f0 + fm) / (h * h)
    for i in range(P):
        for j in range(i + 1, P):
            di = np.zeros(P)
            dj = np.zeros(P)
            di[i] = h
            dj[j] = h
            fpp = float(objective(x + di + dj))
            fpm = float(objective(x + di - dj))
            fmp = float(objective(x - di + dj))
            fmm = float(objective(x - di - dj))
            H[i, j] = H[j, i] = (fpp - fpm - fmp + fmm) / (4.0 * h * h)
    H = np.where(np.isfinite(H), H, 0.0)
    H = 0.5 * (H + H.T)
    try:
        vals, vecs = np.linalg.eigh(H)
    except np.linalg.LinAlgError:
        return np.full(P, np.nan), np.full((P, P), np.nan), H
    max_ev = float(np.max(np.abs(vals)))
    if not np.isfinite(max_ev) or max_ev <= 0:
        return np.full(P, np.nan), np.full((P, P), np.nan), H
    ridge = float(np.clip(max_ev * 1e-6, 1e-12, 1e-4))
    cov = (vecs * (1.0 / (vals + ridge))) @ vecs.T
    var = np.diag(cov)
    var = np.where(var > 0, var, 1.0 / ridge)
    return np.sqrt(var), cov, H


class NumbaRandomCountEvaluator:
    """Numba search evaluator: fixed + RP (ind/cor/grouped) + ZI + GSE (single class)."""

    SUPPORTED_ROLES = (0, 1, 2, 3, 4, 5, 6, 9)

    def __init__(self, df, id_col, y_col, all_variables, allowed_roles,
                 allowed_distributions, mode="single", group_id_col=None,
                 offset_col=None, R=100, draw_method="halton",
                 max_latent_classes=1, seed=42, **_kwargs):
        if mode != "single":
            raise ValueError("numba-rp supports mode='single' only")
        if int(max_latent_classes) != 1:
            raise ValueError("numba-rp supports one latent class only")
        self.id_col = id_col
        self.y_col = y_col
        self.group_id_col = group_id_col
        self.vars = list(all_variables)
        self.mode = mode
        self.offset_col = offset_col
        self.R = int(R)
        self.draw_method = draw_method
        self.seed = int(seed)
        self.allowed_distributions = dict(allowed_distributions or {})
        self.allowed_roles = {}
        for var in self.vars:
            base = list(allowed_roles.get(var, [0, 1, 2]))
            filt = [r for r in base if r in self.SUPPORTED_ROLES]
            self.allowed_roles[var] = filt or [0]
        self.cache = {}
        self._array_cache = {}
        self._fit_cache = {}
        self._last_fit = None
        self.df_train = df.reset_index(drop=True).copy()
        if group_id_col is not None and group_id_col not in self.df_train.columns:
            raise ValueError(f"group_id_col {group_id_col!r} not in dataframe")
        resp = pd.to_numeric(self.df_train[y_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        if np.any(resp < 0):
            raise ValueError("Numba RP evaluator requires non-negative responses")
        self._mean_response = float(np.mean(resp)) if len(resp) else 1.0

    # -- spec --
    def build_spec(self, decision):
        decision = np.asarray(decision, dtype=int).reshape(-1)
        D = len(self.vars)
        if len(decision) < 2 * D + 1:
            return None
        roles = decision[:D]
        dists = decision[D:2 * D]
        use_nb = int(decision[2 * D]) % 2 == 1
        fixed, rdm_ind, rdm_cor, grouped = [], [], [], []
        hetero, hetero_var, zi = [], [], []
        for i, var in enumerate(self.vars):
            role = int(roles[i])
            if role not in self.allowed_roles.get(var, [0]):
                return None
            if role == 0:
                continue
            if role == 1:
                fixed.append(var)
            elif role == 2:
                dist = _decode_distribution(dists[i], self.allowed_distributions.get(var, ["normal"]))
                rdm_ind.append(f"{var}:{dist}")
            elif role == 3:
                dist = _decode_distribution(dists[i], self.allowed_distributions.get(var, ["normal"]))
                rdm_cor.append(f"{var}:{dist}")
            elif role == 4:
                dist = _decode_distribution(dists[i], self.allowed_distributions.get(var, ["normal"]))
                grouped.append(f"{var}:{dist}")
            elif role == 5:
                hetero.append(var)
            elif role == 6:
                zi.append(var)
            elif role == 9:
                hetero_var.append(var)
            else:
                return None
        if len(rdm_cor) == 1:  # mirror JAX demotion
            rdm_ind.extend(rdm_cor)
            rdm_cor = []
        return {
            "fixed_terms": fixed,
            "rdm_terms": rdm_ind,
            "rdm_cor_terms": rdm_cor,
            "grouped_terms": grouped,
            "hetro_in_means": hetero,
            "hetro_in_variances": hetero_var,
            "zi_terms": zi,
            "membership_terms": [],
            "dispersion": 1 if use_nb else 0,
            "latent_classes": 1,
            "group_id_col": self.group_id_col,
        }

    # -- layout (mirrors build_base_index; Ktot excludes grouped) --
    def _layout(self, spec, n_fixed=None):
        Kf = 1 + (len(spec["fixed_terms"]) if n_fixed is None else n_fixed)
        Kc = len(spec.get("rdm_cor_terms", []))
        Kr = len(spec.get("rdm_terms", []))
        Kg = len(spec.get("grouped_terms", []))
        Ktot = Kc + Kr
        Kh = len(spec.get("hetro_in_means", [])) if Ktot else 0
        Kv = len(spec.get("hetro_in_variances", [])) if Ktot else 0
        Kz = len(spec.get("zi_terms", []))
        has_g = bool(spec.get("_has_gse", False)) and Ktot > 0
        Nchol = Kc * (Kc + 1) // 2
        offs = {}
        idx = 0
        offs["beta"] = (idx, idx + Kf)
        idx += Kf
        offs["zi"] = (idx, idx + Kz)
        idx += Kz
        offs["mean_cor"] = (idx, idx + Kc)
        idx += Kc
        offs["chol"] = (idx, idx + Nchol)
        idx += Nchol
        offs["mean_ind"] = (idx, idx + Kr)
        idx += Kr
        offs["sd"] = (idx, idx + Kr)
        idx += Kr
        offs["mean_g"] = (idx, idx + Kg)
        idx += Kg
        offs["sd_g"] = (idx, idx + Kg)
        idx += Kg
        offs["gamma"] = (idx, idx + Kh * Ktot)
        idx += Kh * Ktot
        offs["gamma_var"] = (idx, idx + Kv * Ktot)
        idx += Kv * Ktot
        offs["gse"] = (idx, idx + Ktot) if has_g else (idx, idx)
        idx += Ktot if has_g else 0
        offs["alpha"] = (idx, idx + 1) if spec.get("dispersion", 0) == 1 else (idx, idx)
        idx += int(spec.get("dispersion", 0) == 1)
        return {"Kf": Kf, "Kc": Kc, "Kr": Kr, "Kg": Kg, "Ktot": Ktot,
                "Kh": Kh, "Kv": Kv, "Kz": Kz, "has_g": has_g,
                "Nchol": Nchol, "offs": offs, "total": idx}

    def _n_params(self, spec):
        return self._layout(spec)["total"]

    def _param_names(self, spec):
        lay = self._layout(spec)
        offs = lay["offs"]
        names = [None] * lay["total"]
        for j, c in enumerate(["__INTERCEPT__"] + list(spec["fixed_terms"])):
            names[offs["beta"][0] + j] = f"b_{c}"
        for j, c in enumerate(spec.get("zi_terms", [])):
            names[offs["zi"][0] + j] = f"zi_{c}"
        cor = [t.split(":")[0] for t in spec.get("rdm_cor_terms", [])]
        for j, c in enumerate(cor):
            names[offs["mean_cor"][0] + j] = f"mean_cor_{c}"
        for j in range(lay["Nchol"]):
            names[offs["chol"][0] + j] = f"chol_{j}"
        ind = [t.split(":")[0] for t in spec.get("rdm_terms", [])]
        for j, c in enumerate(ind):
            names[offs["mean_ind"][0] + j] = f"mean_{c}"
            names[offs["sd"][0] + j] = f"sd_raw_{c}"
        grp = [t.split(":")[0] for t in spec.get("grouped_terms", [])]
        for j, c in enumerate(grp):
            names[offs["mean_g"][0] + j] = f"mean_g_{c}"
            names[offs["sd_g"][0] + j] = f"sd_raw_g_{c}"
        rdm_all = cor + ind
        for j, h in enumerate(spec.get("hetro_in_means", [])[:lay["Kh"]]):
            for k, c in enumerate(rdm_all):
                names[offs["gamma"][0] + j * lay["Ktot"] + k] = f"gamma_{h}x{c}"
        for j, h in enumerate(spec.get("hetro_in_variances", [])[:lay["Kv"]]):
            for k, c in enumerate(rdm_all):
                names[offs["gamma_var"][0] + j * lay["Ktot"] + k] = f"gamma_var_{h}x{c}"
        if lay["has_g"]:
            for k, c in enumerate(rdm_all):
                names[offs["gse"][0] + k] = f"gse_{c}"
        if spec.get("dispersion", 0) == 1:
            names[offs["alpha"][0]] = "alpha_raw"
        return names

    # -- arrays --
    def _sorted_panel_ids(self):
        df = self.df_train.sort_values(self.id_col, kind="mergesort").reset_index(drop=True)
        ids = df[self.id_col].to_numpy()
        uniq, counts = np.unique(ids, return_counts=True)
        return df, ids, uniq, counts

    def _build_arrays(self, spec):
        gse = spec.get("gse_scores", None)
        gse_cols = tuple(spec.get("gse_cols", []) or [])
        gse_fp = None
        if gse is not None:
            g = np.asarray(gse, dtype=float)
            gse_fp = ("arr", g.shape, hash(g.tobytes()))
        elif gse_cols:
            gse_fp = ("cols", gse_cols)
        key = (tuple(spec["fixed_terms"]), tuple(spec["rdm_terms"]),
               tuple(spec.get("rdm_cor_terms", [])), tuple(spec.get("grouped_terms", [])),
               tuple(spec["hetro_in_means"]), tuple(spec["hetro_in_variances"]),
               tuple(spec.get("zi_terms", [])), int(spec["dispersion"]), gse_fp)
        if key in self._array_cache:
            return self._array_cache[key]
        df, ids, uniq, counts = self._sorted_panel_ids()
        N = len(uniq)
        P = int(counts.max()) if len(counts) else 0
        fixed = list(spec["fixed_terms"])
        rdm = [t.split(":")[0] for t in spec["rdm_terms"]]
        cor = [t.split(":")[0] for t in spec.get("rdm_cor_terms", [])]
        grp = [t.split(":")[0] for t in spec.get("grouped_terms", [])]
        hetero, hetero_var = spec["hetro_in_means"], spec["hetro_in_variances"]
        zi = list(spec.get("zi_terms", []))
        dists_ind = [t.split(":")[1] if ":" in t else "normal" for t in spec["rdm_terms"]]
        dists_cor = [t.split(":")[1] if ":" in t else "normal" for t in spec.get("rdm_cor_terms", [])]
        dists_g = [t.split(":")[1] if ":" in t else "normal" for t in spec.get("grouped_terms", [])]
        Kf, Kr, Kc, Kg = 1 + len(fixed), len(rdm), len(cor), len(grp)
        Kh, Kv, Kz = len(hetero), len(hetero_var), len(zi)
        Ktot = Kc + Kr
        Xf = np.zeros((N, P, Kf), dtype=np.float64)
        Xr_ind = np.zeros((N, P, Kr), dtype=np.float64)
        Xr_cor = np.zeros((N, P, Kc), dtype=np.float64)
        Xg = np.zeros((N, P, Kg), dtype=np.float64)
        Xh = np.zeros((N, P, Kh), dtype=np.float64)
        Xh_var = np.zeros((N, P, Kv), dtype=np.float64)
        Xzi = np.zeros((N, P, Kz), dtype=np.float64)
        y = np.zeros((N, P), dtype=np.float64)
        mask = np.zeros((N, P), dtype=np.float64)
        offset = np.zeros((N, P), dtype=np.float64)
        id_index = {v: i for i, v in enumerate(uniq)}
        col_cache = {}
        for c in list(fixed) + rdm + cor + grp + hetero + hetero_var + zi:
            col_cache[c] = _num_col(df, c, len(df))
        y_all = _num_col(df, self.y_col, len(df))
        o_all = _num_col(df, self.offset_col, len(df)) if self.offset_col else None
        positions: dict = {}
        for ridx, v in enumerate(ids):
            positions.setdefault(v, []).append(ridx)
        for v, rows in positions.items():
            n = id_index[v]
            for p, ridx in enumerate(rows):
                mask[n, p] = 1.0
                Xf[n, p, 0] = 1.0
                for j, c in enumerate(fixed):
                    Xf[n, p, 1 + j] = col_cache[c][ridx]
                for j, c in enumerate(rdm):
                    Xr_ind[n, p, j] = col_cache[c][ridx]
                for j, c in enumerate(cor):
                    Xr_cor[n, p, j] = col_cache[c][ridx]
                for j, c in enumerate(grp):
                    Xg[n, p, j] = col_cache[c][ridx]
                for j, c in enumerate(hetero):
                    Xh[n, p, j] = col_cache[c][ridx]
                for j, c in enumerate(hetero_var):
                    Xh_var[n, p, j] = col_cache[c][ridx]
                for j, c in enumerate(zi):
                    Xzi[n, p, j] = col_cache[c][ridx]
                y[n, p] = max(y_all[ridx], 0.0)
                if o_all is not None:
                    offset[n, p] = o_all[ridx]
        Xh_mean = np.ascontiguousarray(Xh.mean(axis=1)) if Kh and Ktot else np.zeros((N, 0))
        Xhv_mean = np.ascontiguousarray(Xh_var.mean(axis=1)) if Kv and Ktot else np.zeros((N, 0))
        # GSE scores (N, Ktot), rows follow df sorted by id_col
        if gse is not None:
            Gm = np.ascontiguousarray(np.asarray(gse, dtype=float))
            if Gm.ndim != 2 or Gm.shape[0] != N or Gm.shape[1] != Ktot:
                raise ValueError(f"gse_scores must be (N={N}, Ktot={Ktot}); got {Gm.shape}")
            spec = dict(spec)
            spec["_has_gse"] = True
        elif gse_cols:
            if len(gse_cols) != Ktot:
                raise ValueError(f"gse_cols has {len(gse_cols)} cols but Ktot={Ktot}")
            Gp = np.zeros((N, P, Ktot), dtype=np.float64)
            for j, c in enumerate(gse_cols):
                col = _num_col(df, c, len(df))
                for v, rows in positions.items():
                    n = id_index[v]
                    for p, ridx in enumerate(rows):
                        Gp[n, p, j] = col[ridx]
            Gm = np.ascontiguousarray(Gp.mean(axis=1))
            spec = dict(spec)
            spec["_has_gse"] = True
        else:
            Gm = np.zeros((N, 0), dtype=np.float64)
        # group codes (order of appearance in id-sorted df) + group draws
        if Kg > 0:
            if self.group_id_col is None:
                raise ValueError("Grouped terms require group_id_col")
            gvals = df[self.group_id_col].to_numpy()
            seen: dict = {}
            gid_all = np.zeros(len(df), dtype=np.int64)
            for ridx, gv in enumerate(gvals):
                if gv not in seen:
                    seen[gv] = len(seen)
                gid_all[ridx] = seen[gv]
            G = len(seen)
            gid = np.zeros(N, dtype=np.int64)
            for v, rows in positions.items():
                gid[id_index[v]] = int(gid_all[rows[0]])
            draws_g = generate_normal_draws(G, Kg, self.R, seed=self.seed + 2,
                                            method=self.draw_method)
        else:
            gid = np.zeros(N, dtype=np.int64)
            draws_g = np.zeros((1, 0, 1), dtype=np.float64)
        draws_ind = generate_normal_draws(N, Kr, self.R, seed=self.seed, method=self.draw_method)
        draws_cor = generate_normal_draws(N, Kc, self.R, seed=self.seed + 1, method=self.draw_method)
        dist_ind = np.array([DIST_CODES.get(str(d).lower(), 0) for d in dists_ind], dtype=np.int64)
        dist_cor = np.array([DIST_CODES.get(str(d).lower(), 0) for d in dists_cor], dtype=np.int64)
        dist_g = np.array([DIST_CODES.get(str(d).lower(), 0) for d in dists_g], dtype=np.int64)
        out = (np.ascontiguousarray(Xf), np.ascontiguousarray(Xr_ind),
               np.ascontiguousarray(Xr_cor), np.ascontiguousarray(Xg),
               np.ascontiguousarray(Xh_mean), np.ascontiguousarray(Xhv_mean),
               np.ascontiguousarray(Gm), np.ascontiguousarray(Xzi),
               np.ascontiguousarray(y), np.ascontiguousarray(mask),
               np.ascontiguousarray(offset), np.ascontiguousarray(draws_ind),
               np.ascontiguousarray(draws_cor), np.ascontiguousarray(draws_g),
               np.ascontiguousarray(gid),
               dist_ind, dist_cor, dist_g, spec)
        self._array_cache[key] = out
        return out

    def _init(self, spec):
        lay = self._layout(spec)
        init = np.zeros(lay["total"], dtype=np.float64)
        init[0] = math.log(max(self._mean_response, 1e-6))
        offs = lay["offs"]
        if lay["Kr"]:
            init[offs["mean_ind"][0]:offs["mean_ind"][1]] = 0.0
            init[offs["sd"][0]:offs["sd"][1]] = -1.0
        if spec.get("dispersion", 0) == 1:
            init[offs["alpha"][0]] = 0.541324854612918
        return init

    def _objective_fn(self, arr, spec, weights=None):
        (Xf, Xi, Xc, Xg, Xh_m, Xhv_m, Gm, Xzi, y, mask, offset,
         draws_i, draws_c, draws_g, gid,
         dist_i, dist_c, dist_g, _spec) = arr
        is_nb = spec.get("dispersion", 0) == 1
        has_zi = len(spec.get("zi_terms", [])) > 0
        out = np.empty(Xf.shape[0], dtype=np.float64)
        w = None if weights is None else np.asarray(weights, dtype=float)

        def objective(p):
            p = np.ascontiguousarray(p, dtype=np.float64)
            _rp_ll_ind_core(p, Xf, Xi, Xc, Xg, Xh_m, Xhv_m, Gm, Xzi, y, mask,
                            offset, draws_i, draws_c, draws_g, gid,
                            dist_i, dist_c, dist_g, has_zi, is_nb, out)
            if w is not None:
                return float(-np.dot(w, out))
            return float(-np.sum(out))

        return objective

    # -- fit / fitness --
    def fit_manual(self, spec, maxiter=None, ftol=1e-9, init=None,
                   weights=None, verbose=False):
        """Fit an explicit manual-spec dict (supports gse_scores/gse_cols)."""
        spec = dict(spec)
        arr = self._build_arrays(spec)
        spec = arr[-1]  # _has_gse flag may have been set during build
        is_nb = spec.get("dispersion", 0) == 1
        lay = self._layout(spec)
        if maxiter is None:
            maxiter = 800 if (lay["Kc"] > 0 or lay["Kg"] > 0) else 500
        x0 = self._init(spec) if init is None else np.asarray(init, dtype=float).copy()
        objective = self._objective_fn(arr, spec, weights=weights)
        try:
            result = minimize(objective, x0, method="L-BFGS-B",
                              options={"maxiter": int(maxiter), "ftol": float(ftol)})
            nll = float(result.fun) if np.isfinite(result.fun) else 1e12
            params = np.asarray(result.x, dtype=float)
            success = bool(result.success) or np.isfinite(nll)
            nit = int(getattr(result, "nit", 0) or 0)
            message = str(getattr(result, "message", ""))
        except Exception as exc:  # noqa: BLE001
            nll, params, success, nit, message = 1e12, x0.copy(), False, 0, f"exception: {exc}"
        n_obs = float(max(arr[9].sum(), 1.0))
        bic = 2.0 * nll + x0.size * math.log(n_obs)
        if not np.isfinite(bic):
            bic = 1e12
        res = {"spec": spec, "params": params,
               "param_names": self._param_names(spec),
               "nll": float(nll), "bic": float(bic), "n_obs": n_obs,
               "n_params": int(x0.size), "success": bool(success),
               "nit": nit, "message": message, "is_nb": is_nb,
               "has_zi": len(spec.get("zi_terms", [])) > 0,
               "objective": objective, "arrays": arr}
        self._last_fit = res
        if verbose:
            print(f"  [numba-rp] nll={nll:.3f} bic={bic:.2f} k={x0.size} "
                  f"nit={nit} success={success} msg={message}")
        return res

    def fit(self, decision, maxiter=None, ftol=1e-9, verbose=False):
        """Fit one decision vector; return params/NLL/BIC/convergence dict."""
        decision = np.asarray(decision, dtype=int).reshape(-1)
        key = tuple(decision.tolist())
        lay_key = (key, int(maxiter) if maxiter is not None else None)
        if lay_key in self._fit_cache:
            return self._fit_cache[lay_key]
        spec = self.build_spec(decision)
        if spec is None:
            res = {"spec": None, "params": None, "param_names": [],
                   "nll": 1e12, "bic": 1e12, "n_obs": 0.0,
                   "n_params": 0, "success": False, "nit": 0,
                   "message": "invalid spec (unsupported role)"}
            self._fit_cache[lay_key] = res
            return res
        res = self.fit_manual(spec, maxiter=maxiter, ftol=ftol, verbose=verbose)
        self._fit_cache[lay_key] = res
        return res

    def fitness(self, decision):
        decision = np.asarray(decision, dtype=int).reshape(-1)
        key = tuple(decision.tolist())
        if key in self.cache:
            return self.cache[key]
        res = self.fit(decision)
        self.cache[key] = float(res["bic"])
        return self.cache[key]

    def standard_errors(self, fit_res, h=1e-5):
        """Finite-difference SEs for a fit (mirrors JAX ridge-eigen inversion)."""
        se, cov, H = numeric_hessian_se(fit_res["objective"], fit_res["params"], h=h)
        out = dict(fit_res)
        out["se"] = se
        out["cov"] = cov
        out["hessian"] = H
        return out

    # -- latent classes --
    def _build_lc_arrays(self, lc_spec):
        """Shared panels + per-class fixed slices + membership means."""
        C = int(lc_spec.get("latent_classes", 2))
        if C < 2:
            raise ValueError("latent_classes must be >= 2 for LC fits")
        base = dict(lc_spec)
        base["latent_classes"] = 1
        arr = self._build_arrays(base)
        (Xf, Xi, Xc, Xg, Xh_m, Xhv_m, Gm, Xzi, y, mask, offset,
         draws_i, draws_c, draws_g, gid,
         dist_i, dist_c, dist_g, spec) = arr
        fixed_all = list(spec["fixed_terms"])
        class_fixed = lc_spec.get("class_fixed", None)
        if class_fixed is None:
            class_fixed = [list(fixed_all) for _ in range(C)]
        fidx = {c: j + 1 for j, c in enumerate(fixed_all)}  # +1 for intercept col 0
        Xf_c, Kf_c = [], []
        for c in range(C):
            cols = [0] + [fidx[v] for v in class_fixed[c]]
            Xf_c.append(np.ascontiguousarray(Xf[:, :, cols]))
            Kf_c.append(len(cols))
        mem = list(lc_spec.get("membership_terms", []))
        df, ids, uniq, counts = self._sorted_panel_ids()
        N = len(uniq)
        P = int(counts.max()) if len(counts) else 0
        id_index = {v: i for i, v in enumerate(uniq)}
        positions: dict = {}
        for ridx, v in enumerate(ids):
            positions.setdefault(v, []).append(ridx)
        Xmem = np.zeros((N, P, len(mem)), dtype=np.float64)
        for j, c in enumerate(mem):
            col = _num_col(df, c, len(df))
            for v, rows in positions.items():
                n = id_index[v]
                for p, ridx in enumerate(rows):
                    Xmem[n, p, j] = col[ridx]
        Xmem_m = np.ascontiguousarray(Xmem.mean(axis=1)) if mem else np.zeros((N, 0))
        cm = lc_spec.get("class_membership", None)
        if cm is None:
            cm = [list(mem) for _ in range(C - 1)]
        midx = {c: j for j, c in enumerate(mem)}
        cm_idx = [tuple(midx[v] for v in lst) for lst in cm]
        return {"C": C, "Xf_c": Xf_c, "Kf_c": Kf_c, "class_fixed": class_fixed,
                "shared": (Xi, Xc, Xg, Xh_m, Xhv_m, Gm, Xzi, y, mask, offset,
                           draws_i, draws_c, draws_g, gid,
                           dist_i, dist_c, dist_g),
                "Xmem_m": Xmem_m, "cm_idx": cm_idx, "mem": mem, "spec": spec}

    def _lc_layout(self, lc, spec):
        C = lc["C"]
        K_base, offs = [], []
        idx = 0
        for c in range(C):
            sub = dict(spec)
            sub["fixed_terms"] = lc["class_fixed"][c]
            kb = self._layout(sub)["total"]
            K_base.append(kb)
            offs.append(idx)
            idx += kb
        g_offs, g_len = [], []
        for a in range(C - 1):
            g_offs.append(idx)
            L = 1 + len(lc["cm_idx"][a])
            g_len.append(L)
            idx += L
        return {"K_base": K_base, "offs": offs, "g_offs": g_offs,
                "g_len": g_len, "total": idx}

    def _lc_objective_fn(self, lc, spec, weights=None):
        C = lc["C"]
        lay = self._lc_layout(lc, spec)
        Xf_c = lc["Xf_c"]
        (Xi, Xc, Xg, Xh_m, Xhv_m, Gm, Xzi, y, mask, offset,
         draws_i, draws_c, draws_g, gid, dist_i, dist_c, dist_g) = lc["shared"]
        Xmem_m = lc["Xmem_m"]
        cm_idx = lc["cm_idx"]
        N = y.shape[0]
        is_nb = spec.get("dispersion", 0) == 1
        has_zi = len(spec.get("zi_terms", [])) > 0
        bufs = [np.empty(N, dtype=np.float64) for _ in range(C)]
        llmat = np.empty((N, C), dtype=np.float64)
        w = None if weights is None else np.asarray(weights, dtype=float)

        def sub_spec(c):
            sub = dict(spec)
            sub["fixed_terms"] = lc["class_fixed"][c]
            return sub

        def objective(p):
            p = np.ascontiguousarray(p, dtype=np.float64)
            for c in range(C):
                th = p[lay["offs"][c]:lay["offs"][c] + lay["K_base"][c]]
                _rp_ll_ind_core(th, Xf_c[c], Xi, Xc, Xg, Xh_m, Xhv_m, Gm, Xzi,
                                y, mask, offset, draws_i, draws_c, draws_g, gid,
                                dist_i, dist_c, dist_g, has_zi, is_nb, bufs[c])
                llmat[:, c] = bufs[c]
            logits = np.zeros((N, C - 1), dtype=np.float64)
            for a in range(C - 1):
                g = p[lay["g_offs"][a]:lay["g_offs"][a] + lay["g_len"][a]]
                logits[:, a] = g[0]
                for j, mcol in enumerate(cm_idx[a]):
                    logits[:, a] += Xmem_m[:, mcol] * g[1 + j]
            full = np.concatenate([np.zeros((N, 1)), logits], axis=1)
            log_pi = _log_softmax_rows(full)
            stack = llmat + log_pi
            m = stack.max(axis=1)
            ll_ind = m + np.log(np.exp(stack - m[:, None]).sum(axis=1))
            if w is not None:
                return float(-np.dot(w, ll_ind))
            return float(-np.sum(ll_ind))

        return objective

    def _lc_responsibilities(self, lc, spec, flat):
        C = lc["C"]
        N = lc["shared"][7].shape[0]
        lay = self._lc_layout(lc, spec)
        llmat = np.empty((N, C), dtype=np.float64)
        bufs = [np.empty(N, dtype=np.float64) for _ in range(C)]
        (Xi, Xc, Xg, Xh_m, Xhv_m, Gm, Xzi, y, mask, offset,
         draws_i, draws_c, draws_g, gid, dist_i, dist_c, dist_g) = lc["shared"]
        is_nb = spec.get("dispersion", 0) == 1
        has_zi = len(spec.get("zi_terms", [])) > 0
        for c in range(C):
            th = flat[lay["offs"][c]:lay["offs"][c] + lay["K_base"][c]]
            _rp_ll_ind_core(th, lc["Xf_c"][c], Xi, Xc, Xg, Xh_m, Xhv_m, Gm, Xzi,
                            y, mask, offset, draws_i, draws_c, draws_g, gid,
                            dist_i, dist_c, dist_g, has_zi, is_nb, bufs[c])
            llmat[:, c] = bufs[c]
        logits = np.zeros((N, C - 1), dtype=np.float64)
        for a in range(C - 1):
            g = flat[lay["g_offs"][a]:lay["g_offs"][a] + lay["g_len"][a]]
            logits[:, a] = g[0]
            for j, mcol in enumerate(lc["cm_idx"][a]):
                logits[:, a] += lc["Xmem_m"][:, mcol] * g[1 + j]
        full = np.concatenate([np.zeros((N, 1)), logits], axis=1)
        log_pi = _log_softmax_rows(full)
        stack = llmat + log_pi
        m = stack.max(axis=1, keepdims=True)
        resp = np.exp(stack - m - np.log(np.exp(stack - m).sum(axis=1, keepdims=True)))
        ll_ind = (m[:, 0] + np.log(np.exp(stack - m).sum(axis=1)))
        return resp, ll_ind, log_pi

    def fit_lc(self, lc_spec, maxiter=1000, ftol=1e-9, init=None, verbose=False):
        """Joint MLE for a C-class model with membership covariates."""
        lc = self._build_lc_arrays(lc_spec)
        spec = lc["spec"]
        lay = self._lc_layout(lc, spec)
        if init is None:
            single = self.fit_manual(dict(lc_spec, latent_classes=1,
                                          membership_terms=[],
                                          class_membership=None,
                                          class_fixed=None))
            rng = np.random.default_rng(self.seed)
            parts = []
            full_fixed = list(spec["fixed_terms"])
            for c in range(lc["C"]):
                kb = lay["K_base"][c]
                th = np.zeros(kb)
                # align: intercept + class fixed vars mapped from the full fit
                th[0] = single["params"][0]
                for j, v in enumerate(lc["class_fixed"][c]):
                    if v in full_fixed:
                        th[1 + j] = single["params"][1 + full_fixed.index(v)]
                th = th + rng.normal(0, 0.25, size=kb)
                parts.append(th)
            for a in range(lc["C"] - 1):
                parts.append(np.zeros(lay["g_len"][a]))
            init = np.concatenate(parts)
        objective = self._lc_objective_fn(lc, spec)
        try:
            result = minimize(objective, np.asarray(init, dtype=float),
                              method="L-BFGS-B",
                              options={"maxiter": int(maxiter), "ftol": float(ftol)})
            nll = float(result.fun) if np.isfinite(result.fun) else 1e12
            flat = np.asarray(result.x, dtype=float)
            success = bool(result.success) or np.isfinite(nll)
            message = str(getattr(result, "message", ""))
        except Exception as exc:  # noqa: BLE001
            nll, flat, success, message = 1e12, np.asarray(init, dtype=float), False, f"exception: {exc}"
        n_obs = float(max(lc["shared"][8].sum(), 1.0))
        bic = 2.0 * nll + flat.size * math.log(n_obs)
        resp, _, _ = self._lc_responsibilities(lc, spec, flat)
        res = {"flat": flat, "layout": lay, "lc": lc, "spec": spec,
               "nll": float(nll), "bic": float(bic) if np.isfinite(bic) else 1e12,
               "n_params": int(flat.size), "success": bool(success),
               "message": message, "responsibilities": resp,
               "objective": objective}
        self._last_fit = res
        if verbose:
            print(f"  [numba-rp-lc] C={lc['C']} nll={nll:.3f} bic={res['bic']:.2f} "
                  f"k={flat.size} success={success}")
        return res

    def fit_em(self, lc_spec, max_iter=50, tol=1e-6, verbose=True, polish=True):
        """EM for LC models: warm start, E-step responsibilities, weighted M-steps."""
        lc = self._build_lc_arrays(lc_spec)
        spec = lc["spec"]
        C = lc["C"]
        lay = self._lc_layout(lc, spec)
        single = self.fit_manual(dict(lc_spec, latent_classes=1,
                                      membership_terms=[],
                                      class_membership=None,
                                      class_fixed=None))
        rng = np.random.default_rng(self.seed)
        thetas = []
        base_n = single["params"].size
        for c in range(C):
            kb = lay["K_base"][c]
            th = np.zeros(kb)
            th[:min(kb, base_n)] = single["params"][:min(kb, base_n)]
            thetas.append(th + rng.normal(0, 0.5, size=kb))
        gammas = [np.zeros(L) for L in lay["g_len"]]
        flat = np.concatenate(thetas + gammas)
        prev = np.inf
        trace = []
        Xmem_m = lc["Xmem_m"]
        N = lc["shared"][7].shape[0]
        for it in range(int(max_iter)):
            resp, ll_ind, _ = self._lc_responsibilities(lc, spec, flat)
            nll = float(-np.sum(ll_ind))
            trace.append(nll)
            if verbose:
                print(f"  [em] it={it} nll={nll:.3f}")
            if abs(prev - nll) < tol:
                break
            prev = nll
            # M-step thetas (weighted per-class fits)
            for c in range(C):
                sub = dict(spec)
                sub["fixed_terms"] = lc["class_fixed"][c]
                r = self.fit_manual(sub, init=thetas[c],
                                    weights=resp[:, c])
                thetas[c] = r["params"]
            # M-step gamma (joint MNL cross-entropy)
            g0 = np.concatenate(gammas)

            def mnl(g):
                g = np.asarray(g, dtype=float)
                logits = np.zeros((N, C - 1), dtype=np.float64)
                off = 0
                for a in range(C - 1):
                    L = lay["g_len"][a]
                    ga = g[off:off + L]
                    off += L
                    logits[:, a] = ga[0]
                    for j, mcol in enumerate(lc["cm_idx"][a]):
                        logits[:, a] += Xmem_m[:, mcol] * ga[1 + j]
                full = np.concatenate([np.zeros((N, 1)), logits], axis=1)
                lp = _log_softmax_rows(full)
                return float(-np.sum(resp * lp))

            mr = minimize(mnl, g0, method="L-BFGS-B",
                          options={"maxiter": 500, "ftol": 1e-10})
            g0 = np.asarray(mr.x, dtype=float)
            off = 0
            for a in range(C - 1):
                L = lay["g_len"][a]
                gammas[a] = g0[off:off + L]
                off += L
            flat = np.concatenate(thetas + gammas)
        resp, ll_ind, _ = self._lc_responsibilities(lc, spec, flat)
        nll = float(-np.sum(ll_ind))
        if polish:
            polished = self.fit_lc(lc_spec, init=flat, verbose=verbose)
            if polished["nll"] < nll:
                polished["trace"] = trace
                return polished
        n_obs = float(max(lc["shared"][8].sum(), 1.0))
        bic = 2.0 * nll + flat.size * math.log(n_obs)
        return {"flat": flat, "layout": lay, "lc": lc, "spec": spec,
                "nll": float(nll), "bic": float(bic),
                "n_params": int(flat.size), "success": True,
                "message": "em", "responsibilities": resp, "trace": trace}
