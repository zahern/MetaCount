"""Canonical structural equation for random parameters.

This module is the single reference for how random parameters, heterogeneity
(helping hands), and GSE gradient-score terms enter the metacountregressor
count models.  The JAX engine in :mod:`main_hpc` (:func:`build_eta`,
:func:`unpack_params`, :func:`build_base_index`) and the patched builders in
:mod:`main_hpc_lc_patch` implement exactly this equation; the pure
NumPy/JAX helpers here mirror that implementation so the structure can be
inspected, tested, and re-used without fitting a model.

Structural equation (site ``n``, random coefficient ``k``, simulation draw
``r``) -- independent random parameters::

    mu[n,k]     = beta[k] + sum_j delta[k,j] * z[n,j] + gamma[k] * g[n,k]
    logsd[n,k]  = logsig[k] + sum_l omega[k,l] * w[n,l]
    b[n,k,r]    = T(mu[n,k] + exp(logsd[n,k]) * v[n,k,r]; dist[k])

Correlated random parameters use the same ``mu`` with a Cholesky factor
``L`` (log-diagonal shifted by the variance helpers, as in ``build_eta``)::

    b[n,:,r] = T(mu[n,:] + L_n @ v[n,:,r]; dist)

Notation
--------
``beta``      population mean of each random coefficient.
``z``         helping hands for the *means* (``hetro_in_means``): observed
              covariates that shift the mean of a random parameter.
``delta``     (Kh, K) matrix of mean-heterogeneity loadings (``gamma`` in
              the JAX parameter blocks).
``g``         GSE gradient scores per site and random coefficient
              (standardised, mean 0 / sd 1), following SearchLibrium's
              ``MixedLogitGSE`` idea
              (``beta_nk = mu_k + gamma_k * g_nk + sigma_k * eta_nk``).
              ``None`` disables the term.
``gamma``     (K,) vector of GSE loadings (``gamma_gse`` blocks).
``w``         helping hands for the *variances* (``hetro_in_variances``):
              observed covariates that shift the log-scale of a random
              parameter (``gamma_var`` blocks).
``v``         base simulation draws, standard normal, shape (N, K, R).
``T``         distribution transform per coefficient: normal, lognormal,
              triangular, uniform (codes mirror ``DIST_MAP`` in main_hpc).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

DIST_CODES = {"normal": 0, "lognormal": 1, "triangular": 2, "uniform": 3}
CODE_DISTS = {v: k for k, v in DIST_CODES.items()}

STRUCTURAL_EQUATION_LATEX = (
    r"\beta_{nk} = T\!\left(\beta_k + \boldsymbol{\delta}_k^{\top}\mathbf{z}_n"
    r" + \gamma_k g_{nk} + \exp\!\left(\ln\sigma_k"
    r" + \boldsymbol{\omega}_k^{\top}\mathbf{w}_n\right)\nu_{nk}^{(r)}"
    r"; D_k\right)"
)


def structural_equation_latex() -> str:
    """Return the structural equation as a LaTeX string (for papers/docs)."""
    return STRUCTURAL_EQUATION_LATEX


# ---------------------------------------------------------------------------
# GSE score handling (mirrors SearchLibrium MixedLogitGSE standardisation)
# ---------------------------------------------------------------------------

def standardise_scores(g: np.ndarray) -> np.ndarray:
    """Centre/scale gradient scores to mean 0 / sd 1 per column.

    Constant (zero-variance) columns carry no gradient information and are
    returned as zeros, exactly as in ``MixedLogitGSE.setup``.
    """
    g = np.asarray(g, dtype=float)
    mu = g.mean(axis=0, keepdims=True)
    sd = g.std(axis=0, keepdims=True)
    safe = np.where(sd < 1e-8, 1.0, sd)
    out = (g - mu) / safe
    out[:, (sd < 1e-8).ravel()] = 0.0
    return out


# ---------------------------------------------------------------------------
# Core structural math (NumPy — mirrors build_eta block for block)
# ---------------------------------------------------------------------------

def rp_local_mean(
    base_mean: np.ndarray,
    Z: np.ndarray | None,
    Gamma: np.ndarray | None,
    G: np.ndarray | None = None,
    gamma_gse: np.ndarray | None = None,
) -> np.ndarray:
    """Local means ``mu[n,k]`` of the structural equation (NumPy).

    Parameters match the JAX blocks: ``base_mean`` (K,), ``Z`` (N,Kh),
    ``Gamma`` (Kh,K), ``G`` (N,K) standardised scores, ``gamma_gse`` (K,).
    """
    mu = np.asarray(base_mean, dtype=float).reshape(1, -1)
    if Z is not None and Gamma is not None and np.size(Z) and np.size(Gamma):
        mu = mu + np.asarray(Z, dtype=float) @ np.asarray(Gamma, dtype=float)
    if G is not None and gamma_gse is not None and np.size(G):
        mu = mu + np.asarray(G, dtype=float) * np.asarray(
            gamma_gse, dtype=float
        ).reshape(1, -1)
    return mu


def rp_local_logsd(
    base_logsd: np.ndarray,
    W: np.ndarray | None,
    Omega: np.ndarray | None,
) -> np.ndarray:
    """Local log-scales ``logsd[n,k]`` of the structural equation (NumPy)."""
    ls = np.asarray(base_logsd, dtype=float).reshape(1, -1)
    if W is not None and Omega is not None and np.size(W) and np.size(Omega):
        ls = ls + np.asarray(W, dtype=float) @ np.asarray(Omega, dtype=float)
    return ls


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    from math import erf

    _erf = np.vectorize(erf)
    return 0.5 * (1.0 + _erf(x / np.sqrt(2.0)))


def rp_draws_independent(
    mean: np.ndarray,
    logsd: np.ndarray,
    draws: np.ndarray,
    dists: Sequence[str],
    softplus: bool = True,
) -> np.ndarray:
    """Simulated draws ``b[n,k,r]`` for independent random parameters.

    Mirrors the effective ``transform_draws`` in main_hpc (softplus scale by
    default).  ``mean``/``logsd`` are (K,) or (N,K); ``draws`` is (N,K,R).
    """
    mean = np.asarray(mean, dtype=float)
    logsd = np.asarray(logsd, dtype=float)
    v = np.asarray(draws, dtype=float)
    if mean.ndim == 1:
        mean = mean[None, :, None]
    else:
        mean = mean[:, :, None]
    if logsd.ndim == 1:
        sc = logsd[None, :, None]
    else:
        sc = logsd[:, :, None]
    sc = np.log1p(np.exp(sc)) if softplus else np.exp(sc)
    codes = [DIST_CODES[str(d).lower()] for d in dists]
    out = np.empty_like(v)
    for k, code in enumerate(codes):
        m = mean[:, k, :]
        s = sc[:, k, :]
        z = v[:, k, :]
        if code == 1:  # lognormal
            out[:, k, :] = np.exp(m + s * z)
        elif code == 2:  # triangular
            u = _norm_cdf(z)
            out[:, k, :] = m + s * (2.0 * u - 1.0)
        elif code == 3:  # uniform
            u = _norm_cdf(z) * 2.0 - 1.0
            out[:, k, :] = m + s * u
        else:  # normal
            out[:, k, :] = m + s * z
    return out


def rp_draws_correlated(
    mean: np.ndarray,
    chol: np.ndarray,
    draws: np.ndarray,
    dists: Sequence[str],
) -> np.ndarray:
    """Correlated draws ``b[n,:,r] = T(mu[n,:] + L @ v[n,:,r])`` (NumPy).

    ``chol`` is the lower-triangular Cholesky factor with log-diagonal
    (same convention as ``build_eta``/``random_correlated``); the diagonal
    is exponentiated before use.
    """
    mean = np.asarray(mean, dtype=float)
    L = np.asarray(chol, dtype=float).copy()
    v = np.asarray(draws, dtype=float)
    K = L.shape[0]
    diag = np.diag_indices(K)
    L[diag] = np.exp(np.diag(L))
    zcorr = np.einsum("ij,njr->nir", L, v)
    if mean.ndim == 1:
        mu = np.broadcast_to(mean.reshape(1, K, 1), zcorr.shape).copy()
    else:
        mu = np.broadcast_to(mean[:, :, None], zcorr.shape).copy()
    return _correlated_via_independent(mu, v, zcorr, dists)


def _correlated_via_independent(mu, v, zcorr, dists):
    # Correlated path embeds the Cholesky in zcorr; distribution transforms
    # then apply with the engine's dummy scale, mirroring random_correlated
    # exactly: transform_draws(z_corr, mean, zeros) with softplus(0) = ln 2.
    import numpy as _np

    _LN2 = float(_np.log(2.0))
    N, K, R = zcorr.shape
    codes = [DIST_CODES[str(d).lower()] for d in dists]
    out = _np.empty_like(zcorr)
    for k, code in enumerate(codes):
        m = mu[:, k, :]
        z = zcorr[:, k, :]
        if code == 1:
            out[:, k, :] = _np.exp(m + _LN2 * z)
        elif code == 2:
            u = _norm_cdf(z)
            out[:, k, :] = m + _LN2 * (2.0 * u - 1.0)
        elif code == 3:
            u = _norm_cdf(z) * 2.0 - 1.0
            out[:, k, :] = m + _LN2 * u
        else:
            out[:, k, :] = m + _LN2 * z
    return out


# ---------------------------------------------------------------------------
# JAX variants (lazy import; identical math to build_eta blocks)
# ---------------------------------------------------------------------------

def _jax():
    import jax.numpy as jnp  # type: ignore

    return jnp


def rp_local_mean_jax(base_mean, Z, Gamma, G=None, gamma_gse=None):
    jnp = _jax()
    mu = jnp.asarray(base_mean, dtype=float).reshape(1, -1)
    if Z is not None and Gamma is not None:
        mu = mu + jnp.asarray(Z, dtype=float) @ jnp.asarray(Gamma, dtype=float)
    if G is not None and gamma_gse is not None:
        mu = mu + jnp.asarray(G, dtype=float) * jnp.asarray(
            gamma_gse, dtype=float
        ).reshape(1, -1)
    return mu


def rp_local_logsd_jax(base_logsd, W, Omega):
    jnp = _jax()
    ls = jnp.asarray(base_logsd, dtype=float).reshape(1, -1)
    if W is not None and Omega is not None:
        ls = ls + jnp.asarray(W, dtype=float) @ jnp.asarray(Omega, dtype=float)
    return ls


# ---------------------------------------------------------------------------
# Specification builder (framework wiring)
# ---------------------------------------------------------------------------

def _with_dist(var: str, default: str = "normal") -> str:
    return var if ":" in var else f"{var}:{default}"


@dataclass
class RPStructuralSpec:
    """A random-parameter structure: who is random + which helping hands.

    ``random_vars`` entries are ``"VAR"`` or ``"VAR:dist"`` with dist in
    normal/lognormal/triangular/uniform.  ``mean_helpers`` (z) shift random
    means; ``var_helpers`` (w) shift random log-scales; ``gse`` enables the
    gradient-score loading per random coefficient.
    """

    random_vars: list[str] = field(default_factory=list)
    mean_helpers: list[str] = field(default_factory=list)
    var_helpers: list[str] = field(default_factory=list)
    gse: bool = False
    default_dist: str = "normal"

    def __post_init__(self) -> None:
        for v in list(self.random_vars) + list(self.mean_helpers) + list(
            self.var_helpers
        ):
            base = v.split(":")[0].strip()
            if not base:
                raise ValueError(f"Empty variable name in {v!r}")
        dists = [v.split(":")[1] for v in self.random_vars if ":" in v]
        bad = [d for d in dists if d.lower() not in DIST_CODES]
        if bad:
            raise ValueError(f"Unknown distributions {bad}; use {sorted(DIST_CODES)}")
        if self.default_dist.lower() not in DIST_CODES:
            raise ValueError(f"Unknown default_dist {self.default_dist!r}")

    @property
    def n_random(self) -> int:
        return len(self.random_vars)

    def to_manual_spec(
        self,
        fixed_terms: Sequence[str] | None = None,
        correlated: bool = False,
        dispersion: int = 1,
        latent_classes: int = 1,
        gse_scores: np.ndarray | None = None,
    ) -> dict:
        """Return a manual-spec dict consumable by ``build_model_from_manual_spec``.

        ``gse_scores`` is the (N, K) array of standardised gradient scores
        (see :func:`standardise_scores`), column-aligned with
        ``random_vars`` (correlates first when ``correlated=True``,
        otherwise in listed order).  It is only used when ``gse`` is on;
        a mismatch with ``n_random`` raises ``ValueError``.
        """
        rdm = [_with_dist(v, self.default_dist) for v in self.random_vars]
        spec = {
            "fixed_terms": list(fixed_terms or []),
            "rdm_terms": [] if correlated else rdm,
            "rdm_cor_terms": rdm if correlated else [],
            "grouped_terms": [],
            "hetro_in_means": list(self.mean_helpers),
            "hetro_in_variances": list(self.var_helpers),
            "zi_terms": [],
            "membership_terms": [],
            "dispersion": int(dispersion),
            "latent_classes": int(latent_classes),
        }
        if self.gse:
            if gse_scores is None:
                raise ValueError(
                    "gse is on but no gse_scores array was supplied; pass "
                    "standardised (N, K) scores via standardise_scores()."
                )
            g = np.asarray(gse_scores, dtype=float)
            if g.ndim != 2 or g.shape[1] != self.n_random:
                raise ValueError(
                    f"gse_scores must be (N, K={self.n_random}); got {g.shape}."
                )
            spec["gse_scores"] = g
            spec["gse_random_vars"] = [v.split(":")[0] for v in self.random_vars]
        return spec

    @classmethod
    def from_gse_screen(
        cls,
        gse_df,
        default_dist: str = "normal",
        gse: bool = True,
    ) -> "RPStructuralSpec":
        """Build a structure from ``suggest_random_params_gse`` output.

        The screen's ``candidate`` column becomes the random set and each
        row's ``best_helper`` becomes a mean helper (``'-'`` skipped).
        """
        import pandas as pd  # local import: keeps module dependency-light

        if isinstance(gse_df, pd.DataFrame):
            cands = gse_df["candidate"].astype(str).tolist()
            helpers = (
                gse_df["best_helper"].astype(str).tolist()
                if "best_helper" in gse_df.columns
                else []
            )
        else:  # list of (candidate, helper) pairs
            cands = [str(c) for c, _ in gse_df]
            helpers = [str(h) for _, h in gse_df]
        mean_helpers = sorted({h for h in helpers if h and h != "-"})
        return cls(
            random_vars=list(dict.fromkeys(cands)),
            mean_helpers=mean_helpers,
            var_helpers=[],
            gse=bool(gse),
            default_dist=default_dist,
        )

    def describe(self) -> str:
        lines = [
            "RPStructuralSpec",
            f"  random ({self.n_random}): "
            + (", ".join(self.random_vars) or "-"),
            "  mean helpers (z): " + (", ".join(self.mean_helpers) or "-"),
            "  var helpers (w): " + (", ".join(self.var_helpers) or "-"),
            f"  gse loadings: {'on' if self.gse else 'off'}",
            "  equation: " + STRUCTURAL_EQUATION_LATEX,
        ]
        return "\n".join(lines)
