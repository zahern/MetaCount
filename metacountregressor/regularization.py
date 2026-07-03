"""
Ridge-type regularization for count regression under multicollinearity.
=======================================================================

Implements the Poisson Adjusted Ridge-Type Estimator (PARTE) from:

    Alghamdi et al. (2026).
    "A new regularized Poisson regression estimator for effectively
    modelling multicollinear count data." Statistics.
    https://doi.org/10.1080/02331888.2026.2676679

Core formula (eq 23):
    β̂_PARTE = (F + k(1+d)I)⁻¹ (F + kdI) β̂_MLE

where F = X'WX is the Fisher information matrix (= Hessian of the
negative log-likelihood w.r.t. the fixed-effect parameters at the MLE),
and (k, d) are data-adaptive shrinkage parameters (Section 3.3).

In canonical (eigenvector) space this reduces to an element-wise rescaling:
    α̂_j^PARTE = (q_j + k·d) / (q_j + k·(1+d)) · α̂_j^MLE  (α = θ'β)

which is fully differentiable and JIT-compilable.

Everything here is implemented TWICE:
  1. numpy path  — zero JAX dependency, always available.
  2. JAX-native path (`*_jax` suffix) — JIT-compiled, GPU-ready, returns
     jnp arrays.  Requires JAX ≥ 0.4.

Both paths share the same mathematical logic; only the array primitives differ.
The main_hpc entry-points (compute_regularized_estimates) automatically use
the JAX path when JAX is available.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple, NamedTuple
from functools import partial

try:
    import jax
    import jax.numpy as jnp
    _JAX_AVAILABLE = True
except ImportError:
    _JAX_AVAILABLE = False


# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------

class ParteResult(NamedTuple):
    """Outcome of apply_parte_to_fixed_effects / _jax."""
    beta_mle:         np.ndarray      # original MLE fixed-effect coefficients
    beta_parte:       np.ndarray      # PARTE-shrunk coefficients
    k:                float
    d:                float
    se_mle:           np.ndarray      # sqrt(diag(F⁻¹))
    se_parte:         np.ndarray      # PARTE SEs  (eq 25)
    condition_number: float           # max(q)/min(q) of Fisher information
    eigenvalues:      np.ndarray      # raw eigenvalues of F (ascending)
    alpha:            np.ndarray      # canonical coords: α = θ'β_MLE
    variant:          str             # "k1d1" | "k2d2" | "k3d3"


class MulticollinearityReport(NamedTuple):
    condition_number: float
    eigenvalues:      np.ndarray
    vif_approx:       np.ndarray
    severe:           bool            # κ > 1000
    moderate:         bool            # 30 < κ ≤ 1000


# ===========================================================================
# Section A — JAX-native implementation  (JIT-compiled, GPU-ready)
# ===========================================================================

if _JAX_AVAILABLE:

    @partial(jax.jit, static_argnames=("variant",))
    def _select_params_jax(q: "jnp.ndarray", alpha2: "jnp.ndarray",
                           variant: str = "k3d3"):
        """
        Data-adaptive (k, d) selection — eqs 37-41 — fully in JAX.

        Both k and d are scalars returned as rank-0 jnp arrays.
        All conditional masking uses the double-where pattern to prevent
        NaN gradients in the discarded branch (the classic jnp.where gotcha).
        """
        eps = 1e-12
        INF = jnp.array(1e30)

        # eq 37: initial d  — clipped to (0, 1)
        d_init = jnp.min(q * alpha2 / jnp.maximum(1.0 + q * alpha2, eps))
        d_init = jnp.clip(d_init, eps, 1.0 - eps)

        # Denominator denom_j = α²_j q_j − d̂  (may be ≤ 0 for some j)
        denom_raw = alpha2 * q - d_init
        valid = denom_raw > eps

        # double-where: substitute a safe positive value in invalid slots
        # so that 1/denom never produces NaN/Inf even in the discarded branch.
        safe_denom = jnp.where(valid, denom_raw, jnp.ones_like(denom_raw))

        # eq 38: k̂₁ = min_j sqrt(q_j / denom_j)
        k1_cands = jnp.where(valid, jnp.sqrt(q / safe_denom), INF)
        k1 = jnp.min(k1_cands)
        k1 = jnp.where((k1 > 0) & jnp.isfinite(k1), k1,
                       1.0 / jnp.maximum(jnp.max(alpha2), eps))

        # eq 39: k̂₂ = min_j 2q_j / denom_j
        k2_cands = jnp.where(valid, 2.0 * q / safe_denom, INF)
        k2 = jnp.min(k2_cands)
        k2 = jnp.where((k2 > 0) & jnp.isfinite(k2), k2,
                       1.0 / jnp.maximum(jnp.max(alpha2), eps))

        # eq 40: k̂₃ = (1/p) mean_j 1/denom_j  (zero-out invalid entries)
        k3_cands = jnp.where(valid, 1.0 / safe_denom, 0.0)
        k3 = jnp.mean(k3_cands)
        k3 = jnp.where(k3 > 0, k3,
                       1.0 / jnp.maximum(jnp.max(alpha2), eps))

        # variant is static → plain Python if/elif is fine at trace time
        if variant == "k1d1":
            k = k1
        elif variant == "k2d2":
            k = k2
        else:                         # k3d3 (default)
            k = k3

        # eq 41: update d using the chosen k
        max_alpha2 = jnp.max(alpha2)
        d_cands = q * (k * max_alpha2 - 1.0) / jnp.maximum(k, eps)
        d_updated = jnp.min(d_cands)
        d = jnp.where((d_updated > eps) & (d_updated < 1.0 - eps),
                      d_updated, d_init)

        return k, d


    @partial(jax.jit, static_argnames=("variant",))
    def parte_jax(
        beta_f: "jnp.ndarray",
        fisher_block: "jnp.ndarray",
        variant: str = "k3d3",
    ) -> Tuple["jnp.ndarray", "jnp.ndarray", "jnp.ndarray",
               "jnp.ndarray", "jnp.ndarray"]:
        """
        JIT-compiled PARTE transform.

        Parameters
        ----------
        beta_f       : (p,) fixed-effect MLE coefficients
        fisher_block : (p, p) Fisher information matrix for β_f
                       (= positive sub-block of the Hessian of neg-LL at MLE)
        variant      : "k1d1" | "k2d2" | "k3d3"  (static arg)

        Returns
        -------
        beta_parte   : (p,) shrinkage-corrected coefficients
        se_parte     : (p,) PARTE standard errors  (eq 25)
        se_mle       : (p,) MLE standard errors  sqrt(diag(F⁻¹))
        k            : scalar  shrinkage parameter k
        d            : scalar  shrinkage parameter d
        """
        eps = jnp.array(1e-12)

        # Symmetrise and guard against non-finite entries
        F = (fisher_block + fisher_block.T) * 0.5
        F = jnp.where(jnp.isfinite(F), F, 0.0)

        # F = θ Q θ'  (eigh gives ascending eigenvalues)
        q_raw, theta = jnp.linalg.eigh(F)
        q = jnp.maximum(q_raw, eps)

        # Canonical MLE coordinates: α = θ' β̂_MLE
        alpha = theta.T @ beta_f
        alpha2 = alpha ** 2

        # Data-adaptive k, d
        k, d = _select_params_jax(q, alpha2, variant)

        # PARTE transform in canonical space (eq 23):
        # α̂_j^PARTE = (q_j + k·d) / (q_j + k·(1+d)) · α̂_j
        alpha_parte = alpha * (q + k * d) / jnp.maximum(q + k * (1.0 + d), eps)

        # Back to original space
        beta_parte = theta @ alpha_parte

        # MLE covariance diagonal: diag(F⁻¹) = θ diag(1/q) θ'
        mle_cov = (theta * (1.0 / q)) @ theta.T
        se_mle = jnp.sqrt(jnp.maximum(jnp.diag(mle_cov), 0.0))

        # PARTE covariance diagonal (eq 25):
        # Var_j^PARTE = (q_j + k·d)² / [q_j · (q_j + k·(1+d))²]
        num  = (q + k * d) ** 2
        den  = q * jnp.maximum((q + k * (1.0 + d)) ** 2, eps)
        parte_cov = (theta * (num / den)) @ theta.T
        se_parte = jnp.sqrt(jnp.maximum(jnp.diag(parte_cov), 0.0))

        return beta_parte, se_parte, se_mle, k, d


    @partial(jax.jit, static_argnames=("i0", "i1", "variant"))
    def compute_regularized_estimates_jax(
        params:       "jnp.ndarray",
        hessian:      "jnp.ndarray",
        i0:           int,
        i1:           int,
        variant:      str = "k3d3",
    ) -> Tuple["jnp.ndarray", "jnp.ndarray", "jnp.ndarray",
               "jnp.ndarray", "jnp.ndarray"]:
        """
        JIT-compiled PARTE applied to the fixed-effect block [i0:i1] of the
        full parameter vector, leaving all other parameters at their MLE values.

        Parameters
        ----------
        params   : full MLE parameter vector, shape (K,)
        hessian  : Hessian of the *negative* log-likelihood, shape (K, K)
        i0, i1   : static indices defining the fixed-effect block (from param_index)
        variant  : PARTE variant (static)

        Returns
        -------
        params_parte  : (K,) full vector with β_f replaced by PARTE estimate
        se_parte_full : (K,) SEs — PARTE for β_f, ridge-MLE for the rest
        beta_parte    : (i1-i0,) PARTE coefficients for the fixed-effect block
        k, d          : scalar shrinkage parameters
        """
        beta_f = params[i0:i1]
        F_block = hessian[i0:i1, i0:i1]

        beta_parte, se_parte_block, se_mle_block, k, d = parte_jax(
            beta_f, F_block, variant
        )

        # Assemble full params: PARTE for fixed block, MLE elsewhere
        params_parte = params.at[i0:i1].set(beta_parte)

        # Full SE vector: PARTE for fixed block, ridge-MLE for the rest
        se_full = _ridge_se_jax(hessian)
        se_full = se_full.at[i0:i1].set(se_parte_block)

        return params_parte, se_full, beta_parte, k, d


    @jax.jit
    def _ridge_se_jax(H: "jnp.ndarray") -> "jnp.ndarray":
        """Ridge-regularised SE from Fisher information (JAX version)."""
        H = (H + H.T) * 0.5
        H = jnp.where(jnp.isfinite(H), H, 0.0)
        eigvals, eigvecs = jnp.linalg.eigh(H)
        max_ev = jnp.max(jnp.abs(eigvals))
        ridge = jnp.clip(max_ev * 1e-6, 1e-12, 1e-4)
        inv_ev = 1.0 / (eigvals + ridge)
        cov = (eigvecs * inv_ev) @ eigvecs.T
        diag_cov = jnp.diag(cov)
        diag_cov = jnp.where(diag_cov > 0, diag_cov, 1.0 / ridge)
        return jnp.sqrt(diag_cov)


    @partial(jax.jit, static_argnames=("i0", "i1"))
    def condition_number_jax(
        hessian: "jnp.ndarray",
        i0: int,
        i1: int,
    ) -> "jnp.ndarray":
        """Condition number of the Fisher information fixed-effect block."""
        F = hessian[i0:i1, i0:i1]
        F = (F + F.T) * 0.5
        F = jnp.where(jnp.isfinite(F), F, 0.0)
        q = jnp.linalg.eigvalsh(F)
        q = jnp.maximum(q, 1e-12)
        return q[-1] / q[0]


# ===========================================================================
# Section B — numpy fallback implementation  (no JAX dependency)
# ===========================================================================

def _select_params_np(q: np.ndarray, alpha2: np.ndarray,
                      variant: str = "k3d3") -> Tuple[float, float]:
    """Numpy version of data-adaptive (k, d) selection (eqs 37-41)."""
    eps = 1e-12

    d_init = float(np.min(q * alpha2 / np.maximum(1.0 + q * alpha2, eps)))
    d_init = float(np.clip(d_init, eps, 1.0 - eps))

    denom = alpha2 * q - d_init
    valid = denom > eps
    safe_denom = np.where(valid, denom, 1.0)
    fallback_k = 1.0 / max(float(np.max(alpha2)), eps)

    k1_c = np.where(valid, np.sqrt(q / safe_denom), np.inf)
    k1 = float(np.min(k1_c)); k1 = k1 if (np.isfinite(k1) and k1 > 0) else fallback_k

    k2_c = np.where(valid, 2.0 * q / safe_denom, np.inf)
    k2 = float(np.min(k2_c)); k2 = k2 if (np.isfinite(k2) and k2 > 0) else fallback_k

    k3_c = np.where(valid, 1.0 / safe_denom, 0.0)
    k3 = float(np.mean(k3_c)); k3 = k3 if k3 > 0 else fallback_k

    k = {"k1d1": k1, "k2d2": k2}.get(variant, k3)

    max_alpha2 = float(np.max(alpha2))
    d_cands = q * (k * max_alpha2 - 1.0) / max(k, eps)
    d_updated = float(np.min(d_cands))
    d = d_updated if (eps < d_updated < 1.0 - eps) else d_init

    return float(k), float(d)


def apply_parte_to_fixed_effects(
    beta_mle:     np.ndarray,
    fisher_block: np.ndarray,
    variant:      str = "k3d3",
) -> ParteResult:
    """
    Apply PARTE shrinkage to a fixed-effect coefficient vector (numpy).

    Prefer ``parte_jax`` for JIT-compiled, GPU-accelerated execution.

    Parameters
    ----------
    beta_mle    : (p,) MLE fixed-effect coefficients
    fisher_block: (p, p) Fisher information matrix for β_f
    variant     : "k1d1" | "k2d2" | "k3d3"

    Returns
    -------
    ParteResult  (namedtuple — see class definition above)
    """
    beta_mle = np.asarray(beta_mle, dtype=float)
    F = np.asarray(fisher_block, dtype=float)
    F = np.where(np.isfinite(F), F, 0.0)
    F = (F + F.T) * 0.5

    q_raw, theta = np.linalg.eigh(F)
    q = np.maximum(q_raw, 1e-12)

    alpha  = theta.T @ beta_mle
    alpha2 = alpha ** 2

    k, d = _select_params_np(q, alpha2, variant)

    alpha_parte = alpha * (q + k * d) / np.maximum(q + k * (1.0 + d), 1e-12)
    beta_parte  = theta @ alpha_parte

    mle_cov  = (theta * (1.0 / q)) @ theta.T
    se_mle   = np.sqrt(np.maximum(np.diag(mle_cov), 0.0))

    num      = (q + k * d) ** 2
    den      = q * np.maximum((q + k * (1.0 + d)) ** 2, 1e-300)
    parte_cov = (theta * (num / den)) @ theta.T
    se_parte = np.sqrt(np.maximum(np.diag(parte_cov), 0.0))

    cond = float(q[-1]) / float(q[0]) if q[0] > 0 else np.inf

    return ParteResult(
        beta_mle=beta_mle, beta_parte=beta_parte,
        k=k, d=d, se_mle=se_mle, se_parte=se_parte,
        condition_number=cond, eigenvalues=q_raw, alpha=alpha, variant=variant,
    )


def _ridge_se(H: np.ndarray) -> np.ndarray:
    """Ridge-regularised SE from Fisher information (numpy)."""
    H = np.where(np.isfinite(H), H, 0.0)
    H = (H + H.T) * 0.5
    try:
        ev, evec = np.linalg.eigh(H)
    except np.linalg.LinAlgError:
        return np.full(H.shape[0], np.nan)
    max_ev = float(np.max(np.abs(ev)))
    if max_ev <= 0 or not np.isfinite(max_ev):
        return np.full(H.shape[0], np.nan)
    ridge = float(np.clip(max_ev * 1e-6, 1e-12, 1e-4))
    cov = (evec * (1.0 / (ev + ridge))) @ evec.T
    d = np.diag(cov)
    return np.sqrt(np.where(d > 0, d, 1.0 / ridge))


def compute_regularized_estimates(
    params:      np.ndarray,
    hessian_np:  np.ndarray,
    param_index: dict,
    variant:     str = "k3d3",
) -> Tuple[np.ndarray, np.ndarray, Optional[ParteResult]]:
    """
    PARTE shrinkage for the full parameter vector (numpy fallback).

    Returns (params_parte, se_full, ParteResult|None).
    Prefer compute_regularized_estimates_jax for GPU/JIT use.
    """
    params = np.asarray(params, dtype=float)
    H = np.asarray(hessian_np, dtype=float)

    fixed_key = "fixed" if "fixed" in param_index else "beta_f"
    if fixed_key not in param_index:
        return params.copy(), _ridge_se(H), None

    i0, i1 = param_index[fixed_key]
    if (i1 - i0) < 2:
        return params.copy(), _ridge_se(H), None

    result = apply_parte_to_fixed_effects(
        params[i0:i1], H[i0:i1, i0:i1], variant
    )

    params_parte = params.copy()
    params_parte[i0:i1] = result.beta_parte

    se_full = _ridge_se(H)
    se_full[i0:i1] = result.se_parte

    return params_parte, se_full, result


# ===========================================================================
# Section C — diagnostics and comparison helpers  (numpy, no JAX needed)
# ===========================================================================

def multicollinearity_diagnostics(
    hessian_np:    np.ndarray,
    param_index:   dict,
    feature_names: Optional[list] = None,
) -> MulticollinearityReport:
    """
    Condition number and approximate VIFs from the Fisher information
    fixed-effect sub-block.  κ > 30 indicates moderate collinearity;
    κ > 1000 indicates severe collinearity.
    """
    H = np.asarray(hessian_np, dtype=float)
    H = np.where(np.isfinite(H), H, 0.0)
    H = (H + H.T) * 0.5

    fixed_key = "fixed" if "fixed" in param_index else "beta_f"
    block = H[slice(*param_index[fixed_key])] if fixed_key in param_index else H
    if block.ndim == 2:
        q_raw = np.linalg.eigvalsh(block)
    else:
        q_raw = np.linalg.eigvalsh(H)

    q = np.maximum(q_raw, 1e-12)
    q_desc = np.sort(q)[::-1]
    q_max, q_min = float(q_desc[0]), float(q_desc[-1])
    cond = q_max / max(q_min, 1e-12)

    return MulticollinearityReport(
        condition_number=cond,
        eigenvalues=q_desc,
        vif_approx=q_max / np.maximum(q_desc, 1e-12),
        severe=(cond > 1000.0),
        moderate=(30.0 < cond <= 1000.0),
    )


def compare_estimators(
    beta_mle:     np.ndarray,
    fisher_block: np.ndarray,
) -> dict:
    """
    Compare MLE, PRRE (ridge), PLRE (Liu), PMRTE, and all PARTE variants
    on the same (β̂_MLE, F) pair.  Returns a dict of coefficient vectors + SEs.
    """
    beta_mle = np.asarray(beta_mle, dtype=float)
    F = np.asarray(fisher_block, dtype=float)
    F = np.where(np.isfinite(F), F, 0.0); F = (F + F.T) * 0.5

    q_raw, theta = np.linalg.eigh(F)
    q = np.maximum(q_raw, 1e-12)
    alpha = theta.T @ beta_mle; alpha2 = alpha ** 2; eps = 1e-12

    # PRRE  (eq 8): shrink by q/(q+k)
    k_prre = 1.0 / max(float(np.sum(alpha2)), eps)
    beta_prre = theta @ (q / (q + k_prre) * alpha)

    # PLRE  (eq 13): shrink by (q+d)/(q+1)
    d_plre = float(np.clip(
        np.min((alpha2 - 1.0) / (1.0 / q + alpha2 + eps)), 0.0, 1.0 - eps
    ))
    beta_plre = theta @ ((q + d_plre) / (q + 1.0) * alpha)

    # PMRTE (eq 18): shrink by q/(q+k(1+d))
    d_hat = float(np.min(np.clip(
        q * alpha2 / np.maximum(1.0 + q * alpha2, eps), eps, 1.0 - eps
    )))
    k_pmrte = 1.0 / max(float(np.max(alpha2)), eps)
    beta_pmrte = theta @ (q / (q + k_pmrte * (1.0 + d_hat)) * alpha)

    out = {
        "MLE":   {"beta": beta_mle,   "k": 0.0,      "d": 0.0},
        "PRRE":  {"beta": beta_prre,  "k": k_prre,   "d": 0.0},
        "PLRE":  {"beta": beta_plre,  "k": 0.0,      "d": d_plre},
        "PMRTE": {"beta": beta_pmrte, "k": k_pmrte,  "d": d_hat},
    }
    for v in ("k1d1", "k2d2", "k3d3"):
        r = apply_parte_to_fixed_effects(beta_mle, F, variant=v)
        out[f"PARTE_{v}"] = {"beta": r.beta_parte, "k": r.k, "d": r.d, "se": r.se_parte}

    return out
