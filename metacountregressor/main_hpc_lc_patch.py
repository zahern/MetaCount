# =======================================================================
# main_hpc_lc_patch.py
# =======================================================================
#
# Drop this file next to main_hpc.py and import it INSTEAD of importing
# the patched pieces from main_hpc directly.  The module monkey-patches
# the live objects so everything downstream (StructureEvaluatorLC, etc.)
# picks up the changes automatically.
#
# What is added
# ─────────────
# Membership variables: covariates that explain which latent class an
# individual belongs to.  Instead of fixed class-share constants pi_c,
# the class probability for individual n becomes:
#
#   pi_c(n) = softmax( gamma_c · [1, z_n1, …, z_nK] )
#
# where z_nk are the membership covariates for individual n.
#
# Role encoding (extends the existing role scheme)
# ─────────────────────────────────────────────────
#   Role 7 – Membership only
#             Variable enters the class-probability equation (gamma).
#             It has NO effect in the outcome equation.
#             When latent_classes = 1 → treated as excluded (role 0).
#
#   Role 8 – Membership + class-specific outcome
#             Variable enters BOTH the class-probability equation (gamma)
#             AND the outcome equation as a fixed covariate.
#             Because the model has C classes, each class gets its own
#             fixed coefficient for this variable automatically.
#             When latent_classes = 1 → treated as fixed (role 1).
#
# Parameter layout for a C-class model with K_mem membership variables
# ─────────────────────────────────────────────────────────────────────
#   params = [
#       theta_1  (K_base params for class 1's outcome model)
#       theta_2
#       ...
#       theta_C
#       gamma_flat  ((C-1) * (K_mem + 1) params)
#                   gamma_flat.reshape(C-1, K_mem+1)
#                   columns: [intercept, z1, z2, …, zK_mem]
#                   row c: log-odds coefficients for class c+1 vs class 1
#   ]
#
# Backward compatibility
# ─────────────────────
# When K_mem = 0, gamma_flat has shape (C-1, 1) — only an intercept per
# class — which is identical to the previous constant logits vector.
# All existing code that does not specify membership_terms continues to
# work without modification.
#
# HOW TO APPLY
# ─────────────
# At the top of experiment_package.py (or your run script) add:
#
#   import main_hpc_lc_patch   # applies patches and exports new symbols
#
# Then import from this module instead of main_hpc where needed:
#
#   from main_hpc_lc_patch import (
#       ModelSpec, build_param_index, mixed_model_loglik,
#       build_model_from_manual_spec, print_summary_lc
#   )
# =======================================================================

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import pandas as pd
from dataclasses import dataclass, replace
from functools import partial
from scipy import stats as scipy_stats
from jaxopt import LBFGS

# ── Import the rest of main_hpc unchanged ──────────────────────────────
try:
    from . import main_hpc as _hpc  # type: ignore[attr-defined]
    from .main_hpc import (
        build_jax_data,          # extended below
        build_model_from_manual_spec as _orig_build_model,
        parse_manual_spec,
        balance_panel_dataframe,
        extract_offset,
        generate_halton_normal,
        build_base_index,
        CountModel,
        compute_standard_errors,
        decode_distribution,
        poisson_loglik,
        nb2_loglik,
        gaussian_loglik,
        lognormal_loglik,
        tobit_loglik,
        build_eta,
        ensure_3d,
        unpack_params,
        DIST_MAP,
    )
except ImportError:
    import main_hpc as _hpc
    from main_hpc import (
        build_jax_data,          # extended below
        build_model_from_manual_spec as _orig_build_model,
        parse_manual_spec,
        balance_panel_dataframe,
        extract_offset,
        generate_halton_normal,
        build_base_index,
        CountModel,
        compute_standard_errors,
        decode_distribution,
        poisson_loglik,
        nb2_loglik,
        gaussian_loglik,
        lognormal_loglik,
        tobit_loglik,
        build_eta,
        ensure_3d,
        unpack_params,
        DIST_MAP,
    )

jax.config.update("jax_enable_x64", True)


# ═══════════════════════════════════════════════════════════════════════
# 0.  Tobit OLS initialiser
#     Computes OLS starting values from the non-censored observations.
#     Used by fit_manual_model to bypass the Poisson-style prefit for
#     Tobit models (which would give completely wrong starting values).
# ═══════════════════════════════════════════════════════════════════════

def _tobit_ols_init(data: dict, K_base: int) -> np.ndarray:
    """
    Return an initial parameter vector of length K_base for a Tobit model.

    Strategy
    --------
    1. Average the fixed-effect design matrix Xf and outcomes y over
       the panel dimension to get one row per individual.
    2. Fit OLS on the non-censored rows (y > 0).
    3. Estimate sigma from the OLS residuals.
    4. Pack [beta_ols, sigma_raw] where sigma_raw = log(exp(sigma)-1)
       (inverse of softplus so that softplus(sigma_raw) = sigma).
    """
    y_raw = np.array(data["y"])          # (N, P, 1) or (N, P)
    Xf    = np.array(data["Xf"])         # (N, P, Kf)

    if y_raw.ndim == 3:
        y_flat = y_raw[:, :, 0].mean(axis=1)
    else:
        y_flat = y_raw.mean(axis=1)

    X_flat = Xf.mean(axis=1)             # (N, Kf)
    Kf     = X_flat.shape[1]

    nz = y_flat > 0
    if nz.sum() < Kf + 2:
        nz = np.ones(len(y_flat), dtype=bool)

    y_nz = y_flat[nz]
    X_nz = X_flat[nz]

    try:
        beta_ols = np.linalg.lstsq(X_nz, y_nz, rcond=None)[0]
    except Exception:
        beta_ols = np.zeros(Kf)

    resid     = y_nz - X_nz @ beta_ols
    sigma_hat = max(float(resid.std()), 0.1)
    # inverse-softplus so that softplus(sigma_raw) ≈ sigma_hat
    sigma_raw = float(np.log(np.exp(sigma_hat) - 1.0 + 1e-8))

    params = np.zeros(K_base)
    params[:Kf]        = beta_ols
    params[K_base - 1] = sigma_raw       # sigma is always the last param

    return params


# ═══════════════════════════════════════════════════════════════════════
# 1.  EXTENDED ModelSpec
#     Adds membership_names and K_membership.
#     Replaces the dataclass in main_hpc.
# ═══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ModelSpec:
    Kf:                  int
    Kr_ind:              int
    Kr_cor:              int
    Kg:                  int
    Kh:                  int
    Kzi:                 int
    model:               str
    zero_inflated:       bool
    fixed_names:         tuple
    zi_names:            tuple
    random_ind_names:    tuple
    random_cor_names:    tuple
    grouped_names:       tuple
    hetro_names:         tuple
    random_ind_dists:    tuple
    random_cor_dists:    tuple
    grouped_dists:       tuple
    latent_classes:      int   = 1
    # ── NEW ──────────────────────────────────────────────────────────
    membership_names:    tuple = ()   # variables in class-prob equation
    K_membership:        int   = 0    # len(membership_names)

    @property
    def K_random_total(self):
        return self.Kr_cor + self.Kr_ind


# Monkey-patch so the rest of main_hpc sees the new class
_hpc.ModelSpec = ModelSpec


# ═══════════════════════════════════════════════════════════════════════
# 2.  build_param_index
#     LC tail is now (C-1)*(K_mem+1) instead of (C-1).
# ═══════════════════════════════════════════════════════════════════════

def build_param_index(spec: ModelSpec) -> dict:

    if spec.latent_classes == 1:
        return build_base_index(spec)

    base_spec  = replace(spec, latent_classes=1)
    base_index = build_base_index(base_spec)
    K_base     = base_index["total_params"]
    C          = spec.latent_classes
    K_mem      = spec.K_membership           # NEW

    index = {}
    idx   = 0

    index["class_params"]  = (0, C * K_base)
    idx = C * K_base

    # (C-1) rows × (K_mem+1) cols  [intercept + membership vars]
    gamma_size = (C - 1) * (K_mem + 1)
    index["class_gamma"]   = (idx, idx + gamma_size)    # NEW name
    idx += gamma_size

    index["K_base"]        = K_base
    index["K_mem"]         = K_mem                      # NEW
    index["total_params"]  = idx

    return index


_hpc.build_param_index = build_param_index


# ═══════════════════════════════════════════════════════════════════════
# 3.  Extended build_jax_data
#     Adds membership_cols parameter; appends Xmem to the data dict.
# ═══════════════════════════════════════════════════════════════════════

def build_jax_data(
    df,
    id_col,
    y_col,
    group_id_col=None,
    fixed_cols=None,
    random_ind_cols=None,
    random_cor_cols=None,
    grouped_cols=None,
    hetro_cols=None,
    offset_col=None,
    draws_ind=None,
    draws_cor=None,
    draws_g=None,
    random_ind_dists=None,
    random_cor_dists=None,
    grouped_dists=None,
    zi_cols=None,
    membership_cols=None,   # ← NEW
    R=200,
):
    fixed_cols        = fixed_cols        or []
    random_ind_cols   = random_ind_cols   or []
    random_cor_cols   = random_cor_cols   or []
    grouped_cols      = grouped_cols      or []
    hetro_cols        = hetro_cols        or []
    zi_cols           = zi_cols           or []
    membership_cols   = membership_cols   or []          # NEW
    random_ind_dists  = random_ind_dists  or []
    random_cor_dists  = random_cor_dists  or []
    grouped_dists     = grouped_dists     or []

    intercept_name = "__INTERCEPT__"
    df = df.copy()
    df[intercept_name] = 1.0

    all_features = list(set(
        [intercept_name]
        + fixed_cols + random_ind_cols + random_cor_cols
        + grouped_cols + hetro_cols + zi_cols + membership_cols   # NEW
    ))

    X_all, y, mask = balance_panel_dataframe(df, id_col, y_col, all_features)

    # Group IDs
    if group_id_col is not None and len(grouped_cols) > 0:
        df_sorted   = df.sort_values(id_col)
        group_codes = df_sorted[group_id_col].astype("category").cat.codes.values
        G           = len(np.unique(group_codes))
    else:
        group_codes = None
        G           = 0

    col_map = {col: i for i, col in enumerate(all_features)}

    def extract(cols):
        if len(cols) == 0:
            return np.zeros((X_all.shape[0], X_all.shape[1], 0))
        idx = [col_map[c] for c in cols if c in col_map]
        return X_all[:, :, idx]

    fixed_cols_with_intercept = [intercept_name] + fixed_cols
    Xf   = np.concatenate([extract([intercept_name]), extract(fixed_cols)], axis=2)
    Xr_ind = extract(random_ind_cols)
    Xr_cor = extract(random_cor_cols)
    Xg     = extract(grouped_cols)
    Xh     = extract(hetro_cols)
    Xzi    = extract(zi_cols)
    Xmem   = extract(membership_cols)    # NEW  shape (N, P, K_mem)

    N, P = y.shape[0], y.shape[1]

    if offset_col:
        offset = extract_offset(df, id_col, offset_col)
    else:
        offset = np.zeros((N, P, 1))

    data = {
        "Xf":       jnp.array(Xf),
        "Xr_ind":   jnp.array(Xr_ind),
        "Xr_cor":   jnp.array(Xr_cor),
        "Xg":       jnp.array(Xg),
        "Xh":       jnp.array(Xh),
        "Xzi":      jnp.array(Xzi),
        "Xmem":     jnp.array(Xmem),    # NEW
        "y":        jnp.array(y),
        "mask":     jnp.array(mask),
        "offset":   jnp.array(offset),
        "draws_ind":jnp.zeros((N, 0, R)) if draws_ind is None else jnp.array(draws_ind),
        "draws_cor":jnp.zeros((N, 0, R)) if draws_cor is None else jnp.array(draws_cor),
        "draws_g":  jnp.zeros((N, 0, R)) if draws_g   is None else jnp.array(draws_g),
        "group_ids":jnp.array(group_codes) if group_codes is not None
                    else jnp.zeros(N, dtype=int),
    }

    spec = ModelSpec(
        Kf=Xf.shape[2],
        Kr_ind=Xr_ind.shape[2],
        Kr_cor=Xr_cor.shape[2],
        Kg=Xg.shape[2],
        Kh=Xh.shape[2],
        zi_names=tuple(zi_cols),
        Kzi=Xzi.shape[2],
        zero_inflated=(len(zi_cols) > 0),
        model="poisson",
        fixed_names=tuple(fixed_cols_with_intercept),
        random_ind_names=tuple(random_ind_cols),
        random_cor_names=tuple(random_cor_cols),
        grouped_names=tuple(grouped_cols),
        hetro_names=tuple(hetro_cols),
        random_ind_dists=tuple(random_ind_dists),
        random_cor_dists=tuple(random_cor_dists),
        grouped_dists=tuple(grouped_dists),
        membership_names=tuple(membership_cols),    # NEW
        K_membership=Xmem.shape[2],                # NEW
    )

    return data, spec


_hpc.build_jax_data = build_jax_data


# ═══════════════════════════════════════════════════════════════════════
# 4.  Extended parse_manual_spec
#     Handles "membership_terms" key in the spec dict.
# ═══════════════════════════════════════════════════════════════════════

def parse_manual_spec(manual_spec: dict):
    fixed_cols        = manual_spec.get("fixed_terms", [])
    rdm_terms         = manual_spec.get("rdm_terms", [])
    rdm_cor_terms     = manual_spec.get("rdm_cor_terms", [])
    grouped_terms     = manual_spec.get("grouped_terms", [])
    hetro_terms       = manual_spec.get("hetro_in_means", [])
    zi_cols           = manual_spec.get("zi_terms", [])
    membership_cols   = manual_spec.get("membership_terms", [])   # NEW

    random_ind       = [t.split(":")[0] for t in rdm_terms]
    random_cor       = [t.split(":")[0] for t in rdm_cor_terms]
    grouped_cols     = [t.split(":")[0] for t in grouped_terms]
    hetro_cols       = [t.split(":")[0].strip() for t in hetro_terms]

    random_ind_dists  = [t.split(":")[1] for t in rdm_terms]
    random_cor_dists  = [t.split(":")[1] for t in rdm_cor_terms]
    grouped_dists     = [t.split(":")[1] for t in grouped_terms]

    return (
        fixed_cols, random_ind, random_cor, grouped_cols, hetro_cols,
        random_ind_dists, random_cor_dists, grouped_dists,
        zi_cols, membership_cols,           # NEW: membership_cols returned
    )


_hpc.parse_manual_spec = parse_manual_spec


# ═══════════════════════════════════════════════════════════════════════
# 5.  Extended build_model_from_manual_spec
#     Passes membership_cols to build_jax_data.
# ═══════════════════════════════════════════════════════════════════════

def build_model_from_manual_spec(
    df, manual_spec, id_col, y_col,
    offset_col=None, draws_ind=None, draws_cor=None, draws_g=None, R=200
):
    (
        fixed_cols, random_ind, random_cor, grouped_cols, hetro_cols,
        random_ind_dists, random_cor_dists, grouped_dists,
        zi_cols, membership_cols,
    ) = parse_manual_spec(manual_spec)

    data, spec = build_jax_data(
        df=df,
        id_col=id_col,
        y_col=y_col,
        group_id_col=manual_spec.get("group_id_col", None),
        fixed_cols=fixed_cols,
        random_ind_cols=random_ind,
        random_cor_cols=random_cor,
        grouped_cols=grouped_cols,
        hetro_cols=hetro_cols,
        zi_cols=zi_cols,
        membership_cols=membership_cols,    # NEW
        offset_col=offset_col,
        draws_ind=draws_ind,
        draws_cor=draws_cor,
        draws_g=draws_g,
        random_ind_dists=random_ind_dists,
        random_cor_dists=random_cor_dists,
        grouped_dists=grouped_dists,
        R=R,
    )

    model_type = "nb" if manual_spec.get("dispersion", 0) else "poisson"
    lc         = int(manual_spec.get("latent_classes", 1))
    spec       = replace(spec, model=model_type, latent_classes=lc)

    return data, spec


_hpc.build_model_from_manual_spec = build_model_from_manual_spec


# ═══════════════════════════════════════════════════════════════════════
# 6.  Mixed-model log-likelihood with membership covariates
#
#     The LC branch is the only section changed.  The single-class path
#     is identical to main_hpc so we keep it intact.
#
#     Class-probability model (NEW)
#     ─────────────────────────────
#     gamma  : (C-1, K_mem+1)   row c = log-odds coefficients for class c+1
#     Z_full : (N, K_mem+1)     col 0 = 1 (intercept), cols 1..K = members
#
#     log pi_i = log_softmax( Z_full @ gamma.T , axis=1 )   shape (N, C)
#
#     When K_mem=0, Z_full = ones(N,1) and gamma is just the (C-1) scalar
#     logits — backward-compatible with the old constant-pi behaviour.
# ═══════════════════════════════════════════════════════════════════════

@partial(jax.jit, static_argnames=("spec", "indivi"))
def mixed_model_loglik(params, data, spec: ModelSpec, indivi: bool = False):

    # ── LATENT CLASS BRANCH ─────────────────────────────────────────
    if spec.latent_classes > 1:

        C         = spec.latent_classes
        K_mem     = spec.K_membership
        base_spec = replace(spec, latent_classes=1)
        base_idx  = build_base_index(base_spec)
        K_base    = base_idx["total_params"]

        # Class-specific outcome parameters
        theta_all  = params[:C * K_base].reshape(C, K_base)

        # Membership gamma: (C-1) × (K_mem+1)
        gamma_size = (C - 1) * (K_mem + 1)
        gamma      = params[C * K_base : C * K_base + gamma_size
                            ].reshape(C - 1, K_mem + 1)

        # Build Z_full (N, K_mem+1)
        N = data["y"].shape[0]
        if K_mem > 0:
            # Average membership covariates across panel periods
            Xmem   = data["Xmem"]                          # (N, P, K_mem)
            Z      = jnp.mean(Xmem, axis=1)                # (N, K_mem)
            Z_full = jnp.concatenate(
                [jnp.ones((N, 1)), Z], axis=1
            )                                               # (N, K_mem+1)
        else:
            Z_full = jnp.ones((N, 1))                       # (N, 1) — intercept only

        # Individual-specific log class probabilities
        logits_i    = Z_full @ gamma.T                      # (N, C-1)
        logits_full = jnp.concatenate(
            [jnp.zeros((N, 1)), logits_i], axis=1
        )                                                   # (N, C)
        log_pi      = jax.nn.log_softmax(logits_full, axis=1)  # (N, C)

        # Per-class individual log-likelihoods
        ll_classes = []
        for c in range(C):
            ll_c = mixed_model_loglik(
                theta_all[c], data, base_spec, indivi=True
            )                                               # (N,) log-likelihoods (negative)
            ll_classes.append(ll_c + log_pi[:, c])

        ll_stack = jnp.stack(ll_classes, axis=1)            # (N, C)
        ll_ind   = jsp.special.logsumexp(ll_stack, axis=1)  # (N,)

        if indivi:
            return ll_ind
        return -jnp.sum(ll_ind)

    # ── SINGLE-CLASS BRANCH (unchanged from main_hpc) ───────────────
    blocks = unpack_params(params, spec)
    eta    = build_eta(params, data, spec)
    # Linear-predictor models work in data scale — wider clip to preserve gradients.
    if spec.model in {"gaussian", "tobit"}:
        eta = jnp.clip(eta, -500.0, 500.0)
    else:
        eta = jnp.clip(eta, -25.0, 25.0)

    if eta.ndim == 2:
        eta = eta[..., None]

    y    = ensure_3d(data["y"])
    mask = ensure_3d(data["mask"])
    R    = eta.shape[-1]

    if spec.model == "poisson":
        mu       = jnp.exp(eta)
        ll_count = poisson_loglik(y, mu)

    elif spec.model == "nb":
        alpha    = blocks["alpha"]
        ll_count = nb2_loglik(y, eta, alpha)

    elif spec.model == "lognormal":
        sigma    = blocks["sigma"]
        ll_count = lognormal_loglik(y, eta, sigma)
    elif spec.model == "gaussian":
        sigma    = blocks["sigma"]
        ll_count = gaussian_loglik(y, eta, sigma)
    elif spec.model == "tobit":
        # Left-censored at 0; eta is the linear predictor for the latent Y*
        sigma    = blocks["sigma"]
        ll_count = tobit_loglik(y, eta, sigma)
    else:
        raise ValueError(f"Unknown model: {spec.model}")

    if spec.zero_inflated:
        if spec.Kzi > 0:
            eta_zi = jnp.einsum(
                "npk,k->np", data["Xzi"], blocks["beta_zi"]
            )[..., None]
        else:
            eta_zi = jnp.zeros_like(eta[..., :1])

        pi_zi = jax.nn.sigmoid(eta_zi)
        mu = jnp.exp(eta)

        if spec.model == "poisson":
            f0 = jnp.exp(-mu)
        elif spec.model == "nb":
            alpha_e  = jnp.exp(blocks["alpha"])
            inv_a    = 1.0 / alpha_e
            f0       = jnp.exp(inv_a * (jnp.log(inv_a) - jnp.log(inv_a + mu)))
        elif spec.model == "lognormal":
            f0 = jnp.zeros_like(mu)
        elif spec.model == "gaussian":
            sigma = jax.nn.softplus(blocks["sigma"])
            f0 = jnp.exp(-0.5 * jnp.log(2 * jnp.pi * sigma**2) - (eta**2) / (2 * sigma**2))
        else:
            raise ValueError(f"Unknown zero-inflated model: {spec.model}")

        zero_mask = (y == 0)
        ll_zero   = jnp.log(pi_zi + (1 - pi_zi) * f0 + 1e-12)
        ll_pos    = jnp.log(1 - pi_zi + 1e-12) + ll_count
        ll        = jnp.where(zero_mask, ll_zero, ll_pos)
    else:
        ll = ll_count

    ll       = ll * mask
    ll_panel = jnp.sum(ll, axis=1)

    if R > 1:
        ll_ind = jsp.special.logsumexp(ll_panel, axis=-1) - jnp.log(R)
    else:
        ll_ind = ll_panel.squeeze(-1)

    if indivi:
        return ll_ind
    return -jnp.sum(ll_ind)


_hpc.mixed_model_loglik = mixed_model_loglik


# ═══════════════════════════════════════════════════════════════════════
# 7.  fit_em — EM algorithm aware of membership covariates
#
#     The original fit_em in main_hpc.py treated params[C*K_base:] as a
#     flat (C-1,) logits vector.  With membership variables the gamma
#     section has shape (C-1, K_mem+1) — one intercept + K_mem slopes
#     per class-pair.  This replacement:
#       • E-step: computes individual-specific log_pi via Z_full @ gamma.T
#       • M-step (gamma): minimises the weighted MNL cross-entropy
#       • M-step (theta): unchanged (weighted outcome log-lik per class)
#     When K_mem=0 the behaviour is identical to the original.
# ═══════════════════════════════════════════════════════════════════════

def fit_em(init_params, data, spec: ModelSpec,
           max_iter=100, tol=1e-6, verbose=True):

    from jax.nn import log_softmax as _log_softmax

    assert spec.latent_classes > 1, "EM only needed for latent classes"

    C          = spec.latent_classes
    K_mem      = spec.K_membership
    base_spec  = replace(spec, latent_classes=1)
    base_index = build_base_index(base_spec)
    K_base     = base_index["total_params"]
    gamma_size = (C - 1) * (K_mem + 1)

    params = np.array(init_params)
    N = int(np.array(
        mixed_model_loglik(params[:K_base], data, base_spec, indivi=True)
    ).shape[0])

    # Build membership design matrix Z_full (N, K_mem+1) — fixed for all iters
    if K_mem > 0:
        Xmem   = np.array(data["Xmem"])            # (N, P, K_mem)
        Z      = np.mean(Xmem, axis=1)              # (N, K_mem)
        Z_full = np.concatenate(
            [np.ones((N, 1)), Z], axis=1
        )                                           # (N, K_mem+1)
    else:
        Z_full = np.ones((N, 1))                    # (N, 1) — intercept only

    for iteration in range(max_iter):

        params_old = params.copy()

        # ==========================================================
        # E-STEP
        # ==========================================================

        theta_all = params[:C * K_base].reshape(C, K_base)
        gamma     = params[C * K_base:].reshape(C - 1, K_mem + 1)

        # Individual-specific log class probabilities  (N, C)
        logits_i    = Z_full @ gamma.T                          # (N, C-1)
        logits_full = np.concatenate(
            [np.zeros((N, 1)), logits_i], axis=1
        )                                                       # (N, C)
        log_pi = _log_softmax(logits_full, axis=1)              # (N, C)

        # Per-class individual log-likelihoods  (N, C)
        logL = np.zeros((N, C))
        for c in range(C):
            ll_ind = mixed_model_loglik(
                theta_all[c], data, base_spec, indivi=True
            )
            logL[:, c] = np.array(ll_ind)

        log_num = logL + log_pi                                 # (N, C)

        # Posterior class membership weights
        max_log = log_num.max(axis=1, keepdims=True)
        w = np.exp(log_num - max_log)
        w /= w.sum(axis=1, keepdims=True)

        # Collapse guard: if any class captures < 2% of total weight the
        # solution has degenerated — stop early rather than waste M-step
        # budget pushing classes further apart.
        mean_w = w.mean(axis=0)                     # (C,)
        if np.any(mean_w < 0.02):
            if verbose:
                print(f"  [EM] class collapse detected at iter {iteration} "
                      f"(min mean weight {mean_w.min():.4f}) — stopping early")
            break

        # ==========================================================
        # M-STEP
        # ==========================================================

        # ✅ Update class-specific outcome parameters
        theta_new = []
        for c in range(C):
            wc = w[:, c].copy()

            def weighted_objective(theta_c, _wc=wc):
                ll_ind = mixed_model_loglik(
                    theta_c, data, base_spec, indivi=True
                )
                return -jnp.sum(jnp.array(_wc) * jnp.array(ll_ind))

            solver_theta = LBFGS(fun=weighted_objective, maxiter=300)
            result = solver_theta.run(jnp.array(theta_all[c]))
            theta_new.append(np.array(result.params))

        theta_new = np.concatenate(theta_new)

        # ✅ Update gamma (membership / class-probability coefficients)
        def gamma_objective(gamma_flat, _w=w, _Zf=Z_full):
            gc = gamma_flat.reshape(C - 1, K_mem + 1)
            li = _Zf @ gc.T                                     # (N, C-1)
            lf = np.concatenate([np.zeros((N, 1)), li], axis=1)
            lp = _log_softmax(lf, axis=1)                       # (N, C)
            return -jnp.sum(jnp.array(_w) * lp)

        solver_gamma = LBFGS(fun=gamma_objective, maxiter=300)
        result_gamma = solver_gamma.run(jnp.array(gamma.flatten()))
        gamma_new = np.array(result_gamma.params)

        params = np.concatenate([theta_new, gamma_new])

        # ==========================================================
        # Convergence Check
        # ==========================================================

        diff = np.max(np.abs(params - params_old))

        if verbose:
            total_ll = float(mixed_model_loglik(params, data, spec))
            print(f"EM iter {iteration:3d} | max Δ = {diff:.3e} | LL = {-total_ll:.6f}")

        if diff < tol:
            if verbose:
                print(f"\n✅ EM converged in {iteration} iterations\n")
            break

    return params


_hpc.fit_em = fit_em


# ═══════════════════════════════════════════════════════════════════════
# 7b. _seed_classes_from_clusters
#
#     Given a single-class warm-start theta_1, cluster observations in
#     the space of (fixed covariates, per-obs LL) and fit a per-cluster
#     weighted model.  Returns C genuinely differentiated theta vectors,
#     which prevents EM from collapsing all classes onto the same solution.
# ═══════════════════════════════════════════════════════════════════════

def _seed_classes_from_clusters(
    theta_1: np.ndarray,
    data: dict,
    base_spec: ModelSpec,
    C: int,
    K_base: int,
    rng: np.random.Generator,
) -> list:
    """
    Returns a list of C numpy arrays (each shape (K_base,)) to use as
    per-class starting parameters.

    Strategy
    --------
    1. Compute per-observation log-likelihoods under the single-class fit.
    2. Build a feature matrix from the mean fixed-effect covariates Xf
       and the individual LLs.
    3. K-means cluster observations into C groups.
    4. For each cluster, shift only the intercept (params[0]) by the
       log-ratio of cluster mean outcome to overall mean outcome.
       All other parameters stay at theta_1 + small noise.
       This is numerically stable and avoids per-cluster LBFGS divergence.
    """
    from sklearn.cluster import KMeans

    ll_ind = np.array(
        mixed_model_loglik(theta_1, data, base_spec, indivi=True)
    )                                                       # (N,)

    Xf = np.array(data["Xf"])                              # (N, P, Kf)
    if Xf.ndim == 3:
        Xf_mean = Xf.mean(axis=1)                          # (N, Kf)
    else:
        Xf_mean = Xf

    features = np.concatenate([Xf_mean, ll_ind[:, None]], axis=1)
    col_std  = features.std(0) + 1e-8
    features_scaled = (features - features.mean(0)) / col_std

    try:
        km = KMeans(
            n_clusters=C,
            n_init=10,
            max_iter=300,
            random_state=int(rng.integers(2**31)),
        )
        labels = km.fit_predict(features_scaled)
    except Exception:
        labels = np.arange(features_scaled.shape[0]) % C

    # Overall mean outcome (collapse panel dimension first)
    y_all = np.array(data["y"])                            # (N, P, 1) or (N,)
    if y_all.ndim == 3:
        y_all = y_all.mean(axis=1).squeeze(-1)             # (N,)
    elif y_all.ndim == 2:
        y_all = y_all.mean(axis=1)
    y_global_mean = float(np.maximum(y_all.mean(), 1e-3))

    thetas = []
    for c in range(C):
        in_cluster = labels == c
        n_c = int(in_cluster.sum())

        theta_c = theta_1.copy()

        if n_c >= 3:
            y_c_mean = float(np.maximum(y_all[in_cluster].mean(), 1e-3))
            # Shift intercept so the class predicts y_c_mean at the
            # global covariate average — prevents EM collapse.
            delta_intercept = np.log(y_c_mean) - np.log(y_global_mean)
            theta_c[0] = theta_1[0] + delta_intercept
        else:
            theta_c[0] = theta_1[0] + rng.normal(0, 0.3)

        # Small noise on all other structural parameters (not dispersion)
        n_struct = min(K_base - 1, base_spec.Kf - 1)
        if n_struct > 0:
            theta_c[1:1 + n_struct] += rng.normal(0, 0.05, n_struct)

        thetas.append(theta_c)

    return thetas


# ═══════════════════════════════════════════════════════════════════════
# 8.  print_summary — extended with membership gamma section
# ═══════════════════════════════════════════════════════════════════════

def print_summary(result, objective, data, spec: ModelSpec,
                  param_index, se=None, return_df=None):
    """
    Full model summary.  For LC models with membership variables the
    gamma matrix is printed as a proper table (one column per membership
    variable, one row per class).
    """

    # ── LC DISPATCH ─────────────────────────────────────────────────
    if spec.latent_classes > 1:
        C         = spec.latent_classes
        K_mem     = spec.K_membership
        base_spec = replace(spec, latent_classes=1)
        base_idx  = build_base_index(base_spec)
        K_base    = base_idx["total_params"]

        params_np = np.asarray(result.params if hasattr(result, "params")
                               else result.x)

        if se is None:
            se_np = np.asarray(compute_standard_errors(params_np, objective))
        else:
            se_np = np.asarray(se)

        theta_all = params_np[:C * K_base].reshape(C, K_base)
        se_all    = se_np[:C * K_base].reshape(C, K_base)

        gamma_flat = params_np[C * K_base:]
        se_gamma   = se_np[C * K_base:]
        gamma      = gamma_flat.reshape(C - 1, K_mem + 1)
        se_g       = se_gamma.reshape(C - 1, K_mem + 1)

        logits_full = np.concatenate([[0.0], gamma[:, 0]])
        pi          = np.exp(logits_full) / np.exp(logits_full).sum()

        print("\n" + "=" * 65)
        print("   LATENT CLASS MIXED MODEL SUMMARY")
        if K_mem > 0:
            print(f"   Membership covariates: {list(spec.membership_names)}")
        print("=" * 65)

        # ── Per-class outcome params ──────────────────────────────
        class DummyRes:
            pass

        for c in range(C):
            print(f"\n{'#' * 20}  CLASS {c+1}  (pi = {pi[c]:.4f})  {'#' * 20}\n")
            dummy       = DummyRes()
            dummy.params = theta_all[c]
            dummy.x      = theta_all[c]

            obj_c = partial(
                mixed_model_loglik, data=data,
                spec=replace(base_spec, latent_classes=1)
            )
            print_summary(
                result=dummy, objective=obj_c, data=data,
                spec=replace(base_spec, latent_classes=1),
                param_index=base_idx, se=se_all[c]
            )

        # ── Membership gamma ─────────────────────────────────────
        print("\n" + "=" * 65)
        print("   CLASS-MEMBERSHIP EQUATION")
        print("   log[pi_c(n) / pi_1(n)] = g_c0  +  sum_k g_ck * z_nk")
        print("   (Class 1 is the reference; all coefficients vs class 1)")
        print("=" * 65)

        mem_cols = ["(intercept)"] + list(spec.membership_names)

        # Header
        col_w    = max(16, max(len(c) for c in mem_cols) + 2)
        hdr      = f"  {'':>{col_w}}"
        for c in range(1, C):
            hdr += f"  {'Class '+str(c+1)+' coef':>14}  {'SE':>8}  {'z':>7}  {'p':>7}"
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))

        for k, col_name in enumerate(mem_cols):
            row = f"  {col_name:>{col_w}}"
            for c in range(C - 1):
                g     = gamma[c, k]
                sg    = se_g[c, k]
                z_val = g / sg if sg > 1e-12 else 0.0
                p_val = 2 * (1 - scipy_stats.norm.cdf(abs(z_val)))
                stars = "***" if p_val < 0.01 else "**" if p_val < 0.05 \
                        else "*" if p_val < 0.10 else ""
                row  += f"  {g:>+14.4f}  {sg:>8.4f}  {z_val:>7.3f}  {p_val:>7.4f}{stars}"
            print(row)

        print(f"\n  NOTE: g_c0 is the class-{c+1} log-odds intercept vs class 1.")
        if K_mem > 0:
            print("  g_ck > 0: higher value of z_k -> higher probability of class c+1.")

        # ── Class shares ─────────────────────────────────────────
        print("\n" + "-" * 65)
        print("  MARGINAL CLASS PROBABILITIES (at sample-mean covariates)\n")
        for c in range(C):
            print(f"  pi_{c+1} = {pi[c]:.6f}")

        print("\n" + "=" * 65 + "\n")

        # ── Build summary dict for LC models ─────────────────────
        lc_ll = float(-objective(params_np))
        lc_k  = len(params_np)
        lc_n  = data["y"].shape[0]
        return {
            "loglik":    lc_ll,
            "num_parm":  lc_k,
            "n_obs":     lc_n,
            "aic":       2 * lc_k - 2 * lc_ll,
            "bic":       lc_k * np.log(lc_n) - 2 * lc_ll,
            "latent_classes": C,
            "class_probs":    pi.tolist(),
        }

    # ── SINGLE-CLASS SUMMARY (delegate to main_hpc's print_summary) ─
    _orig_print = _hpc.__dict__.get("_orig_print_summary")
    if _orig_print is None:
        # Fall back: minimal inline summary
        params_np = np.asarray(result.params if hasattr(result, "params")
                               else result.x)
        if se is None:
            se_np = np.asarray(compute_standard_errors(params_np, objective))
        else:
            se_np = np.asarray(se)

        z_vals = params_np / np.where(se_np > 1e-12, se_np, 1e-12)
        p_vals = 2 * (1 - scipy_stats.norm.cdf(np.abs(z_vals)))

        final_ll = float(-objective(params_np))
        k, n     = len(params_np), data["y"].shape[0]

        df_out = pd.DataFrame({
            "Estimate": params_np,
            "Std.Err":  se_np,
            "z-value":  z_vals,
            "p-value":  p_vals,
        })

        print("\n================ MODEL SUMMARY ================\n")
        print(df_out.to_string(float_format="%.4f"))
        print(f"\nLog-Likelihood: {final_ll:.4f}")
        print(f"AIC: {2*k - 2*final_ll:.4f}")
        print(f"BIC: {k*np.log(n) - 2*final_ll:.4f}\n")

        return {
            "loglik":   final_ll,
            "num_parm": k,
            "n_obs":    n,
            "aic":      2 * k - 2 * final_ll,
            "bic":      k * np.log(n) - 2 * final_ll,
        }
    else:
        _orig_print(result, objective, data, spec, param_index, se=se)
        params_np = np.asarray(result.params if hasattr(result, "params")
                               else result.x)
        final_ll = float(-objective(params_np))
        k, n = len(params_np), data["y"].shape[0]
        return {
            "loglik":   final_ll,
            "num_parm": k,
            "n_obs":    n,
            "aic":      2 * k - 2 * final_ll,
            "bic":      k * np.log(n) - 2 * final_ll,
        }


# Keep the original print_summary available under a private alias so the
# LC dispatcher can delegate back to it.
_hpc._orig_print_summary = _hpc.print_summary
_hpc.print_summary       = print_summary


# ═══════════════════════════════════════════════════════════════════════
# Public re-exports for experiment_package.py
# ═══════════════════════════════════════════════════════════════════════

__all__ = [
    "ModelSpec",
    "build_param_index",
    "build_jax_data",
    "build_model_from_manual_spec",
    "parse_manual_spec",
    "mixed_model_loglik",
    "fit_em",
    "print_summary",
]
